"""Descriptor-pinned filesystem staging: open once, then never trust a name again.

Every function here exists because guarding a path by NAME does not guard the open
that follows it. A name is validated, then re-opened, and in the window between the
two anything running as this user -- which in this product includes an agent -- can
swap the final component, or an ancestor DIRECTORY, for a link pointing somewhere
else. The validated path and the opened inode are then not the same thing.

The discipline is one sentence: resolve once, open once, and address everything
downstream through the descriptor you already hold. A descriptor cannot be
re-pointed, so a component that is open is fixed; a component reached by name is
not.

Why this module exists rather than a check at each call site: two closed pull
requests (#2446, #2447) tried to add snapshot components while hardening staging in
the same change. Each review round named one more validated-by-name path use --
source root, each ancestor, each file, the destination tree, the pre-restore backup
pass -- and neither converged. The mechanism belongs in one place with one set of
invariants, and callers become thin consumers of it.

The mechanical half of this was first written for the benchmark harness in
``kiro_crew.eval.bench.safepath``, which now imports it from here. Its invariants
survived several rounds of review there and are preserved verbatim; the docstrings
explaining WHY each flag is load-bearing came with them.

Two things this module deliberately does NOT do:

* It holds no policy. It does not know which locations are protected, and it does
  not decide whether a refusal should abort a command or be reported and skipped.
  Callers pass their own refusal type (so an existing CLI error contract does not
  change) and their own ``on_skip`` reporter (so user-facing wording stays theirs).
* It does not silently substitute a weaker mechanism. Where a platform cannot pin
  (see :func:`supports_pinned_walk`), the caller is told so and decides; nothing
  here falls back to a by-name walk on its own.
"""

from __future__ import annotations

import errno
import os
import shutil
import stat as _stat
from pathlib import Path, PurePath
from typing import Callable

__all__ = [
    "PinnedPathRefusal",
    "SKIP_NOT_REGULAR",
    "SKIP_SYMLINK",
    "SKIP_VANISHED",
    "SkipReporter",
    "copy_file_pinned",
    "create_and_open_dir_pinned",
    "fatal_skip_reporter",
    "is_reparse_point",
    "is_regular_at",
    "stat_at",
    "open_dir_pinned",
    "open_in_pinned_parent",
    "pin_parent",
    "refuse_hardlink_alias",
    "stage_tree_pinned",
    "supports_pinned_tree_walk",
    "supports_pinned_walk",
]


class PinnedPathRefusal(Exception):
    """Raised instead of completing an operation that could not be pinned.

    Neutral on purpose. A caller with its own refusal taxonomy passes that type as
    ``refusal=`` so its existing error contract is unchanged -- the benchmark
    harness keeps raising its ``UnsafePathError``, and a snapshot refusal stays
    something the CLI boundary already knows how to contain.
    """


#: Reason codes handed to an ``on_skip`` reporter. The primitive classifies; the
#: caller words the message, so user-facing output stays the caller's own.
SKIP_SYMLINK = "symlink"
SKIP_VANISHED = "vanished"
SKIP_NOT_REGULAR = "not_regular"

#: ``(reason_code, by_name_path)``. The path is for the message only -- it is never
#: re-opened, because re-opening it is the bug this module exists to prevent.
SkipReporter = Callable[[str, str], None]


def _noop_skip(_reason: str, _path: str) -> None:
    return None


def fatal_skip_reporter(what: str, *, refusal: type[Exception] = PinnedPathRefusal) -> SkipReporter:
    """A reporter that REFUSES instead of recording, for paths where a skip loses data.

    Skipping is the right answer while producing an archive: the entry is omitted, the
    omission is recorded, and nothing of the operator's is touched. It is the wrong answer
    on every path that has already moved or deleted the original -- there the skip means
    the live copy is gone AND the replacement was never written, so the operation
    "succeeds" having destroyed data.

    That distinction cost three separate review findings on this change (a backup pass
    whose skips preceded an ``rmtree``, a restore source skipped after the live file was
    moved aside, and a destination subtree that could not be opened). They were three
    instances of one rule, so the rule is now a parameter a caller passes rather than a
    condition each site re-implements: archive paths keep the recording reporter, mutating
    paths pass this one, and which kind a call site is becomes visible at the call site.
    """

    def _refuse(reason: str, path: str) -> None:
        raise refusal(
            f"refusing to continue the {what}: {Path(path).name!r} could not be copied "
            f"({reason}). This path has already moved or removed what it is replacing, so "
            "skipping would finish with the original gone and the replacement missing. "
            "Resolve that entry -- a hardlink alias or a symbolic link is the usual cause "
            "-- and re-run."
        )

    return _refuse


def supports_pinned_walk() -> bool:
    """Whether this platform can open relative to a directory descriptor.

    ``O_NOFOLLOW`` is part of the requirement, not an extra: a pinned walk without
    it would open each ancestor happily through whatever link sits there, which is
    the hole being closed. Found by the Windows-simulation tests, which delete
    ``os.O_NOFOLLOW`` and would otherwise have taken this path and crashed.
    """
    return (
        hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW") and os.open in os.supports_dir_fd
    )


def supports_pinned_tree_walk() -> bool:
    """Whether a whole TREE can be walked without ever re-opening a name.

    Strictly more than :func:`supports_pinned_walk`: descending a tree also needs to
    list and stat through a descriptor. Without ``os.listdir`` on an fd the walk would
    have to re-list by name, which reintroduces exactly the ancestor swap the pinned
    open just refused.

    Note which stat is probed. ``os.lstat`` is NOT a member of
    ``os.supports_dir_fd`` even on Linux -- the capability belongs to ``os.stat``, and
    ``lstat(p, dir_fd=fd)`` is documented as ``stat(p, dir_fd=fd,
    follow_symlinks=False)``. Probing ``os.lstat`` reports False on a platform that
    fully supports the walk, which would have made every snapshot on Linux refuse and
    demand the by-name opt-in. The walk below calls ``os.stat`` with
    ``follow_symlinks=False`` so the call and the probe are the same function.
    """
    return supports_pinned_walk() and os.listdir in os.supports_fd and os.stat in os.supports_dir_fd


def _dir_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def pin_parent(
    resolved_parent: str,
    *,
    what: str,
    refusal: type[Exception] = PinnedPathRefusal,
) -> int:
    """Return a descriptor for *resolved_parent*, refusing a component that is now a link.

    One ``openat`` per component, each relative to the previous component's
    descriptor and each carrying ``O_NOFOLLOW``. Two properties come out of that:

    * a component that became a symlink after *resolved_parent* was computed fails
      ``O_NOFOLLOW`` and is refused -- this is the check-to-use swap, and it is the
      reason a single ``os.open(parent, O_DIRECTORY)`` is not enough: that call
      follows such a link silently and then pins its target;
    * once a component is open, its descriptor cannot be re-pointed, so everything
      already traversed is fixed.

    *resolved_parent* must be resolved by the CALLER, once, before this runs.
    Resolving it here would re-follow whatever an ancestor points at by now, which
    is the exact mistake that made an earlier version of this defensible-looking and
    useless.

    The descriptor is returned OPEN and the caller must close it. Handing it back
    rather than doing one open inside is what lets a durable write create its
    temporary file and rename it over the destination through the same pinned
    directory, so the swap cannot be redirected between the two steps.

    Not closed: a component swapped BEFORE *resolved_parent* was computed is
    followed by that resolution. Refusing every symlinked ancestor would close it
    and would also break paths under ``/tmp`` on macOS, where ``/tmp`` is itself a
    link.
    """
    parts = PurePath(resolved_parent).parts
    if not parts:  # pragma: no cover - a resolved path always has parts
        raise refusal(f"refusing to open the {what}: empty parent path")

    if os.path.isabs(resolved_parent):
        dir_fd = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY)
        rest = parts[1:]
    else:  # pragma: no cover - realpath returns absolute paths
        dir_fd = os.open(".", os.O_RDONLY | os.O_DIRECTORY)
        rest = parts

    try:
        for component in rest:
            try:
                nxt = os.open(component, _dir_flags(), dir_fd=dir_fd)
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise refusal(
                        f"refusing to write the {what}: the directory {component!r} on "
                        "the way to it became a symbolic link after the path was "
                        "checked. A parent swapped for a link redirects the write "
                        "however carefully the final name is opened, so it is refused."
                    ) from exc
                raise
            os.close(dir_fd)
            dir_fd = nxt
    except BaseException:
        os.close(dir_fd)
        raise
    return dir_fd


def open_in_pinned_parent(
    resolved_parent: str,
    name: str,
    *,
    flags: int,
    mode: int,
    what: str,
    refusal: type[Exception] = PinnedPathRefusal,
) -> int:
    """Open *name* under *resolved_parent* with the parent chain pinned.

    *name* is opened as given, so a link at the final name is refused by
    ``O_NOFOLLOW`` in *flags*. See :func:`pin_parent` for what pinning buys.
    """
    dir_fd = pin_parent(resolved_parent, what=what, refusal=refusal)
    try:
        return os.open(name, flags, mode, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


def open_dir_pinned(
    path: str | Path,
    *,
    what: str,
    refusal: type[Exception] = PinnedPathRefusal,
) -> int:
    """Open a DIRECTORY with its whole ancestor chain pinned, final component included.

    This is the one the preserved staging branches did not have, and its absence is
    the finding that closed #2446: ``os.open(str(src), O_DIRECTORY | O_NOFOLLOW)``
    refuses a link at the root's own name but reaches that name by walking every
    ancestor BY NAME, so swapping a validated ancestor for a link to a credential
    directory redirects the whole traversal and the ``O_NOFOLLOW`` on the final
    component never fires -- what it finds there is a perfectly ordinary directory.

    Here the parent chain is resolved once and pinned component by component, and
    the root's own name is then opened relative to the pinned parent. Nothing in the
    subtree is ever addressed by a path again.
    """
    as_given = Path(path)
    resolved_parent = os.path.realpath(as_given.parent or Path("."))
    try:
        return open_in_pinned_parent(
            resolved_parent,
            as_given.name,
            flags=_dir_flags(),
            mode=0o700,
            what=what,
            refusal=refusal,
        )
    except OSError as exc:
        # Same translation as `create_and_open_dir_pinned`. Review found this sibling
        # still leaking the raw errno: `O_NOFOLLOW` refuses the link correctly, but a
        # direct caller (the data home, a backup root, a restore destination) got an
        # `ELOOP`/`ENOTDIR` traceback instead of the one refusal type every other path on
        # this surface produces and the CLI boundary contains.
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise refusal(
                f"refusing to use the {what}: {as_given.name!r} is a symbolic link or "
                "not a directory, so working through it would follow whatever it points "
                "at. Remove it and re-run."
            ) from exc
        raise


def refuse_hardlink_alias(
    fd: int,
    *,
    what: str,
    name: str,
    refusal: type[Exception] = PinnedPathRefusal,
) -> None:
    """Reject a descriptor that is one of several names for the same inode.

    A hardlink is invisible to every path-based guard: it shares the target's inode,
    so ``realpath`` yields the alias's own name, ``is_symlink()`` is False, and
    ``O_NOFOLLOW`` has no link to refuse. A planted alias therefore let an O_TRUNC
    write destroy a protected file, and let a read hand back its bytes.

    Checked on the DESCRIPTOR rather than the path, which is what makes it
    race-free: this fd already refers to the inode being judged.

    The cost is honest and small: a file that legitimately has more than one link --
    a dedup-ing backup tool, a deliberate alias -- is refused. Copy it instead.

    Closes *fd* before raising, so a caller's ``except BaseException: os.close(fd)``
    must not run for this refusal.
    """
    links = os.fstat(fd).st_nlink
    if links > 1:
        os.close(fd)
        raise refusal(
            f"refusing to use the {what}: {name!r} has {links} hard links, so it is "
            "another name for a file this command was not pointed at. A path guard "
            "cannot see that -- the alias shares the target's inode -- so it is "
            "refused on the open descriptor instead. Remove the extra link or use a "
            "different path."
        )


def is_reparse_point(path: str | Path) -> bool:
    """True for a symlink or a Windows junction.

    ``os.path.islink`` is False for a junction -- it is a reparse point but not a
    symlink -- so the tag is checked as well. Comparing ``realpath`` against
    ``abspath`` would be simpler and wrong: on Windows a temp directory is handed
    back as an 8.3 short path, which differs from its resolved form with nothing
    linked anywhere.
    """
    if os.path.islink(path):
        return True
    try:
        return bool(getattr(os.lstat(path), "st_reparse_tag", 0))
    except OSError:  # pragma: no cover - a component that vanished mid-walk
        return False


def copy_file_pinned(
    by_name: str,
    dst: str | None = None,
    *,
    dir_fd: int | None = None,
    name: str | None = None,
    dst_dir_fd: int | None = None,
    dst_name: str | None = None,
    skip_existing: bool = False,
    force_mode: int | None = None,
    on_skip: SkipReporter = _noop_skip,
) -> bool:
    """Copy one file's bytes from a descriptor pinned to a validated inode.

    Returns True when bytes were copied, False when the source was skipped.

    ``shutil.copy2`` cannot be used on a user-writable tree: it dereferences a
    hardlink into innocent-looking regular bytes, and a later tar-level hardlink
    screen never sees a link to reject -- so a hardlink to a credential planted
    inside an otherwise allowlisted directory would ride along as plain content.
    The order here is open first, judge the DESCRIPTOR second: ``O_NOFOLLOW`` where
    the platform has it, then ``fstat`` on the fd, so the inode that is validated is
    exactly the inode whose bytes are copied and no check-to-use window remains.
    Mode and timestamps are applied from that same ``fstat`` result rather than from
    a fresh by-name stat.

    BOTH ends can be pinned, and on a destination the caller does not own they MUST
    be. Pass *dir_fd* + *name* for a pinned source and *dst_dir_fd* + *dst_name* for
    a pinned destination; each side falls back to the by-name form when its pair is
    absent, which is only appropriate for a path this process just created. A
    destination reached by name is an ancestor swap away from landing the bytes
    somewhere else entirely -- that was a real gap in the first version of this
    module, caught in review, and it is why the by-name destination is now the
    exception rather than the only form.

    The destination is created with ``O_EXCL``, so anything already at that name is a
    planted link or alias rather than a file to overwrite and creation refuses it
    without a separate check. With *skip_existing* an occupied name is reported as
    skipped instead, which is what a merge that must not overwrite needs: exclusive
    creation makes "it did not exist" and "this call created it" one statement rather
    than two with a window between them.

    ``FileNotFoundError`` propagates so a caller can tolerate a source that vanished
    mid-walk; every other ``OSError`` propagates so real failures still abort.
    """
    if dst is None and dst_name is None:  # pragma: no cover - caller bug
        raise ValueError("copy_file_pinned needs either dst or dst_name")
    # O_NONBLOCK is not about performance. Opening a FIFO for reading BLOCKS until a
    # writer appears, so without it a single named pipe -- in an extracted archive, or
    # planted in a staged tree -- hangs the whole snapshot or restore forever with no
    # timeout and no message. Found when a mutation probe removed a caller's own
    # `is_file()` guard and the test run stalled until a watchdog killed it, which is
    # exactly how an operator would experience it. The fstat below still rejects the FIFO;
    # this only guarantees we reach that check. On a regular file the flag has no effect.
    src_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        if dir_fd is not None and name is not None:
            fd = os.open(name, src_flags, dir_fd=dir_fd)
        else:
            fd = os.open(by_name, src_flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            # A symlink final component that appeared after the listing-time link
            # screen -- refuse it the same way the screen would have.
            on_skip(SKIP_SYMLINK, by_name)
            return False
        raise
    try:
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            on_skip(SKIP_NOT_REGULAR, by_name)
            return False
        dst_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            if dst_dir_fd is not None and dst_name is not None:
                dst_fd = os.open(dst_name, dst_flags, 0o600, dir_fd=dst_dir_fd)
            else:
                dst_fd = os.open(str(dst), dst_flags, 0o600)
        except FileExistsError:
            if skip_existing:
                return False
            raise
        # The destination is finished through its OWN descriptor and never by name
        # again. Two reasons, both raised in review:
        #
        # * `os.chmod(name, dir_fd=...)` re-resolves the final component under the
        #   pinned directory, so a name swapped between the write and the chmod would
        #   have the mode applied to the replacement. `fchmod`/`futimes` on the open
        #   descriptor cannot be redirected, because that fd already refers to the
        #   inode whose bytes were just written.
        # * A copy that fails part-way (ENOSPC, a read error) must not leave a partial
        #   file behind. On the merge path `skip_existing` would then treat that
        #   fragment as an existing file and skip the archive's real one on a retry --
        #   a corrupt file that survives the fix. So the destination this call created
        #   is unlinked on any failure, through the same pinned directory.
        #
        # The descriptor is closed BEFORE that unlink, not after. Windows refuses to
        # delete a file that still has an open handle, so unlinking first left the
        # partial file exactly where the cleanup was supposed to remove it -- which is
        # the whole defect, reappearing on the one platform the fix had not run on. My
        # own Windows shard caught it.
        try:
            with os.fdopen(fd, "rb") as fsrc:
                fd = -1  # ownership passed to the file object
                # fdopen takes ownership and closes what it is given, so it gets a
                # duplicate: dst_fd itself has to outlive the write for the two
                # descriptor-based metadata calls below.
                with os.fdopen(os.dup(dst_fd), "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
            _apply_metadata(
                dst_fd,
                st,
                dst=dst,
                dst_dir_fd=dst_dir_fd,
                dst_name=dst_name,
                mode=force_mode,
            )
        except BaseException:
            os.close(dst_fd)
            dst_fd = -1
            if dst_dir_fd is not None and dst_name is not None:
                _unlink_quietly(dst_name, dir_fd=dst_dir_fd)
            else:
                _unlink_quietly(str(dst))
            raise
        finally:
            if dst_fd >= 0:
                os.close(dst_fd)
        return True
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def stat_at(dir_fd: int, name: str) -> os.stat_result | None:
    """`lstat` *name* relative to *dir_fd*, or ``None`` if it is not there.

    The descriptor-relative answer to "what is this, and is it there?". A plain
    ``Path.is_file()`` re-resolves the whole path, so between the question and the use of
    the answer the object can be replaced -- and a guard that inspects the replacement
    while the code acts on the original is worse than no guard, because it reports success.
    Review found three such guards in code that was already holding the right descriptor.

    Never follows a link: the caller wants to know what the NAME is, and a link is one of
    the answers it needs to be able to see.
    """
    try:
        return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError):
        return None


def is_regular_at(dir_fd: int, name: str) -> bool:
    """True when *name* under *dir_fd* is a plain file -- not a link, FIFO, or directory."""
    st = stat_at(dir_fd, name)
    return st is not None and _stat.S_ISREG(st.st_mode)


def _apply_dir_metadata(dst_fd: int, st: os.stat_result) -> None:
    """Copy a directory's mode and timestamps onto the destination descriptor.

    Directories need this as much as files do -- `shutil.copytree` did it and the pinned
    walk that replaced it did not, so restored trees came back mode 0700 with fresh mtimes.
    Applied by descriptor for the same reason files are: a name re-resolves, an fd cannot.

    Best-effort on the timestamps only. `fchmod` on a directory is supported wherever the
    pinned walk runs at all, but some filesystems refuse `utime` on a directory handle, and
    a wrong mtime is not worth failing a whole snapshot over.
    """
    if hasattr(os, "fchmod"):
        os.fchmod(dst_fd, _stat.S_IMODE(st.st_mode))
    if os.utime in os.supports_fd:
        try:
            os.utime(dst_fd, ns=(st.st_atime_ns, st.st_mtime_ns))
        except OSError:  # pragma: no cover - filesystem-dependent
            pass


def _apply_metadata(
    dst_fd: int,
    st: os.stat_result,
    *,
    dst: str | Path | None,
    dst_dir_fd: int | None,
    dst_name: str | None,
    mode: int | None = None,
) -> None:
    """Copy mode and timestamps onto the destination, by descriptor where possible.

    ``os.fchmod`` does not exist on Windows and ``os.utime`` only accepts a descriptor
    where ``os.utime in os.supports_fd``, so the fd form cannot be unconditional: an
    earlier revision called it always and crashed a Windows snapshot with
    ``AttributeError`` the moment it reached a core file. Caught in review.

    The fd form is preferred wherever it exists, because a name re-resolves: a final
    component swapped between the write and the chmod would have the mode applied to the
    replacement. Where it does not exist the by-name form is the only option available,
    and it is the same platform that already cannot pin a directory at all -- so this
    adds no exposure that the declared by-name traversal does not already carry.

    *mode* overrides the source's mode. Used for a restored security file, which must end
    up owner-only regardless of what the archive recorded.
    """
    want = _stat.S_IMODE(st.st_mode) if mode is None else mode
    if hasattr(os, "fchmod"):
        os.fchmod(dst_fd, want)
    elif dst_dir_fd is not None and dst_name is not None:  # pragma: no cover - Windows
        os.chmod(dst_name, want, dir_fd=dst_dir_fd)
    else:  # pragma: no cover - Windows
        os.chmod(str(dst), want)

    times = (st.st_atime_ns, st.st_mtime_ns)
    if os.utime in os.supports_fd:
        os.utime(dst_fd, ns=times)
    elif dst_dir_fd is not None and dst_name is not None:  # pragma: no cover - Windows
        os.utime(dst_name, ns=times, dir_fd=dst_dir_fd)
    else:  # pragma: no cover - Windows
        os.utime(str(dst), ns=times)


def _create_and_open_dir_reporting(
    path: str | Path,
    *,
    what: str,
    refusal: type[Exception] = PinnedPathRefusal,
) -> tuple[int, bool]:
    """Create a directory through its PINNED parent; return its descriptor and whether
    this call is what created it.

    The `created` half exists because the ONLY trustworthy answer to "is this directory
    mine to stamp metadata onto?" is whether our own `mkdir` succeeded. Sampling
    `dst.exists()` beforehand is a name-based check with a window after it, and review
    found the window: a forced restore removes the workspace, the gateway recreates it
    before the `mkdir` runs, and the live directory is then treated as newly created and
    has the archive's mode and timestamps written over it. `FileExistsError` cannot race
    -- the kernel decided it.

    ``Path(p).mkdir(parents=True)`` creates every missing component by name, so a link
    already sitting at an ancestor is followed and the directories are created inside
    whatever it points at -- a write through an attacker-controlled path, which is
    strictly worse than a read through one. Review flagged the by-name creation; the
    probe that settled it showed a link AT the final component is already refused by
    ``O_NOFOLLOW`` but an ancestor link is not.

    So the parent chain is pinned first and only the final component is created,
    relative to that descriptor. What remains is this module's documented and
    deliberate residual, stated in :func:`pin_parent`: a component that was already a
    link when the parent was resolved is followed by that resolution, because refusing
    every symlinked ancestor would break a destination under ``/tmp`` on macOS. The
    parent must therefore already exist -- callers create their own tree roots.
    """
    as_given = Path(path)
    parent_fd = pin_parent(
        os.path.realpath(as_given.parent or Path(".")), what=what, refusal=refusal
    )
    try:
        try:
            os.mkdir(as_given.name, 0o700, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            created = False
        try:
            return os.open(as_given.name, _dir_flags(), dir_fd=parent_fd), created
        except OSError as exc:
            # A link (or a plain file) at the destination's own name. O_NOFOLLOW already
            # refuses it -- the gap review found was that it escaped as a raw OSError, so
            # a restore ended in a traceback instead of the refusal every other path on
            # this surface produces. Translated here so callers have one type to contain.
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise refusal(
                    f"refusing to use the {what}: {as_given.name!r} is a symbolic link "
                    "or not a directory, so creating the tree there would write through "
                    "whatever it points at. Remove it and re-run."
                ) from exc
            raise
    finally:
        os.close(parent_fd)


def create_and_open_dir_pinned(
    path: str | Path,
    *,
    what: str,
    refusal: type[Exception] = PinnedPathRefusal,
) -> int:
    """Create a directory through its PINNED parent and return its descriptor.

    The public form, unchanged: callers that only need the descriptor are not made to
    unpack a tuple they will discard. Use :func:`_create_and_open_dir_reporting` inside
    this module when the answer to "did I create it?" has to be atomic.
    """
    return _create_and_open_dir_reporting(path, what=what, refusal=refusal)[0]


def _unlink_quietly(name: str, *, dir_fd: int | None = None) -> None:
    """Remove a destination this module created, ignoring an already-gone name.

    Only ever called on a path this process just created with ``O_EXCL``, on a failure
    path, so there is nothing of the caller's to lose and a second error here would
    only mask the original one.
    """
    try:
        if dir_fd is not None:
            os.unlink(name, dir_fd=dir_fd)
        else:
            os.unlink(name)
    except OSError:  # pragma: no cover - the name was already gone
        pass


def stage_tree_pinned(
    src: str | Path,
    dst: str | Path,
    *,
    what: str,
    ignore: Callable[[str, list[str]], set[str]] | None = None,
    on_skip: SkipReporter = _noop_skip,
    skip_existing: bool = False,
    refusal: type[Exception] = PinnedPathRefusal,
) -> None:
    """Copy a tree with BOTH traversals pinned end to end.

    A by-name walk's link screens protect only each final component: swapping an
    allowlisted ancestor DIRECTORY for a link to a credential directory mid-walk
    redirects every deeper open through the link, and the per-file ``O_NOFOLLOW``
    never fires because what it finds inside the replaced tree is a plain regular
    file. Here every directory is opened ``O_NOFOLLOW|O_DIRECTORY`` relative to its
    PARENT's descriptor and every file is opened relative to its pinned parent, so
    the directory that was validated is exactly the directory used. Both roots go
    through :func:`open_dir_pinned`, which pins the chain ABOVE each root too.

    The DESTINATION is pinned for a reason, not for symmetry. The first version of
    this function pinned only the source, which was defensible while the only
    destination was a private temporary directory -- and wrong the moment a restore
    used it, because then the destination IS the live data home and an ancestor
    swapped there lands the archive's bytes outside it. Caught in review.

    With *skip_existing* an occupied destination name is reported and skipped rather
    than refused, which is what a merge that must not overwrite needs. Without it an
    occupied name raises, because in a staging directory this process just created,
    the only thing that can be sitting there is something planted.

    Symlinks and non-regular files are reported through *on_skip* and skipped;
    entries that vanish mid-walk are reported and skipped; *ignore* sees
    ``(directory_by_name, contents)`` exactly as ``shutil.copytree``'s does. Every
    other error propagates, so a staging pass never silently ships without files it
    failed to read.

    Refuses outright on a platform that cannot pin a tree. Callers that must still
    function there are expected to say so explicitly rather than have this module
    quietly hand them a by-name walk -- see :func:`supports_pinned_tree_walk`.
    """
    if not supports_pinned_tree_walk():
        raise refusal(
            f"refusing to stage the {what}: this platform cannot open a directory "
            "relative to a descriptor, so the traversal would have to re-open every "
            "component by name and could be redirected by an ancestor swapped "
            "mid-walk. Staging by name is a caller's decision to declare, not this "
            "helper's to make silently."
        )

    def _walk(src_fd: int, dst_fd: int, by_name: str) -> None:
        names = os.listdir(src_fd)
        skipped = set(ignore(by_name, names)) if ignore else set()
        for entry in sorted(names):
            if entry in skipped:
                continue
            path = os.path.join(by_name, entry)
            try:
                st = os.stat(entry, dir_fd=src_fd, follow_symlinks=False)
            except FileNotFoundError:
                on_skip(SKIP_VANISHED, path)
                continue
            if _stat.S_ISLNK(st.st_mode):
                on_skip(SKIP_SYMLINK, path)
            elif _stat.S_ISDIR(st.st_mode):
                child_src = _open_child_dir(src_fd, entry, path, on_skip)
                if child_src is None:
                    continue
                try:
                    created = True
                    try:
                        os.mkdir(entry, 0o700, dir_fd=dst_fd)
                    except FileExistsError:
                        created = False
                        # Only a merge legitimately meets an existing destination
                        # directory. Anywhere else the destination tree is one this
                        # process just created, so a name already occupying it is a
                        # planted link or file -- and swallowing that made the pinned
                        # open below refuse the subtree and the whole restore report
                        # success with the archive's subtree missing. Raised in
                        # review, and the same silent-partial shape this change fixes
                        # elsewhere, so it is now a refusal rather than a skip.
                        if not skip_existing:
                            raise refusal(
                                f"refusing to stage into {path!r}: a name already "
                                "occupies that directory in a destination tree this "
                                "operation created, so it is a link or a file planted "
                                "there rather than a directory to merge into. Writing "
                                "past it would silently omit everything below it."
                            )
                    child_dst = _open_child_dir(dst_fd, entry, path, on_skip)
                    if child_dst is None:
                        # A SOURCE entry that stopped being a directory is skipped --
                        # there is nothing left to copy. A DESTINATION that stopped
                        # being one is different: the archive's subtree still exists
                        # and now has nowhere to go, so continuing would report success
                        # with that subtree missing. Raised in review. A merge is the
                        # one caller that may legitimately meet a foreign destination
                        # tree, so it keeps the skip.
                        if not skip_existing:
                            raise refusal(
                                f"refusing to stage into {path!r}: the destination "
                                "directory stopped being a plain directory after it was "
                                "created, so the archive's contents below it could not "
                                "be written. Continuing would report success with that "
                                "subtree missing."
                            )
                        continue
                    try:
                        _walk(child_src, child_dst, path)
                        # Applied AFTER the contents, through the destination's OWN
                        # descriptor, and ONLY to a directory this walk created.
                        #
                        # `shutil.copytree` preserved directory mode and timestamps; the
                        # walk that replaced it did not, so a restored 0755 directory came
                        # back 0700 with a fresh mtime. Fixing that unconditionally then
                        # broke the merge case: a live 0700 directory had the ARCHIVE's
                        # 0755 stamped onto it, which both clobbers the user's metadata and
                        # loosens permissions from an untrusted source. Both caught in
                        # review, one round apart.
                        #
                        # The archive-is-untrusted half is not a new rule -- it is why a
                        # security file is forced to 0600 rather than given the archive's
                        # mode. That rule was applied to files and not carried to
                        # directories.
                        #
                        # Ordering is load-bearing: after the children, because writing
                        # them updates the directory mtime, and because a restrictive
                        # source mode applied first would block those writes.
                        if created:
                            _apply_dir_metadata(child_dst, st)
                    finally:
                        os.close(child_dst)
                finally:
                    os.close(child_src)
            elif _stat.S_ISREG(st.st_mode):
                try:
                    copy_file_pinned(
                        path,
                        dir_fd=src_fd,
                        name=entry,
                        dst_dir_fd=dst_fd,
                        dst_name=entry,
                        skip_existing=skip_existing,
                        on_skip=on_skip,
                    )
                except FileNotFoundError:
                    on_skip(SKIP_VANISHED, path)
            else:
                on_skip(SKIP_NOT_REGULAR, path)

    try:
        root_src = open_dir_pinned(src, what=what, refusal=refusal)
    except (OSError, refusal) as exc:
        # A SOURCE root that was swapped or removed after the caller's listing-time
        # screen is omitted, not fatal -- the same treatment every other unusable source
        # entry gets, and it now reaches MANIFEST.json rather than only the console. The
        # refusal type is caught alongside the raw errno because `open_dir_pinned` now
        # translates ELOOP/ENOTDIR (review asked for that so DIRECT callers stop getting
        # tracebacks); this call site is the one place that wants the softer outcome.
        if isinstance(exc, OSError) and exc.errno not in (
            errno.ELOOP,
            errno.ENOTDIR,
            errno.ENOENT,
        ):
            raise
        on_skip(SKIP_SYMLINK, str(src))
        return
    try:
        # No parents=True here, deliberately. Creating a missing ancestor chain by name
        # is exactly what this replaced: every caller's destination parent already
        # exists (a staging directory this process made, the data home, or a backup
        # directory the caller created), so a missing parent means the caller is
        # pointing somewhere it has not validated, and that should surface rather than
        # be materialised through whatever the path resolves to.
        # Whether the ROOT is ours to stamp comes from the kernel, not from a name.
        # `not dst.exists()` was a check with a window after it: a forced restore removes
        # the workspace, the gateway recreates it before the mkdir, and the live directory
        # is then stamped with the archive's metadata. Review caught it, and it is the
        # third instance of this change's first invariant -- no write, and no decision
        # governing a write, may be made through a path name.
        root_dst, root_is_ours = _create_and_open_dir_reporting(
            dst, what=f"{what} destination", refusal=refusal
        )
        try:
            _walk(root_src, root_dst, str(src))
            if root_is_ours:
                _apply_dir_metadata(root_dst, os.fstat(root_src))
        finally:
            os.close(root_dst)
    finally:
        os.close(root_src)


def _open_child_dir(parent_fd: int, entry: str, by_name: str, on_skip: SkipReporter) -> int | None:
    """Open a child directory through *parent_fd*, or report why it was skipped.

    Returns ``None`` for the two races worth tolerating -- the entry vanished, or it
    stopped being a plain directory between the stat and this open. ``ELOOP`` and
    ``ENOTDIR`` are exactly the swap the pinned open exists to refuse, so they are
    reported as a skipped symlink rather than raised: the listing-time screen would
    have said the same thing a moment earlier. Every other error propagates.

    Extracted because the source and destination sides need identical handling, and
    the version of this code that had it on one side only let the swap the source
    skipped escape the destination as a raw ``OSError``.
    """
    try:
        return os.open(entry, _dir_flags(), dir_fd=parent_fd)
    except FileNotFoundError:
        on_skip(SKIP_VANISHED, by_name)
        return None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            on_skip(SKIP_SYMLINK, by_name)
            return None
        raise
