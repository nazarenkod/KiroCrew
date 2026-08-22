"""Channel-neutral keyword commands — one copy of the reply text.

``sessions``, ``spawn``, ``cron`` and ``task run`` are the *path-independent*
commands: they need no LLM turn, they carry no channel state, and their answer
is one string. Slack has had them since before the transport abstraction
existed, which is why they were written inside ``slack/handler.py``; nothing in
them is Slack-shaped, so a second channel that wanted them had only the choice
of a second copy.

Each function here is ``(text, service) -> reply | None``: ``None`` means the
text is not that command and normal routing continues, a string means the
caller posts it and returns without starting a turn. The caller owns delivery,
which is what keeps the module free of a client, a chat id, a thread id, and a
markup dialect.

**Services are duck-typed on purpose.** ``kiro_crew.subagent`` and
``kiro_crew.taskrunner`` both reach ``kiro_crew.slack`` transitively, so
importing them at runtime here would reintroduce the ``messaging -> slack``
edge the abstraction exists to remove. They are typed under ``TYPE_CHECKING``
and consumed through the handful of attributes named below.

Dependency direction stays one-way: this module imports ``kiro_crew.cron``
(itself free of any channel import), ``security`` and ``stats`` — never
``kiro_crew.slack`` or ``kiro_crew.dashboard``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kiro_crew.cron import (
    CronStoreBusy,
    compute_next_run_ts,
    format_schedule,
    get_local_tz,
)
from kiro_crew.security import redact

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import edge
    from kiro_crew.cron import CronService
    from kiro_crew.subagent import SubagentManager
    from kiro_crew.taskrunner import TaskRunner

logger = logging.getLogger(__name__)

#: How much of a cron job's message body a list row shows.
_CRON_MESSAGE_PREVIEW_CHARS = 50
#: How much of a subagent's task a list row shows.
_SPAWN_TASK_PREVIEW_CHARS = 60
#: How much of the spawned task the confirmation echoes back.
_SPAWN_ECHO_CHARS = 100

#: The retryable answer for a store another writer currently holds. One string,
#: because a caller that sees a different wording per verb cannot tell "retry
#: this" from "this failed".
_CRON_BUSY = "⏳ Cron store busy — try again in a moment."


def _redact(text: str) -> str:
    """Both redaction passes, over text that may be ``None``.

    Cron job names/messages and subagent tasks are free-form text a user OR the
    LLM wrote, and every caller posts the result to an external surface AND
    persists it, so the scan runs before the string leaves this module.
    """
    return redact(text or "")


# ── spawn / bg ─────────────────────────────────────────────────────────────


def spawn_command_reply(
    text: str, manager: "SubagentManager | Any", session_key: str = ""
) -> str | None:
    """Handle ``spawn <task>`` / ``bg <task>`` / ``spawn list`` / ``spawn status``.

    Reads ``manager.running`` (an iterable of records with ``id``/``started``/
    ``task``), ``manager.max_concurrent`` and ``manager.spawn(task,
    parent_session_key=)``.
    """
    stripped = text.strip()
    low = stripped.lower()
    for prefix in ("spawn ", "bg "):
        if low.startswith(prefix):
            return spawn_task_reply(stripped[len(prefix) :].strip(), manager, session_key)
    return None


def spawn_task_reply(
    task: str, manager: "SubagentManager | Any", session_key: str = ""
) -> str | None:
    """Handle an ALREADY-PARSED spawn argument (``list``/``status``/a task).

    Public because a channel whose command grammar carries its own prefix (a
    Telegram ``/spawn <task>``, a Discord ``!spawn <task>``) has the argument in
    hand and must not have to re-synthesize ``"spawn " + arg`` just to have it
    stripped off again.
    """
    if not task:
        return None
    if task.lower() in ("list", "status"):
        running = list(manager.running)
        if not running:
            return "No subagents running."
        now = time.time()
        lines = ["*Running subagents:*"]
        for agent in running:
            elapsed = int(now - agent.started)
            lines.append(
                f"🔹 `{agent.id}` | {elapsed}s | {_redact(agent.task)[:_SPAWN_TASK_PREVIEW_CHARS]}"
            )
        return "\n".join(lines)
    info = manager.spawn(task, parent_session_key=session_key)
    if not info:
        return f"⚠️ Subagent capacity reached ({manager.max_concurrent}). Try again later."
    return f"🚀 Spawned subagent `{info.id}`\n_{_redact(task)[:_SPAWN_ECHO_CHARS]}_"


# ── cron ───────────────────────────────────────────────────────────────────


def _relative(delta: float) -> str:
    """A cron job's next run as a coarse relative duration."""
    if delta >= 86400:
        return f"in {int(delta // 86400)}d {int((delta % 86400) // 3600)}h"
    if delta >= 3600:
        return f"in {int(delta // 3600)}h {int((delta % 3600) // 60)}m"
    if delta > 0:
        minutes = int(delta // 60)
        return f"in {minutes}m" if minutes >= 1 else "in <1m"
    return "now"


def _cron_list(cron_service: "CronService | Any") -> str:
    jobs = cron_service.list_jobs(include_disabled=True)
    if not jobs:
        return "No cron jobs scheduled."
    now = time.time()
    tz_name, _ = get_local_tz()
    lines = ["*Your cron jobs:*"]
    for job in jobs:
        status = "✅" if job.enabled else "⏸️"
        schedule = _redact(format_schedule(job.schedule, tz_name=job.timezone or tz_name))
        last = ""
        if job.last_status == "ok":
            last = " ✓"
        elif job.last_status == "error":
            last = " ❌"
        nxt = compute_next_run_ts(job, now=now)
        next_part = f" | ⏭ {_relative(nxt - now)}" if nxt is not None else ""
        message = _redact(job.message)[:_CRON_MESSAGE_PREVIEW_CHARS]
        lines.append(f"{status} `{job.id}` | `{schedule}` | {message}{last}{next_part}")
    return "\n".join(lines)


async def cron_remove_all_reply(cron_service: "CronService | Any") -> str:
    """Remove every cron job (enabled or not) and summarize what went.

    Public for the same reason as :func:`spawn_task_reply`: a channel that
    parsed ``remove all`` itself should not have to rebuild the sentence.
    """
    jobs = cron_service.list_jobs(include_disabled=True)
    if not jobs:
        return "No cron jobs to remove."
    # ``job.name`` is free-form user/LLM text reaching a chat reply and the
    # persisted conversation log; ``job.id`` is a generated UUID and is left as-is.
    lines = [f"- `{job.id}` — {_redact(job.name)}" for job in jobs]
    # One batch lock/reload/save, offloaded by the service itself, so a chat
    # gateway's loop is never parked on the store lock.
    try:
        await cron_service.remove_jobs([job.id for job in jobs])
    except CronStoreBusy:
        return _CRON_BUSY
    return f"✅ Removed {len(lines)} cron job(s):\n" + "\n".join(lines)


async def cron_command_reply(text: str, cron_service: "CronService | Any") -> str | None:
    """Handle ``cron list`` / ``cron remove <id>|all`` / ``cron pause|resume <id>``.

    Async because the mutators run through the store's event-loop-safe
    ``*_async`` variants; a contended store answers "busy, retry" rather than
    parking the caller's loop on the lock.
    """
    parts = text.strip().lower().split()
    if len(parts) < 2 or parts[0] != "cron":
        return None
    action = parts[1]
    if action == "list":
        return _cron_list(cron_service)
    if len(parts) < 3:
        return None
    job_id = parts[2]
    if action == "remove":
        if job_id == "all":
            return await cron_remove_all_reply(cron_service)
        try:
            removed = await cron_service.remove_job_async(job_id)
        except CronStoreBusy:
            return _CRON_BUSY
        return f"✅ Removed cron job `{job_id}`" if removed else f"❌ Job `{job_id}` not found"
    if action in ("pause", "resume"):
        enabled = action == "resume"
        try:
            changed = await cron_service.enable_job_async(job_id, enabled=enabled)
        except CronStoreBusy:
            return _CRON_BUSY
        if not changed:
            return f"❌ Job `{job_id}` not found"
        return f"▶️ Resumed cron job `{job_id}`" if enabled else f"⏸️ Paused cron job `{job_id}`"
    return None


# ── task run ───────────────────────────────────────────────────────────────


async def task_command_reply(
    text: str, runner: "TaskRunner | Any", *, session_key: str = ""
) -> str | None:
    """Handle the KEYWORD grammar: ``task run <spec>`` / ``status`` / ``cancel``.

    ``project run <spec>`` is accepted as an alias for ``task run <spec>``.

    ``session_key`` is keyword-only, defaults to empty, and is forwarded
    verbatim to :func:`task_arg_reply` — see there for what the runner does with
    it. It exists on this entry point too because the two grammars reach the same
    runner: without it a ``task run <spec>`` typed as a keyword escalates its
    approval notices only to the owner DM, while the same run started from a
    ``/task`` command reaches the conversation the operator is watching.
    """
    stripped = text.strip()
    low = stripped.lower()
    if low.startswith("project run "):
        stripped = "task run " + stripped[len("project run ") :]
        low = stripped.lower()
    if not low.startswith("task run "):
        return None
    return await task_arg_reply(stripped[len("task run ") :], runner, session_key=session_key)


async def task_arg_reply(
    arg: str, runner: "TaskRunner | Any", *, session_key: str = ""
) -> str | None:
    """Handle an ALREADY-PARSED task argument: a spec path, ``status``, ``cancel``.

    Public for the same reason as :func:`spawn_task_reply`, and load-bearing here
    in a way it is not there: the keyword grammar spells the verb ``task run``,
    but a channel whose command IS ``/task`` receives ``run <spec>`` as its
    argument. Re-composing ``"task run " + arg`` then yields ``task run run
    <spec>`` and the runner is handed a spec file literally named ``run <spec>``,
    which cannot exist — so a leading ``run`` is absorbed HERE, once, rather than
    at each channel's call site.

    ``session_key`` is keyword-only and defaults to empty so the positional
    signature every existing caller uses is unchanged. It names the conversation
    the command arrived in and is forwarded to the runner, which hands it to its
    notify sink; that is what lets an approval notice go back to the channel the
    operator is watching rather than only to the Slack owner DM. A caller that
    omits it keeps exactly the previous behaviour.

    Reads ``runner.running``, ``runner.status()``, ``runner.cancel()`` and
    ``runner.start_background(path, source=, session_key=)``.
    """
    arg = arg.strip()
    if arg.lower() == "run" or arg.lower().startswith("run "):
        arg = arg[len("run") :].strip()
    if not arg:
        return None

    if arg.lower() == "status":
        status = runner.status()
        if not status.get("running"):
            return "No task running."
        # Progress is per-run: ``status()`` puts only ``running``/``agent``/
        # ``runs`` at the top level, so prefer the live run when several are tracked.
        runs = status.get("runs") or []
        run = next((r for r in runs if r.get("running")), runs[0] if runs else {})
        return (
            "*Task Runner*\n"
            f"Status: {run.get('status', 'idle')}\n"
            f"Steps: {run.get('completed', 0)}/{run.get('tasks', 0)}\n"
            f"Current: step {run.get('current_task', 0)}"
        )

    if arg.lower() == "cancel":
        if not runner.running:
            return "No task running."
        runner.cancel()
        return "🛑 Task cancelled."

    if runner.running:
        return "⚠️ Task runner is already running. Use `task run cancel` first."
    # The spec is READ and its contents reach the model, so an arbitrary path is
    # an exfiltration primitive: `task run ~/.ssh/id_rsa` would hand a private key
    # to a third-party LLM. Gate it on the shared ``validate_file_path``, which
    # applies the Windows UNC trusted-root check BEFORE resolving (realpath on a
    # UNC path is itself the outbound SMB probe), canonicalizes through every
    # symlink, and refuses a resolved target under a sensitive root -- so a
    # workspace symlink into ~/.ssh is refused through the link. Hand-rolling a
    # prefix test here would miss both the symlink and the UNC case.
    #
    # The CANONICAL path is what gets used from here on, not the raw argument:
    # validating one string and then acting on another is how a guard ends up
    # ornamental.
    #
    # Off-loop: realpath and stat on a user-supplied path can block on a stalled
    # network mount, and this runs on the gateway's single event loop.
    from kiro_crew.hooks import validate_file_path

    canonical = await asyncio.to_thread(validate_file_path, arg)
    if canonical is None:
        # Deliberately does not echo the path or say WHY. A refusal that
        # distinguishes "sensitive" from "malformed" is an oracle for probing
        # which roots exist on the host.
        return "❌ That path cannot be used as a task spec."
    spec_path = Path(canonical)
    if not await asyncio.to_thread(spec_path.exists):
        return f"❌ Spec file not found: `{_redact(str(spec_path))}`"
    try:
        # Widen the call only when there IS a conversation to carry. ``runner``
        # is duck-typed (this module must not import ``TaskRunner`` at runtime),
        # so passing ``session_key=`` unconditionally would break every narrower
        # stand-in that accepts only ``(path, source=)`` — and it would do so on
        # the start path, turning a working command into "Failed to start".
        if session_key:
            await runner.start_background(spec_path, source="chat", session_key=session_key)
        else:
            await runner.start_background(spec_path, source="chat")
    except Exception as exc:
        logger.warning("task run: start_background failed", exc_info=True)
        return f"❌ Failed to start: {_redact(str(exc))}"
    return (
        f"🚀 Task started: `{_redact(spec_path.name)}`\n" "Use `task run status` to check progress."
    )
