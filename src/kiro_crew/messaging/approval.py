"""Layer 2c -- the channel-neutral interactive-approval awaiter.

``TurnDriver`` resolves an INTERACTIVE tool permission by awaiting a
``decider`` (:data:`kiro_crew.messaging.driver.ApprovalDecider`). Every
channel's decider is the same object: a pending :class:`asyncio.Future` that
the channel's inbound path resolves when the user answers. What differs is only
how the user answers -- a button press, an inline keyboard tap, or a typed
reply -- and that half stays channel-local.

This module owns the shared half so a third channel does not become a third
copy. Telegram (``telegram/renderer.py``) and Discord (``discord/renderer.py``)
each carry their own registry today and may migrate here at their own pace;
nothing here forces them to, and Discord's per-prompt nonce guard deliberately
stays channel-local because it is coupled to a `custom_id` round trip this
module knows nothing about.

Why a PROCESS-GLOBAL registry rather than per-dispatcher state: one gateway
serves many conversations, and a channel's inbound path (a websocket frame, an
interaction callback) is a different call stack from the turn that is waiting.
The registry is what connects them. Keys are ``session_key:request_id``
because ACP request ids restart at 1 for each session, so a bare request id
would let one conversation resolve another's prompt.

Dependency direction is ``<channel> -> messaging`` (never the reverse), so this
module imports no channel package.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from typing import Any

logger = logging.getLogger(__name__)

#: How long a pending approval waits before denying by default.
#:
#: Deny-on-timeout is the security-relevant half: an unanswered prompt must not
#: leave the tool approved, and it must not hold the session semaphore forever.
#: The window is generous because a human has to read the prompt and type a
#: reply, and short enough that an abandoned conversation frees its session.
APPROVAL_TIMEOUT_S = 300.0


class PendingApprovals:
    """Registry of in-flight approval decisions for one channel.

    One instance per channel (as a module-level singleton in the channel's own
    package), so two channels cannot collide on a session key that happens to
    match. Instance state rather than class state for the same reason: a
    class-level dict shared by subclasses is how the existing per-channel
    copies became hard to reason about.
    """

    __slots__ = ("_channel_type", "_nonces", "_pending")

    def __init__(self, channel_type: str) -> None:
        self._channel_type = channel_type
        self._pending: dict[str, "asyncio.Future[bool]"] = {}
        #: key -> the nonce minted for that prompt's widget, if it has one.
        self._nonces: dict[str, str] = {}

    @staticmethod
    def key(session_key: str, request_id: str | int) -> str:
        """The registry address for one prompt.

        Namespaced by session because ACP request ids restart at 1 per session:
        a bare request id would let a reply in one conversation resolve a
        pending prompt in another.
        """
        return f"{session_key}:{request_id}"

    def _first_pending(
        self, session_key: str, request_id: str | int | None = None
    ) -> tuple[str, "asyncio.Future[bool]"] | None:
        """The unresolved prompt this answer belongs to, or ``None``.

        One predicate for both readers, because it IS the isolation guarantee:
        the session prefix carries its ``:`` separator, so ``webex:a`` cannot
        match a prompt pending for ``webex:ab``.

        With *request_id* the match is exact — the shape a widget channel needs,
        since a button press carries the correlation id. Without it, the oldest
        unresolved entry wins, which is what a TYPED answer needs: the user
        replies to "the question on screen" and names no id. Oldest-first is
        well-defined because ``TurnDriver`` is sequential over the event stream,
        so a turn awaits one prompt at a time and insertion order is decision
        order (dicts preserve it).
        """
        if request_id is not None:
            k = self.key(session_key, request_id)
            fut = self._pending.get(k)
            return (k, fut) if fut is not None and not fut.done() else None
        prefix = f"{session_key}:"
        for k, fut in self._pending.items():
            if k.startswith(prefix) and not fut.done():
                return (k, fut)
        return None

    def has_pending(self, session_key: str) -> bool:
        """Whether *session_key* has an unresolved prompt waiting.

        Lets a channel's inbound path decide whether to read the next message
        as an approval answer at all, instead of consuming an ordinary message
        that merely looks like one.
        """
        return self._first_pending(session_key) is not None

    def resolve(
        self,
        session_key: str,
        approved: bool,
        *,
        request_id: str | int | None = None,
        expected_nonce: str = "",
    ) -> bool:
        """Resolve a pending prompt for *session_key*; return whether one waited.

        The return value is load-bearing: it lets a caller tell "your answer was
        applied" from "that prompt already expired" and say so, instead of
        reporting a decision that never reached the provider.

        *expected_nonce* is the WIDGET path's guard, and it is checked BEFORE
        anything is resolved. That ordering is the whole point: a channel that
        resolved first and validated after would have already approved the tool by
        the time it decided the press was stale, and the only thing the guard
        could still suppress is the confirmation message. Compared in constant
        time against the nonce minted for THIS entry, so a press on a card whose
        decision already resolved cannot answer a later prompt — which matters
        because a platform that refuses to edit a message carrying an attachment
        (Webex) leaves a resolved card's buttons clickable forever.

        A typed answer passes no nonce (``""``) and skips the check: there is no
        widget to have gone stale, and the sender was authorized upstream.
        """
        found = self._first_pending(session_key, request_id)
        if found is None:
            return False
        key, fut = found
        if expected_nonce:
            minted = self._nonces.get(key, "")
            if not minted or not secrets.compare_digest(expected_nonce, minted):
                return False
        fut.set_result(bool(approved))
        return True

    def reserve(self, session_key: str, request_id: str | int) -> str:
        """Open the decision window BEFORE the prompt is sent; return its nonce.

        Called by the renderer as it renders the prompt, because ``TurnDriver``
        dispatches ``PROMPT_CHOICE`` and only then awaits the decider: between the
        prompt becoming visible in the room and ``decide`` registering, an answer
        that arrived would find nothing pending, fall through to the mid-turn path,
        and be discarded — the user would watch their decision do nothing and the
        tool deny itself minutes later. Reserving first means the window is open
        for the whole time the prompt is answerable.

        Never replaces a LIVE future, in either direction: a reservation that
        follows ``decide`` (or a second reservation for the same address) keeps the
        future already being awaited and only ensures a nonce exists for it.
        Replacing it would orphan the object the waiter is blocked on, and a
        resolved answer would set a future nobody reads — the prompt would hang for
        its whole window and then deny.
        """
        k = self.key(session_key, request_id)
        pending = self._pending.get(k)
        if pending is None or pending.done():
            self._pending[k] = asyncio.get_running_loop().create_future()
        nonce = self._nonces.get(k) or secrets.token_hex(8)
        self._nonces[k] = nonce
        return nonce

    def discard_reservations(self, session_key: str) -> None:
        """Drop *session_key*'s unawaited reservations.

        ``decide`` retires its own entry in a ``finally``, so this only matters for
        a reservation that was never awaited — the prompt rendered and the turn
        then failed before the driver reached the decider. Called from the
        channel's per-turn teardown, so a reservation cannot outlive the turn that
        opened it and be resolved by a stray answer to a LATER prompt.
        """
        prefix = f"{session_key}:"
        for k in [k for k in self._pending if k.startswith(prefix)]:
            if not self._pending[k].done():
                self._pending.pop(k, None)
                self._nonces.pop(k, None)

    async def decide(self, session_key: str, event: Any) -> bool:
        """Await the user's decision for *event*; deny on timeout.

        The channel's renderer has already presented the prompt by the time
        this is awaited (``TurnDriver`` dispatches ``PROMPT_CHOICE`` before
        calling the decider), so this only waits.
        """
        k = self.key(session_key, getattr(event, "request_id", ""))
        # Adopt the renderer's reservation. It opened this window before the prompt
        # was sent, which is the whole point: the answer may ALREADY have arrived
        # and resolved it, in which case that decision is the answer and there is
        # nothing left to await. Creating a fresh future in either case would
        # discard a decision the user already made and then wait out the full
        # window before denying.
        reserved = self._pending.get(k)
        if reserved is not None and reserved.done():
            try:
                return bool(reserved.result())
            finally:
                self._pending.pop(k, None)
                self._nonces.pop(k, None)
        fut = reserved
        if fut is None:
            fut = asyncio.get_running_loop().create_future()
            self._pending[k] = fut
        try:
            return bool(await asyncio.wait_for(fut, APPROVAL_TIMEOUT_S))
        except asyncio.TimeoutError:
            logger.info(
                "%s: approval prompt unanswered after %.0fs; denying",
                self._channel_type,
                APPROVAL_TIMEOUT_S,
            )
            _notify_approval_stalled(session_key)
            return False
        finally:
            # Retire the address AND its nonce with the decision window, so a late
            # answer can never resolve a LATER prompt that reused this request id
            # and a stale widget's nonce can never match a live entry.
            self._pending.pop(k, None)
            self._nonces.pop(k, None)


def _notify_approval_stalled(session_key: str) -> None:
    """Tell AutoNudge that a prompt in *session_key* went unanswered.

    An unanswered prompt is the only evidence available that an UNATTENDED loop
    can no longer act: without it a monitor loop bound to this conversation keeps
    firing, is denied every interactive tool, and burns its whole cycle budget
    while reporting itself healthy — the per-turn cap is measured in tens of
    minutes and the approval window in minutes, so every remaining cycle is spent
    waiting to be denied. With it the loop deactivates naming the remedy.

    Resolved through ``binding_key_for`` so it is inert for a key no loop could be
    bound to, lazily imported (autonudge imports channel packages, so a module
    import here would close a cycle), and best-effort: a monitoring convenience
    must never change how this turn's denial is reported.
    """
    try:
        from kiro_crew.autonudge import binding_key_for
        from kiro_crew.autonudge import get_instance as _autonudge_get

        slot_key = binding_key_for(session_key)
        instance = _autonudge_get() if slot_key else None
        if instance is not None and slot_key:
            instance.notify_approval_stalled(slot_key)
    except Exception:
        logger.debug("autonudge.notify_approval_stalled failed", exc_info=True)


class SessionApprovalDecider:
    """An :data:`ApprovalDecider` bound to one session's registry entry.

    ``TurnDriver`` calls the decider with just the event, so the session key is
    captured here rather than passed per call.
    """

    __slots__ = ("_pending", "_session_key")

    def __init__(self, pending: PendingApprovals, *, session_key: str) -> None:
        self._pending = pending
        self._session_key = session_key

    async def __call__(self, event: Any) -> bool:
        return await self._pending.decide(self._session_key, event)
