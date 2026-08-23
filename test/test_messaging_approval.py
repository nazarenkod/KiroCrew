"""Tests for kiro_crew.messaging.approval — the shared approval awaiter.

The properties here are security-relevant, not cosmetic: an unanswered prompt
must DENY, and one conversation must never resolve another's prompt. ACP request
ids restart at 1 per session, so the second is a live collision risk rather than
a theoretical one.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from kiro_crew.messaging import approval
from kiro_crew.messaging.approval import PendingApprovals, SessionApprovalDecider


def _event(request_id: str | int = 1) -> SimpleNamespace:
    return SimpleNamespace(request_id=request_id, title="fs_write", tool_kind="write")


class TestResolution:
    @pytest.mark.asyncio
    async def test_an_approve_resolves_the_waiting_prompt(self) -> None:
        pending = PendingApprovals("webex")
        task = asyncio.create_task(pending.decide("s1", _event()))
        await _until(lambda: pending.has_pending("s1"))

        assert pending.resolve("s1", True) is True
        assert await task is True

    @pytest.mark.asyncio
    async def test_a_deny_resolves_the_waiting_prompt(self) -> None:
        pending = PendingApprovals("webex")
        task = asyncio.create_task(pending.decide("s1", _event()))
        await _until(lambda: pending.has_pending("s1"))

        assert pending.resolve("s1", False) is True
        assert await task is False

    @pytest.mark.asyncio
    async def test_resolve_reports_false_when_nothing_is_waiting(self) -> None:
        # The caller needs to tell "your answer was applied" from "that prompt
        # already expired", or it reports a decision that never reached the
        # provider.
        pending = PendingApprovals("webex")
        assert pending.resolve("s1", True) is False

    @pytest.mark.asyncio
    async def test_the_entry_is_retired_once_decided(self) -> None:
        # A late answer must not be able to resolve a LATER prompt that happens
        # to reuse this request id.
        pending = PendingApprovals("webex")
        task = asyncio.create_task(pending.decide("s1", _event()))
        await _until(lambda: pending.has_pending("s1"))
        pending.resolve("s1", True)
        await task

        assert pending.has_pending("s1") is False
        assert pending.resolve("s1", True) is False


class TestIsolation:
    @pytest.mark.asyncio
    async def test_two_sessions_sharing_a_request_id_do_not_cross_resolve(self) -> None:
        """The reason the key is namespaced by session.

        kiro-cli numbers permission requests from 1 within each session, so two
        concurrent conversations routinely hold a pending request_id=1. Without
        the namespace, one user's "approve" would approve the other's tool.
        """
        pending = PendingApprovals("webex")
        a = asyncio.create_task(pending.decide("session-a", _event(1)))
        b = asyncio.create_task(pending.decide("session-b", _event(1)))
        await _until(lambda: pending.has_pending("session-a") and pending.has_pending("session-b"))

        assert pending.resolve("session-a", True) is True
        assert await a is True
        assert pending.has_pending("session-b") is True  # untouched
        assert b.done() is False

        pending.resolve("session-b", False)
        assert await b is False

    @pytest.mark.asyncio
    async def test_has_pending_is_scoped_to_the_session(self) -> None:
        pending = PendingApprovals("webex")
        task = asyncio.create_task(pending.decide("session-a", _event()))
        await _until(lambda: pending.has_pending("session-a"))

        assert pending.has_pending("session-b") is False
        pending.resolve("session-a", True)
        await task

    @pytest.mark.asyncio
    async def test_a_session_key_that_is_a_prefix_of_another_is_not_matched(self) -> None:
        # Keys are compared with the ":" separator attached, so "webex:a" must
        # not resolve a prompt pending for "webex:ab".
        pending = PendingApprovals("webex")
        task = asyncio.create_task(pending.decide("webex:ab", _event()))
        await _until(lambda: pending.has_pending("webex:ab"))

        assert pending.has_pending("webex:a") is False
        assert pending.resolve("webex:a", True) is False

        pending.resolve("webex:ab", True)
        await task


class TestDenyByDefault:
    @pytest.mark.asyncio
    async def test_an_unanswered_prompt_denies_and_stops_waiting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deny-on-timeout, driven by a patched window rather than a real sleep.

        Waiting out the production timeout would make this test take minutes;
        asserting on the RESULT of an elapsed window is the property that
        matters, so shrink the window instead of sleeping.
        """
        monkeypatch.setattr("kiro_crew.messaging.approval.APPROVAL_TIMEOUT_S", 0.01)
        pending = PendingApprovals("webex")

        assert await pending.decide("s1", _event()) is False
        # And the entry is gone, so a reply arriving after the window is told the
        # prompt expired rather than silently doing nothing.
        assert pending.resolve("s1", True) is False

    @pytest.mark.asyncio
    async def test_an_event_with_no_request_id_still_resolves(self) -> None:
        # A backend that omits request_id must not make the prompt unanswerable:
        # the key degrades to the session plus an empty id, which is still unique
        # per session.
        pending = PendingApprovals("webex")
        task = asyncio.create_task(pending.decide("s1", SimpleNamespace()))
        await _until(lambda: pending.has_pending("s1"))

        assert pending.resolve("s1", True) is True
        assert await task is True


class TestDecider:
    @pytest.mark.asyncio
    async def test_the_decider_binds_one_session(self) -> None:
        pending = PendingApprovals("webex")
        decider = SessionApprovalDecider(pending, session_key="s1")
        task = asyncio.create_task(decider(_event()))
        await _until(lambda: pending.has_pending("s1"))

        assert pending.resolve("s1", True) is True
        assert await task is True

    def test_key_is_session_scoped(self) -> None:
        assert PendingApprovals.key("webex:a", 7) == "webex:a:7"


async def _until(predicate, timeout: float = 1.0) -> None:
    """Poll *predicate* until true. Polling, not sleeping.

    A fixed sleep long enough to be reliable on a loaded CI box is also long
    enough to dominate the suite, and a short one is a flake.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition not reached")


class TestExactIdResolution:
    """The affordance-agnostic half: a widget channel resolves by id.

    A typed reply names no request id — the user answers "the question on
    screen" — so the default is oldest-first. A button press DOES carry the
    correlation id, and that is the shape a later card or button channel needs to
    migrate onto this registry rather than growing a fourth copy.
    """

    @pytest.mark.asyncio
    async def test_an_exact_request_id_resolves_only_that_prompt(self) -> None:
        pending = PendingApprovals("webex")
        first = asyncio.create_task(pending.decide("s1", _event(1)))
        await _until(lambda: pending.has_pending("s1"))
        second = asyncio.create_task(pending.decide("s1", _event(2)))
        await _until(lambda: len(_pending_keys(pending)) == 2)

        assert pending.resolve("s1", True, request_id=2) is True
        assert await second is True
        assert first.done() is False

        pending.resolve("s1", False, request_id=1)
        assert await first is False

    @pytest.mark.asyncio
    async def test_an_unknown_request_id_resolves_nothing(self) -> None:
        pending = PendingApprovals("webex")
        task = asyncio.create_task(pending.decide("s1", _event(1)))
        await _until(lambda: pending.has_pending("s1"))

        assert pending.resolve("s1", True, request_id=99) is False
        assert task.done() is False

        pending.resolve("s1", True)
        await task

    @pytest.mark.asyncio
    async def test_without_an_id_the_oldest_prompt_resolves_first(self) -> None:
        pending = PendingApprovals("webex")
        first = asyncio.create_task(pending.decide("s1", _event(1)))
        await _until(lambda: pending.has_pending("s1"))
        second = asyncio.create_task(pending.decide("s1", _event(2)))
        await _until(lambda: len(_pending_keys(pending)) == 2)

        assert pending.resolve("s1", True) is True
        assert await first is True
        assert second.done() is False

        pending.resolve("s1", False)
        assert await second is False


def _pending_keys(pending: PendingApprovals) -> list[str]:
    """The registry's live keys. Reaches into private state on purpose: the
    alternative is a public accessor that exists only for tests."""
    return list(pending._pending)


class TestWidgetNonce:
    """A widget's nonce is validated INSIDE resolve, as a precondition.

    A channel that resolved first and validated after would have already approved
    the tool by the time it decided the press was stale — the only thing left to
    suppress is the confirmation message. This matters on a platform that cannot
    retire a resolved widget (Webex refuses to edit a message carrying an
    attachment), where the buttons stay clickable forever.
    """

    @pytest.mark.asyncio
    async def test_the_minted_nonce_resolves_the_prompt_it_was_minted_for(self) -> None:
        pending = PendingApprovals("webex")
        task = asyncio.create_task(pending.decide("s1", _event(1)))
        await _until(lambda: pending.has_pending("s1"))
        nonce = pending.reserve("s1", 1)

        assert pending.resolve("s1", True, request_id=1, expected_nonce=nonce) is True
        assert await task is True

    @pytest.mark.asyncio
    async def test_a_wrong_nonce_leaves_the_prompt_pending(self) -> None:
        pending = PendingApprovals("webex")
        task = asyncio.create_task(pending.decide("s1", _event(1)))
        await _until(lambda: pending.has_pending("s1"))
        pending.reserve("s1", 1)

        assert pending.resolve("s1", True, request_id=1, expected_nonce="wrong") is False
        assert task.done() is False

        pending.resolve("s1", False, request_id=1)
        assert await task is False

    @pytest.mark.asyncio
    async def test_an_unminted_prompt_refuses_every_nonce(self) -> None:
        # Fail closed: a prompt with no widget has no press to honour.
        pending = PendingApprovals("webex")
        task = asyncio.create_task(pending.decide("s1", _event(1)))
        await _until(lambda: pending.has_pending("s1"))

        assert pending.resolve("s1", True, request_id=1, expected_nonce="anything") is False
        pending.resolve("s1", False, request_id=1)
        assert await task is False

    @pytest.mark.asyncio
    async def test_a_nonce_dies_with_the_decision_it_guards(self) -> None:
        """The reason it is minted against the pending ENTRY.

        A renderer-owned nonce outlives the turn; this one is retired by the same
        ``finally`` that retires the future, so a press on a spent widget cannot
        answer a LATER prompt that reused the request id.
        """
        pending = PendingApprovals("webex")
        first = asyncio.create_task(pending.decide("s1", _event(1)))
        await _until(lambda: pending.has_pending("s1"))
        stale = pending.reserve("s1", 1)
        pending.resolve("s1", True, request_id=1, expected_nonce=stale)
        assert await first is True

        second = asyncio.create_task(pending.decide("s1", _event(1)))
        await _until(lambda: pending.has_pending("s1"))

        assert pending.resolve("s1", True, request_id=1, expected_nonce=stale) is False
        pending.resolve("s1", False, request_id=1)
        assert await second is False

    @pytest.mark.asyncio
    async def test_a_typed_answer_passes_no_nonce_and_is_unaffected(self) -> None:
        # There is no widget to have gone stale, and the sender was authorized
        # upstream — so the guard must not apply to the typed path.
        pending = PendingApprovals("webex")
        task = asyncio.create_task(pending.decide("s1", _event(1)))
        await _until(lambda: pending.has_pending("s1"))
        pending.reserve("s1", 1)

        assert pending.resolve("s1", True) is True
        assert await task is True


class TestApprovalStallSignalling:
    @pytest.mark.asyncio
    async def test_a_timeout_signals_autonudge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unanswered prompt is the only evidence an unattended loop can no
        longer act.

        Without the signal a monitor loop bound to this conversation keeps firing,
        is denied every interactive tool, and burns its whole cycle budget while
        reporting itself healthy — its per-turn cap is measured in tens of minutes
        and the approval window in minutes.
        """
        from kiro_crew import autonudge

        stalled: list[str] = []
        monkeypatch.setattr(approval, "APPROVAL_TIMEOUT_S", 0.01)
        monkeypatch.setattr(
            autonudge,
            "get_instance",
            lambda: SimpleNamespace(notify_approval_stalled=stalled.append),
        )

        pending = PendingApprovals("webex")
        assert await pending.decide("webex:agent:direct:a@b.com", _event(1)) is False
        assert stalled == ["webex:agent:direct:a@b.com"]

    @pytest.mark.asyncio
    async def test_a_key_no_loop_can_bind_to_is_not_signalled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kiro_crew import autonudge

        stalled: list[str] = []
        monkeypatch.setattr(approval, "APPROVAL_TIMEOUT_S", 0.01)
        monkeypatch.setattr(
            autonudge,
            "get_instance",
            lambda: SimpleNamespace(notify_approval_stalled=stalled.append),
        )

        pending = PendingApprovals("webex")
        assert await pending.decide("cron:nightly", _event(1)) is False
        assert stalled == []

    @pytest.mark.asyncio
    async def test_a_signalling_failure_never_changes_the_denial(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A monitoring convenience must not be able to turn a denied tool into a
        # raised exception inside the turn.
        from kiro_crew import autonudge

        def _boom() -> object:
            raise RuntimeError("autonudge is unavailable")

        monkeypatch.setattr(approval, "APPROVAL_TIMEOUT_S", 0.01)
        monkeypatch.setattr(autonudge, "get_instance", _boom)

        pending = PendingApprovals("webex")
        assert await pending.decide("webex:agent:direct:a@b.com", _event(1)) is False


class TestReservationRace:
    """The decision window opens BEFORE the prompt is sent.

    ``TurnDriver`` dispatches ``PROMPT_CHOICE`` and only then awaits the decider,
    so the prompt is visible in the room for a whole REST round trip before
    ``decide`` would have registered anything. An answer arriving in that window
    used to find nothing pending, fall through to the mid-turn path, and be
    discarded — the user watched their decision do nothing and the tool denied
    itself minutes later.
    """

    @pytest.mark.asyncio
    async def test_an_answer_before_decide_still_resolves(self) -> None:
        pending = PendingApprovals("webex")
        nonce = pending.reserve("s1", 1)

        # The reply lands while the prompt is still being delivered.
        assert pending.has_pending("s1")
        assert pending.resolve("s1", True, request_id=1, expected_nonce=nonce) is True

        # The driver reaches the decider afterwards and reads the decision.
        assert await pending.decide("s1", _event(1)) is True

    @pytest.mark.asyncio
    async def test_reserving_does_not_orphan_a_live_future(self) -> None:
        """Never replace a future someone is already awaiting.

        A second reservation — or one that follows ``decide`` — must keep the
        existing future, or a resolved answer sets an object nobody reads and the
        prompt hangs for its whole window before denying.
        """
        pending = PendingApprovals("webex")
        task = asyncio.create_task(pending.decide("s1", _event(1)))
        await _until(lambda: pending.has_pending("s1"))

        nonce = pending.reserve("s1", 1)
        again = pending.reserve("s1", 1)

        assert nonce == again
        assert pending.resolve("s1", True, request_id=1, expected_nonce=nonce) is True
        assert await task is True

    @pytest.mark.asyncio
    async def test_an_unawaited_reservation_is_discarded_with_the_turn(self) -> None:
        """The prompt rendered and the turn then failed before the decider.

        Left behind, that reservation outlives its turn and a stray answer to a
        LATER prompt could resolve it.
        """
        pending = PendingApprovals("webex")
        pending.reserve("s1", 1)
        assert pending.has_pending("s1")

        pending.discard_reservations("s1")

        assert not pending.has_pending("s1")
        assert pending.resolve("s1", True, request_id=1) is False

    @pytest.mark.asyncio
    async def test_discarding_leaves_another_sessions_reservation_alone(self) -> None:
        pending = PendingApprovals("webex")
        pending.reserve("s1", 1)
        pending.reserve("s2", 1)

        pending.discard_reservations("s1")

        assert not pending.has_pending("s1")
        assert pending.has_pending("s2")

    @pytest.mark.asyncio
    async def test_discarding_does_not_disturb_an_awaited_prompt(self) -> None:
        # ``decide`` retires its own entry, so a teardown running while a decision
        # is genuinely in flight must not cancel it.
        pending = PendingApprovals("webex")
        task = asyncio.create_task(pending.decide("s1", _event(1)))
        await _until(lambda: pending.has_pending("s1"))

        pending.discard_reservations("s1")

        assert pending.resolve("s1", True) is False
        assert task.done() is False
        task.cancel()
