"""Finalization contract of the shared channel turn pipeline.

``drive_turn`` owns the semaphore lifetime for every adopted channel, so a bug
in its ``finally`` is a bug in all of them at once. These tests pin the part
that is invisible on the happy path: what happens to ``release()`` when
finalization itself fails.
"""

from __future__ import annotations

import asyncio
from typing import Any

from kiro_crew.messaging import dispatch as D
from kiro_crew.messaging.dispatch import ChannelTurn, drive_turn
from kiro_crew.messaging.renderer import SilentRenderer


class _Sessions:
    """Minimal stand-in that counts the calls this contract is about."""

    def __init__(self, raise_on_acquire: bool = False):
        self.released = 0
        self.successes = 0
        self.failures = 0
        self._raise_on_acquire = raise_on_acquire

    async def get_or_create(self, key, agent=None, channel_id=None):
        if self._raise_on_acquire:
            raise RuntimeError("cold start failed")
        return object(), False, False

    async def set_channel(self, key, channel_id):
        pass

    def record_success(self, key):
        self.successes += 1

    async def record_failure(self, key):
        self.failures += 1

    def release(self, key):
        self.released += 1

    def get_provider(self, key):
        return object()


class _Renderer:
    """Renderer whose ``close`` can fail the way a real one can mid-flush."""

    def __init__(self, close_raises: bool = False):
        self.close_raises = close_raises
        self.closed = 0

    async def on_turn_start(self):
        pass

    async def close(self):
        self.closed += 1
        if self.close_raises:
            raise RuntimeError("renderer finalization failed")


class _CtxBuilder:
    def build_message(self, text, is_new, session_key, **kw):
        return text, None


class _Driver:
    def __init__(self, *a, **kw):
        pass

    async def run(self, message):
        return "the reply"


def _turn(renderer: Any) -> ChannelTurn:
    return ChannelTurn(
        channel_type="weixin",
        session_key="weixin:agentA:direct:userA",
        conversation_id="weixin:userA",
        agent="agentA",
        user_text="hi",
        renderer=renderer,
        approval_mode="auto",
    )


def _patch_pipeline(monkeypatch, *, permitted: bool = True):
    """Stub everything drive_turn touches except the finalization under test."""

    async def _permitted(_channel_type):
        return permitted

    async def _publish(_sessions, _key):
        pass

    async def _embed(fn, *args, **kw):
        return fn(*args, **kw)

    monkeypatch.setattr(D, "inbound_permitted", _permitted)
    monkeypatch.setattr(D, "publish_turn_identity", _publish)
    monkeypatch.setattr(D, "run_in_embed_pool", _embed)
    monkeypatch.setattr(D, "TurnDriver", _Driver)


def test_release_still_runs_when_renderer_close_fails(monkeypatch) -> None:
    """A failed renderer.close must NOT strand the session semaphore.

    The semaphore is keyed by SESSION, so leaking it does not merely lose this
    turn -- every later message for that conversation blocks forever and any
    queued turn never drains, until the gateway restarts.
    """
    _patch_pipeline(monkeypatch)
    sessions = _Sessions()
    renderer = _Renderer(close_raises=True)

    asyncio.run(drive_turn(_turn(renderer), sessions=sessions, ctx_builder=_CtxBuilder()))

    assert renderer.closed == 1, "close should still be attempted"
    assert sessions.released == 1, (
        "renderer.close raised and the session was never released -- the "
        "conversation is now permanently busy"
    )


def test_a_failing_close_does_not_escape_drive_turn(monkeypatch) -> None:
    """The failure is logged and swallowed, not raised at the caller.

    Adopters call drive_turn from a per-message task; letting finalization
    raise would surface as an unhandled task exception for a turn that already
    delivered its reply.
    """
    _patch_pipeline(monkeypatch)
    sessions = _Sessions()

    # asyncio.run re-raises anything drive_turn lets escape.
    asyncio.run(
        drive_turn(
            _turn(_Renderer(close_raises=True)),
            sessions=sessions,
            ctx_builder=_CtxBuilder(),
        )
    )

    assert sessions.successes == 1, "the turn itself succeeded"


def test_release_is_not_called_when_the_semaphore_was_never_acquired(monkeypatch) -> None:
    """The _acquired gate must survive the new guard.

    A cold-start failure raises before get_or_create returns, so nothing was
    ever held -- releasing here would hand back a permit that does not exist.
    """
    _patch_pipeline(monkeypatch)
    sessions = _Sessions(raise_on_acquire=True)
    renderer = _Renderer(close_raises=True)

    asyncio.run(drive_turn(_turn(renderer), sessions=sessions, ctx_builder=_CtxBuilder()))

    assert renderer.closed == 1, "finalization still runs on the failure path"
    assert sessions.released == 0, "nothing was acquired, so nothing may be released"
    assert sessions.failures == 0, "record_failure is also gated on _acquired"


def test_the_happy_path_releases_exactly_once(monkeypatch) -> None:
    """Guard rail: the new try/except must not double-release."""
    _patch_pipeline(monkeypatch)
    sessions = _Sessions()
    renderer = _Renderer()

    asyncio.run(drive_turn(_turn(renderer), sessions=sessions, ctx_builder=_CtxBuilder()))

    assert renderer.closed == 1
    assert sessions.released == 1
    assert sessions.successes == 1


def test_a_denied_turn_neither_renders_nor_releases(monkeypatch) -> None:
    """Governance backstop returns before any side effect."""
    _patch_pipeline(monkeypatch, permitted=False)
    sessions = _Sessions()
    renderer = _Renderer()

    asyncio.run(drive_turn(_turn(renderer), sessions=sessions, ctx_builder=_CtxBuilder()))

    assert renderer.closed == 0
    assert sessions.released == 0
    assert sessions.successes == 0


class _PauseSessions(_Sessions):
    """Interface parity with the real SessionManager for the pause lookup.

    Extended here rather than leaning on production's fail-open: that fallback
    exists for the bare ``MagicMock`` managers elsewhere in the suite, and a test
    about the gate must not be silently exercising the fallback instead.
    """

    def __init__(self, paused: bool = False):
        super().__init__()
        self.paused = paused
        self.pause_calls: list[tuple[str, bool]] = []

    def is_mirror_paused(self, key, *, origin=False):
        self.pause_calls.append((key, origin))
        return self.paused


class _CountingRenderer(_Renderer):
    """Records the turn-start the user would SEE as a typing indicator."""

    def __init__(self):
        super().__init__()
        self.started = 0

    async def on_turn_start(self):
        self.started += 1


def _capture_driver(box: list) -> type:
    class _Capturing(_Driver):
        def __init__(self, provider, renderer, **kw):
            super().__init__()
            box.append(renderer)

    return _Capturing


def _turn_with_key(renderer: Any, session_key: str) -> ChannelTurn:
    return ChannelTurn(
        channel_type="weixin",
        session_key=session_key,
        conversation_id="weixin:userA",
        agent="agentA",
        user_text="hi",
        renderer=renderer,
        approval_mode="auto",
    )


def test_a_disconnected_conversation_is_silenced(monkeypatch) -> None:
    """Disconnect stops the replies, which for a non-Slack channel happens HERE.

    Slack enforces a disconnect on its own streaming mirror. Every other channel
    answers through this pipeline, so before this gate a disconnected channel
    kept replying and the dashboard control changed nothing but its own label.

    The turn still runs and the semaphore is still released: the binding is
    retained by design, so the inbound message must still land in the session.
    """
    box: list[Any] = []
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(D, "TurnDriver", _capture_driver(box))
    sessions = _PauseSessions(paused=True)
    renderer = _CountingRenderer()

    asyncio.run(
        drive_turn(
            _turn_with_key(renderer, "weixin:agentA:direct:userA"),
            sessions=sessions,
            ctx_builder=_CtxBuilder(),
        )
    )

    assert isinstance(box[0], SilentRenderer), "the driver must stream into the silent one"
    assert renderer.started == 0, "a disconnected conversation must not even show typing"
    assert renderer.closed == 0, "the real renderer was never used, so it has nothing to close"
    assert sessions.successes == 1, "the turn still ran"
    assert sessions.released == 1, "and the session semaphore was still released"


def test_a_connected_conversation_keeps_its_real_renderer(monkeypatch) -> None:
    """The non-vacuity half: without it, deleting the gate would still pass above."""
    box: list[Any] = []
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(D, "TurnDriver", _capture_driver(box))
    sessions = _PauseSessions(paused=False)
    renderer = _CountingRenderer()

    asyncio.run(
        drive_turn(
            _turn_with_key(renderer, "weixin:agentA:direct:userA"),
            sessions=sessions,
            ctx_builder=_CtxBuilder(),
        )
    )

    assert box[0] is renderer
    assert renderer.started == 1
    assert renderer.closed == 1


def test_the_pause_is_read_for_the_role_the_turn_arrived_on(monkeypatch) -> None:
    """Two non-Slack deliveries mute independently, so the ROLE decides the flag.

    A channel-BORN session's key IS its conversation, so a turn arriving in that
    namespace is the origin. Anything else reaching this pipeline came over a
    mirror/resume binding. Reading the wrong flag would let one row's disconnect
    silence the other's conversation.
    """
    _patch_pipeline(monkeypatch)

    born = _PauseSessions(paused=False)
    asyncio.run(
        drive_turn(
            _turn_with_key(_CountingRenderer(), "weixin:agentA:direct:userA"),
            sessions=born,
            ctx_builder=_CtxBuilder(),
        )
    )
    assert born.pause_calls == [("weixin:agentA:direct:userA", True)], "born-in reads origin"

    mirrored = _PauseSessions(paused=False)
    asyncio.run(
        drive_turn(
            _turn_with_key(_CountingRenderer(), "dashboard:chat-1"),
            sessions=mirrored,
            ctx_builder=_CtxBuilder(),
        )
    )
    assert mirrored.pause_calls == [("dashboard:chat-1", False)], "a mirror reads the mirror flag"


# ---------------------------------------------------------------------------
# What the pipeline forwards to the driver, and what it binds per turn.
#
# Both of these were asymmetries rather than missing features: the field existed
# on the driver and the helper existed in ``link``, but the shared pipeline never
# passed them, so every channel riding ``drive_turn`` (webex, wecom, teams,
# weixin, imessage) silently lost a capability the forked channels had.
# ---------------------------------------------------------------------------


class _MirrorSessions(_Sessions):
    """Adds the origin/mirror surface ``drive_turn`` binds through."""

    def __init__(self, *, opt_out: bool = False, existing=None, raises: bool = False):
        super().__init__()
        self.origin_links: dict = {}
        self.mirror_links: dict = {} if existing is None else dict(existing)
        self._opt_out = opt_out
        self._raises = raises

    def set_origin_link(self, key, link):
        if self._raises:
            raise RuntimeError("session map unavailable")
        self.origin_links[key] = link

    def mirror_opt_out(self, key) -> bool:
        return self._opt_out

    def get_mirror_link(self, key):
        return self.mirror_links.get(key)

    def set_mirror_link(self, key, link, *, reason=""):
        self.mirror_links[key] = link


def _capture_turn_driver(box: dict) -> type:
    class _Capturing(_Driver):
        def __init__(self, provider, renderer, **kw):
            box.update(kw)
            super().__init__(provider, renderer, **kw)

    return _Capturing


def test_session_auto_approve_is_forwarded_to_the_driver(monkeypatch) -> None:
    """Without this, ``/yolo`` and per-session Trust are inert on four channels.

    The driver has had the rung all along; the shared pipeline simply never
    handed it the predicate, so the grant an operator took could not reach a tool
    on any channel that rides ``drive_turn``.
    """
    kwargs: dict = {}
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(D, "TurnDriver", _capture_turn_driver(kwargs))
    turn = _turn(_Renderer())
    turn.auto_approve_session = lambda: True

    asyncio.run(drive_turn(turn, sessions=_Sessions(), ctx_builder=_CtxBuilder()))

    assert kwargs["auto_approve_session"] is turn.auto_approve_session
    assert kwargs["auto_approve_session"]() is True


def test_a_turn_that_omits_the_predicate_still_runs(monkeypatch) -> None:
    # The field is additive with a safe default, so no existing adopter changes
    # behaviour by not setting it.
    kwargs: dict = {}
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(D, "TurnDriver", _capture_turn_driver(kwargs))

    asyncio.run(drive_turn(_turn(_Renderer()), sessions=_Sessions(), ctx_builder=_CtxBuilder()))

    assert kwargs["auto_approve_session"] is None


def test_the_origin_conversation_is_recorded_and_bound(monkeypatch) -> None:
    from kiro_crew.messaging.link import ChannelLink

    _patch_pipeline(monkeypatch)
    sessions = _MirrorSessions()
    turn = _turn(_Renderer())
    turn.origin_conversation = ChannelLink("weixin", channel_id="ROOM", thread_id=None)

    asyncio.run(drive_turn(turn, sessions=sessions, ctx_builder=_CtxBuilder()))

    assert sessions.origin_links[turn.session_key].channel_id == "ROOM"
    assert sessions.mirror_links[turn.session_key].channel_id == "ROOM"


def test_a_turn_that_omits_the_origin_conversation_binds_nothing(monkeypatch) -> None:
    _patch_pipeline(monkeypatch)
    sessions = _MirrorSessions()

    asyncio.run(drive_turn(_turn(_Renderer()), sessions=sessions, ctx_builder=_CtxBuilder()))

    assert sessions.origin_links == {}
    assert sessions.mirror_links == {}


def test_the_persisted_opt_out_is_honoured(monkeypatch) -> None:
    """An in-channel unlink has to survive the user's next message.

    The bind is re-asserted every turn, so without reading the opt-out "off"
    would last exactly until they typed again.
    """
    from kiro_crew.messaging.link import ChannelLink

    _patch_pipeline(monkeypatch)
    sessions = _MirrorSessions(opt_out=True)
    turn = _turn(_Renderer())
    turn.origin_conversation = ChannelLink("weixin", channel_id="ROOM", thread_id=None)

    asyncio.run(drive_turn(turn, sessions=sessions, ctx_builder=_CtxBuilder()))

    assert sessions.mirror_links == {}


def test_a_binding_aimed_elsewhere_is_not_repointed(monkeypatch) -> None:
    # The dashboard can aim a session's mirror at any surface; overwriting it
    # would silently redirect the user's replies into this conversation.
    from kiro_crew.messaging.link import ChannelLink

    _patch_pipeline(monkeypatch)
    elsewhere = ChannelLink("discord", channel_id="99", thread_id=None)
    sessions = _MirrorSessions(existing={"weixin:agentA:direct:userA": elsewhere})
    turn = _turn(_Renderer())
    turn.origin_conversation = ChannelLink("weixin", channel_id="ROOM", thread_id=None)

    asyncio.run(drive_turn(turn, sessions=sessions, ctx_builder=_CtxBuilder()))

    assert sessions.mirror_links["weixin:agentA:direct:userA"] is elsewhere


def test_a_bind_failure_does_not_drop_the_turn(monkeypatch) -> None:
    """This is the widest call site in the codebase — five channels route here.

    Losing the mirror costs a dashboard convenience; raising costs the user the
    answer they are waiting for.
    """
    from kiro_crew.messaging.link import ChannelLink

    _patch_pipeline(monkeypatch)
    sessions = _MirrorSessions(raises=True)
    turn = _turn(_Renderer())
    turn.origin_conversation = ChannelLink("weixin", channel_id="ROOM", thread_id=None)

    asyncio.run(drive_turn(turn, sessions=sessions, ctx_builder=_CtxBuilder()))

    assert sessions.successes == 1
    assert sessions.released == 1


class _KnownProviderSessions(_Sessions):
    """Returns an identifiable provider, so the hook's argument can be asserted."""

    def __init__(self) -> None:
        super().__init__()
        self.provider = object()

    async def get_or_create(self, key, agent=None, channel_id=None, **kw):
        return self.provider, False, False


def test_the_live_provider_is_handed_to_the_channel(monkeypatch) -> None:
    """A channel that uploads local files needs the provider's own cwd as the
    extraction root, and that is unknowable until ``get_or_create`` returns.

    Reading it from the session map BEFORE the turn yields ``None`` on the first
    message of every session generation, so the feature is silently off for
    exactly the turn that introduces it and mysteriously on afterwards.
    """
    seen: list = []
    _patch_pipeline(monkeypatch)
    sessions = _KnownProviderSessions()
    turn = _turn(_Renderer())
    turn.bind_provider = seen.append

    asyncio.run(drive_turn(turn, sessions=sessions, ctx_builder=_CtxBuilder()))

    assert seen == [sessions.provider]


def test_the_hook_runs_before_the_driver(monkeypatch) -> None:
    # Whatever it authorizes has to be in place for the turn it belongs to, not
    # the next one.
    order: list[str] = []
    _patch_pipeline(monkeypatch)

    class _OrderedDriver(_Driver):
        def __init__(self, *a, **kw) -> None:
            order.append("driver")
            super().__init__(*a, **kw)

    monkeypatch.setattr(D, "TurnDriver", _OrderedDriver)
    turn = _turn(_Renderer())
    turn.bind_provider = lambda _p: order.append("bind")

    asyncio.run(drive_turn(turn, sessions=_Sessions(), ctx_builder=_CtxBuilder()))

    assert order == ["bind", "driver"]


def test_a_failing_hook_degrades_the_feature_not_the_turn(monkeypatch) -> None:
    # Guarded like the origin bind: what it authorizes is an enhancement, so a
    # failure must not drop an answer the user is waiting for.
    _patch_pipeline(monkeypatch)
    sessions = _Sessions()
    turn = _turn(_Renderer())

    def _boom(_provider) -> None:
        raise RuntimeError("no cwd")

    turn.bind_provider = _boom

    asyncio.run(drive_turn(turn, sessions=sessions, ctx_builder=_CtxBuilder()))

    assert sessions.successes == 1
    assert sessions.released == 1


def test_a_turn_that_omits_the_hook_still_runs(monkeypatch) -> None:
    _patch_pipeline(monkeypatch)
    sessions = _Sessions()

    asyncio.run(drive_turn(_turn(_Renderer()), sessions=sessions, ctx_builder=_CtxBuilder()))

    assert sessions.successes == 1
