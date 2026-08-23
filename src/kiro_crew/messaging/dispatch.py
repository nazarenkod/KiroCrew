"""Shared channel turn pipeline — one copy of the dispatch skeleton.

This module owns the sequence every non-Slack channel dispatcher runs around
:class:`TurnDriver`:

    governance gate
    -> renderer.on_turn_start()          (typing indicator before cold start)
    -> sessions.get_or_create + set_channel
    -> publish_turn_identity
    -> ctx_builder.build_message         (off-loop, embeds block)
    -> TurnDriver.run                    (shared redaction + approval ladder)
    -> post-turn: record_success, persist, threshold notice, SEL audit
                                         (each guarded independently)
    -> finally: renderer.close() + release (release gated on acquire)

What stays per-channel is what actually differs between them: the wire
protocol, event normalization, ``authorize()`` semantics, rendering, command
vocabulary, and the ack strings. Channels inject those through
:class:`ChannelTurn` rather than subclassing, so a capability this protocol
lacks widens the protocol once instead of forking the pipeline.

Dependency direction is ``<channel> -> messaging`` (never the reverse), so this
module must not import any channel package.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from kiro_crew.executors import run_in_embed_pool
from kiro_crew.hooks import TOOL_AUTO_APPROVE, TOOL_DENY
from kiro_crew.messaging.driver import DirectiveConsumer, TurnDriver
from kiro_crew.messaging.identity import channel_inbound_permitted, publish_turn_identity
from kiro_crew.messaging.link import (
    ChannelLink,
    bind_origin_mirror,
    channel_namespace_of,
    is_channel_session_key,
)
from kiro_crew.messaging.renderer import SilentRenderer
from kiro_crew.security import redact_credentials, redact_exfiltration_urls
from kiro_crew.sel import sel

logger = logging.getLogger(__name__)


@dataclass
class ChannelTurn:
    """Everything the pipeline needs that varies per channel.

    ``persist`` runs in a worker thread (it does blocking history I/O), so it
    must be a plain sync callable. ``notice`` and ``renderer`` are async.
    """

    channel_type: str
    """Governance member name, e.g. ``"weixin"``. Gates every inbound message.

    Invariant: this MUST equal the first ``:``-segment of ``session_key`` —
    the surface name is the routing authority for governance and (future)
    control-plane operations. Holds by construction today because every
    adopter builds its key via ``build_dm_session_key(channel_type, ...)``.
    """

    session_key: str
    """The session address. OPAQUE to this pipeline: it is passed through to
    ``sessions.*`` verbatim and never parsed, split, or rebuilt here. Keys are
    constructed channel-side via :func:`kiro_crew.messaging.link.build_dm_session_key`
    (``{surface}:{agent}:{chat_type}:{scope}[:genN]``). Keeping the pipeline
    address-agnostic is what lets the address grammar evolve (deeper scope
    paths, new surfaces) without touching dispatch.
    """

    conversation_id: str
    """Stable identity of the conversation on its transport (e.g.
    ``"weixin:{user_id}"``), used for session attribution/UI. Feeds the
    legacy ``channel_id=`` kwarg of ``sessions.get_or_create`` /
    ``set_channel`` — the old name survives at that API boundary only.
    Distinct from BOTH ``channel_type`` (the transport) and the app-platform
    ``channel:`` concept.
    """

    agent: str
    user_text: str
    renderer: Any
    """A :class:`kiro_crew.messaging.renderer.Renderer`."""

    approval_mode: str
    decider: Optional[Any] = None
    """``None`` for channels with no interactive approval affordance
    (deny-by-default for INTERACTIVE mode; ``auto``/``trust`` still work)."""

    auto_approve_session: Optional[Callable[[], bool]] = None
    """``() -> bool``: auto-approve every permission request in this turn.

    Carries per-session Trust / global YOLO into the driver. A channel that
    omits it gets the deny-by-default ladder, so an operator who turned YOLO on
    would find it silently inert on that channel — the grant is read PER
    REQUEST rather than captured at turn start so a mid-turn revoke takes
    effect. The PreToolUse ``tool_gate`` still runs first, so a hard deny can
    never be overridden by it.
    """

    bind_provider: Optional[Callable[[Any], None]] = None
    """``(provider) -> None``, called once the session's provider exists.

    The hook for anything a renderer can only be told AFTER the session is
    resolved — a channel that uploads local files needs the provider's own cwd as
    the extraction root, and that is unknowable until ``get_or_create`` returns.
    A channel reading it BEFORE the turn gets ``None`` on the first message of
    every session generation, so the feature is silently off for exactly the turn
    that introduces it and mysteriously on afterwards.

    Guarded like the origin bind: whatever it authorizes is an enhancement to the
    turn, so a failure here degrades that one feature rather than dropping an
    answer the user is waiting for.
    """

    persist: Optional[Callable[[str, str, bool], None]] = None
    """``(user_text, reply_text, is_new) -> None``, called off the event loop."""

    notice: Optional[Callable[[str, Any], Awaitable[None]]] = None
    """``(session_key, provider) -> None`` post-turn threshold handling."""

    after_persist: Optional[Callable[[], Awaitable[None]]] = None
    """Optional loop-side callback after persistence, such as dashboard surfacing."""

    directive_consumer: Optional[DirectiveConsumer] = None
    """Session-directive consumer for this turn (``build_directive_consumer``).
    ``None`` leaves directive-tool results inert — the pre-consumer behavior."""

    model: Optional[str] = None
    """Model id for a NEW session, or ``None`` to let config decide.

    Reaches a session only at CREATION: ``get_or_create`` returns a reused
    session from its fast path before it consults this, so a channel-side model
    pick applies to the next fresh conversation rather than retroactively to the
    running one. A channel that offers a model command has to say so in its reply
    or the user reads the switch as broken.

    Never a hardcoded id — the value comes from what the session's backend
    advertised, which is the set THIS account may actually use.
    """

    origin_conversation: Optional[ChannelLink] = None
    """A :class:`~kiro_crew.messaging.link.ChannelLink` naming THIS conversation.

    Supplying it makes the conversation both the session's origin (so unattended
    output about the session — the auto-compact notice — has somewhere to go) and
    its own outbound mirror (so a turn the user later takes from the dashboard
    comes back here instead of leaving the chat looking dead).

    ``None`` means the channel opts out, and its conversations stay unmirrored.
    Both writes are steady-state READS after the first turn, so this costs a map
    lookup per turn rather than a rewrite; see
    :func:`kiro_crew.messaging.link.bind_origin_mirror` for why the bind must be
    re-asserted on every turn rather than only on a new session, and for the
    binding it deliberately declines to overwrite.

    It MUST be the same value the channel's own unlink command hands
    ``release_conversation_location``, which matches an occupied location by
    VALUE — a second spelling of "this conversation" would let the release miss
    the binding this wrote. Channels define it once and reuse it for both.
    """

    audit_caller: str = ""
    """SEL audit caller label; defaults to ``<channel_type>:unknown``."""


async def inbound_permitted(channel_type: str) -> bool:
    """Per-message governance gate.

    Rechecked on every message (not just at connect) so a host-profile deny
    added while the transport is live stops dispatch without a restart. The
    pipeline calls this itself, so a channel cannot forget it.
    """
    if await channel_inbound_permitted(channel_type):
        return True
    logger.info("%s inbound dropped: denied by channels governance policy", channel_type)
    return False


def build_tool_gate(ctx_builder: Any, *, session_key: str, agent: str) -> Callable[[Any], str]:
    """PreToolUse security gate, channel-neutral (off ``ctx_builder.hooks``).

    Sensitive-path keystone + governance ceiling + deny-list. Returns ``"deny"``
    (un-overridable), ``"auto_approve"``, or ``""`` (passthrough). Built here so
    no channel package needs to import ``kiro_crew.slack``.
    """

    def _tool_gate(event: Any) -> str:
        result = ctx_builder.hooks.on_tool_call(
            getattr(event, "title", "") or "",
            session_key=session_key,
            agent=agent,
            tool_kind=getattr(event, "tool_kind", "") or "",
            raw_params=getattr(event, "raw_tool_params", None),
            command=getattr(event, "shell_command", None),
            is_shell=bool(getattr(event, "is_shell", False)),
        )
        if result.action == TOOL_DENY:
            return "deny"
        if result.action == TOOL_AUTO_APPROVE:
            return "auto_approve"
        return ""

    return _tool_gate


def build_auto_approve(ctx_builder: Any) -> Callable[[str], bool]:
    """Preserve the ``auto_approve_subagent_spawn`` hook for ``spawn_run``."""

    def _auto_approve(title: str) -> bool:
        return bool(
            ctx_builder
            and ctx_builder.hooks
            and ctx_builder.hooks.auto_approve_subagent_spawn
            and title == "spawn_run"
        )

    return _auto_approve


@dataclass
class _ChannelDirectiveState:
    """Minimal ``NudgeAuthzState`` stand-in for a turn with no gateway state.

    Carries the one thing the monitor-trio authorizer can validate for a
    channel session — ``sessions`` (Slack routability). The empty ``_slots`` /
    ``channel_transports`` make every other lookup fail CLOSED (deny), never
    crash.
    """

    sessions: Any
    channel_transports: dict[str, Any] = field(default_factory=dict)
    _slots: dict[str, Any] = field(default_factory=dict)


def build_directive_consumer(
    *,
    session_key: str,
    sessions: Any,
    dispatcher: Any = None,
) -> DirectiveConsumer:
    """Session-directive consumer for one channel turn (``TurnDriver`` injection).

    Applies a decoded directive against THIS turn's *session_key* via the shared
    ``apply_session_directive`` core — the same applier the dashboard's
    ``chat_runner`` consumer uses, so the security boundaries (the
    dashboard-only denial, the monitor-trio authorization chokepoint) live in
    exactly one place. A channel turn has no dashboard chat slot, so
    ``slot=None`` is passed and the dashboard-only directives are refused there
    (fail-closed).

    *dispatcher* supplies the live gateway state: each channel dispatcher gets
    ``dashboard_state`` attached at boot (``register_channel_transport``), and
    it is re-read per directive so a consumer built before that attachment
    still sees it. Slack's function-style dispatch has no dispatcher object;
    the minimal *sessions*-backed stand-in covers the Slack routability check,
    and everything it cannot answer fails CLOSED in the authorizer.
    """

    async def _consume(kind: str, args: dict[str, Any]) -> None:
        # Deferred import: the dashboard package imports every channel package
        # at boot, and the channel packages import this module (cycle).
        from kiro_crew.dashboard.session_directive_apply import apply_session_directive

        state: Any = getattr(dispatcher, "dashboard_state", None)
        if state is None:
            state = _ChannelDirectiveState(sessions=sessions)
        result = await apply_session_directive(state, None, session_key, kind, args)
        # The channel surface never renders tool results, so the applier's
        # confirmation has no user-facing sink here; this log is the operator's
        # record (the applier itself SEL-audits every outcome). Failures log at
        # WARNING because a silently dropped effect is the defect this consumer
        # exists to remove. The string interpolates LLM-derived text (a stop
        # reason, a rejected path, exception args), so scrub it like every
        # other LLM-influenced output before it lands anywhere.
        result, _ = redact_exfiltration_urls(result)
        result, _ = redact_credentials(result)
        log = logger.warning if result.startswith(("Error", "Failed")) else logger.info
        log("session directive %s on %s: %s", kind, session_key, result)

    return _consume


def delivery_is_muted(sessions: Any, session_key: str, channel_type: str) -> bool:
    """True when output to *channel_type* must NOT be written back for this session.

    The primitive behind :func:`conversation_is_muted`, taking explicit arguments
    because Discord and Telegram run their OWN copies of the turn loop rather
    than going through :func:`drive_turn`, so they have no ``ChannelTurn`` to
    pass. Every channel that can be disconnected must consult this, or the
    dashboard control is a label with nothing behind it.

    ``origin`` is resolved rather than passed because a session can hold two
    non-Slack deliveries at once, and they mute independently: the conversation
    it was BORN in, and an explicit mirror it was told to post to. This turn came
    from the born-in conversation exactly when the session key IS a channel key
    in this turn's own namespace — a channel-born session's key is its
    conversation. Anything else arriving here is a mirror/resume binding, so it
    reads the mirror flag.

    Slack never reaches these pipelines (it drives its own gateway and is gated by
    ``slack_mirror_is_paused``), so no Slack special-case is needed here.

    Fails OPEN, matching the dashboard-side predicates: ``sessions`` is a bare
    ``MagicMock`` across much of the suite and would return a truthy child for an
    unstubbed accessor, which would silence every channel in the test suite. A
    muted conversation that stays noisy is a visible bug; a live conversation
    silently dead is a much worse one.
    """
    origin = is_channel_session_key(session_key) and (
        channel_namespace_of(session_key) == channel_type
    )
    try:
        return sessions.is_mirror_paused(session_key, origin=origin) is True
    except Exception:
        logger.debug(
            "%s: mirror pause lookup failed session=%s",
            channel_type,
            session_key,
            exc_info=True,
        )
        return False


def conversation_is_muted(sessions: Any, turn: ChannelTurn) -> bool:
    """:func:`delivery_is_muted` for a turn on the shared pipeline."""
    return delivery_is_muted(sessions, turn.session_key, turn.channel_type)


async def drive_turn(turn: ChannelTurn, *, sessions: Any, ctx_builder: Any) -> None:
    """Run one authorized inbound message end to end.

    Everything acquire-dependent runs INSIDE the try so ``finally`` always
    finalizes the turn (``renderer.close``), even when ``get_or_create`` raises
    on a cold-start failure. ``release()`` is gated on ``_acquired`` so a
    semaphore that was never held is never released.
    """
    renderer = turn.renderer
    session_key = turn.session_key
    _acquired = False
    # Enforced governance backstop. Channels SHOULD gate earlier (before any
    # side effect such as a command ack or a generation bump — see the weixin
    # dispatcher, which checks before parse_command), but the pipeline rechecks
    # so an adopter that forgets cannot execute a policy-denied turn. Denied
    # messages are dropped silently, before the typing indicator and before any
    # session is acquired.
    if not await inbound_permitted(turn.channel_type):
        return
    # Substituted BEFORE on_turn_start so a disconnected conversation never even
    # shows a typing indicator, and before TurnDriver so nothing streams. The
    # local name is what the driver and the finally's close() both use, so the
    # real renderer is left completely untouched -- it opened nothing, so there
    # is nothing of its own to finalize.
    if conversation_is_muted(sessions, turn):
        renderer = SilentRenderer(
            getattr(renderer, "capabilities", None),
            getattr(renderer, "channel_type", "") or turn.channel_type,
        )
    try:
        # Typing indicator first (before the potentially slow cold start);
        # on_turn_start is idempotent so the driver's later call no-ops.
        await renderer.on_turn_start()
        # ``model`` is passed ONLY when the channel set one, so an adopter that
        # does not offer a model command calls this with exactly the arguments it
        # always did. Widening the call for everyone would make the new field's
        # cost fall on channels that gain nothing from it.
        extra: dict[str, Any] = {"model": turn.model} if turn.model else {}
        provider, is_new, resumed = await sessions.get_or_create(
            session_key, agent=turn.agent, channel_id=turn.conversation_id, **extra
        )
        _acquired = True
        if is_new:
            await sessions.set_channel(session_key, turn.conversation_id)
        # Bind this conversation as the session's origin AND its own mirror, so
        # unattended notices and dashboard-side turns both reach the user here.
        # After get_or_create, because a cold-start failure leaves no session to
        # bind to; on EVERY turn, because the binding is what a restart, an
        # unlink elsewhere, or a rival claim can take away, and only a
        # self-healing bind cannot leave a live conversation silently unmirrored.
        #
        # Deliberately NOT gated on ``resumed``. Discord skips its bind for a
        # resumed session, but its flag is a mirror-binding LOOKUP ("this turn is
        # answering a dashboard-owned session"), whereas ``resumed`` here means
        # "restored via ACP session/load" — a cold-start recovery of this very
        # conversation, which is exactly the case a self-healing bind exists for.
        # Skipping on it would leave every post-restart session unmirrored.
        #
        # Guarded as a pair. An unbound conversation is a degraded turn — the
        # user still gets their answer here, they just lose the dashboard mirror
        # — whereas a raise on this line drops a turn they are waiting on.
        # ``bind_origin_mirror`` promises not to raise, but that promise covers
        # the ownership conflict it names, not a session accessor failing, and
        # this is the widest call site in the codebase: every channel on the
        # shared pipeline routes through it.
        if turn.origin_conversation is not None:
            try:
                sessions.set_origin_link(session_key, turn.origin_conversation)
                bind_origin_mirror(sessions, key=session_key, location=turn.origin_conversation)
            except Exception:
                logger.warning(
                    "%s: origin/mirror bind failed session=%s",
                    turn.channel_type,
                    session_key,
                    exc_info=True,
                )
        # Hand the live provider to whatever the channel could not resolve before
        # the session existed. Before the driver runs, so the first turn of a
        # generation behaves like every later one.
        if turn.bind_provider is not None:
            try:
                turn.bind_provider(provider)
            except Exception:
                logger.warning(
                    "%s: bind_provider failed session=%s",
                    turn.channel_type,
                    session_key,
                    exc_info=True,
                )
        # Publish this turn's session identity so managed MCP tools resolve
        # X-Session-Key; one shared writer lives in messaging.identity.
        await publish_turn_identity(sessions, session_key)
        # Off-loop: build_message embeds the episodic query (blocking urllib).
        full_message, _ = await run_in_embed_pool(
            ctx_builder.build_message,
            turn.user_text,
            is_new,
            session_key,
            channel_id=turn.conversation_id,
            agent=turn.agent,
            resumed=resumed,
            runtime_source=turn.channel_type,
        )

        driver = TurnDriver(
            provider,
            renderer,
            approval_mode=turn.approval_mode,
            decider=turn.decider,
            auto_approve_tool=build_auto_approve(ctx_builder),
            auto_approve_session=turn.auto_approve_session,
            tool_gate=build_tool_gate(ctx_builder, session_key=session_key, agent=turn.agent),
            directive_consumer=turn.directive_consumer,
        )
        accumulated = await driver.run(full_message)

        # ── Post-turn bookkeeping. Each step is guarded independently so a
        # failure here cannot fall through to the except and re-record a turn
        # that actually succeeded. ──
        try:
            sessions.record_success(session_key)
        except Exception:
            logger.warning(
                "%s: record_success failed session=%s",
                turn.channel_type,
                session_key,
                exc_info=True,
            )
        if turn.persist is not None:
            try:
                await asyncio.to_thread(turn.persist, turn.user_text, accumulated, is_new)
            except Exception:
                logger.warning(
                    "%s: persist_turn failed session=%s",
                    turn.channel_type,
                    session_key,
                    exc_info=True,
                )
        if is_new and turn.after_persist is not None:
            try:
                await turn.after_persist()
            except Exception:
                logger.warning(
                    "%s: post-persist callback failed session=%s",
                    turn.channel_type,
                    session_key,
                    exc_info=True,
                )
        if turn.notice is not None:
            try:
                await turn.notice(session_key, provider)
            except Exception:
                logger.warning(
                    "%s: maybe_notice failed session=%s",
                    turn.channel_type,
                    session_key,
                    exc_info=True,
                )
        try:
            sel().log_api_access(
                caller=turn.audit_caller or f"{turn.channel_type}:unknown",
                operation="transport_dispatch.handle",
                outcome="success",
                source=turn.channel_type,
                resources=f"session={session_key}",
            )
        except Exception:
            logger.debug("%s: success audit failed", turn.channel_type, exc_info=True)
    except Exception:
        logger.exception("%s transport_dispatch: error handling message", turn.channel_type)
        if _acquired:
            await sessions.record_failure(session_key)
    finally:
        # Always finalize the turn, even if get_or_create raised before the
        # semaphore was held. Only release if we actually acquired it.
        #
        # ``renderer.close()`` is best-effort and must NEVER prevent the release
        # below. A renderer that fails to finalize -- a malformed vendor
        # response, a dropped socket mid-flush -- would otherwise leave the
        # semaphore held with no path to give it back. Because the semaphore is
        # keyed by SESSION, that does not just lose this turn: every later
        # message for that conversation blocks forever, and any queued turn
        # never drains. The channel looks permanently busy until the gateway
        # restarts.
        #
        # Discord already guards this in its own dispatcher, which is how the
        # hazard was found; the guard belongs here so every channel on the
        # shared pipeline inherits it instead of re-deriving it.
        try:
            await renderer.close()
        except Exception:
            logger.warning(
                "%s: renderer.close failed session=%s",
                turn.channel_type,
                session_key,
                exc_info=True,
            )
        if _acquired:
            sessions.release(session_key)
