"""Per-session tool Trust — one set, shared by every channel.

"Trust the rest of this session's tools" is a channel-neutral grant: the
``TurnDriver``'s ``auto_approve_session`` predicate is what consumes it, and the
button that writes it is just a widget. It lived in ``slack/handler.py`` only
because Slack's approval prompt was the first to offer one, which meant a second
channel could not read it — ``messaging`` may not import ``slack``.

Distinct from global YOLO (``safety_override``) in scope and in lifetime: this
grant covers ONE session and is held in memory only, so it dies with the process.
It does not weaken the PreToolUse gate — the sensitive-path keystone, the
governance ceiling and the deny-list all run ahead of the approval ladder in
``TurnDriver``, so a hard DENY still refuses a trusted session's tool.

Named ``session_trust`` rather than ``trust`` because "trust" in a messaging
package reads as connection admission — WHICH principals may attach — and that is
a different, operator-owned decision with a different failure mode. This module
answers only "may THIS session's remaining tools skip the prompt", so an
admission roster can keep the shorter name without either being mistaken for the
other.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kiro_crew.session import SessionManager

logger = logging.getLogger(__name__)

#: Session keys granted Trust this process. In memory only, deliberately: an
#: ad-hoc auto-approve grant must not survive a restart.
_trusted_sessions: set[str] = set()


def is_session_trusted(session_key: str) -> bool:
    """Whether *session_key* has been granted per-session Trust."""
    return bool(session_key) and session_key in _trusted_sessions


def add_trusted_session(session_key: str, sessions: "SessionManager | Any | None" = None) -> None:
    """Grant per-session Trust for *session_key*.

    Adds the key to the in-memory set and, when a ``SessionManager`` is supplied,
    sets that session's approval policy to ``auto`` so spawned subagents inherit
    the trust — a subagent reads its parent's approval policy, never this set, so
    without the second half a trusted session's children still stop to ask.
    """
    if not session_key:
        return
    _trusted_sessions.add(session_key)
    if sessions is None:
        return
    try:
        sessions.set_approval_policy(session_key, "auto")
    except Exception:
        logger.warning(
            "Failed to propagate trust approval policy for %s", session_key, exc_info=True
        )


def clear_trusted_sessions() -> None:
    """Drop every grant. Used when the approval mode changes under the operator."""
    _trusted_sessions.clear()
