"""Tests for the babysit skill's zero-token PR watch script cron.

Pins the watch contract: silent (Skip) while nothing needs a brain, one wake
(Report) per anomaly per head, terminal Done on merge/close, per-head alert
reset on force-push, known-inherited reds filtered, gh failures quiet until
the consecutive-error alert, and hostile check names sanitized before they
reach a wake brief.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest
from skill_script_helpers import load_skill_script

from kiro_crew.cron_script import Done, Report, Skip

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "src"
    / "kiro_crew"
    / "builtin_skills"
    / "kirocrew-dev"
    / "babysit"
    / "scripts"
    / "pr_watch.py"
)


class _Job:
    id = "job-e2e-1"


class _Ctx:
    def __init__(self, message: str) -> None:
        self.message = message
        self.job = _Job()


def _check(name: str, conclusion: str = "", status: str = "COMPLETED") -> dict:
    return {"name": name, "conclusion": conclusion, "status": status}


def _payload(
    checks: list[dict],
    *,
    state: str = "OPEN",
    merged_at: str | None = None,
    mergeable: str = "MERGEABLE",
    merge_state: str = "BLOCKED",
    head: str = "a" * 40,
) -> dict:
    return {
        "state": state,
        "mergedAt": merged_at,
        "mergeable": mergeable,
        "mergeStateStatus": merge_state,
        "headRefOid": head,
        "statusCheckRollup": checks,
    }


@pytest.fixture()
def module(monkeypatch, tmp_path) -> ModuleType:
    mod = load_skill_script("babysit_pr_watch", SCRIPT)
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "home"))
    return mod


def _wire(monkeypatch, module: ModuleType, payload: dict | None) -> None:
    def _fake_run_gh(args):
        if payload is None:
            return 1, ""
        return 0, json.dumps(payload)

    monkeypatch.setattr(module, "_run_gh", _fake_run_gh)


def _msg(**overrides) -> str:
    base = {"repo": "acme/widgets", "pr": 42}
    base.update(overrides)
    return json.dumps(base)


def _tick(module: ModuleType, message: str):
    return module.watch(_Ctx(message))


# ── terminal states ───────────────────────────────────────────────────────


def test_merged_pr_completes_the_watch(monkeypatch, module):
    _wire(monkeypatch, module, _payload([], merged_at="2026-08-23T00:00:00Z"))
    with pytest.raises(Done, match="MERGED"):
        _tick(module, _msg())


def test_closed_unmerged_completes_the_watch(monkeypatch, module):
    _wire(monkeypatch, module, _payload([], state="CLOSED"))
    with pytest.raises(Done, match="CLOSED"):
        _tick(module, _msg())


def test_invalid_message_is_terminal_not_a_retry_loop(monkeypatch, module):
    _wire(monkeypatch, module, _payload([]))
    with pytest.raises(Done, match="not valid JSON"):
        _tick(module, "{oops")
    with pytest.raises(Done, match="JSON object"):
        _tick(module, "[]")  # valid JSON, wrong shape — must not crash-loop
    with pytest.raises(Done, match="needs"):
        _tick(module, json.dumps({"repo": "no-slash", "pr": 1}))
    with pytest.raises(Done, match="needs"):
        _tick(module, json.dumps({"repo": "a/b", "pr": "not-an-int"}))


# ── quiet paths ───────────────────────────────────────────────────────────


def test_pending_checks_skip_silently(monkeypatch, module):
    _wire(
        monkeypatch,
        module,
        _payload([_check("CI", status="IN_PROGRESS"), _check("Lint", "SUCCESS")]),
    )
    with pytest.raises(Skip):
        _tick(module, _msg())


def test_known_inherited_red_never_wakes_while_others_pend(monkeypatch, module):
    checks = [_check("Frontend Tests (4)", "FAILURE"), _check("CI", status="QUEUED")]
    _wire(monkeypatch, module, _payload(checks))
    with pytest.raises(Skip):
        _tick(module, _msg(known_reds=["Frontend Tests (4)"]))


def test_cancelled_runs_are_noise_not_failures(monkeypatch, module):
    checks = [_check("GPT Review", "CANCELLED"), _check("CI", "SUCCESS")]
    _wire(monkeypatch, module, _payload(checks))
    # CANCELLED neither fails nor pends: with everything else green this is
    # the ready wake, not a new-red wake.
    with pytest.raises(Report, match="all checks green"):
        _tick(module, _msg())


# ── wake paths, deduped per head ──────────────────────────────────────────


def test_conflict_wakes_once_per_head(monkeypatch, module):
    payload = _payload([_check("CI", status="QUEUED")], mergeable="CONFLICTING")
    _wire(monkeypatch, module, payload)
    with pytest.raises(Report, match="merge conflict"):
        _tick(module, _msg())
    with pytest.raises(Skip):
        _tick(module, _msg())


def test_alert_rearms_after_the_dedupe_window(monkeypatch, module):
    """Dedupe is time-bounded, not a permanent acknowledgement: a delivery
    lost to a gateway failure must cost a bounded delay, never a permanently
    suppressed signal."""
    payload = _payload([_check("CI", status="QUEUED")], mergeable="CONFLICTING")
    _wire(monkeypatch, module, payload)
    t = [1_000_000.0]
    monkeypatch.setattr(module.time, "time", lambda: t[0])
    with pytest.raises(Report):
        _tick(module, _msg())
    with pytest.raises(Skip):
        _tick(module, _msg())
    t[0] += module._REALERT_SECS + 1
    with pytest.raises(Report):  # condition persists -> re-delivered
        _tick(module, _msg())


def test_same_check_name_across_workflows_keeps_distinct_identity(monkeypatch, module):
    """Allowlisting one workflow's 'test' must not silence another
    workflow's failing 'test'."""
    checks = [
        dict(_check("test", "FAILURE"), workflowName="Alpha", startedAt="2026-08-23T10:00:00Z"),
        dict(_check("test", "FAILURE"), workflowName="Beta", startedAt="2026-08-23T10:00:00Z"),
    ]
    _wire(monkeypatch, module, _payload(checks))
    with pytest.raises(Report, match="Beta / test"):
        _tick(module, _msg(known_reds=["Alpha / test"]))


def test_new_red_wakes_and_names_the_check_then_goes_quiet(monkeypatch, module):
    checks = [_check("Backend Tests (3.12, 2)", "FAILURE"), _check("CI", "SUCCESS")]
    _wire(monkeypatch, module, _payload(checks))
    with pytest.raises(Report, match=r"Backend Tests \(3.12, 2\)"):
        _tick(module, _msg())
    with pytest.raises(Skip):
        _tick(module, _msg())


def test_second_distinct_red_wakes_again(monkeypatch, module):
    _wire(monkeypatch, module, _payload([_check("A", "FAILURE")]))
    with pytest.raises(Report):
        _tick(module, _msg())
    _wire(
        monkeypatch,
        module,
        _payload([_check("A", "FAILURE"), _check("B", "TIMED_OUT")]),
    )
    with pytest.raises(Report, match="B"):
        _tick(module, _msg())


def test_green_wakes_once_and_respects_known_red_filter(monkeypatch, module):
    checks = [_check("Frontend Tests (4)", "FAILURE"), _check("CI", "SUCCESS")]
    _wire(monkeypatch, module, _payload(checks))
    with pytest.raises(Report, match="review-ready"):
        _tick(module, _msg(known_reds=["Frontend Tests (4)"]))
    with pytest.raises(Skip):
        _tick(module, _msg(known_reds=["Frontend Tests (4)"]))


def test_wake_on_green_false_stays_quiet(monkeypatch, module):
    _wire(monkeypatch, module, _payload([_check("CI", "SUCCESS")]))
    with pytest.raises(Skip):
        _tick(module, _msg(wake_on_green=False))


def test_empty_rollup_never_reports_ready(monkeypatch, module):
    """A PR whose checks have not dispatched yet has an empty rollup — that
    is 'nothing ran', not 'everything passed'."""
    _wire(monkeypatch, module, _payload([]))
    with pytest.raises(Skip):
        _tick(module, _msg())


def test_force_push_resets_alert_memory(monkeypatch, module):
    _wire(
        monkeypatch,
        module,
        _payload([_check("A", "FAILURE")], head="a" * 40),
    )
    with pytest.raises(Report):
        _tick(module, _msg())
    _wire(
        monkeypatch,
        module,
        _payload([_check("A", "FAILURE")], head="b" * 40),
    )
    with pytest.raises(Report, match="A"):
        _tick(module, _msg())


def test_unknown_conclusion_vocabulary_wakes_a_brain(monkeypatch, module):
    _wire(monkeypatch, module, _payload([_check("Odd", "SOMETHING_NEW")]))
    with pytest.raises(Report, match="Odd"):
        _tick(module, _msg())


# ── watch health ──────────────────────────────────────────────────────────


def test_gh_failures_stay_quiet_then_alert_once(monkeypatch, module):
    _wire(monkeypatch, module, None)
    for _ in range(module._MAX_CONSECUTIVE_ERRORS - 1):
        with pytest.raises(Skip):
            _tick(module, _msg())
    with pytest.raises(Report, match="consecutive"):
        _tick(module, _msg())
    with pytest.raises(Skip):
        _tick(module, _msg())


def test_gh_recovery_resets_the_error_streak(monkeypatch, module):
    _wire(monkeypatch, module, None)
    with pytest.raises(Skip):
        _tick(module, _msg())
    _wire(monkeypatch, module, _payload([_check("CI", status="QUEUED")]))
    with pytest.raises(Skip):
        _tick(module, _msg())
    _wire(monkeypatch, module, None)
    # Streak restarted at 1, not continuing from 2.
    with pytest.raises(Skip):
        _tick(module, _msg())


# ── hygiene ───────────────────────────────────────────────────────────────


def test_hostile_check_names_are_sanitized_in_the_brief(monkeypatch, module):
    evil = "Evil\ncheck\x1b[31m<script>"
    _wire(monkeypatch, module, _payload([_check(evil, "FAILURE")]))
    with pytest.raises(Report) as exc:
        _tick(module, _msg())
    text = str(exc.value)
    assert "\x1b" not in text
    assert "<script>" not in text
    assert "Evil" in text


def test_state_survives_corrupt_state_file(monkeypatch, module, tmp_path):
    _wire(monkeypatch, module, _payload([_check("A", "FAILURE")]))
    with pytest.raises(Report):
        _tick(module, _msg())
    spath = module._state_path("acme/widgets", 42, "job-e2e-1")
    spath.write_text("{broken", encoding="utf-8")
    # Corrupt state reads as fresh: the red alerts again rather than crashing.
    with pytest.raises(Report):
        _tick(module, _msg())


def test_malformed_state_field_types_read_as_fresh(monkeypatch, module):
    _wire(monkeypatch, module, _payload([_check("A", "FAILURE")]))
    with pytest.raises(Report):
        _tick(module, _msg())
    spath = module._state_path("acme/widgets", 42, "job-e2e-1")
    spath.write_text(json.dumps({"head": 7, "alerted": "yes", "errors": "x"}), encoding="utf-8")
    with pytest.raises(Report):  # wrong types coerce to fresh, never crash
        _tick(module, _msg())


def test_malformed_known_reds_parameter_is_terminal(monkeypatch, module):
    _wire(monkeypatch, module, _payload([]))
    with pytest.raises(Done, match="known_reds"):
        _tick(module, _msg(known_reds=1))


def test_boolean_and_nonpositive_pr_numbers_are_terminal(monkeypatch, module):
    _wire(monkeypatch, module, _payload([]))
    with pytest.raises(Done, match="positive int"):
        _tick(module, _msg(pr=True))  # bool passes isinstance(int) checks
    with pytest.raises(Done, match="positive int"):
        _tick(module, _msg(pr=0))
    with pytest.raises(Done, match="positive int"):
        _tick(module, json.dumps({"repo": "host/owner/name", "pr": 1}))
    with pytest.raises(Done, match="positive int"):  # host segments refused
        _tick(module, json.dumps({"repo": "ghe.corp.example/o/r", "pr": 1}))


def test_queued_rerun_without_timestamp_blocks_false_ready(monkeypatch, module):
    """A just-queued rerun row has no startedAt; it must not lose to the
    older green row and produce a false all-checks-green wake."""
    checks = [
        dict(_check("CI", "SUCCESS"), startedAt="2026-08-23T10:00:00Z", workflowName="CI"),
        dict(_check("CI", status="QUEUED", conclusion=""), startedAt="", workflowName="CI"),
    ]
    _wire(monkeypatch, module, _payload(checks))
    with pytest.raises(Skip):  # pending, not ready
        _tick(module, _msg())


def test_double_failure_alerts_immediately(monkeypatch, module):
    """gh failing AND state unwritable: the counted threshold can never fire,
    so the watch says it is inoperative on the first tick."""
    _wire(monkeypatch, module, None)
    monkeypatch.setattr(module, "_save_state", lambda *a: False)
    with pytest.raises(Report, match="inoperative"):
        _tick(module, _msg())


def test_two_watches_on_one_pr_keep_independent_state(monkeypatch, module):
    """One session's alert must not suppress the other's delivery."""
    _wire(monkeypatch, module, _payload([_check("A", "FAILURE")]))
    with pytest.raises(Report):
        _tick(module, _msg())

    class OtherJob:
        id = "job-other-2"

    class OtherCtx:
        message = _msg()
        job = OtherJob()

    with pytest.raises(Report):  # second watch alerts independently
        module.watch(OtherCtx())


def test_unwritable_state_degrades_to_repeats_not_removal(monkeypatch, module):
    """An unwritable state dir must not remove the watch (later signals would
    be lost) and must not silence it: the alert still fires, carrying the
    degraded-dedupe warning, and repeats on the next tick."""
    _wire(monkeypatch, module, _payload([_check("A", "FAILURE")]))
    monkeypatch.setattr(module, "_save_state", lambda *a: False)
    with pytest.raises(Report, match="unwritable"):
        _tick(module, _msg())
    with pytest.raises(Report):  # duplicate wake, never a lost signal
        _tick(module, _msg())


def test_rerun_green_supersedes_stale_red_row_of_same_name(monkeypatch, module):
    checks = [
        dict(_check("CI", "FAILURE"), startedAt="2026-08-23T10:00:00Z", workflowName="CI"),
        dict(_check("CI", "SUCCESS"), startedAt="2026-08-23T11:00:00Z", workflowName="CI"),
    ]
    _wire(monkeypatch, module, _payload(checks))
    # The stale red row must not wake; with the rerun green this is ready.
    with pytest.raises(Report, match="all checks green"):
        _tick(module, _msg())


def test_rerun_red_supersedes_stale_green_row_of_same_name(monkeypatch, module):
    """Recency arbitrates BOTH directions: an older success must not mask a
    newer failing rerun into a false-ready."""
    checks = [
        dict(_check("CI", "SUCCESS"), startedAt="2026-08-23T10:00:00Z", workflowName="CI"),
        dict(_check("CI", "FAILURE"), startedAt="2026-08-23T11:00:00Z", workflowName="CI"),
    ]
    _wire(monkeypatch, module, _payload(checks))
    with pytest.raises(Report, match="new failing check"):
        _tick(module, _msg())
