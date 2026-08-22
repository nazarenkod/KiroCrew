"""The channel-neutral keyword commands (``messaging/commands.py``).

These are the reply strings Slack has had since before the transport abstraction
existed, hoisted so a second channel can have them without a second copy. The
tests here pin the CONTRACT the hoist has to preserve — the ``None`` sentinel
that means "not this command, keep routing", the retryable busy answer, and the
redaction every reply owes an external surface — plus the two primitives a
channel with its own command prefix calls directly.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew.cron import CronJob, CronSchedule, CronStoreBusy
from kiro_crew.messaging.commands import (
    _CRON_BUSY,
    cron_command_reply,
    cron_remove_all_reply,
    spawn_command_reply,
    spawn_task_reply,
    task_arg_reply,
    task_command_reply,
)

_AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"

#: A non-Slack conversation key. Namespaced rather than a bare id because this is
#: the shape every channel's session key has, and the value is forwarded verbatim.
_TG_KEY = "telegram:kirocrew:direct:U9"


def _job(job_id: str = "j1", **kw: Any) -> CronJob:
    """A real ``CronJob``, not a namespace.

    ``compute_next_run_ts`` reads ``schedule.kind``/``.cron_expr``, so a duck-typed
    stand-in passes the ``list_jobs`` boundary and then fails inside the relative-time
    arithmetic — which is exactly the row the listing has to render.
    """
    return CronJob(
        id=job_id,
        name=kw.get("name", "nightly digest"),
        message=kw.get("message", "summarize the day"),
        schedule=CronSchedule(kind="cron", cron_expr=kw.get("cron_expr", "0 9 * * *")),
        enabled=kw.get("enabled", True),
        last_status=kw.get("last_status", "ok"),
    )


class TestNotThisCommand:
    """``None`` is the sentinel that keeps normal routing going."""

    @pytest.mark.parametrize("text", ["", "hello", "spawnish thing", "  ", "bgone"])
    def test_spawn_declines_text_that_is_not_a_spawn(self, text: str) -> None:
        assert spawn_command_reply(text, MagicMock()) is None

    @pytest.mark.parametrize("text", ["", "cron", "crond list", "not cron list"])
    @pytest.mark.asyncio
    async def test_cron_declines_text_that_is_not_a_cron_command(self, text: str) -> None:
        assert await cron_command_reply(text, MagicMock()) is None

    @pytest.mark.parametrize("text", ["", "task", "task runner", "run the thing"])
    @pytest.mark.asyncio
    async def test_task_declines_text_that_is_not_a_task_command(self, text: str) -> None:
        assert await task_command_reply(text, MagicMock()) is None

    @pytest.mark.asyncio
    async def test_an_unknown_cron_verb_declines_rather_than_guessing(self) -> None:
        assert await cron_command_reply("cron obliterate j1", MagicMock()) is None


class TestSpawn:
    def test_both_prefixes_reach_the_same_spawn(self) -> None:
        manager = MagicMock(max_concurrent=2)
        manager.spawn.return_value = SimpleNamespace(id="z9")
        for text in ("spawn do it", "bg do it", "SPAWN do it"):
            manager.spawn.reset_mock()
            assert "z9" in (spawn_command_reply(text, manager) or "")
            assert manager.spawn.call_args.args[0] == "do it"

    def test_the_parsed_form_is_public_for_a_prefixed_command_grammar(self) -> None:
        # A channel whose own grammar carries the prefix (/spawn, !spawn) has the
        # argument already; it must not have to rebuild "spawn " + arg.
        manager = MagicMock(max_concurrent=2)
        manager.spawn.return_value = SimpleNamespace(id="q1")
        assert "q1" in (spawn_task_reply("do it", manager) or "")

    def test_an_empty_argument_declines(self) -> None:
        assert spawn_task_reply("", MagicMock()) is None
        assert spawn_command_reply("spawn    ", MagicMock()) is None

    @pytest.mark.parametrize("verb", ["list", "status", "LIST"])
    def test_the_list_verbs_report_an_empty_roster(self, verb: str) -> None:
        assert spawn_task_reply(verb, MagicMock(running=[])) == "No subagents running."

    def test_a_running_subagent_is_listed_with_its_elapsed_time(self) -> None:
        agent = SimpleNamespace(id="a7", started=time.time() - 5, task="reindex the corpus")
        out = spawn_task_reply("list", MagicMock(running=[agent])) or ""
        assert "a7" in out and "reindex the corpus" in out

    def test_capacity_is_reported_with_the_limit_that_was_reached(self) -> None:
        manager = MagicMock(max_concurrent=3)
        manager.spawn.return_value = None
        assert "capacity reached (3)" in (spawn_task_reply("work", manager) or "")

    def test_the_echoed_task_is_redacted(self) -> None:
        # The echo goes to an external surface and into the persisted log, and the
        # task is free-form text a user typed or an LLM proposed.
        manager = MagicMock(max_concurrent=2)
        manager.spawn.return_value = SimpleNamespace(id="r1")
        out = spawn_task_reply(f"push with {_AWS_KEY}", manager) or ""
        assert _AWS_KEY not in out

    def test_a_listed_task_is_redacted(self) -> None:
        agent = SimpleNamespace(id="a1", started=time.time(), task=f"key {_AWS_KEY}")
        out = spawn_task_reply("list", MagicMock(running=[agent])) or ""
        assert _AWS_KEY not in out


class TestCron:
    @pytest.mark.asyncio
    async def test_an_empty_roster_says_so(self) -> None:
        svc = MagicMock()
        svc.list_jobs.return_value = []
        assert await cron_command_reply("cron list", svc) == "No cron jobs scheduled."

    @pytest.mark.asyncio
    async def test_a_disabled_job_is_marked_and_still_listed(self) -> None:
        svc = MagicMock()
        svc.list_jobs.return_value = [_job("j1", enabled=False)]
        out = await cron_command_reply("cron list", svc) or ""
        assert "⏸️" in out and "`j1`" in out
        # include_disabled is what makes a paused job visible enough to resume.
        assert svc.list_jobs.call_args.kwargs == {"include_disabled": True}

    @pytest.mark.asyncio
    async def test_a_job_message_is_redacted_in_the_listing(self) -> None:
        svc = MagicMock()
        svc.list_jobs.return_value = [_job("j1", message=f"post {_AWS_KEY}")]
        assert _AWS_KEY not in (await cron_command_reply("cron list", svc) or "")

    @pytest.mark.parametrize(
        "verb,enabled,mark",
        [("pause", False, "⏸️"), ("resume", True, "▶️")],
    )
    @pytest.mark.asyncio
    async def test_pause_and_resume_pass_the_right_enabled_flag(
        self, verb: str, enabled: bool, mark: str
    ) -> None:
        svc = MagicMock()
        svc.enable_job_async = AsyncMock(return_value=True)
        out = await cron_command_reply(f"cron {verb} j1", svc) or ""
        assert mark in out and "`j1`" in out
        assert svc.enable_job_async.await_args.kwargs == {"enabled": enabled}

    @pytest.mark.asyncio
    async def test_a_missing_job_is_reported_not_claimed_as_done(self) -> None:
        svc = MagicMock()
        svc.remove_job_async = AsyncMock(return_value=False)
        assert "not found" in (await cron_command_reply("cron remove j9", svc) or "")

    @pytest.mark.parametrize("text", ["cron remove j1", "cron pause j1", "cron resume j1"])
    @pytest.mark.asyncio
    async def test_a_contended_store_answers_retryably(self, text: str) -> None:
        # One wording for every verb: a caller who sees a different string per verb
        # cannot tell "retry this" from "this failed".
        svc = MagicMock()
        svc.remove_job_async = AsyncMock(side_effect=CronStoreBusy())
        svc.enable_job_async = AsyncMock(side_effect=CronStoreBusy())
        assert await cron_command_reply(text, svc) == _CRON_BUSY

    @pytest.mark.asyncio
    async def test_remove_all_reports_each_job_and_batches_one_write(self) -> None:
        svc = MagicMock()
        svc.list_jobs.return_value = [_job("j1"), _job("j2")]
        svc.remove_jobs = AsyncMock()
        out = await cron_command_reply("cron remove all", svc) or ""
        assert "Removed 2 cron job(s)" in out and "`j1`" in out and "`j2`" in out
        assert svc.remove_jobs.await_args.args[0] == ["j1", "j2"]

    @pytest.mark.asyncio
    async def test_remove_all_redacts_each_job_name(self) -> None:
        svc = MagicMock()
        svc.list_jobs.return_value = [_job("j1", name=f"leak {_AWS_KEY}")]
        svc.remove_jobs = AsyncMock()
        assert _AWS_KEY not in (await cron_remove_all_reply(svc) or "")

    @pytest.mark.asyncio
    async def test_remove_all_on_an_empty_roster_touches_nothing(self) -> None:
        svc = MagicMock()
        svc.list_jobs.return_value = []
        svc.remove_jobs = AsyncMock()
        assert await cron_remove_all_reply(svc) == "No cron jobs to remove."
        svc.remove_jobs.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_remove_all_survives_a_contended_store(self) -> None:
        svc = MagicMock()
        svc.list_jobs.return_value = [_job()]
        svc.remove_jobs = AsyncMock(side_effect=CronStoreBusy())
        assert await cron_remove_all_reply(svc) == _CRON_BUSY


class TestTaskRunner:
    @pytest.mark.asyncio
    async def test_project_run_is_accepted_as_an_alias(self, tmp_path: Any) -> None:
        spec = tmp_path / "plan.yaml"
        spec.write_text("steps: []", encoding="utf-8")
        runner = MagicMock(running=False)
        runner.start_background = AsyncMock()
        assert "plan.yaml" in (await task_command_reply(f"project run {spec}", runner) or "")

    @pytest.mark.asyncio
    async def test_an_absent_spec_is_refused_before_the_runner_is_touched(
        self, tmp_path: Any
    ) -> None:
        runner = MagicMock(running=False)
        runner.start_background = AsyncMock()
        out = await task_command_reply(f"task run {tmp_path / 'nope.yaml'}", runner) or ""
        assert "not found" in out
        runner.start_background.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_second_run_is_refused_while_one_is_live(self) -> None:
        out = await task_command_reply("task run /tmp/x.yaml", MagicMock(running=True)) or ""
        assert "already running" in out

    @pytest.mark.asyncio
    async def test_status_reports_the_live_run_not_the_first_one(self) -> None:
        runner = MagicMock()
        runner.status.return_value = {
            "running": True,
            "runs": [
                {"running": False, "status": "done", "completed": 3, "tasks": 3},
                {
                    "running": True,
                    "status": "working",
                    "completed": 1,
                    "tasks": 4,
                    "current_task": 2,
                },
            ],
        }
        out = await task_command_reply("task run status", runner) or ""
        assert "working" in out and "1/4" in out and "step 2" in out

    @pytest.mark.asyncio
    async def test_status_with_nothing_running_says_so(self) -> None:
        runner = MagicMock()
        runner.status.return_value = {"running": False}
        assert await task_command_reply("task run status", runner) == "No task running."

    @pytest.mark.asyncio
    async def test_cancel_only_cancels_when_something_runs(self) -> None:
        idle = MagicMock(running=False)
        assert await task_command_reply("task run cancel", idle) == "No task running."
        idle.cancel.assert_not_called()
        live = MagicMock(running=True)
        assert "cancelled" in (await task_command_reply("task run cancel", live) or "")
        live.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_start_failure_is_reported_redacted_not_raised(self, tmp_path: Any) -> None:
        spec = tmp_path / "plan.yaml"
        spec.write_text("steps: []", encoding="utf-8")
        runner = MagicMock(running=False)
        runner.start_background = AsyncMock(side_effect=RuntimeError(f"bad {_AWS_KEY}"))
        out = await task_command_reply(f"task run {spec}", runner) or ""
        assert "Failed to start" in out and _AWS_KEY not in out

    @pytest.mark.asyncio
    async def test_the_keyword_grammar_carries_the_session_key(self, tmp_path: Any) -> None:
        """``task run <spec>`` must escalate where ``/task run <spec>`` does.

        The runner hands ``session_key`` to its notify sink, which is what sends an
        approval notice back to the conversation the operator is watching instead
        of only to the Slack owner DM. Both grammars reach the same runner, so a
        key carried on one entry point and dropped on the other means the same run
        escalates to a different place depending on how it was typed.
        """
        spec = tmp_path / "plan.yaml"
        spec.write_text("steps: []", encoding="utf-8")
        runner = MagicMock(running=False)
        runner.start_background = AsyncMock()
        out = await task_command_reply(f"task run {spec}", runner, session_key=_TG_KEY) or ""
        assert "plan.yaml" in out
        assert runner.start_background.await_args.kwargs["session_key"] == _TG_KEY

    @pytest.mark.asyncio
    async def test_omitting_the_session_key_keeps_the_narrow_runner_call(
        self, tmp_path: Any
    ) -> None:
        """A stand-in accepting only ``(path, source=)`` must still start.

        ``runner`` is duck-typed here — this module may not import ``TaskRunner``
        at runtime — so widening the call unconditionally would turn a working
        command into "Failed to start" for every narrower runner.
        """
        spec = tmp_path / "plan.yaml"
        spec.write_text("steps: []", encoding="utf-8")
        runner = MagicMock(running=False)
        runner.start_background = AsyncMock()
        assert "plan.yaml" in (await task_command_reply(f"task run {spec}", runner) or "")
        assert "session_key" not in runner.start_background.await_args.kwargs


class TestTaskArgReply:
    """The already-parsed entry point, which is where the ``run`` verb is absorbed.

    The keyword grammar spells the verb ``task run``, but a channel whose command
    IS ``/task`` receives ``run <spec>`` as its argument — re-composing
    ``"task run " + arg`` handed the runner a spec named ``run <spec>``.
    """

    @pytest.mark.asyncio
    async def test_a_leading_run_verb_is_absorbed(self, tmp_path: Any) -> None:
        spec = tmp_path / "plan.yaml"
        spec.write_text("steps: []", encoding="utf-8")
        runner = MagicMock(running=False)
        runner.start_background = AsyncMock()
        assert "plan.yaml" in (await task_arg_reply(f"run {spec}", runner) or "")
        assert runner.start_background.await_args.args[0].name == "plan.yaml"

    @pytest.mark.asyncio
    async def test_a_bare_spec_works_too(self, tmp_path: Any) -> None:
        spec = tmp_path / "plan.yaml"
        spec.write_text("steps: []", encoding="utf-8")
        runner = MagicMock(running=False)
        runner.start_background = AsyncMock()
        assert "plan.yaml" in (await task_arg_reply(str(spec), runner) or "")

    @pytest.mark.parametrize("arg", ["", "   ", "run", "run   "])
    @pytest.mark.asyncio
    async def test_a_verb_with_no_argument_declines(self, arg: str) -> None:
        runner = MagicMock(running=False)
        runner.start_background = AsyncMock()
        assert await task_arg_reply(arg, runner) is None
        runner.start_background.assert_not_awaited()

    @pytest.mark.parametrize("arg", ["status", "cancel"])
    @pytest.mark.asyncio
    async def test_the_bare_verbs_reach_their_branches(self, arg: str) -> None:
        runner = MagicMock(running=False)
        runner.status.return_value = {"running": False}
        assert await task_arg_reply(arg, runner) == "No task running."


class TestTaskSpecPathIsGated:
    """A task spec is READ and its contents reach the model.

    So an unvalidated path is an exfiltration primitive rather than a usability
    question: ``task run ~/.ssh/id_rsa`` would hand a private key to a third-party
    LLM. Both grammars route through this one module, so Slack's ``task run`` and a
    channel's ``/task`` argument are covered by the same gate.
    """

    @pytest.mark.asyncio
    async def test_a_sensitive_path_is_refused_before_the_runner_sees_it(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        runner = MagicMock(running=False)
        runner.start_background = AsyncMock()
        # Point the sensitive-root check at a real directory this test owns, so
        # the assertion does not depend on the host having ~/.ssh.
        secret_dir = tmp_path / "dot-ssh"
        secret_dir.mkdir()
        key = secret_dir / "id_rsa"
        key.write_text("PRIVATE KEY", encoding="utf-8")
        monkeypatch.setattr(
            "kiro_crew.hooks.is_sensitive_path",
            lambda p: str(secret_dir) in str(p),
        )

        reply = await task_arg_reply(f"run {key}", runner)

        runner.start_background.assert_not_awaited()
        assert reply is not None
        # The refusal must not echo the path back into the channel, and must not
        # say WHY: distinguishing "sensitive" from "missing" is a probing oracle.
        assert str(key) not in reply
        assert "id_rsa" not in reply

    @pytest.mark.asyncio
    async def test_a_symlink_into_a_sensitive_root_is_refused_through_the_link(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """The reason the shared helper is used instead of a prefix test.

        A path that looks innocent resolves into a blocked root, so the check has
        to run on the RESOLVED target.
        """
        runner = MagicMock(running=False)
        runner.start_background = AsyncMock()
        secret_dir = tmp_path / "dot-ssh"
        secret_dir.mkdir()
        (secret_dir / "id_rsa").write_text("PRIVATE KEY", encoding="utf-8")
        link = tmp_path / "plan.yaml"
        link.symlink_to(secret_dir / "id_rsa")
        monkeypatch.setattr(
            "kiro_crew.hooks.is_sensitive_path",
            lambda p: str(secret_dir) in str(p),
        )

        assert await task_arg_reply(f"run {link}", runner) is not None
        runner.start_background.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_runner_receives_the_canonical_path_not_the_raw_argument(
        self, tmp_path: Any
    ) -> None:
        """Validating one string and acting on another is an ornamental guard."""
        runner = MagicMock(running=False)
        runner.start_background = AsyncMock()
        real = tmp_path / "real.yaml"
        real.write_text("steps: []", encoding="utf-8")
        link = tmp_path / "alias.yaml"
        link.symlink_to(real)

        await task_arg_reply(f"run {link}", runner)

        handed = runner.start_background.await_args.args[0]
        assert handed.name == "real.yaml"

    @pytest.mark.asyncio
    async def test_an_ordinary_spec_still_runs(self, tmp_path: Any) -> None:
        """Non-vacuity: the gate must not refuse everything."""
        runner = MagicMock(running=False)
        runner.start_background = AsyncMock()
        spec = tmp_path / "plan.yaml"
        spec.write_text("steps: []", encoding="utf-8")
        assert "plan.yaml" in (await task_arg_reply(f"run {spec}", runner) or "")
        runner.start_background.assert_awaited_once()


class TestSlackKeywordCarriesTheSessionKey:
    """The `task run <spec>` KEYWORD grammar carries the session key too.

    ``task_arg_reply`` (the ``/task`` slash grammar) gained ``session_key`` so a
    blocked task could report back to the conversation the operator is watching.
    ``task_command_reply`` (the bare ``task run …`` keyword grammar, which is Slack's
    route) did not, so the same task typed as a keyword still escalated only to the
    owner DM — the kwarg was passed by one of its two callers.
    """

    @pytest.mark.asyncio
    async def test_the_slack_handler_forwards_the_session_key(self) -> None:
        from kiro_crew.slack import handler as sh

        seen: list[dict] = []

        async def _reply(text: str, runner: Any, *, session_key: str = "") -> str:
            seen.append({"text": text, "session_key": session_key})
            return "started"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sh, "task_command_reply", _reply)
            out = await sh._handle_run_command(
                "task run spec.md",
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                "C1",
                "1.2",
                session_key="slack:kirocrew:direct:U1",
            )
        assert out == "started"
        assert seen and seen[0]["session_key"] == "slack:kirocrew:direct:U1"

    @pytest.mark.asyncio
    async def test_omitting_it_reproduces_the_old_behaviour(self) -> None:
        # Keyword-only with a default, so the ~25 existing positional call sites are
        # unchanged and an omitted key means owner-DM-only exactly as before.
        from kiro_crew.slack import handler as sh

        seen: list[str] = []

        async def _reply(text: str, runner: Any, *, session_key: str = "") -> str:
            seen.append(session_key)
            return "started"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sh, "task_command_reply", _reply)
            await sh._handle_run_command(
                "task run spec.md", object(), object(), "C1", "1.2"  # type: ignore[arg-type]
            )
        assert seen == [""]
