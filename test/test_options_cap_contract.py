"""Cross-channel contract: ``max_buttons`` is ENFORCED, per channel.

The capability ledger (``test_capability_ledger.py``) says the field is
enforced; THIS file is what makes that claim unforgeable. For every channel
declaring ``max_buttons > 0`` it drives the real options path with an
over-cap list and pins:

1. exactly ``max_buttons`` choices render interactively, and
2. the overflow degrades to a numbered text list (numbering continues after
   the widget slots) instead of being silently dropped — the pre-enforcement
   behavior lost choices without any user-visible signal.

``test_every_widget_channel_is_pinned_here`` is the ratchet: a channel that
starts declaring ``max_buttons > 0`` without a pin in this file fails it.
"""

from __future__ import annotations

import asyncio

from kiro_crew.messaging.renderer import (
    append_options_text,
    apply_options_cap,
    cap_choices,
    split_options_trailer,
)
from kiro_crew.messaging.transport import TransportCapabilities

#: channel_type -> the test class below that pins its enforcement.
PINNED_WIDGET_CHANNELS = {"slack", "discord", "telegram", "webex"}


def _all_channel_capabilities() -> dict[str, TransportCapabilities]:
    from kiro_crew.discord.transport import DISCORD_CAPABILITIES
    from kiro_crew.imessage.transport import IMESSAGE_CAPABILITIES
    from kiro_crew.slack.transport import SLACK_CAPABILITIES
    from kiro_crew.teams.transport import TEAMS_CAPABILITIES
    from kiro_crew.telegram.transport import TELEGRAM_CAPABILITIES
    from kiro_crew.webex.transport import WEBEX_CAPABILITIES
    from kiro_crew.wecom.transport import WECOM_CAPABILITIES
    from kiro_crew.weixin.transport import WEIXIN_CAPABILITIES

    return {
        "slack": SLACK_CAPABILITIES,
        "discord": DISCORD_CAPABILITIES,
        "telegram": TELEGRAM_CAPABILITIES,
        "teams": TEAMS_CAPABILITIES,
        "webex": WEBEX_CAPABILITIES,
        "wecom": WECOM_CAPABILITIES,
        "weixin": WEIXIN_CAPABILITIES,
        "imessage": IMESSAGE_CAPABILITIES,
    }


class TestRatchet:
    def test_every_widget_channel_is_pinned_here(self) -> None:
        widget_channels = {
            name for name, caps in _all_channel_capabilities().items() if caps.max_buttons > 0
        }
        assert widget_channels == PINNED_WIDGET_CHANNELS, (
            "A channel's max_buttons declaration changed. Every channel "
            "declaring max_buttons > 0 must have an enforcement pin in this "
            f"file. unpinned={widget_channels - PINNED_WIDGET_CHANNELS} "
            f"stale={PINNED_WIDGET_CHANNELS - widget_channels}"
        )


class TestSharedHelper:
    def test_under_cap_is_byte_identical(self) -> None:
        caps = TransportCapabilities(max_buttons=3)
        body, kept = apply_options_cap("Choose.", ["A", "B"], caps)
        assert body == "Choose."
        assert kept == ["A", "B"]

    def test_overflow_degrades_to_numbered_text_continuing_the_widget_slots(self) -> None:
        caps = TransportCapabilities(max_buttons=2)
        body, kept = apply_options_cap("Pick one.", ["A", "B", "C", "D"], caps)
        assert kept == ["A", "B"]
        assert body == "Pick one.\n\n3. C\n4. D"

    def test_zero_cap_keeps_nothing_and_leaves_body_alone(self) -> None:
        caps = TransportCapabilities(max_buttons=0)
        body, kept = apply_options_cap("Text.", ["A", "B"], caps)
        assert body == "Text."
        assert kept == []

    def test_cap_choices_splits_without_formatting(self) -> None:
        caps = TransportCapabilities(max_buttons=1)
        kept, overflow = cap_choices(["A", "B", "C"], caps)
        assert kept == ["A"]
        assert overflow == ["B", "C"]

    def test_overflow_neutralizes_mass_mention_syntax(self) -> None:
        # Regression (review round 2): overflow lands in the message BODY
        # where platforms parse mentions — unlike widget labels, which render
        # as plain text. A prompt-injected choice must not mass-notify.
        from kiro_crew.messaging.renderer import format_overflow

        out = format_overflow(["ping @everyone now", "or <!channel> maybe"], start=1)
        assert "@everyone" not in out
        assert "<!channel>" not in out
        # The text stays human-readable — only the trigger syntax is broken.
        assert "everyone" in out and "channel" in out

    def test_overflow_redacts_credentials_in_their_DISPLAY_form(self) -> None:
        # Regression (review round 5): overflow lands in the markdown-parsed
        # BODY, so a key split by a code span or emphasis is broken to every
        # byte-level scan (the driver's stream redactor included) and WHOLE on
        # screen once the platform drops the delimiters. Slack's widget path
        # already routes choices through the display redactor for exactly this
        # reason; the shared sink has to close the same hole for telegram and
        # discord, which have no display-state pass of their own.
        from kiro_crew.messaging.renderer import format_overflow

        split = "AKIA`" + "`IOSFODNN7EXAMPLE"
        out = format_overflow([f"Retry with {split}"], start=1)
        assert "IOSFODNN7EXAMPLE" not in out, (
            "a backtick-split key survived the overflow sink — the platform "
            "strips the delimiters and shows the reader an intact credential"
        )

    def test_overflow_redaction_runs_before_mention_defanging(self) -> None:
        # Both sanitisations transform the text; if the ZWSP went in first it
        # could split a key so the regex stops matching while the platform
        # still renders it whole. Pin the order with a choice that needs both.
        from kiro_crew.messaging.renderer import format_overflow

        out = format_overflow(["@everyone use AKIA*IOSFODNN7EXAMPLE*"], start=0)
        assert "@everyone" not in out
        assert "IOSFODNN7EXAMPLE" not in out

    def test_overflow_redacts_a_spoiler_split_key(self) -> None:
        # Regression (review round 6): ``||…||`` is Discord's spoiler. The
        # reader clicks it, the delimiters vanish and the halves join — the
        # same splitter property as ``**``, but it was missing from the
        # canonicaliser's delimiter run, so round 5's fix had a hole exactly
        # one delimiter family wide.
        from kiro_crew.messaging.renderer import format_overflow

        out = format_overflow(["Retry with AKIA||IOSFODNN7EXAMPLE||"], start=0)
        assert "IOSFODNN7EXAMPLE" not in out, (
            "a spoiler-split key survived — Discord joins the halves when the "
            "reader reveals the spoiler"
        )

    def test_overflow_redacts_an_invisible_character_split_key(self) -> None:
        # The invisible half of the same hazard, and worse than the markup half:
        # a zero-width character renders as NOTHING, so the reader sees an
        # intact key with no click and no markup while every literal scan sees
        # it broken. Pre-existing in the display redactor; closed here because
        # this sink is what puts LLM-authored choice text into the body.
        from kiro_crew.messaging.renderer import format_overflow

        for name, ch in (
            ("ZWSP", "\u200b"),
            ("ZWNJ", "\u200c"),
            ("word joiner", "\u2060"),
            ("BOM", "\ufeff"),
            ("soft hyphen", "\u00ad"),
        ):
            out = format_overflow([f"Retry with AKIA{ch}IOSFODNN7EXAMPLE"], start=0)
            assert "IOSFODNN7EXAMPLE" not in out, f"{name} split the key past the scan"

    def test_non_ascii_text_is_not_mangled(self) -> None:
        """The format-character filter must not touch visible non-ASCII text."""
        from kiro_crew.messaging.renderer import format_overflow

        out = format_overflow(["重新部署到主分支", "café — naïve"], start=0)
        assert out == "1. 重新部署到主分支\n2. café — naïve"

    def test_a_lone_pipe_is_left_alone(self) -> None:
        """The pipe counts only in pairs — pinned so the boundary is deliberate.

        A single ``|`` is literal on every channel here, so collapsing it would
        widen the canonical form with no rendering that matches it. This also
        keeps ordinary table-ish text intact.
        """
        from kiro_crew.messaging.display_safety import canonicalize_display

        assert canonicalize_display("a|b") == "a|b"
        assert canonicalize_display("a||b") == "ab"

    def test_clean_choices_are_untouched_by_the_redactor(self) -> None:
        """The sink must not mangle ordinary text — no false-positive damage."""
        from kiro_crew.messaging.renderer import format_overflow

        out = format_overflow(["Rebase onto main", "Skip the `--force` flag"], start=2)
        assert out == "3. Rebase onto main\n4. Skip the `--force` flag"


class TestSlackEnforcement:
    def _choices(self, n: int) -> list[str]:
        return [f"Choice {i}" for i in range(1, n + 1)]

    def test_widget_caps_at_declared_and_overflow_is_visible(self) -> None:
        from kiro_crew.slack.format import build_options_blocks
        from kiro_crew.slack.transport import SLACK_CAPABILITIES

        n = SLACK_CAPABILITIES.max_buttons
        blocks = build_options_blocks(self._choices(n + 3))
        actions = next(b for b in blocks if b["type"] == "actions")
        opts = actions["elements"][0]["options"]
        assert len(opts) == n
        overflow = next(b for b in blocks if b["type"] == "context")
        text = overflow["elements"][0]["text"]
        # Numbering continues after the widget slots; every dropped choice shows.
        assert f"{n + 1}. Choice {n + 1}" in text
        assert f"{n + 3}. Choice {n + 3}" in text

    def test_under_cap_emits_no_overflow_block(self) -> None:
        from kiro_crew.slack.format import build_options_blocks

        blocks = build_options_blocks(self._choices(2))
        assert [b["type"] for b in blocks] == ["actions"]

    def test_huge_overflow_is_chunked_not_sliced(self) -> None:
        # Regression (review round 1): a single [:2900] slice re-created the
        # silent data loss the cap exists to remove. Every overflow choice
        # must reach the wire, across as many context blocks as needed.
        from kiro_crew.slack.format import build_options_blocks
        from kiro_crew.slack.transport import SLACK_CAPABILITIES

        n = SLACK_CAPABILITIES.max_buttons
        long = [f"Choice {i} " + "x" * 140 for i in range(1, n + 41)]
        blocks = build_options_blocks(long)
        ctx = [b for b in blocks if b["type"] == "context"]
        assert len(ctx) >= 2, "one sliced block would drop tail choices"
        joined = "".join(b["elements"][0]["text"] for b in ctx)
        assert f"{n + 40}." in joined, "the LAST overflow choice must survive"

    def test_pathological_overflow_is_bounded_with_visible_truncation(self) -> None:
        # Regression (review round 3): unbounded context blocks blow Slack's
        # 50-block message limit — the API rejects the WHOLE message and every
        # choice disappears. The block budget is capped and the tail drop is
        # VISIBLE (counted marker), never silent.
        from kiro_crew.slack.format import build_options_blocks
        from kiro_crew.slack.transport import SLACK_CAPABILITIES

        n = SLACK_CAPABILITIES.max_buttons
        huge = [f"Choice {i} " + "x" * 140 for i in range(1, n + 201)]
        blocks = build_options_blocks(huge)
        ctx = [b for b in blocks if b["type"] == "context"]
        assert len(ctx) <= 4, "block budget must be bounded"
        assert len(blocks) <= 5
        marker = ctx[-1]["elements"][0]["text"]
        assert "omitted" in marker
        # The marker counts what was dropped — no silent loss.
        assert any(ch.isdigit() for ch in marker)

    def test_single_oversized_choice_truncates_with_visible_marker(self) -> None:
        # Regression (review round 4): one absurd >2900-char choice was
        # sliced with no signal. The cut must be visible.
        from kiro_crew.slack.format import build_options_blocks
        from kiro_crew.slack.transport import SLACK_CAPABILITIES

        n = SLACK_CAPABILITIES.max_buttons
        choices = [f"Choice {i}" for i in range(1, n + 1)] + ["y" * 4000]
        blocks = build_options_blocks(choices)
        ctx = [b for b in blocks if b["type"] == "context"]
        text = ctx[0]["elements"][0]["text"]
        assert len(text) <= 2900
        assert text.endswith("…"), "truncation must be visible, not silent"


class TestTelegramEnforcement:
    def test_steer_seal_near_limit_with_overflow_stays_under_transport_cap(self) -> None:
        # Regression (review round 1): on_steer_consumed ran _rotate_on_length
        # BEFORE apply_options_cap expanded the body with numbered overflow, so
        # a near-limit pre-steer answer sealed past the transport cap.
        from test_telegram import FakeClient

        from kiro_crew.messaging.renderer import STEER_CONSUMED, TEXT_CHUNK, OutputEvent
        from kiro_crew.telegram.client import TELEGRAM_CHUNK_LIMIT
        from kiro_crew.telegram.renderer import TelegramRenderer
        from kiro_crew.telegram.transport import TELEGRAM_CAPABILITIES

        n = TELEGRAM_CAPABILITIES.max_buttons
        trailer = " | ".join(f"Choice number {i} with a long label" for i in range(1, n + 9))
        near_limit = "x" * (TELEGRAM_CHUNK_LIMIT - 60)
        cli = FakeClient()
        r = TelegramRenderer(  # type: ignore[arg-type]
            cli, 55, TELEGRAM_CAPABILITIES, session_key="telegram:1:0"
        )

        async def _go() -> None:
            await r.on_turn_start()
            await r.dispatch(
                OutputEvent(kind=TEXT_CHUNK, text=f"{near_limit}\n\n[OPTIONS: {trailer}]")
            )
            await r.dispatch(OutputEvent(kind=STEER_CONSUMED, text="steered"))

        asyncio.run(_go())
        for text, _ in cli.sent:
            assert len(text) <= TELEGRAM_CHUNK_LIMIT
        for _, text, _ in cli.edits:
            assert len(text) <= TELEGRAM_CHUNK_LIMIT

    def test_keyboard_caps_at_declared_and_overflow_is_visible(self) -> None:
        from test_telegram import FakeClient

        from kiro_crew.messaging.renderer import DONE, TEXT_CHUNK, OutputEvent
        from kiro_crew.telegram.renderer import TelegramRenderer
        from kiro_crew.telegram.transport import TELEGRAM_CAPABILITIES

        n = TELEGRAM_CAPABILITIES.max_buttons
        trailer = " | ".join(f"Choice {i}" for i in range(1, n + 4))
        cli = FakeClient()
        r = TelegramRenderer(  # type: ignore[arg-type]
            cli, 55, TELEGRAM_CAPABILITIES, session_key="telegram:1:0"
        )

        async def _go() -> None:
            await r.on_turn_start()
            await r.dispatch(OutputEvent(kind=TEXT_CHUNK, text=f"Pick.\n\n[OPTIONS: {trailer}]"))
            await r.dispatch(OutputEvent(kind=DONE, stop_reason=""))

        asyncio.run(_go())
        kb = cli.final_markup()
        labels = [b["text"] for row in kb["inline_keyboard"] for b in row]
        assert len(labels) == n, "telegram keyboard was uncapped before enforcement"
        assert labels == [f"Choice {i}" for i in range(1, n + 1)]
        final = cli.final_text()
        assert f"{n + 1}. Choice {n + 1}" in final
        assert f"{n + 3}. Choice {n + 3}" in final


class TestDiscordEnforcement:
    def test_buttons_cap_at_declared_and_overflow_is_visible(self) -> None:
        import pytest_asyncio  # noqa: F401  (asyncio runner parity with test_discord)
        from test_discord import FakeClient

        from kiro_crew.discord.renderer import DiscordRenderer
        from kiro_crew.discord.transport import DISCORD_CAPABILITIES

        n = DISCORD_CAPABILITIES.max_buttons
        trailer = " | ".join(f"Choice {i}" for i in range(1, n + 4))
        cli = FakeClient()
        r = DiscordRenderer(  # type: ignore[arg-type]
            cli, "chan1", DISCORD_CAPABILITIES, session_key="discord:u:c"
        )

        async def _go() -> None:
            await r.on_turn_start()
            await r.on_text_chunk(f"Pick.\n\n[OPTIONS: {trailer}]")
            await r.on_done()

        asyncio.run(_go())
        comps = cli.final_components()
        labels = [b["label"] for row in comps for b in row["components"]]
        assert len(labels) == n
        final = cli.final_text()
        assert f"{n + 1}. Choice {n + 1}" in final
        assert f"{n + 3}. Choice {n + 3}" in final

    def test_overflow_credential_is_redacted_on_the_real_render_path(self) -> None:
        """End-to-end: discord has no display-state pass of its own.

        Before enforcement the 26th+ choices were dropped entirely, so there
        was no exposure; routing them into the parsed body is what opened the
        surface this closes.
        """
        import pytest_asyncio  # noqa: F401  (asyncio runner parity with test_discord)
        from test_discord import FakeClient

        from kiro_crew.discord.renderer import DiscordRenderer
        from kiro_crew.discord.transport import DISCORD_CAPABILITIES

        n = DISCORD_CAPABILITIES.max_buttons
        leaked = "AKIA`" + "`IOSFODNN7EXAMPLE"
        choices = [f"Choice {i}" for i in range(1, n + 1)] + [f"Retry with {leaked}"]
        cli = FakeClient()
        r = DiscordRenderer(  # type: ignore[arg-type]
            cli, "chan1", DISCORD_CAPABILITIES, session_key="discord:u:c"
        )

        async def _go() -> None:
            await r.on_turn_start()
            await r.on_text_chunk("Pick.\n\n[OPTIONS: " + " | ".join(choices) + "]")
            await r.on_done()

        asyncio.run(_go())
        assert "IOSFODNN7EXAMPLE" not in cli.final_text()


class TestZeroWidgetTextFallback:
    """``append_options_text`` — the ``max_buttons=0`` counterpart of the cap.

    A channel with no interactive widget used to DELETE the trailer, so the user
    read a question with no visible answers. Numbering them into the body keeps
    them answerable by typing, which every channel supports.

    The sanitisations are the reason this lives in shared code rather than per
    channel: the body is markdown-parsed, so a credential split by a code span is
    whole on screen, and the body is also where platforms parse mentions.
    """

    def test_choices_are_numbered_from_one(self) -> None:
        # Numbering starts at 1, not after a widget's slots: there is no widget.
        assert append_options_text("Pick one", ["A", "B", "C"]) == "Pick one\n\n1. A\n2. B\n3. C"

    def test_no_choices_leaves_the_body_byte_identical(self) -> None:
        assert append_options_text("Body", []) == "Body"

    def test_an_empty_body_yields_just_the_list(self) -> None:
        assert append_options_text("", ["A"]) == "1. A"

    def test_a_body_ending_in_one_newline_still_gets_a_paragraph_break(self) -> None:
        # A single trailing newline is mid-paragraph in every markdown dialect,
        # so the list would render inline without the extra break.
        assert append_options_text("Body\n", ["A"]) == "Body\n\n1. A"

    def test_a_credential_split_by_markup_is_redacted_in_display_form(self) -> None:
        """The driver's byte-level redactor saw the key broken; the reader will
        not. Both halves must be scrubbed at this sink."""
        out = append_options_text("Pick", ["use AKIA`" + "A" * 16 + "`"])
        assert "AKIA" + "A" * 16 not in out.replace("`", "")

    def test_mass_mention_syntax_is_defanged(self) -> None:
        # A prompt-injected choice would otherwise mass-notify a whole workspace.
        out = append_options_text("Pick", ["@everyone", "<!channel>"])
        assert "@everyone" not in out
        assert "<!channel>" not in out

    def test_non_ascii_choices_are_not_mangled(self) -> None:
        out = append_options_text("Pick", ["日本語", "café"])
        assert "日本語" in out and "café" in out

    def test_the_zero_cap_branch_of_apply_options_cap_is_unchanged(self) -> None:
        """The new helper is additive.

        ``apply_options_cap``'s "keep nothing, leave the body alone" contract is
        what the widget-capable callers rely on, so it stays exactly as pinned.
        """
        caps = TransportCapabilities(max_buttons=0)
        assert apply_options_cap("Body", ["A", "B"], caps) == ("Body", [])


class TestSplitOptionsTrailer:
    """The shared trailer parse, used by every zero-widget channel.

    Its second branch is a leak guard, not a nicety: a trailer still streaming in
    has no knowable choices, so it must be HIDDEN rather than rendered. That rule
    used to be reimplemented per channel, which is how a channel ends up leaking
    reserved protocol into the conversation as raw text.
    """

    def test_a_complete_trailer_is_split_off(self) -> None:
        assert split_options_trailer("Pick one\n\n[OPTIONS: A | B | C]") == (
            "Pick one",
            ["A", "B", "C"],
        )

    def test_choices_are_stripped_and_blanks_dropped(self) -> None:
        assert split_options_trailer("Q [OPTIONS:  A  |  | B ]")[1] == ["A", "B"]

    def test_an_unterminated_trailer_is_hidden_with_no_choices(self) -> None:
        assert split_options_trailer("Pick one\n\n[OPTIONS: A | B") == ("Pick one", [])

    def test_text_with_no_trailer_is_returned_unchanged(self) -> None:
        assert split_options_trailer("just an answer") == ("just an answer", [])

    def test_a_trailer_mid_text_is_not_a_trailer(self) -> None:
        # The grammar is end-anchored: a bracketed literal a user typed mid-answer
        # is content, not protocol.
        body, choices = split_options_trailer("see [OPTIONS: a] above and more")
        assert body == "see [OPTIONS: a] above and more" and choices == []

    def test_leading_whitespace_is_preserved(self) -> None:
        # rstrip only: stripping the left would silently re-indent code.
        assert split_options_trailer("    indented\n[OPTIONS: A]")[0] == "    indented"

    def test_an_unterminated_tag_is_not_redos(self) -> None:
        # Regression (py/polynomial-redos): a lazy ``\s*(.*?)`` body could consume
        # a "[" that ALSO starts the outer "[OPTIONS:" literal, so over text with
        # many "[OPTIONS:" prefixes search() re-explored the body from each
        # position — polynomial. The tempered OPTIONS_RE_TRAILER forbids only a
        # re-occurring "[OPTIONS:", making the body unambiguous (linear). A
        # whitespace-padded unterminated tag and many repeated prefixes (the real
        # pump) must both return promptly.
        import time

        evil = "[OPTIONS:" + ("\t" * 200_000) + "x"
        start = time.perf_counter()
        assert split_options_trailer(evil)[0] == ""
        assert time.perf_counter() - start < 1.0, "possible ReDoS"

        evil = "[OPTIONS:" * 100_000 + "x"
        start = time.perf_counter()
        split_options_trailer(evil)
        assert time.perf_counter() - start < 1.0, "possible ReDoS"

    def test_every_zero_widget_channel_routes_through_it(self) -> None:
        """The reason the parse was hoisted: four copies became one caller each.

        Weixin is deliberately absent — its variant has different whitespace
        semantics and no fragment guard, so converging it is a behaviour change
        that belongs in its own commit rather than riding along here.
        """
        from kiro_crew.imessage import renderer as imessage_renderer
        from kiro_crew.teams import renderer as teams_renderer
        from kiro_crew.wecom import renderer as wecom_renderer

        for mod in (teams_renderer, imessage_renderer, wecom_renderer):
            assert mod._strip_options("Answer\n\n[OPTIONS: a | b]") == "Answer"
            assert mod._strip_options("Answer\n\n[OPTIONS: a | b") == "Answer"
            assert mod._strip_options("plain") == "plain"


class TestWebexEnforcement:
    def test_card_actions_cap_at_declared_and_overflow_is_visible(self) -> None:
        """Drive the REAL render path, not the helper.

        The ratchet exists because a renderer can call the shared cap and then
        build its widget from the uncapped list, so the only assertion worth
        making is against what the client was actually asked to send.
        """
        import asyncio

        from test_webex_renderer import FakeClient

        from kiro_crew.messaging.renderer import DONE, TEXT_CHUNK, OutputEvent
        from kiro_crew.webex.renderer import WebexRenderer
        from kiro_crew.webex.transport import WEBEX_CAPABILITIES

        n = WEBEX_CAPABILITIES.max_buttons
        trailer = " | ".join(f"Choice {i}" for i in range(1, n + 4))
        cli = FakeClient()
        # A card is rendered only when its press can be resolved, so the real path
        # needs the dispatcher's choice store — the card is the last thing a turn
        # sends, and a renderer-owned map is gone before any press arrives.
        r = WebexRenderer(
            cli,
            "ROOM",
            WEBEX_CAPABILITIES,  # type: ignore[arg-type]
            publish_choices=lambda _nonce, _choices: None,
        )

        async def _go() -> None:
            await r.on_turn_start()
            await r.dispatch(OutputEvent(kind=TEXT_CHUNK, text=f"Pick.\n\n[OPTIONS: {trailer}]"))
            await r.dispatch(OutputEvent(kind=DONE, stop_reason=""))

        asyncio.run(_go())

        card = next(kw for (_, _, kw) in cli.sent_full if kw.get("attachments"))
        actions = card["attachments"][0]["content"]["actions"]
        labels = [a["title"] for a in actions]
        assert len(labels) == n, "webex card actions were uncapped"
        assert labels == [f"Choice {i}" for i in range(1, n + 1)]
        # Overflow is numbered CONTINUING the widget slots, never dropped.
        final = cli.edits[-1][2]
        assert f"{n + 1}. Choice {n + 1}" in final
        assert f"{n + 3}. Choice {n + 3}" in final

    def test_a_press_resolves_by_index_not_by_text(self) -> None:
        """A crafted press must not be able to inject words into the turn.

        The button carries an index into the choices the renderer rendered, so a
        forged ``kirocrew_choice`` either indexes a real choice or resolves to
        nothing — it can never become arbitrary text.
        """
        import asyncio

        from test_webex_renderer import FakeClient

        from kiro_crew.messaging.renderer import DONE, TEXT_CHUNK, OutputEvent
        from kiro_crew.webex.cards import KEY_CHOICE, KEY_NONCE, LiveChoices, read_press
        from kiro_crew.webex.renderer import WebexRenderer
        from kiro_crew.webex.transport import WEBEX_CAPABILITIES

        cli = FakeClient()
        live = LiveChoices()
        r = WebexRenderer(
            cli,
            "ROOM",
            WEBEX_CAPABILITIES,  # type: ignore[arg-type]
            publish_choices=lambda nonce, choices: live.publish("S", nonce, choices),
        )

        async def _go() -> None:
            await r.on_turn_start()
            await r.dispatch(OutputEvent(kind=TEXT_CHUNK, text="Pick.\n\n[OPTIONS: Yes | No]"))
            await r.dispatch(OutputEvent(kind=DONE, stop_reason=""))

        asyncio.run(_go())
        card = next(kw for (_, _, kw) in cli.sent_full if kw.get("attachments"))
        data = card["attachments"][0]["content"]["actions"][1]["data"]
        _, choice, nonce, _ = read_press(data)

        # A forged index, a forged nonce, and injected text all resolve to nothing.
        assert live.take("S", "99", nonce) == ""
        assert live.take("S", choice, "deadbeefdeadbeef") == ""
        assert live.take("S", "rm -rf /", nonce) == ""
        assert read_press({KEY_CHOICE: "x", KEY_NONCE: "y"})[0] == ""
        # The real press resolves once, and only once: the platform cannot retire
        # the buttons, so the ENTRY is what has to expire.
        assert live.take("S", choice, nonce) == "No"
        assert live.take("S", choice, nonce) == ""
