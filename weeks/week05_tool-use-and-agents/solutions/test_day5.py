"""Tests for Week 5 Day 5: the task, the scripted reader, and what each context strategy preserves."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))

import day5_solution as d5  # noqa: E402

from common import context as cx  # noqa: E402
from common.chat import ToolCall  # noqa: E402
from common.tools import ToolRegistry  # noqa: E402

BY_NAME = {s.name: s for s in d5.STRATEGIES}


@pytest.fixture(scope="module")
def results():
    env = d5.Env()  # approximate token counter, drop summariser: fast and offline
    return {s.name: d5.run_strategy(s, env) for s in d5.STRATEGIES if "Qwen" not in s.name}


# ----------------------------------------------------------------------------- the task


def test_pages_are_deterministic_long_and_carry_the_code_at_the_end():
    assert d5.page_text(7) == d5.page_text(7) != d5.page_text(8)
    for n in range(1, d5.N_PAGES + 1):
        text = d5.page_text(n)
        assert 2300 < len(text) < 3200 and text.startswith(f"=== PAGE {n} ===")
        has_fact = "FACT:" in text
        assert has_fact == (n in d5.NEEDLES)
        if has_fact:
            assert text.rstrip().endswith(f"is {d5.NEEDLES[n]}."), "the fact is the LAST line"
            assert len(re.findall(r"access code for page", text)) == 1
        assert "not an access code" in text, "a decoy ticket reference on every page"


def test_get_page_tool_validates_the_range():
    reg = ToolRegistry([d5.get_page])
    assert reg.execute(ToolCall("1", "get_page", {"n": 3})).content.startswith("=== PAGE 3 ===")
    for bad in (0, 51, -1):
        r = reg.execute(ToolCall("1", "get_page", {"n": bad}))
        assert r.is_error and "between 1 and 50" in r.content


def test_needles_are_distinct_and_the_codes_look_like_the_decoys_so_a_regex_cannot_cheat():
    assert len(set(d5.NEEDLES.values())) == 5 and all(
        d5.CODE_RE.fullmatch(c) for c in d5.NEEDLES.values()
    )
    decoy = re.search(r"(QWE|RTY|UIO)-\d{3}", d5.page_text(1))
    assert decoy and d5.CODE_RE.fullmatch(decoy.group())


# ----------------------------------------------------------------------------- reading what the context shows


def test_visible_pairs_reads_results_notes_and_free_text_summaries():
    msgs = [
        {
            "role": "user",
            "content": "task\n\n[Summary of earlier work]\nPages 1-5 read. On page 3 the code was TANGO-417.\n\n[Your notes so far]\n- code_17: OSCAR-902",
        },
        {
            "role": "tool",
            "content": "...\nFACT: the access code for page 29 is LIMA-058.",
            "tool_call_id": "x",
            "name": "get_page",
            "is_error": False,
        },
        {
            "role": "tool",
            "content": "Ticket reference QWE-123 (not an access code).",
            "tool_call_id": "y",
            "name": "get_page",
            "is_error": False,
        },
    ]
    assert d5.visible_pairs(msgs) == {3: "TANGO-417", 17: "OSCAR-902", 29: "LIMA-058"}
    assert d5.visible_pairs([]) == {}
    first_wins = d5.visible_pairs(
        [
            {"role": "user", "content": "code_3: AAA-111"},
            {"role": "user", "content": "code_3: BBB-222"},
        ]
    )
    assert first_wins == {3: "AAA-111"}


def test_last_page_called_finds_the_most_recent_get_page_call():
    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "a", "name": "get_page", "args": {"n": 4}}],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "b", "name": "remember", "args": {"key": "k", "value": "v"}}],
        },
    ]
    assert d5.last_page_called(msgs) == 4 and d5.last_page_called([]) == 0


# ----------------------------------------------------------------------------- the strategies


def test_every_strategy_finishes_and_never_breaks_the_call_result_invariant(results):
    assert len(results) == 5
    for r in results.values():
        assert r.run.status == "done" and r.invariant_ok, r.name
        assert r.wrong_codes == 0, "no strategy may report a WRONG code"


def test_naive_history_finds_everything_but_its_cost_grows_quadratically(results):
    naive = results["naive (no trimming)"]
    assert naive.codes_right == 5 and len(naive.run.steps) == 51
    ctx = naive.context_tokens
    assert all(b > a for a, b in zip(ctx, ctx[1:], strict=False)), (
        "every call carries every earlier page"
    )
    assert ctx[-1] / ctx[1] > 25, "about 50 pages by the end versus 1 at the start"
    # total input ~ n^2/2 pages: far more than the n pages that were actually read
    assert naive.total_input > 20 * ctx[-1] / 2 * 0.9 and naive.total_input / ctx[-1] > 20


def test_head_truncation_destroys_the_information_it_was_meant_to_cap(results):
    r = results["truncate results to 800 chars"]
    assert r.codes_right == 0 and "could not find" in r.run.answer, (
        "the codes are at the END of each page"
    )
    assert r.peak < results["naive (no trimming)"].peak


def test_clearing_old_results_bounds_the_context_but_forgets_all_but_the_recent_codes(results):
    r = results["clear old results (keep 3)"]
    assert r.peak < 4000 and r.peak < results["naive (no trimming)"].peak / 6
    assert r.codes_right == 1 and r.found == {48: "ECHO-266"}, (
        "only page 48 is among the last three results"
    )
    slope = (r.context_tokens[-1] - r.context_tokens[10]) / (len(r.context_tokens) - 11)
    naive_slope = (
        results["naive (no trimming)"].context_tokens[-1]
        - results["naive (no trimming)"].context_tokens[10]
    ) / (len(r.context_tokens) - 11)
    assert 0 < slope < 60, (
        "not flat: every cleared result leaves a ~25-token placeholder, and each call is still listed"
    )
    assert slope < naive_slope / 10


def test_compaction_without_a_summary_is_just_forgetting(results):
    r = results["compaction, no summary"]
    assert r.found == {44: "ZULU-731", 48: "ECHO-266"} and r.codes_right == 2, (
        "only the pages still inside the window"
    )
    assert r.peak < 6500 and r.peak > results["clear old results (keep 3)"].peak, (
        "bounded by the trigger, not by 3 results"
    )
    assert any(b < a for a, b in zip(r.context_tokens, r.context_tokens[1:], strict=False)), (
        "the context shrinks when it compacts"
    )


def test_notes_keep_all_five_codes_at_a_fraction_of_the_naive_cost(results):
    r, naive = results["clear old + scratchpad notes"], results["naive (no trimming)"]
    assert r.codes_right == 5 and len(r.run.steps) == 56, "5 extra steps to save the 5 codes"
    assert r.total_input < 0.25 * naive.total_input and r.peak < naive.peak / 6
    assert [t.name for t in [type("T", (), {"name": c.name}) for c in r.run.calls]].count(
        "remember"
    ) == 5


def test_a_summary_that_preserves_the_facts_makes_compaction_work_in_the_full_loop():
    def faithful(transcript: str) -> str:
        pairs = re.findall(r"access code for page (\d+) is ([A-Z]{3,5}-\d{3})", transcript)
        pairs += re.findall(
            r"On page (\d+) the code was ([A-Z]{3,5}-\d{3})", transcript
        )  # carry the PREVIOUS summary
        return (
            " ".join(f"On page {p} the code was {c}." for p, c in dict(pairs).items())
            or "Nothing notable yet."
        )

    env = d5.Env(summarizer=faithful)
    strat = d5.Strategy(
        "faithful",
        lambda e: (
            cx.compact_history(e.summarizer, trigger_tokens=d5.COMPACT_AT, keep_last=3),
            None,
        ),
    )
    r = d5.run_strategy(strat, env)
    assert r.codes_right == 5 and r.invariant_ok and r.peak < 7000
    assert r.run.status == "done"


def test_a_summariser_that_forgets_to_carry_the_previous_summary_loses_facts_on_the_second_compaction():
    def forgetful(transcript: str) -> str:
        pairs = re.findall(
            r"access code for page (\d+) is ([A-Z]{3,5}-\d{3})", transcript
        )  # new results only
        return (
            " ".join(f"On page {p} the code was {c}." for p, c in pairs) or "Nothing notable yet."
        )

    strat = d5.Strategy(
        "forgetful",
        lambda e: (
            cx.compact_history(e.summarizer, trigger_tokens=d5.COMPACT_AT, keep_last=3),
            None,
        ),
    )
    r = d5.run_strategy(strat, d5.Env(summarizer=forgetful))
    assert r.codes_right < 5 and r.wrong_codes == 0


def test_clear_then_compact_is_the_wrong_order_because_the_summariser_never_sees_the_results():
    seen = []

    def spy(transcript: str) -> str:
        seen.append(transcript)
        return "S"

    wrong = d5.Strategy(
        "wrong order",
        lambda e: (
            cx.chain(
                cx.clear_old_tool_results(3),
                cx.compact_history(e.summarizer, trigger_tokens=2500, keep_last=3),
            ),
            None,
        ),
    )
    d5.run_strategy(wrong, d5.Env(summarizer=spy))
    assert seen and not any("FACT:" in t for t in seen), (
        "the facts were already cleared when compaction ran"
    )
    seen.clear()
    right = d5.Strategy(
        "right order",
        lambda e: (
            cx.compact_history(e.summarizer, trigger_tokens=d5.COMPACT_AT, keep_last=3),
            None,
        ),
    )
    d5.run_strategy(right, d5.Env(summarizer=spy))
    assert any("FACT:" in t for t in seen)


def test_cost_ranking_matches_the_lesson(results):
    t = {n: r.total_input for n, r in results.items()}
    assert (
        t["naive (no trimming)"]
        > t["truncate results to 800 chars"]
        > t["compaction, no summary"]
        > t["clear old results (keep 3)"]
    )
    assert t["clear old + scratchpad notes"] < 0.25 * t["naive (no trimming)"]


def test_scripted_reader_records_one_measurement_per_model_call(results):
    r = results["naive (no trimming)"]
    assert len(r.context_tokens) == len(r.run.steps) == 51
    assert d5.ScriptedReader().counter is cx.approx_tokens


def test_strategy_names_are_unique():
    assert len({s.name for s in d5.STRATEGIES}) == len(d5.STRATEGIES) == 6
