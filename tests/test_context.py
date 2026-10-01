"""Tests for common/context.py: every strategy keeps the call/result structure valid and does what it says."""

from __future__ import annotations

import pytest

from common import context as cx
from common.agent import run_agent
from common.chat import ToolCall
from common.fake import fake_llm, tool_calls
from common.tools import ToolRegistry, tool


def history(n: int, size: int = 100) -> list[dict]:
    """task + n exchanges, each: assistant calls get_page(i) -> tool result of `size` chars."""
    msgs = [{"role": "user", "content": "TASK: read the pages"}]
    for i in range(n):
        msgs.append(
            {
                "role": "assistant",
                "content": f"thinking {i}",
                "tool_calls": [{"id": f"c{i}", "name": "get_page", "args": {"n": i}}],
            }
        )
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": f"c{i}",
                "name": "get_page",
                "content": f"page {i} " + "x" * size,
                "is_error": False,
            }
        )
    return msgs


# ----------------------------------------------------------------------------- measuring and structure


def test_token_counts_are_monotone_and_include_tool_call_arguments():
    h = history(3)
    assert cx.count_tokens(h) < cx.count_tokens(history(4)) < cx.count_tokens(history(4, 400))
    assert 'get_page({"n": 0})' in cx.message_text(h[1])
    assert (
        cx.approx_tokens("") == 0
        and cx.approx_tokens("abcd") == 1
        and cx.approx_tokens("abcde") == 2
    )
    assert cx.count_tokens(h, counter=lambda t: 1) == len(h) * 5


def test_exchanges_keep_assistant_calls_with_their_results():
    units = cx.exchanges(history(3))
    assert (
        [len(u) for u in units] == [1, 2, 2, 2]
        and units[1][0]["role"] == "assistant"
        and units[1][1]["role"] == "tool"
    )
    parallel = [
        {"role": "user", "content": "t"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "name": "f", "args": {}},
                {"id": "b", "name": "f", "args": {}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "name": "f", "content": "1", "is_error": False},
        {"role": "tool", "tool_call_id": "b", "name": "f", "content": "2", "is_error": False},
    ]
    assert [len(u) for u in cx.exchanges(parallel)] == [1, 3]


def test_check_invariants_detects_each_kind_of_breakage():
    good = history(3)
    assert cx.check_invariants(good) == []
    orphan_result = [good[0], good[2]]  # result without its call
    assert "no matching call" in cx.check_invariants(orphan_result)[0]
    unanswered = good[:2]
    assert "never got results" in cx.check_invariants(unanswered)[0]
    interrupted = [good[0], good[1], {"role": "user", "content": "hi"}]
    assert "unanswered" in cx.check_invariants(interrupted)[0]
    double = [good[0], good[1], good[1]]
    assert "never got results" in cx.check_invariants(double)[0]


# ----------------------------------------------------------------------------- truncate and clear


def test_truncate_caps_long_results_with_a_visible_note_and_leaves_others_alone():
    h = history(2, size=3000)
    out = cx.truncate_tool_results(500)(h, None)
    assert (
        len(out[2]["content"]) < 560
        and "[truncated: " in out[2]["content"]
        and out[2]["content"].startswith("page 0 ")
    )
    assert out[0] == h[0] and out[1] == h[1]
    short = cx.truncate_tool_results(5000)(h, None)
    assert short == h
    assert len(h[2]["content"]) > 3000, "the input history is not modified"
    exact = [
        {"role": "tool", "tool_call_id": "c", "name": "f", "content": "x" * 10, "is_error": False}
    ]
    assert cx.truncate_tool_results(10)(exact, None) == exact, (
        "exactly at the limit is not truncated"
    )


def test_clear_old_keeps_the_last_n_results_and_all_calls():
    h = history(6)
    out = cx.clear_old_tool_results(keep_last=2)(h, None)
    results = [m for m in out if m["role"] == "tool"]
    assert [m["content"] == cx.CLEARED for m in results] == [True, True, True, True, False, False]
    assert [m for m in out if m["role"] == "assistant"] == [
        m for m in h if m["role"] == "assistant"
    ]
    assert cx.check_invariants(out) == [] and results[0]["tool_call_id"] == "c0", (
        "ids survive clearing"
    )
    assert cx.count_tokens(out) < cx.count_tokens(h)
    assert cx.clear_old_tool_results(keep_last=10)(h, None) == h
    assert all(
        m["content"] == cx.CLEARED
        for m in cx.clear_old_tool_results(keep_last=0)(h, None)
        if m["role"] == "tool"
    )
    again = cx.clear_old_tool_results(keep_last=2)(out, None)
    assert again == out, "idempotent"


# ----------------------------------------------------------------------------- scratchpad


def test_scratchpad_save_update_forget_and_limits():
    pad = cx.Scratchpad(max_chars=60)
    assert pad.remember("a", "1") == "Saved note 'a'."
    assert pad.remember("a", "22").startswith("Updated note 'a' (was: '1')")
    pad.remember("b", "x" * 20)
    with pytest.raises(ValueError, match="notes are full.*keys now: a, b"):
        pad.remember("c", "y" * 50)
    assert "c" not in pad.notes, "a rejected note is not stored"
    assert pad.forget("b") == "Deleted note 'b'."
    with pytest.raises(ValueError, match="no note 'zzz'. Existing keys: a"):
        pad.forget("zzz")
    assert pad.render() == "[Your notes so far]\n- a: 22" and cx.Scratchpad().render() == ""
    big = cx.Scratchpad(max_chars=10)
    big.remember("k", "123456789")
    with pytest.raises(ValueError):
        big.remember("k2", "1")  # 10+... over the limit by key length too


def test_scratchpad_tools_work_through_the_registry():
    pad = cx.Scratchpad()
    reg = ToolRegistry(pad.tools())
    assert (
        reg.execute(ToolCall("1", "remember", {"key": "code_3", "value": "XYZ-9"})).content
        == "Saved note 'code_3'."
    )
    assert pad.notes == {"code_3": "XYZ-9"}
    full = ToolRegistry(cx.Scratchpad(max_chars=5).tools()).execute(
        ToolCall("1", "remember", {"key": "long_key", "value": "v"})
    )
    assert full.is_error and "notes are full" in full.content
    assert (
        reg.execute(ToolCall("2", "forget", {"key": "code_3"})).content == "Deleted note 'code_3'."
        and pad.notes == {}
    )


def test_inject_notes_replaces_rather_than_stacks_and_never_adds_a_message():
    pad = cx.Scratchpad()
    hook = cx.inject_notes(pad)
    h = history(1)
    assert hook(h, None) == h, "no notes yet: nothing injected"
    pad.remember("a", "1")
    once = hook(h, None)
    assert (
        len(once) == len(h)
        and once[0]["content"] == "TASK: read the pages\n\n[Your notes so far]\n- a: 1"
    )
    pad.remember("a", "2")
    twice = hook(once, None)
    assert twice[0]["content"].count("[Your notes so far]") == 1 and twice[0]["content"].endswith(
        "- a: 2"
    )
    assert h[0]["content"] == "TASK: read the pages", "input untouched"
    pad.forget("a")
    assert hook(twice, None)[0]["content"] == "TASK: read the pages"


# ----------------------------------------------------------------------------- compaction


def test_compaction_waits_for_the_trigger_keeps_the_task_and_recent_exchanges():
    seen = []

    def summarizer(t):
        seen.append(t)
        return "S: pages 0-3 read; code=ABC"

    h = history(8)
    hook = cx.compact_history(summarizer, trigger_tokens=10**6, keep_last=3)
    assert hook(h, None) == h and seen == [], "below the trigger: untouched, summarizer not called"
    out = cx.compact_history(summarizer, trigger_tokens=10, keep_last=3)(h, None)
    assert (
        len(seen) == 1 and "get_page" in seen[0] and "page 0 " in seen[0] and "page 4 " in seen[0]
    )
    assert "page 5 " not in seen[0], "the last 3 exchanges are not summarised"
    assert (
        out[0]["content"]
        == "TASK: read the pages\n\n[Summary of earlier work]\nS: pages 0-3 read; code=ABC"
    )
    assert out[1:] == h[1 + 2 * 5 :], "the last three exchanges survive verbatim"
    assert cx.check_invariants(out) == [] and out[1]["role"] == "assistant"
    assert cx.count_tokens(out) < cx.count_tokens(h)


def test_compaction_folds_the_previous_summary_into_the_next_and_does_nothing_when_too_short():
    calls = []
    hook = cx.compact_history(
        lambda t: calls.append(t) or f"summary#{len(calls)}", trigger_tokens=10, keep_last=2
    )
    first = hook(history(6), None)
    longer = first + history(4)[1:]  # more exchanges after the summary
    longer = [m for m in longer]
    second = hook(longer, None)
    assert "PREVIOUS SUMMARY:\nsummary#1" in calls[1] and "NEW STEPS:" in calls[1]
    assert second[0]["content"].count("[Summary of earlier work]") == 1 and second[0][
        "content"
    ].endswith("summary#2")
    short = history(2)
    assert cx.compact_history(lambda t: "x", trigger_tokens=1, keep_last=2)(short, None) == short, (
        "nothing old enough to fold"
    )


def test_compaction_caps_the_summary_and_keeps_notes_out_of_the_summary():
    pad = cx.Scratchpad()
    pad.remember("k", "v")
    h = cx.inject_notes(pad)(history(6), None)
    got = []
    out = cx.compact_history(
        lambda t: got.append(t) or "z" * 5000, trigger_tokens=10, keep_last=2, max_summary_chars=100
    )(h, None)
    task, summary, notes = cx.split_first(out[0]["content"])
    assert task == "TASK: read the pages" and len(summary) == 100 and notes == ""
    assert "[Your notes so far]" not in got[0], (
        "the notes block is re-injected each turn, not summarised"
    )
    again = cx.inject_notes(pad)(out, None)
    assert cx.split_first(again[0]["content"])[2] == "[Your notes so far]\n- k: v"


def test_split_and_join_first_roundtrip():
    for task, summary, notes in [
        ("t", "", ""),
        ("t", "s", ""),
        ("t", "", "[Your notes so far]\n- a: 1"),
        ("t", "s", "[Your notes so far]\n- a: 1"),
    ]:
        assert cx.split_first(cx.join_first(task, summary, notes)) == (task, summary, notes)


def test_drop_summarizer_says_how_much_was_removed():
    assert cx.drop_summarizer("USER: t\nASSISTANT: a\nTOOL: r\nASSISTANT: b\nTOOL: r2").startswith(
        "(2 earlier steps"
    )
    assert cx.drop_summarizer("ASSISTANT: only").startswith("(1 earlier steps")


def test_chain_applies_hooks_in_order():
    order = []
    mk = lambda tag: lambda m, r: order.append(tag) or m  # noqa: E731
    cx.chain(mk("a"), mk("b"), mk("c"))([], None)
    assert order == ["a", "b", "c"]


def test_llm_summarizer_sends_the_transcript_and_a_fact_preserving_instruction():
    with fake_llm([(r"(?s).*", "the summary")]) as f:
        out = cx.llm_summarizer("anthropic")("ASSISTANT: found code ABC-1")
    assert (
        out == "the summary"
        and "ABC-1" in f.calls[0].prompt
        and "EVERY concrete" in f.calls[0].system
    )


# ----------------------------------------------------------------------------- in the real loop


@tool
def get_page(n: int) -> str:
    """Return page n.

    Args:
        n: Page number.
    """
    return f"page {n} " + "lorem " * 200


def test_hooks_keep_a_long_agent_run_valid_and_small_while_the_transcript_keeps_everything():
    hook = cx.chain(
        cx.clear_old_tool_results(keep_last=2),
        cx.compact_history(cx.drop_summarizer, trigger_tokens=600, keep_last=3),
    )
    steps = [tool_calls(("get_page", {"n": i})) for i in range(15)] + ["done"]
    sizes, valid = [], []

    def spy(messages, run):
        out = hook(messages, run)
        sizes.append(cx.count_tokens(out))
        valid.append(cx.check_invariants(out) == [])
        return out

    with fake_llm([(r"(?s).*", steps)]):
        run = run_agent(
            "read", ToolRegistry([get_page]), provider="anthropic", max_steps=20, context_hook=spy
        )
    assert run.ok and all(valid) and len(run.transcript) == 1 + 15 * 2
    assert max(sizes) < 1500 and sizes[-1] < 1.5 * sizes[3], "the context stops growing"
    naive = cx.count_tokens(run.transcript)
    assert naive > 3 * max(sizes), "an untrimmed history would be several times larger"
    assert len(run.messages) < len(run.transcript)


def test_an_orphan_tool_result_is_its_own_unit_not_glued_to_a_user_message():
    msgs = [
        {"role": "user", "content": "t"},
        {"role": "tool", "tool_call_id": "x", "name": "f", "content": "r", "is_error": False},
    ]
    assert [len(u) for u in cx.exchanges(msgs)] == [1, 1]


def test_compaction_triggers_only_when_strictly_over_the_threshold():
    h = history(8)
    exact = cx.count_tokens(h, counter=lambda t: 1)  # every message costs 1 + 4
    at = cx.compact_history(lambda t: "S", trigger_tokens=exact, keep_last=2, counter=lambda t: 1)(
        h, None
    )
    over = cx.compact_history(
        lambda t: "S", trigger_tokens=exact - 1, keep_last=2, counter=lambda t: 1
    )(h, None)
    assert at == h and over != h


def test_inject_notes_leaves_a_history_that_does_not_start_with_a_user_message_alone():
    pad = cx.Scratchpad()
    pad.remember("a", "1")
    odd = [{"role": "assistant", "content": "hello"}]
    assert cx.inject_notes(pad)(odd, None) == odd and cx.inject_notes(pad)([], None) == []
