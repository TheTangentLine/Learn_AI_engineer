"""Tests for common/agent.py: the loop, its budgets, its loop-breakers and its history discipline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fake_llm_server import FakeLLMServer  # noqa: E402

from common import llm  # noqa: E402
from common.agent import LAST_STEP_NOTE, AgentRun, run_agent  # noqa: E402
from common.fake import fake_llm, tool_calls  # noqa: E402
from common.tools import ToolRegistry, tool  # noqa: E402

CALLS: list[str] = []


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First.
        b: Second.
    """
    CALLS.append(f"add({a},{b})")
    return a + b


@tool
def boom() -> str:
    """Always fails."""
    raise RuntimeError("disk on fire")


REG = ToolRegistry([add, boom])
ANY = r"(?s).*"


def seq(*turns):
    """One rule that plays the given turns in order (the last one repeats)."""
    return [(ANY, list(turns))]


def test_happy_path_tool_then_answer_keeps_history_in_order():
    CALLS.clear()
    with fake_llm(seq(tool_calls(("add", {"a": 2, "b": 3})), "The sum is 5.")) as f:
        run = run_agent("2+3?", REG, provider="anthropic")
    assert run.ok and run.status == "done" and run.answer == "The sum is 5."
    assert [m["role"] for m in run.messages] == ["user", "assistant", "tool"]
    assert (
        run.messages[2]["content"] == "5"
        and run.messages[2]["tool_call_id"] == run.messages[1]["tool_calls"][0]["id"]
    )
    assert run.tool_names == ["add"] and CALLS == ["add(2,3)"] and run.errors == 0
    assert len(run.steps) == 2 and len(f.calls) == 2
    assert run.input_tokens > 0 and run.output_tokens > 0 and run.cost_usd > 0 and run.seconds >= 0


def test_parallel_calls_all_answered_in_order_including_failures():
    turn = tool_calls(
        ("add", {"a": 1, "b": 1}), ("boom", {}), ("add", {"a": 5, "b": 5}), ("nope", {})
    )
    with fake_llm(seq(turn, "done")):
        run = run_agent("x", REG, provider="anthropic")
    contents = [m["content"] for m in run.messages if m["role"] == "tool"]
    assert (
        contents[0] == "2"
        and "disk on fire" in contents[1]
        and contents[2] == "10"
        and "Unknown tool" in contents[3]
    )
    assert [m["is_error"] for m in run.messages if m["role"] == "tool"] == [
        False,
        True,
        False,
        True,
    ]
    assert run.ok and run.errors == 2


def test_last_step_hides_tools_and_a_model_that_still_asks_has_not_finished():
    with fake_llm(seq(tool_calls(("add", {"a": 1, "b": 1})))) as f:  # always asks for a tool
        run = run_agent("x", REG, provider="anthropic", max_steps=3)
    assert run.status == "max_steps" and len(run.steps) == 3 and "still asked" in run.error
    assert [c.kwargs.get("tool_choice") for c in f.calls] == [None, None, "none"]
    assert LAST_STEP_NOTE in f.calls[-1].system and LAST_STEP_NOTE not in f.calls[0].system
    assert run.steps[-1].calls == [], (
        "a request made on the last step is never executed, so it is not a call"
    )
    assert len(run.calls) == 2 and len(run.results) == 2 and run.answer == ""
    assert [m["role"] for m in run.messages] == ["user", "assistant", "tool", "assistant", "tool"]


def test_model_that_answers_on_the_last_step_is_done():
    with fake_llm(seq(tool_calls(("add", {"a": 1, "b": 1})), "It is 2.")):
        run = run_agent("x", REG, provider="anthropic", max_steps=2)
    assert run.status == "done" and run.answer == "It is 2."


def test_single_step_budget_means_tools_are_never_available():
    with fake_llm(seq("direct answer")) as f:
        run = run_agent("x", REG, provider="anthropic", max_steps=1)
    assert run.ok and f.calls[0].kwargs["tool_choice"] == "none"


def test_repeated_identical_calls_are_blocked_with_the_earlier_result_in_the_message():
    CALLS.clear()
    same = tool_calls(("add", {"a": 2, "b": 2}))
    with fake_llm(seq(same, same, same, "ok, 4")):
        run = run_agent("x", REG, provider="anthropic", repeat_limit=2)
    assert run.ok and CALLS == ["add(2,2)", "add(2,2)"], "the third identical call is NOT executed"
    third = run.steps[2]
    assert third.blocked == 1 and third.results[0].is_error
    assert "Blocked" in third.results[0].content and "'4'" in third.results[0].content
    different = tool_calls(("add", {"a": 2, "b": 3}))
    CALLS.clear()
    with fake_llm(seq(same, different, same, different, "done")):
        run = run_agent("x", REG, provider="anthropic", repeat_limit=2, max_steps=8)
    assert run.ok and len(CALLS) == 4 and all(s.blocked == 0 for s in run.steps), (
        "different args are not repeats"
    )


def test_argument_order_does_not_hide_a_repeat():
    a = tool_calls(("add", {"a": 1, "b": 2}))
    b = tool_calls(("add", {"b": 2, "a": 1}))
    CALLS.clear()
    with fake_llm(seq(a, b, a, "x")):
        run = run_agent("x", REG, provider="anthropic", repeat_limit=2)
    assert len(CALLS) == 2 and run.steps[2].blocked == 1


def test_duplicates_inside_one_step_count_toward_the_limit():
    CALLS.clear()
    dup = tool_calls(("add", {"a": 1, "b": 1}), ("add", {"a": 1, "b": 1}))
    with fake_llm(seq(dup, "x")):
        run = run_agent("x", REG, provider="anthropic", repeat_limit=1)
    assert CALLS == ["add(1,1)"] and run.steps[0].blocked == 1
    assert "same step" in run.steps[0].results[1].content


def test_two_fully_blocked_steps_in_a_row_end_the_run_as_stuck():
    same = tool_calls(("add", {"a": 1, "b": 1}))
    with fake_llm(seq(same)) as f:  # repeats forever
        run = run_agent("x", REG, provider="anthropic", repeat_limit=2, max_steps=20)
    assert run.status == "stuck" and len(run.steps) == 4 and len(f.calls) == 4
    assert run.answer == ""


def test_a_single_blocked_step_followed_by_progress_is_not_stuck():
    same = tool_calls(("add", {"a": 1, "b": 1}))
    other = tool_calls(("add", {"a": 9, "b": 9}))
    with fake_llm(seq(same, same, same, other, same, "fine")):
        run = run_agent("x", REG, provider="anthropic", repeat_limit=2, max_steps=10)
    assert run.status == "done" and sum(s.blocked for s in run.steps) == 2


def test_cost_and_token_budgets_stop_the_run_after_the_step_that_crossed_them():
    same = lambda i: tool_calls(("add", {"a": i, "b": 1}))  # noqa: E731
    with fake_llm(seq(*[same(i) for i in range(20)])):
        run = run_agent("x", REG, provider="anthropic", max_cost_usd=1e-12, max_steps=10)
    assert run.status == "budget" and len(run.steps) == 1 and "cost" in run.error
    with fake_llm(seq(*[same(i) for i in range(20)])):
        run = run_agent("x", REG, provider="anthropic", max_total_tokens=1, max_steps=10)
    assert run.status == "budget" and len(run.steps) == 1 and "tokens" in run.error
    with fake_llm(seq(*[same(i) for i in range(3)], "end")):
        run = run_agent("x", REG, provider="anthropic", max_cost_usd=100, max_total_tokens=10**9)
    assert run.status == "done", "generous budgets must not trip"


def test_provider_errors_are_reported_not_raised():
    with fake_llm(seq(tool_calls(("add", {"a": 1, "b": 1})), RuntimeError("503 overloaded"))):
        run = run_agent("x", REG, provider="anthropic")
    assert (
        run.status == "error"
        and "RuntimeError: 503 overloaded" in run.error
        and len(run.steps) == 1
    )
    assert not run.ok and run.answer == ""


def test_truncated_final_answer_is_flagged():
    from common import chat

    with fake_llm(seq("a partial ans")):
        real = chat.turn  # the fake, installed by fake_llm

        def cut(*a, **k):
            t = real(*a, **k)
            t.stop_reason = "max_tokens"
            return t

        chat.turn = cut
        run = run_agent("x", REG, provider="anthropic")
    assert run.status == "truncated" and run.answer == "a partial ans"


def test_context_hook_runs_before_every_model_call_and_its_output_is_used():
    seen: list[int] = []

    def hook(messages, run):
        seen.append(len(messages))
        return (
            [m for m in messages if m["role"] != "tool"] + [{"role": "user", "content": "HOOKED"}]
            if len(seen) > 1
            else messages
        )

    with fake_llm(seq(tool_calls(("add", {"a": 1, "b": 1})), "ok")) as f:
        run = run_agent("x", REG, provider="anthropic", context_hook=hook)
    assert seen == [1, 3], "the hook sees the full history each time"
    assert "HOOKED" in f.calls[1].prompt and run.ok


def test_on_step_sees_every_step_including_the_final_one_and_prior_history_is_kept():
    steps = []
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "noted"}]
    with fake_llm(seq(tool_calls(("add", {"a": 1, "b": 1})), "fin")) as f:
        run = run_agent("now", REG, provider="anthropic", messages=history, on_step=steps.append)
    assert (
        [s.index for s in steps] == [1, 2]
        and steps[0].results[0].content == "2"
        and steps[1].text == "fin"
    )
    assert "earlier" in f.calls[0].prompt and run.messages[2]["content"] == "now"
    assert history == [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "noted"},
    ], "input not mutated"


def test_trace_is_readable_and_clips_long_values():
    with fake_llm(
        seq(tool_calls(("add", {"a": 1, "b": 1}), ("boom", {}), text="Adding."), "x" * 500)
    ):
        run = run_agent("A task", REG, provider="anthropic")
    tr = run.trace(width=60)
    assert tr.startswith("TASK: A task") and "say:  Adding." in tr and "call: add(" in tr
    assert "-> 2" in tr and "ERR Tool boom failed" in tr and "STATUS: done" in tr
    assert max(len(line) for line in tr.splitlines() if line.startswith("ANSWER")) <= 60 + len(
        "ANSWER: "
    )
    assert isinstance(AgentRun("t").trace(), str)


# ------------------------------------------------------------------ the last-step switch on the real wire formats


@pytest.fixture()
def server(monkeypatch):
    with FakeLLMServer() as srv:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", srv.url)
        monkeypatch.setenv("OPENAI_BASE_URL", srv.url + "/v1")
        monkeypatch.setenv("OLLAMA_BASE_URL", srv.url + "/v1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "t")
        monkeypatch.setenv("OPENAI_API_KEY", "t")
        monkeypatch.setenv("LLM_MAX_RETRIES", "0")
        for fn in (llm._anthropic, llm._openai, llm._ollama):
            fn.cache_clear()
        yield srv
        for fn in (llm._anthropic, llm._openai, llm._ollama):
            fn.cache_clear()


@pytest.mark.parametrize("provider", ["anthropic", "openai", "ollama"])
def test_final_step_sends_tool_choice_none_in_each_providers_shape(server, provider):
    server.script = [
        {"text": "", "tool_calls": [{"name": "add", "args": {"a": 1, "b": 2}}]},
        {"text": "3"},
    ]
    run = run_agent("1+2?", REG, provider=provider, max_steps=2)
    assert run.status == "done" and run.answer == "3"
    first, last = server.requests[0]["body"], server.requests[-1]["body"]
    assert "tool_choice" not in first
    expected = {"type": "none"} if provider == "anthropic" else "none"
    assert last["tool_choice"] == expected
    assert LAST_STEP_NOTE.strip()[:20] in str(
        last.get("system") or last.get("instructions") or last["messages"][0]
    )


def test_a_step_with_one_repeat_and_one_new_call_is_progress_not_stuck():
    mixed = lambda i: tool_calls(("add", {"a": 1, "b": 1}), ("add", {"a": 100 + i, "b": 1}))  # noqa: E731
    with fake_llm(seq(*[mixed(i) for i in range(5)], "done")):
        run = run_agent("x", REG, provider="anthropic", repeat_limit=1, max_steps=10)
    assert run.status == "done" and [s.blocked for s in run.steps[:5]] == [0, 1, 1, 1, 1]


def test_budget_limits_trip_exactly_when_crossed_not_later():
    def turns(**kw):
        steps = [tool_calls(("add", {"a": i, "b": 1})) for i in range(6)]
        with fake_llm(seq(*steps)):
            return run_agent("x", REG, provider="anthropic", max_steps=5, **kw)

    probe = turns(max_total_tokens=0)  # trips after the first step: tells us what one step costs
    s1 = probe.steps[0]
    one_step_tokens = s1.usage.input_tokens + s1.usage.output_tokens
    assert turns(max_cost_usd=s1.cost_usd * 0.9).status == "budget", "just under one step's cost"
    assert len(turns(max_cost_usd=s1.cost_usd * 1.01).steps) > 1, (
        "just over one step's cost: keeps going"
    )
    assert len(turns(max_total_tokens=one_step_tokens - 1).steps) == 1
    assert len(turns(max_total_tokens=one_step_tokens + 1).steps) > 1
