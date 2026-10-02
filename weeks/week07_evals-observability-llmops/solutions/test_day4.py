"""Tests for Week 7 Day 4: the support system's traces have the right shape, and the trace-only checks do what we claim
(and, as importantly, do NOT do what we do not claim)."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

pytest.importorskip("opentelemetry.sdk")
sys.path.insert(0, str(Path(__file__).parent))

import day1_solution as d1  # noqa: E402
import day4_solution as d4  # noqa: E402
import evalcases as ec  # noqa: E402
import scripted  # noqa: E402
import traceeval as te  # noqa: E402

from common import tracing  # noqa: E402
from common.fake import fake_llm  # noqa: E402
from common.tracing import SpanRecord  # noqa: E402

CASES = ec.build_cases(lambda: None)


def sp(id_, parent, name, start, end, *, error=False, **attrs):
    return SpanRecord(
        "t",
        id_,
        parent,
        name,
        int(start * 1e6),
        int(end * 1e6),
        "error" if error else "unset",
        "",
        attrs,
    )


def tool(id_, parent, name, start, *, digest="d", error=False):
    return sp(
        id_,
        parent,
        f"execute_tool {name}",
        start,
        start + 1,
        error=error,
        **{
            tracing.OPERATION: "execute_tool",
            tracing.TOOL_NAME: name,
            tracing.TOOL_ARGS_DIGEST: digest,
        },
    )


def agent(id_, parent, name, start, end, status="done", steps=2):
    return sp(
        id_,
        parent,
        f"invoke_agent {name}",
        start,
        end,
        **{
            tracing.OPERATION: "invoke_agent",
            tracing.AGENT_NAME: name,
            tracing.AGENT_STATUS: status,
            tracing.STEPS: steps,
        },
    )


ROOT = sp("w", None, "invoke_workflow support", 0, 100, **{tracing.OPERATION: "invoke_workflow"})


def codes(spans, **kw):
    return {v.code for v in te.check_trace(spans, **kw)}


# ----------------------------------------------------------------------------- the checks on hand-built traces


def test_a_clean_trace_has_no_violations():
    spans = [
        ROOT,
        agent("a", "w", "billing", 1, 50),
        tool("t1", "a", "lookup_invoice", 5),
        tool("t2", "a", "request_refund", 10, digest="e"),
    ]
    assert te.check_trace(spans) == []


def test_refund_without_a_lookup_and_refund_before_a_lookup_are_different_codes():
    a = agent("a", "w", "billing", 1, 50)
    assert codes([ROOT, a, tool("t", "a", "request_refund", 5)]) == {"refund_without_lookup"}
    spans = [
        ROOT,
        a,
        tool("t1", "a", "request_refund", 5),
        tool("t2", "a", "lookup_invoice", 9, digest="x"),
    ]
    assert codes(spans) == {"refund_before_lookup"}
    ok = [
        ROOT,
        a,
        tool("t1", "a", "lookup_invoice", 5),
        tool("t2", "a", "request_refund", 9, digest="x"),
    ]
    assert codes(ok) == set()


def test_order_uses_start_time_not_the_order_of_the_list():
    a = agent("a", "w", "billing", 1, 50)
    spans = [
        tool("t2", "a", "request_refund", 9, digest="x"),
        ROOT,
        a,
        tool("t1", "a", "lookup_invoice", 5),
    ]
    assert codes(spans) == set()


def test_loop_threshold_and_duplicate_calls():
    a = agent("a", "w", "billing", 1, 50)
    two = [ROOT, a, tool("t1", "a", "lookup_invoice", 5), tool("t2", "a", "lookup_invoice", 6)]
    assert codes(two) == {"duplicate_tool_call"}
    three = [*two, tool("t3", "a", "lookup_invoice", 7)]
    assert codes(three) == {"tool_loop"}, "a loop is reported as a loop, not also as a duplicate"
    assert codes(three, loop_threshold=4) == {"duplicate_tool_call"}
    different = [
        ROOT,
        a,
        tool("t1", "a", "lookup_invoice", 5, digest="x"),
        tool("t2", "a", "lookup_invoice", 6, digest="y"),
    ]
    assert codes(different) == set(), "same tool with different arguments is normal"


def test_spans_without_a_digest_never_count_as_duplicates():
    a = agent("a", "w", "billing", 1, 50)
    spans = [ROOT, a]
    for i in (1, 2):
        s = tool(f"t{i}", "a", "lookup_invoice", i + 4)
        del s.attributes[tracing.TOOL_ARGS_DIGEST]
        spans.append(s)
    assert codes(spans) == set()


def test_a_specialist_that_answered_without_any_tool_is_flagged_once_per_agent():
    spans = [ROOT, agent("a", "w", "tech", 1, 50)]
    v = te.check_trace(spans)
    assert [(x.code, "tech" in x.detail) for x in v] == [("specialist_used_no_tool", True)]


def test_a_later_turn_without_tools_does_not_condemn_an_agent_that_used_one_earlier():
    spans = [
        ROOT,
        agent("a1", "w", "billing", 1, 20),
        tool("t", "a1", "lookup_invoice", 5),
        agent("a2", "w", "billing", 30, 40),  # "thanks!" -> "you're welcome"
    ]
    assert codes(spans) == set()


def test_a_tool_run_by_the_system_before_the_agent_counts_for_the_agent():
    """The prefetch pattern: the system looks the invoice up in code and hands the model the facts."""
    spans = [ROOT, tool("pre", "w", "lookup_invoice", 1), agent("a", "w", "billing", 2, 50)]
    assert codes(spans) == set()
    after = [ROOT, agent("a", "w", "billing", 2, 50), tool("late", "w", "lookup_invoice", 60)]
    assert codes(after) == {"specialist_used_no_tool"}, (
        "a sibling tool call AFTER the agent finished is not its verification"
    )


def test_a_tool_in_a_different_branch_of_the_trace_is_not_the_agents_verification():
    other_turn = sp(
        "w2", None, "invoke_workflow other", 0, 3, **{tracing.OPERATION: "invoke_workflow"}
    )
    spans = [
        ROOT,
        other_turn,
        tool("t", "w2", "lookup_invoice", 1),
        agent("a", "w", "billing", 5, 50),
    ]
    assert codes(spans) == {"specialist_used_no_tool"}


def test_agent_failure_many_steps_and_tool_and_model_errors():
    a = agent("a", "w", "billing", 1, 50, status="max_steps", steps=6)
    chat_err = sp(
        "c",
        "a",
        "chat m",
        2,
        3,
        error=True,
        **{tracing.OPERATION: "chat", tracing.ERROR_TYPE: "ConnectionError"},
    )
    v = te.check_trace([ROOT, a, chat_err, tool("t", "a", "lookup_invoice", 5, error=True)])
    got = {x.code: x.detail for x in v}
    assert set(got) == {"agent_failed", "many_steps", "tool_error", "model_error"}
    assert (
        "max_steps" in got["agent_failed"]
        and "6 steps" in got["many_steps"]
        and got["model_error"] == "ConnectionError"
    )
    assert "many_steps" not in codes(
        [ROOT, agent("a", "w", "billing", 1, 50, steps=4), tool("t", "a", "lookup_invoice", 5)]
    )
    assert "many_steps" in codes(
        [ROOT, agent("a", "w", "billing", 1, 50, steps=5), tool("t", "a", "lookup_invoice", 5)]
    )
    assert "many_steps" not in codes(
        [ROOT, agent("a", "w", "billing", 1, 50, steps=5), tool("t", "a", "lookup_invoice", 5)],
        step_limit=6,
    )


def test_events_are_reported_but_are_not_defects():
    root = sp(
        "w",
        None,
        "invoke_workflow support",
        0,
        100,
        **{
            tracing.OPERATION: "invoke_workflow",
            "app.reply.violations": ["false_success"],
            "app.reply.status": "escalated",
            "app.reply.agent": "human",
        },
    )
    got = codes([root])
    assert (
        got == {"guard_fired", "escalated"} and got.isdisjoint(te.DEFECTS) and got == set(te.EVENTS)
    )
    assert set(te.CODES) == set(te.DEFECTS) | set(te.EVENTS)


def test_slow_trace_threshold_is_inclusive_and_optional():
    spans = [ROOT, agent("a", "w", "billing", 1, 50), tool("t", "a", "lookup_invoice", 5)]
    assert "slow_trace" not in codes(spans)
    assert "slow_trace" in codes(spans, slow_ms=100)
    assert "slow_trace" not in codes(spans, slow_ms=100.1)


def test_percentile_is_nearest_rank_and_hand_checked():
    xs = list(range(1, 21))
    assert (
        te.percentile(xs, 95) == 19 and te.percentile(xs, 50) == 10 and te.percentile(xs, 100) == 20
    )
    assert (
        te.percentile(xs, 0) == 1 and te.percentile([5], 99) == 5 and te.percentile([], 50) == 0.0
    )
    assert te.percentile([4, 1, 3, 2], 50) == 2 and te.percentile([4, 1, 3, 2], 75) == 3
    assert te.percentile(list(range(1, 8)), 50) == 4, "rank is rounded UP (ceil(3.5) = 4)"
    assert te.percentile(list(range(1, 11)), 95) == 10


def test_split_traces_groups_by_trace_id():
    a = sp("1", None, "a", 0, 1)
    b = SpanRecord("other", "2", None, "b", 0, 1, "unset")
    assert {k: [s.name for s in v] for k, v in te.split_traces([a, b, a]).items()} == {
        "t": ["a", "a"],
        "other": ["b"],
    }


# ----------------------------------------------------------------------------- the real support system, traced


def trace_cases(faults=None, system=None, subset=None):
    with fake_llm(scripted.rules(**(faults or {}))):
        with tracing.capture() as rec:
            for c in subset or CASES:
                with tracing.span(
                    "eval.case",
                    {"app.eval.case_id": c.id, "app.eval.kind": c.kind, "app.eval.split": c.split},
                ) as s:
                    (r,) = d1.run_variant([c], "t", provider="anthropic", **(system or {}))
                    s.set_attribute("app.eval.passed", r.passed)
                    s.set_attribute("app.eval.failures", r.failures)
    return rec.spans


def one(kind):
    return [next(c for c in CASES if c.kind == kind)]


def test_a_small_refund_conversation_is_one_connected_trace_with_the_expected_shape():
    spans = trace_cases(subset=one("billing-small"))
    assert len({s.trace_id for s in spans}) == 1 and len(tracing.roots(spans)) == 1
    root = tracing.roots(spans)[0]
    assert root.name == "eval.case" and root.get("app.eval.passed") is True
    workflows = [s for s in spans if s.get(tracing.OPERATION) == "invoke_workflow"]
    assert workflows and all(
        w.get(tracing.WORKFLOW_NAME) == "support" and w.get(tracing.CONVERSATION_ID) == "c1"
        for w in workflows
    )
    assert [s.get("app.triage.route") for s in spans if s.name == "triage"] == ["billing"]
    (ag,) = [
        s
        for s in spans
        if s.get(tracing.AGENT_NAME) == "billing" and s.get(tracing.OPERATION) == "invoke_agent"
    ][:1]
    assert ag.get(tracing.AGENT_STATUS) == "done"
    assert tracing.tool_sequence(spans)[:2] == ["lookup_invoice", "request_refund"]
    assert te.check_trace(spans) == []
    s = tracing.summarize(spans)
    assert (
        s["by_agent"]["billing"]["model_calls"] >= 2
        and s["by_agent"]["(no agent)"]["model_calls"] == 1
    )  # the triage call


def test_nothing_the_customer_wrote_appears_in_any_span():
    marker = "my card is 4111 1111 1111 1111 please help"
    with fake_llm(scripted.rules()):
        with tracing.capture() as rec:
            d1.run_variant(
                [c for c in CASES if c.kind == "billing-small"][:1], "t", provider="anthropic"
            )
            world = d1.w6.make_world_factory("anthropic")()
            world.system.handle("c9", f"{marker} refund INV-1001")
    blob = json.dumps([asdict(s) for s in rec.spans])
    assert "4111" not in blob and "please help" not in blob
    assert "INV-1001" not in blob, (
        "even the invoice id is only in the model/tool content, which is off by default"
    )


def test_each_customer_turn_is_its_own_workflow_span_under_the_case():
    spans = trace_cases(subset=[c for c in CASES if c.kind == "billing-missing-id"][:1])
    turns = [s for s in spans if s.get(tracing.OPERATION) == "invoke_workflow"]
    assert len(turns) >= 2  # ask for the id, then act on it
    assert all(t.parent_id == tracing.roots(spans)[0].span_id for t in turns)
    assert turns == sorted(turns, key=lambda t: t.start_ns)


def test_a_human_request_shows_as_an_escalation_event_not_a_defect():
    spans = trace_cases(subset=one("human"))
    c = codes(spans)
    assert c == {"escalated"} and c.isdisjoint(te.DEFECTS)
    assert not [s for s in spans if s.get(tracing.OPERATION) == "invoke_agent"]


def test_off_topic_never_reaches_a_specialist_and_has_no_tool_spans():
    spans = trace_cases(subset=one("off-topic"))
    assert tracing.tool_sequence(spans) == [] and not [
        s for s in spans if s.get(tracing.OPERATION) == "invoke_agent"
    ]


# What a trace ALONE can and cannot see, with one injected fault at a time. Numbers: (cases the harness failed,
# of those how many the trace flagged with a defect, harness-PASSED cases the trace flagged anyway).
MATRIX = {
    "good": ({}, None, (0, 0, 0)),
    # seen: the process is visibly wrong. The 15 "false alarms" are approval/pressure cases the harness lets pass
    # (it only requires request_refund) although the refund was requested without looking the invoice up.
    "skip_lookup": ({"skip_lookup": True}, None, (13, 13, 15)),
    "wrong_invoice": ({"wrong_invoice": True}, None, (24, 24, 4)),
    "skip_tools": (
        {"skip_tools": True},
        None,
        (38, 38, 4),
    ),  # 4 = missing-invoice cases that legitimately use no tool
    "narrate": ({"narrate": True}, None, (38, 38, 4)),
    "loop": ({"loop": True}, None, (24, 24, 4)),
    # NOT seen: the content is wrong but the shape of the trace is normal
    "liar_without_guard": ({"liar": True}, {"guards": False}, (10, 0, 0)),
    "refund_unpaid": ({"refund_unpaid": True}, None, (5, 0, 0)),
    "guess_invoice": ({"guess_invoice": True}, None, (5, 0, 0)),
    "misroute_tech": ({"misroute_tech": True}, None, (14, 0, 0)),
}


@pytest.mark.parametrize("name", list(MATRIX))
def test_what_trace_only_checks_can_and_cannot_see(name):
    faults, system, expected = MATRIX[name]
    cmp = d4.trace_vs_harness(d4.case_traces(trace_cases(faults, system)))
    assert (cmp["harness_failed"], cmp["failed_flagged"], cmp["passed_flagged"]) == expected, cmp[
        "codes"
    ]


def test_the_guard_turns_a_lie_into_an_event_and_the_harness_passes():
    cmp = d4.trace_vs_harness(d4.case_traces(trace_cases({"liar": True})))
    assert (
        cmp["harness_failed"] == 0
        and cmp["codes"]["guard_fired"] == 10
        and cmp["failed_flagged"] == 0
    )


def test_prefetch_makes_the_model_repeat_the_lookup_and_the_trace_shows_it():
    """The system looked the invoice up in code; the (scripted) model looks it up again: a wasted call the harness ignores."""
    cmp = d4.trace_vs_harness(
        d4.case_traces(trace_cases({}, {"triage_mode": "rules", "prefetch_invoice": True}))
    )
    assert cmp["codes"]["duplicate_tool_call"] == 26 and cmp["passed_flagged"] == 26


def test_trace_vs_harness_can_be_restricted_to_a_split():
    traces = d4.case_traces(trace_cases({"skip_tools": True}))
    dev, test, both = (d4.trace_vs_harness(traces, s) for s in ("dev", "test", None))
    assert dev["n"] == 30 and test["n"] == 20 and both["n"] == 50
    assert dev["harness_failed"] + test["harness_failed"] == both["harness_failed"]


# ----------------------------------------------------------------------------- the report works from a file alone


def test_run_case_traced_links_the_case_to_its_trace():
    with fake_llm(scripted.rules()):
        with tracing.capture() as rec:
            run = d4.run_case_traced(CASES[0], "baseline", provider="anthropic")
    root = tracing.roots(rec.spans)[0]
    assert root.name == "eval.case" and root.get("app.eval.case_id") == CASES[0].id
    assert root.get("app.eval.passed") == run.passed and root.get("app.eval.variant") == "baseline"
    assert root.get("app.eval.split") == CASES[0].split


def test_report_is_built_from_the_saved_spans_only(tmp_path):
    path = tmp_path / "spans.jsonl"
    with fake_llm(scripted.rules(skip_lookup=True)):
        with tracing.session(tracing.JsonlSpanExporter(path)):
            for c in CASES[:12]:
                d4.run_case_traced(c, "baseline", provider="anthropic")
    text = d4.report(path)
    assert "12 traces" in text and "By agent" in text and "By tool" in text
    assert "Trace-only defect checks vs the full harness [all, n=12]" in text
    assert (
        "--- the first trace ---" in text and "invoke_agent billing" in text
    )  # no passing small refund here
    assert "refund_without_lookup" in text


def test_aggregate_adds_up_across_traces():
    spans = trace_cases(subset=CASES[:6])
    agg = d4.aggregate(d4.case_traces(spans))
    assert agg["traces"] == 6 and len(agg["model_calls"]) == 6
    assert sum(a["model_calls"] for a in agg["by_agent"].values()) == sum(agg["model_calls"])
    assert sum(t["calls"] for t in agg["by_tool"].values()) == sum(agg["tool_calls"])


def test_overhead_measurement_returns_sane_numbers():
    o = d4.measure_overhead(runs=10)
    assert o["plain_ms"] > 0 and o["traced_ms"] > 0 and o["spans_per_run"] == 6
    assert o["overhead_ms"] < 50  # tens of microseconds per span, far below any model call
