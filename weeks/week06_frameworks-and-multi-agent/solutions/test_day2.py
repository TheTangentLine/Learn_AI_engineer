"""Tests for Week 6 Day 2: the graph matches the Week 2 pipeline, survives crashes, persists, forks, and fans out."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import day2_solution as d2  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from common.fake import fake_llm  # noqa: E402

w2 = d2.w2


@pytest.fixture()
def model():
    with fake_llm(w2.offline_responder_rules()) as f:
        yield f


def trace_names(result):
    return [s.name for s in result.trace]


# ----------------------------------------------------------------------------- parity with Week 2


@pytest.mark.parametrize("ticket", w2.TICKETS)
def test_the_graph_makes_the_same_decisions_as_the_week_2_function(model, ticket):
    expected = w2.handle_ticket(ticket)
    app = d2.build_ticket_graph(MemorySaver())
    got = d2.handle_ticket(app, ticket, "t")
    assert got.get("category") == expected.category
    assert got.get("escalated", False) == expected.escalated
    assert got.get("revisions", 0) == expected.revisions
    assert got.get("reply") == expected.reply if not expected.escalated else "reply" not in got
    assert got["trace"] == [("escalate" if False else n) for n in trace_names(expected)] + (
        ["escalate"] if expected.escalated else []
    )


def test_without_a_checkpointer_the_graph_still_runs(model):
    app = d2.build_ticket_graph()
    out = app.invoke({"ticket": w2.TICKETS[1], "trace": []})
    assert out["trace"] == ["route", "draft", "review"] and not out.get("escalated")


# ----------------------------------------------------------------------------- the edges are pure and tested at their boundaries


def test_gate_escalates_other_and_low_confidence_and_passes_the_boundary():
    floor = w2.CONFIDENCE_FLOOR
    assert d2.after_route({"category": "other", "confidence": 0.99}) == "escalate"
    assert d2.after_route({"category": "billing", "confidence": floor - 0.01}) == "escalate"
    assert d2.after_route({"category": "billing", "confidence": floor}) == "draft", (
        "exactly at the floor is trusted"
    )
    assert d2.after_route({"category": "technical", "confidence": 0.95}) == "draft"


def test_review_edge_accepts_revises_or_escalates_at_the_cap():
    assert d2.after_review({"passed": True, "revisions": 0}) == "accept"
    assert d2.after_review({"passed": False, "revisions": 0}) == "revise"
    assert d2.after_review({"passed": False, "revisions": d2.MAX_REVISIONS - 1}) == "revise"
    assert d2.after_review({"passed": False, "revisions": d2.MAX_REVISIONS}) == "escalate"
    assert d2.after_review({"passed": True, "revisions": 99}) == "accept", (
        "a pass wins over the cap"
    )


def test_a_reviewer_that_never_passes_escalates_after_exactly_the_revision_cap(model, monkeypatch):
    monkeypatch.setattr(
        w2, "review", lambda t, r: w2.Review(passes=False, problems=["still wrong"])
    )
    app = d2.build_ticket_graph(MemorySaver())
    out = d2.handle_ticket(app, w2.TICKETS[0], "t")
    assert out["trace"] == [
        "route",
        "draft",
        "review",
        "revise",
        "review",
        "revise",
        "review",
        "escalate",
    ]
    assert (
        out["escalated"]
        and out["revisions"] == d2.MAX_REVISIONS
        and "failed review after 2 revisions" in out["reason"]
    )


def test_the_gate_stops_the_run_before_any_drafting_tokens_are_spent(model):
    app = d2.build_ticket_graph(MemorySaver())
    out = d2.handle_ticket(app, w2.TICKETS[3], "t")
    assert (
        out["trace"] == ["route", "escalate"]
        and "reply" not in out
        and out["reason"].startswith("router: other")
    )
    assert not model.calls_matching("support specialist"), "no specialist prompt was ever sent"


# ----------------------------------------------------------------------------- durable execution


class Outage(RuntimeError):
    pass


def crash_once_in_review(monkeypatch):
    real = w2.review
    state = {"crashed": False}

    def flaky(ticket, reply):
        if not state["crashed"]:
            state["crashed"] = True
            raise Outage("provider outage")
        return real(ticket, reply)

    monkeypatch.setattr(w2, "review", flaky)


def test_a_crash_mid_run_resumes_at_the_failed_node_without_repeating_earlier_model_calls(
    model, monkeypatch, tmp_path
):
    crash_once_in_review(monkeypatch)
    with SqliteSaver.from_conn_string(str(tmp_path / "t.sqlite")) as saver:
        app = d2.build_ticket_graph(saver)
        with pytest.raises(Outage):
            d2.handle_ticket(app, w2.TICKETS[1], "t1")
        snap = app.get_state(d2.config_for("t1"))
        assert (
            snap.next == ("review",)
            and snap.values["trace"] == ["route", "draft"]
            and snap.values["reply"]
        )
        routes_before = len(model.calls_matching("You classify customer support tickets"))
        drafts_before = len(model.calls_matching("technical support specialist"))
        out = d2.handle_ticket(app, w2.TICKETS[1], "t1")  # same thread id: RESUME
    assert out["trace"] == ["route", "draft", "review"] and not out.get("escalated")
    assert (
        len(model.calls_matching("You classify customer support tickets")) == routes_before == 1
    ), "route was not repeated"
    assert len(model.calls_matching("technical support specialist")) == drafts_before == 1, (
        "the draft was not regenerated"
    )


def test_state_survives_a_process_restart(model, monkeypatch, tmp_path):
    db = str(tmp_path / "durable.sqlite")
    crash_once_in_review(monkeypatch)
    with SqliteSaver.from_conn_string(db) as saver:
        app = d2.build_ticket_graph(saver)
        with pytest.raises(Outage):
            d2.handle_ticket(app, w2.TICKETS[0], "billing-1")
    reader = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        import day2_solution as d2
        from langgraph.checkpoint.sqlite import SqliteSaver
        with SqliteSaver.from_conn_string({db!r}) as saver:
            snap = d2.build_ticket_graph(saver).get_state(d2.config_for("billing-1"))
            print(json.dumps({{"next": list(snap.next), "trace": snap.values["trace"], "category": snap.values["category"]}}))
        """
    )
    r = subprocess.run([sys.executable, "-c", reader], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == {
        "next": ["review"],
        "trace": ["route", "draft"],
        "category": "billing",
    }


def test_threads_are_isolated_and_a_finished_thread_is_idempotent(model):
    app = d2.build_ticket_graph(MemorySaver())
    a = d2.handle_ticket(app, w2.TICKETS[0], "a")
    b = d2.handle_ticket(app, w2.TICKETS[1], "b")
    assert (
        a["category"] == "billing"
        and b["category"] == "technical"
        and "billing" not in str(b["trace"])
    )
    calls = len(model.calls)
    again = d2.handle_ticket(app, "a completely different ticket text", "a")
    assert again == a and len(model.calls) == calls, (
        "a finished thread returns its stored result; the model is not called"
    )
    assert app.get_state(d2.config_for("never-used")).values == {}


def test_history_lists_every_checkpoint_in_order(model):
    app = d2.build_ticket_graph(MemorySaver())
    d2.handle_ticket(app, w2.TICKETS[0], "h")
    h = d2.history(app, "h")
    nexts = [n for n, _ in h]
    assert nexts[0] == ("__start__",) and nexts[-1] == () and ("revise",) in nexts
    assert [len(t) for _, t in h] == sorted(len(t) for _, t in h), "the trace only grows"
    assert h[-1][1] == ("route", "draft", "review", "revise", "review")


def test_time_travel_fork_from_a_past_checkpoint_with_an_edited_reply(model):
    app = d2.build_ticket_graph(MemorySaver())
    d2.handle_ticket(app, w2.TICKETS[0], "fork")
    cfg = d2.config_for("fork")
    before_review = next(
        s
        for s in app.get_state_history(cfg)
        if s.next == ("review",) and s.values["trace"] == ["route", "draft"]
    )
    assert "guarantee" in before_review.values["reply"]
    fork = app.update_state(
        before_review.config,
        {
            "reply": "Our billing team will review the duplicate charge; please send your invoice number."
        },
    )
    out = app.invoke(None, fork)
    assert (
        out["trace"] == ["route", "draft", "review"]
        and out["passed"]
        and "billing team" in out["reply"]
    )
    # forking never deletes the original branch: its checkpoints are still in the thread's history
    originals = [
        snap
        for snap in app.get_state_history(cfg)
        if snap.values.get("trace") == ["route", "draft", "review", "revise", "review"]
    ]
    assert (
        originals
        and originals[0].values["revisions"] == 1
        and "guarantee" not in originals[0].values["reply"]
    )
    assert app.get_state(cfg).values["trace"] == ["route", "draft", "review"], (
        "but the thread's latest checkpoint is now the fork's"
    )


# ----------------------------------------------------------------------------- map-reduce with Send


def test_send_fans_out_in_parallel_and_the_merge_restores_input_order(model):
    app = d2.build_summary_graph()
    threads = [f"thread number {i}: customer asks about order {i}" for i in range(6)]
    out = app.invoke({"threads": threads})
    assert out["summaries"] == [f"SUMMARY: {t[:25].strip()}" for t in threads]
    assert len(model.calls_matching("Summarise this support thread")) == 6


def test_send_with_no_threads_does_not_hang(model):
    out = d2.build_summary_graph().invoke({"threads": []})
    assert out.get("summaries", []) == []


def test_the_graph_can_be_drawn(model):
    mermaid = d2.build_ticket_graph().get_graph().draw_mermaid()
    for node in ("route", "draft", "review", "revise", "escalate"):
        assert node in mermaid
    assert "review -.-> revise;" in mermaid and "revise --> review;" in mermaid


def test_results_come_back_in_input_order_even_when_workers_finish_in_reverse_order():
    """Finding: LangGraph applies a reducer's updates in TASK order, not completion order, so ``parts`` is already
    ordered here. The ``sorted()`` in merge() is a guard that does not depend on that internal detail (so deleting it
    is an *equivalent* mutation under today's LangGraph)."""
    import re
    import time

    def slow_for_early_threads(prompt):
        n = int(re.search(r"thread number (\d+)", prompt).group(1))
        time.sleep(0.05 * (5 - n))  # thread 0 finishes LAST
        return f"S{n}"

    with fake_llm([(r"Summarise this support thread", slow_for_early_threads)]):
        out = d2.build_summary_graph().invoke({"threads": [f"thread number {i}" for i in range(6)]})
    assert out["summaries"] == [f"S{i}" for i in range(6)]
    assert [i for i, _ in out["parts"]] == list(range(6)), (
        "reducer updates are applied in task order"
    )
