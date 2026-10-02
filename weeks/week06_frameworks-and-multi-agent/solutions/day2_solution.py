"""Week 6 Day 2 - Solution: the Week 2 ticket pipeline as a LangGraph graph, with durable checkpoints.

The same five steps as Week 2 Day 4 (route, gate, draft, review, revise-until-it-passes), but now:
  * the control flow is a GRAPH you can draw, inspect and test node by node
  * every step is CHECKPOINTED to SQLite, so a crash mid-run resumes at the failed node instead of starting over
  * the state of any ticket can be read later (even from another process) and its full history replayed
  * a second graph shows the map-reduce pattern (``Send``): summarise N threads in parallel, merge in order

The model-calling functions are Week 2's (``route``, ``draft``, ``review``), reused unchanged: the graph owns the
control flow, not the prompts.

  uv run python weeks/week06_frameworks-and-multi-agent/solutions/day2_solution.py --offline
"""

from __future__ import annotations

import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from _weeks import load  # noqa: E402

w2 = load("week02_prompting-and-structured-outputs", "day4_solution")

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Send  # noqa: E402

MAX_REVISIONS = 2


class TicketState(TypedDict, total=False):
    ticket: str
    category: str
    confidence: float
    reply: str
    problems: list[str]
    passed: bool
    revisions: int
    escalated: bool
    reason: str
    trace: Annotated[list[str], operator.add]  # a reducer: each node APPENDS its name


# ----------------------------------------------------------------------------- nodes (each returns a partial update)


def route_node(state: TicketState) -> dict:
    r = w2.route(state["ticket"])
    return {
        "category": r.category,
        "confidence": r.confidence,
        "reason": r.reason,
        "trace": ["route"],
    }


def draft_node(state: TicketState) -> dict:
    return {
        "reply": w2.draft(state["ticket"], state["category"]),
        "revisions": 0,
        "trace": ["draft"],
    }


def review_node(state: TicketState) -> dict:
    v = w2.review(state["ticket"], state["reply"])
    return {"passed": v.passes, "problems": v.problems, "trace": ["review"]}


def revise_node(state: TicketState) -> dict:
    reply = w2.draft(state["ticket"], state["category"], state["problems"])
    return {"reply": reply, "revisions": state["revisions"] + 1, "trace": ["revise"]}


def escalate_node(state: TicketState) -> dict:
    if "reply" in state and not state.get("passed"):
        why = f"reply failed review after {state.get('revisions', 0)} revisions"
    else:
        why = f"router: {state['category']} @ {state['confidence']:.2f} ({state.get('reason', '')})"
    return {"escalated": True, "reason": why, "trace": ["escalate"]}


# ----------------------------------------------------------------------------- edges (pure functions of the state)


def after_route(state: TicketState) -> str:
    """The GATE: a cheap programmatic check; do not spend tokens drafting for a doubtful route."""
    return (
        "escalate"
        if state["category"] == "other" or state["confidence"] < w2.CONFIDENCE_FLOOR
        else "draft"
    )


def after_review(state: TicketState) -> str:
    if state["passed"]:
        return "accept"
    return "escalate" if state["revisions"] >= MAX_REVISIONS else "revise"


def build_ticket_graph(checkpointer=None, *, interrupt_before: list[str] | None = None):
    g = StateGraph(TicketState)
    for name, fn in (
        ("route", route_node),
        ("draft", draft_node),
        ("review", review_node),
        ("revise", revise_node),
        ("escalate", escalate_node),
    ):
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, {"draft": "draft", "escalate": "escalate"})
    g.add_edge("draft", "review")
    g.add_conditional_edges(
        "review", after_review, {"accept": END, "revise": "revise", "escalate": "escalate"}
    )
    g.add_edge("revise", "review")
    g.add_edge("escalate", END)
    return g.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def handle_ticket(app, ticket: str, thread_id: str) -> dict:
    """Run (or resume) one ticket. The thread_id is the durable identity of the run."""
    cfg = config_for(thread_id)
    existing = app.get_state(cfg)
    if existing.values and existing.next:  # a previous run stopped part-way: continue it
        return app.invoke(None, cfg)
    if existing.values:  # finished earlier: handling is idempotent per thread_id (re-running would APPEND to the old state)
        return existing.values
    return app.invoke({"ticket": ticket, "trace": []}, cfg)


def history(app, thread_id: str) -> list[tuple[str, ...]]:
    """(next nodes, trace so far) for every checkpoint of a thread, oldest first."""
    snaps = list(app.get_state_history(config_for(thread_id)))
    return [(tuple(s.next), tuple(s.values.get("trace", []))) for s in reversed(snaps)]


# ----------------------------------------------------------------------------- map-reduce with Send


class SummaryState(TypedDict, total=False):
    threads: list[str]
    parts: Annotated[list[tuple[int, str]], operator.add]
    summaries: list[str]


def build_summary_graph():
    from common import llm

    def fan_out(state: SummaryState):
        return [
            Send("summarise_one", {"i": i, "thread": t}) for i, t in enumerate(state["threads"])
        ]

    def summarise_one(payload: dict) -> dict:  # runs once per Send, in parallel
        r = llm.complete(
            f"<thread>\n{payload['thread']}\n</thread>\n\nSummarise this support thread in one sentence for a manager.",
            max_tokens=150,
        )
        return {"parts": [(payload["i"], r.text.strip())]}

    def merge(
        state: SummaryState,
    ) -> dict:  # parts arrive in completion order; restore the input order
        return {"summaries": [text for _, text in sorted(state["parts"])]}

    g = StateGraph(SummaryState)
    g.add_node("summarise_one", summarise_one)
    g.add_node("merge", merge)
    g.add_conditional_edges(START, fan_out, ["summarise_one"])
    g.add_edge("summarise_one", "merge")
    g.add_edge("merge", END)
    return g.compile()


# ----------------------------------------------------------------------------- demo


def main() -> None:
    import tempfile

    from common.fake import fake_llm

    offline = "--offline" in sys.argv
    if not offline:
        sys.exit(
            "This demo runs with the scripted Week 2 model; use --offline (see the tests for the crash/resume scenarios)."
        )
    with fake_llm(w2.offline_responder_rules()), tempfile.TemporaryDirectory() as tmp:
        with SqliteSaver.from_conn_string(str(Path(tmp) / "tickets.sqlite")) as saver:
            app = build_ticket_graph(saver)
            for i, t in enumerate(w2.TICKETS):
                out = handle_ticket(app, t, f"ticket-{i}")
                print(
                    f"{out['category']:<9} escalated={out.get('escalated', False)!s:<5} {' > '.join(out['trace'])}"
                )
            print("\ncheckpoints for ticket-0:")
            for nxt, trace in history(app, "ticket-0"):
                print(f"  next={nxt!s:<14} trace={list(trace)}")
            print("\n" + app.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
