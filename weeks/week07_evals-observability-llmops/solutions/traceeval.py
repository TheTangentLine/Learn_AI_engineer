"""Checks that read ONLY a trace (a list of ``SpanRecord``): the monitoring half of evaluation.

A test-time eval knows the right answer; production does not. A trace still tells you a lot without it: a refund
requested before any lookup, the same tool called over and over, a specialist that answered without using one tool,
a guard that had to rewrite the reply, an escalation, a slow or failed model call. These checks run on live traffic
(sampled or all of it), on saved JSONL spans, and on the eval run itself, where we can measure how much of the full
harness's verdict a trace alone recovers.

    violations = check_trace(spans)           # [Violation("refund_before_lookup", "..."), ...]
    codes = {v.code for v in violations}
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common import tracing  # noqa: E402
from common.tracing import SpanRecord  # noqa: E402

# DEFECTS say something went wrong. EVENTS are things worth COUNTING on a dashboard (an escalation is the right outcome
# for "let me speak to a person"; a guard that fires means the guard worked): they are not failures by themselves.
DEFECTS = (
    "refund_before_lookup",
    "refund_without_lookup",
    "duplicate_tool_call",
    "tool_loop",
    "specialist_used_no_tool",
    "tool_error",
    "model_error",
    "agent_failed",
    "slow_trace",
    "many_steps",
)
EVENTS = ("guard_fired", "escalated")
CODES = DEFECTS + EVENTS


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str = ""


def tool_spans(spans: Sequence[SpanRecord]) -> list[SpanRecord]:
    return sorted(
        (s for s in spans if s.get(tracing.OPERATION) == "execute_tool"), key=lambda s: s.start_ns
    )


def check_trace(
    spans: Sequence[SpanRecord],
    *,
    loop_threshold: int = 3,
    slow_ms: float | None = None,
    step_limit: int = 5,
) -> list[Violation]:
    """Everything a single trace (one customer turn, or one whole case) says about itself."""
    out: list[Violation] = []
    tools = tool_spans(spans)
    names = [t.get(tracing.TOOL_NAME) for t in tools]

    if "request_refund" in names:
        first_refund = names.index("request_refund")
        if "lookup_invoice" not in names:
            out.append(
                Violation(
                    "refund_without_lookup",
                    "request_refund was called but the invoice was never looked up",
                )
            )
        elif names.index("lookup_invoice") > first_refund:
            out.append(
                Violation("refund_before_lookup", "request_refund started before lookup_invoice")
            )

    for name, n in Counter(names).items():
        if n >= loop_threshold:
            out.append(Violation("tool_loop", f"{name} called {n} times"))
    same = Counter((t.get(tracing.TOOL_NAME), t.get(tracing.TOOL_ARGS_DIGEST)) for t in tools)
    for (name, digest), n in same.items():
        if n >= 2 and digest is not None and Counter(names)[name] < loop_threshold:
            out.append(
                Violation(
                    "duplicate_tool_call", f"{name} called {n} times with identical arguments"
                )
            )

    agents: dict[str, list[bool]] = {}  # agent name -> per invocation: did it use any tool?
    for s in spans:
        op = s.get(tracing.OPERATION)
        if op == "invoke_agent":
            name = s.get(tracing.AGENT_NAME)
            agents.setdefault(name, []).append(any(_verified_for(s, t, spans) for t in tools))
            if s.get(tracing.AGENT_STATUS) != "done":
                out.append(Violation("agent_failed", f"{name}: {s.get(tracing.AGENT_STATUS)}"))
            if (s.get(tracing.STEPS) or 0) >= step_limit:
                out.append(Violation("many_steps", f"{name} took {s.get(tracing.STEPS)} steps"))
        elif op == "execute_tool" and s.is_error:
            out.append(Violation("tool_error", f"{s.get(tracing.TOOL_NAME)} returned an error"))
        elif op == "chat" and s.is_error:
            out.append(Violation("model_error", s.get(tracing.ERROR_TYPE, "")))
        elif op == "invoke_workflow":
            if s.get("app.reply.violations"):
                out.append(Violation("guard_fired", ", ".join(s.get("app.reply.violations"))))
            if s.get("app.reply.status") == "escalated":
                out.append(Violation("escalated", s.get("app.reply.agent", "")))

    for (
        name,
        used,
    ) in agents.items():  # across the whole trace: a later "thanks" turn legitimately uses no tool
        if not any(used):
            out.append(
                Violation("specialist_used_no_tool", f"{name} answered without any tool call")
            )

    if slow_ms is not None:
        total = tracing.summarize(spans)["duration_ms"]
        if total >= slow_ms:
            out.append(Violation("slow_trace", f"{total:.0f} ms"))
    return out


def _verified_for(agent: SpanRecord, tool: SpanRecord, spans: Sequence[SpanRecord]) -> bool:
    """A tool call counts for an agent if it ran inside the agent's span, or the SYSTEM ran it for the agent just before
    (the prefetch pattern: same parent, started earlier)."""
    return _inside(tool, agent, spans) or (
        tool.parent_id == agent.parent_id and tool.start_ns <= agent.start_ns
    )


def _inside(child: SpanRecord, ancestor: SpanRecord, spans: Sequence[SpanRecord]) -> bool:
    by_id = {s.span_id: s for s in spans}
    p = by_id.get(child.parent_id) if child.parent_id else None
    while p is not None:
        if p.span_id == ancestor.span_id:
            return True
        p = by_id.get(p.parent_id) if p.parent_id else None
    return False


def split_traces(spans: Sequence[SpanRecord]) -> dict[str, list[SpanRecord]]:
    out: dict[str, list[SpanRecord]] = {}
    for s in spans:
        out.setdefault(s.trace_id, []).append(s)
    return out


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100); the definition is deliberate and tested: p95 of 20 values is the 19th."""
    if not values:
        return 0.0
    xs = sorted(values)
    rank = max(1, -(-len(xs) * q // 100))  # ceil
    return xs[int(rank) - 1]
