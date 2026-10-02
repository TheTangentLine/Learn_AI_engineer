"""Week 7 Day 4 - Solution: full traces (model calls + tool calls) for the Week 6 support system.

1. RUN the 50 Day 1 cases on the real local model with tracing on; every case is one trace
   (an ``eval.case`` span containing each customer turn: triage, specialist loop, model calls, tool calls).
2. READ the traces: one example tree, totals by agent and by tool, where the time and tokens went.
3. CHECK the traces without the answer key (``traceeval.py``) and measure how much of the full harness's
   verdict a trace alone recovers.
4. MEASURE what tracing costs (overhead per agent run, scripted model so the model time is zero).

uv run python weeks/week07_evals-observability-llmops/solutions/day4_solution.py run
uv run python weeks/week07_evals-observability-llmops/solutions/day4_solution.py report
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import day1_solution as d1  # noqa: E402
import evalcases as ec  # noqa: E402
import traceeval as te  # noqa: E402

from common import tracing  # noqa: E402

OUT = ROOT / "outputs"
SPANS = OUT / "w7d4_spans.jsonl"
DEFECTS = set(te.DEFECTS)


def run_case_traced(case, variant: str, *, provider: str, model: str | None = None):
    """One case = one trace. The case id is an attribute, so an eval report can link to its trace and back."""
    attrs = {
        "app.eval.case_id": case.id,
        "app.eval.kind": case.kind,
        "app.eval.variant": variant,
        "app.eval.split": case.split,
    }
    with tracing.span("eval.case", attrs) as sp:
        (run,) = d1.run_variant(
            [case], variant, provider=provider, model=model, **d1.VARIANTS[variant]
        )
        sp.set_attribute("app.eval.passed", run.passed)
        sp.set_attribute("app.eval.failures", run.failures)
    return run


def run_all(variant: str = "rules+prefetch", spans_path: Path = SPANS) -> list:
    from common import llm
    from common.local_server import LocalOpenAIServer

    if spans_path.exists():
        spans_path.unlink()
    runs = []
    with LocalOpenAIServer() as srv:
        os.environ["OLLAMA_BASE_URL"] = srv.url
        llm._ollama.cache_clear()
        with tracing.session(tracing.JsonlSpanExporter(spans_path), service_name="support-eval"):
            for case in ec.build_cases(lambda: None):
                runs.append(run_case_traced(case, variant, provider="ollama", model="local-qwen"))
    return runs


# ----------------------------------------------------------------------------- reading the traces


def case_traces(spans):
    """trace id -> (case id, passed, spans), for traces rooted at an eval.case span."""
    out = {}
    for tid, group in te.split_traces(spans).items():
        root = next((s for s in group if s.name == "eval.case"), None)
        if root is not None:
            out[tid] = (root.get("app.eval.case_id"), bool(root.get("app.eval.passed")), group)
    return out


def aggregate(traces) -> dict:
    sums = [tracing.summarize(g) for _, _, g in traces.values()]
    by_agent: dict[str, dict] = {}
    by_tool: dict[str, dict] = {}
    for s in sums:
        for name, a in s["by_agent"].items():
            t = by_agent.setdefault(name, {"model_calls": 0, "input_tokens": 0, "output_tokens": 0})
            for k in t:
                t[k] += a[k]
        for name, a in s["by_tool"].items():
            t = by_tool.setdefault(name, {"calls": 0, "errors": 0, "ms": 0.0})
            for k in t:
                t[k] += a[k]
    return {
        "traces": len(sums),
        "model_calls": [s["model_calls"] for s in sums],
        "input_tokens": [s["input_tokens"] for s in sums],
        "output_tokens": [s["output_tokens"] for s in sums],
        "tool_calls": [s["tool_calls"] for s in sums],
        "duration_ms": [s["duration_ms"] for s in sums],
        "by_agent": by_agent,
        "by_tool": by_tool,
    }


def trace_vs_harness(traces, split: str | None = None) -> dict:
    """How much of the full harness's verdict does a trace alone recover? A case is "flagged" if the trace shows any
    DEFECT (events such as an escalation are counted separately: they are not failures)."""
    rows = []
    for case_id, passed, group in traces.values():
        root = next(s for s in group if s.name == "eval.case")
        if split and root.get("app.eval.split") != split:
            continue
        v = te.check_trace(group)
        codes = sorted({x.code for x in v})
        rows.append((case_id, passed, codes, any(c in DEFECTS for c in codes)))
    failed = [r for r in rows if not r[1]]
    ok = [r for r in rows if r[1]]
    codes: dict[str, int] = {}
    for _, _, cs, _ in rows:
        for c in cs:
            codes[c] = codes.get(c, 0) + 1
    return {
        "n": len(rows),
        "harness_failed": len(failed),
        "failed_flagged": sum(1 for r in failed if r[3]),
        "harness_passed": len(ok),
        "passed_flagged": sum(1 for r in ok if r[3]),
        "codes": dict(sorted(codes.items(), key=lambda kv: -kv[1])),
        "missed": [r[0] for r in failed if not r[3]],
        "false_alarms": [(r[0], r[2]) for r in ok if r[3]],
    }


def report(spans_path: Path = SPANS) -> str:
    spans = tracing.load_spans(spans_path)
    traces = case_traces(spans)
    agg = aggregate(traces)
    lines = [f"{len(spans)} spans, {agg['traces']} traces (one per case)\n"]

    def example(prefix: str, passed: bool | None):
        return next(
            (
                g
                for cid, p, g in traces.values()
                if cid.startswith(prefix) and (passed is None or p == passed)
            ),
            None,
        )

    shown = False
    for title, group in (
        ("A PASSING small refund", example("billing-small", True)),
        ("A FAILING approval case", example("billing-approval", False)),
        ("A technical question", example("tech-howto", None)),
    ):
        if group:
            shown = True
            lines += [f"--- {title} ---", tracing.render_tree(group), ""]
    if not shown and traces:  # a different dataset or a very broken system: still show one trace
        lines += [
            "--- the first trace ---",
            tracing.render_tree(next(iter(traces.values()))[2]),
            "",
        ]

    def q(xs, p):
        return te.percentile(xs, p)

    lines.append(f"Per case (n={agg['traces']}):  median / p95 / max")
    for label, key in (
        ("model calls", "model_calls"),
        ("input tokens", "input_tokens"),
        ("output tokens", "output_tokens"),
        ("tool calls", "tool_calls"),
        ("duration ms", "duration_ms"),
    ):
        xs = agg[key]
        lines.append(f"  {label:<14}{q(xs, 50):>8.0f} {q(xs, 95):>8.0f} {max(xs):>8.0f}")
    total_in = sum(a["input_tokens"] for a in agg["by_agent"].values()) or 1
    lines.append("\nBy agent (share of input tokens):")
    for name, a in sorted(agg["by_agent"].items(), key=lambda kv: -kv[1]["input_tokens"]):
        lines.append(
            f"  {name:<12} {a['model_calls']:>4} model calls  {a['input_tokens']:>7} in  {a['output_tokens']:>6} out  {a['input_tokens'] / total_in:>5.0%}"
        )
    lines.append("\nBy tool:")
    for name, t in sorted(agg["by_tool"].items(), key=lambda kv: -kv[1]["calls"]):
        lines.append(
            f"  {name:<18} {t['calls']:>4} calls  {t['errors']:>3} errors  {t['ms'] / max(t['calls'], 1):.1f} ms avg"
        )
    for split in (None, "dev", "test"):
        cmp = trace_vs_harness(traces, split)
        lines.append(
            f"\nTrace-only defect checks vs the full harness [{split or 'all'}, n={cmp['n']}]: of {cmp['harness_failed']} failed cases "
            f"the trace flagged {cmp['failed_flagged']}; of {cmp['harness_passed']} passed cases it flagged {cmp['passed_flagged']} (false alarms)."
        )
        if split is None:
            lines.append(
                "Codes across all cases: " + ", ".join(f"{c} x{n}" for c, n in cmp["codes"].items())
            )
            lines.append(f"Failed but NOT flagged by the trace: {cmp['missed']}")
            lines.append(f"Flagged although the harness passed: {cmp['false_alarms']}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- the cost of tracing


def measure_overhead(runs: int = 300) -> dict:
    """Per agent run (3 model calls, 2 tool calls) with a scripted model, so everything measured is OUR code."""
    from common.agent import run_agent
    from common.fake import fake_llm, tool_calls
    from common.tools import ToolRegistry, tool

    @tool
    def add(a: int, b: int) -> int:
        """Add.

        Args:
            a: x
            b: y
        """
        return a + b

    reg = ToolRegistry([add])
    script = [
        (
            r"(?s).*",
            [tool_calls(("add", {"a": 1, "b": 2})), tool_calls(("add", {"a": 3, "b": 4})), "done"]
            * runs,
        )
    ]

    def timed(traced: bool) -> float:
        with fake_llm(script):
            if traced:
                from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                    InMemorySpanExporter,
                )

                with tracing.session(InMemorySpanExporter()):
                    t0 = time.perf_counter()
                    for _ in range(runs):
                        run_agent("x", reg, provider="anthropic")
                    return (time.perf_counter() - t0) / runs
            t0 = time.perf_counter()
            for _ in range(runs):
                run_agent("x", reg, provider="anthropic")
            return (time.perf_counter() - t0) / runs

    plain = statistics.median(timed(False) for _ in range(5))
    traced = statistics.median(timed(True) for _ in range(5))
    return {
        "plain_ms": plain * 1e3,
        "traced_ms": traced * 1e3,
        "overhead_ms": (traced - plain) * 1e3,
        "spans_per_run": 6,
    }


def main(argv: list[str]) -> None:
    if argv[:1] == ["run"]:
        runs = run_all(argv[1] if len(argv) > 1 else "rules+prefetch")
        print(f"ran {len(runs)} cases; spans -> {SPANS}\n")
        print(report())
    elif argv[:1] == ["report"]:
        print(report())
    elif argv[:1] == ["overhead"]:
        o = measure_overhead()
        print(
            f"agent run with 3 model calls + 2 tool calls: {o['plain_ms']:.3f} ms plain, {o['traced_ms']:.3f} ms traced "
            f"(+{o['overhead_ms']:.3f} ms for {o['spans_per_run']} spans)"
        )
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
