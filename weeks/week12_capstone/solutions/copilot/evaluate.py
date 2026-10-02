"""Run the product over the golden set and report what happened, with intervals.

    runs = run_golden(copilot, golden.load(), split="dev")
    table = report(runs)                      # pass rates by kind with Wilson intervals, retrieval and attribution rates, latency percentiles, error counts

Scores come from ``copilot.golden.score`` (code, not a model). Nothing here calls a model except through the ``Copilot`` under test.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from common.abtest import wilson  # noqa: E402

from . import golden as G  # noqa: E402
from .core import Copilot, Result  # noqa: E402


@dataclass
class Run:
    item: G.Item
    result: Result
    score: dict


def outcome_of(item: G.Item, r: Result) -> G.Outcome:
    return G.Outcome(
        item.id,
        r.answer,
        r.abstained,
        [s.doc for s in (r.retrieved or r.sources)],
        r.cited_docs,
        invalid_citation=False,  # the pipeline removes or refuses invalid citations; a violation would show up as an unattributed answer
        blocked=r.blocked,
        error=r.error,
        seconds=r.seconds,
    )


def run_golden(
    copilot: Copilot,
    items: list[G.Item],
    split: str | None = None,
    kinds: tuple[str, ...] | None = None,
) -> list[Run]:
    runs = []
    for it in items:
        if (split and it.split != split) or (kinds and it.kind not in kinds):
            continue
        r = copilot.ask(it.question, request_id=it.id)
        runs.append(Run(it, r, G.score(it, outcome_of(it, r))))
    return runs


def rate(successes: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    lo, hi = wilson(successes, n)
    return successes / n, lo, hi


def fmt(ci: tuple[float, float, float]) -> str:
    p, lo, hi = ci
    return "n/a" if p != p else f"{p:.0%} [{lo:.0%}, {hi:.0%}]"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    lo, hi = int(k // 1), int(-(-k // 1))
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def report(runs: list[Run]) -> dict:
    """Aggregate numbers. Every rate is (rate, low, high)."""
    by = {k: [r for r in runs if r.item.kind == k] for k in G.KINDS}
    answerable = by["single"] + by["multi"]
    out: dict = {"n": len(runs)}
    out["overall"] = rate(sum(r.score["passed"] for r in runs), len(runs))
    for k, rs in by.items():
        out[k] = rate(sum(r.score["passed"] for r in rs), len(rs))
        out[f"{k}_n"] = len(rs)
    out["facts_ok"] = rate(sum(r.score["facts_ok"] for r in answerable), len(answerable))
    out["attributed"] = rate(sum(r.score["attributed"] for r in answerable), len(answerable))
    out["retrieved_any"] = rate(sum(r.score["retrieved_any"] for r in answerable), len(answerable))
    out["retrieved_all"] = rate(
        sum(r.score["retrieved_all"] for r in by["multi"]), len(by["multi"])
    )
    out["wrong_abstention"] = rate(
        sum(r.score["wrong_abstention"] for r in answerable), len(answerable)
    )
    out["leaks"] = sum(r.score.get("leaked", False) for r in by["adversarial"])
    out["errors"] = sum(r.score["error"] for r in runs)
    secs = [r.result.seconds for r in runs if not r.result.cached]
    out["latency"] = {
        "p50": percentile(secs, 50),
        "p95": percentile(secs, 95),
        "mean": statistics.fmean(secs) if secs else float("nan"),
    }
    out["tokens"] = {
        "prompt": sum(r.result.prompt_tokens for r in runs),
        "completion": sum(r.result.completion_tokens for r in runs),
        "model_calls": sum(1 for r in runs if r.result.prompt_tokens or r.result.completion_tokens),
    }
    stage: dict[str, list[float]] = {}
    for r in runs:
        for k, v in r.result.timings.items():
            stage.setdefault(k, []).append(v)
    out["stages_ms"] = {
        k: {"p50": percentile(v, 50) * 1000, "p95": percentile(v, 95) * 1000}
        for k, v in stage.items()
    }
    return out
