"""The decision procedure: split -> evaluate all candidates on DEV -> pick -> confirm on TEST -> verdict.

Why this shape: with ~50 questions you cannot both choose among many options and report an honest
score on the same questions (selection bias, Week 2 Day 6). So:
  1. split the golden set into dev / test, stratified by question source (manual / synthetic)
  2. evaluate every candidate on DEV and choose ONE (best MRR within a context-cost cap)
  3. evaluate baseline and winner on TEST once, with a PAIRED bootstrap
  4. a verdict that needs BOTH a significant gain and an acceptable cost
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from common.evalkit import (
    GoldQuery,
    RetrievalResult,
    bootstrap_ci,
    evaluate_retriever,
    fmt_ci,
    paired_bootstrap,
)

Retrieve = Callable[[str], list[tuple[str, str]]]  # query -> [(doc, text)] best-first


@dataclass
class Config:
    name: str
    retrieve: Retrieve
    k: int  # how many chunks this config would put in the LLM prompt
    rerank_pairs: float = 0.0  # cross-encoder pairs scored per query
    llm_calls: float = 0.0  # query-time LLM calls per query
    index_rows: int = 0  # vectors stored
    note: str = ""


@dataclass
class Evaluated:
    config: Config
    result: RetrievalResult
    context_chars: float  # mean characters in the k chunks sent to the LLM
    hit_at_4: list[float] = field(
        default_factory=list
    )  # common cut-off for apples-to-apples comparison


def split_golden(
    golden: list[GoldQuery], seed: int = 0, dev_fraction: float = 0.5
) -> tuple[list, list]:
    """Deterministic split, stratified by `kind` so dev and test have the same manual/synthetic mix."""
    rng = random.Random(seed)
    dev, test = [], []
    for kind in sorted({g.kind for g in golden}):
        group = [g for g in golden if g.kind == kind]
        rng.shuffle(group)
        cut = round(len(group) * dev_fraction)
        dev += group[:cut]
        test += group[cut:]
    return dev, test


def evaluate_config(cfg: Config, queries: list[GoldQuery]) -> Evaluated:
    """Run the config once per query (memoised), then derive every metric from the same ranked lists."""
    memo: dict[str, list[tuple[str, str]]] = {}

    def retrieve(q: str) -> list[tuple[str, str]]:
        if q not in memo:
            memo[q] = cfg.retrieve(q)
        return memo[q]

    res = evaluate_retriever(cfg.name, retrieve, queries)
    hit4 = [float(any(q.is_relevant(d, t) for d, t in retrieve(q.query)[:4])) for q in queries]
    ctx = [sum(len(t) for _, t in retrieve(q.query)[: cfg.k]) for q in queries]
    costs = getattr(cfg.retrieve, "per_query_cost", None)
    if costs:  # retrievers that count their own work (CRAG) report measured costs
        cfg.rerank_pairs, cfg.llm_calls = costs()
    return Evaluated(cfg, res, float(np.mean(ctx)), hit4)


def select_best(dev: dict[str, Evaluated], baseline: str, max_context_ratio: float) -> str:
    """Highest dev MRR among candidates whose context size is within `max_context_ratio` of the baseline's.
    Ties go to the cheaper configuration (smaller context, then fewer rerank pairs)."""
    base_ctx = dev[baseline].context_chars
    ok = [e for e in dev.values() if e.context_chars <= max_context_ratio * base_ctx]
    best = max(
        ok,
        key=lambda e: (
            round(float(np.mean(e.result.mrr)), 4),
            -e.context_chars,
            -e.config.rerank_pairs,
        ),
    )
    return best.config.name


def verdict(paired: dict, context_ratio: float, max_context_ratio: float) -> str:
    """ADOPT needs a significant gain (95% interval excludes 0) AND an acceptable cost."""
    if paired["ci_low"] > 0 and context_ratio <= max_context_ratio:
        return "ADOPT"
    if paired["ci_low"] > 0:
        return "GAIN BUT TOO COSTLY"
    if paired["ci_high"] < 0:
        return "REJECT: significantly worse"
    return "REJECT: no significant improvement"


def render_report(
    dev: dict[str, Evaluated],
    test: dict[str, Evaluated],
    baseline: str,
    winner: str,
    n_dev: int,
    n_test: int,
    max_context_ratio: float,
) -> tuple[str, dict]:
    lines = [
        "# RAG upgrade report",
        "",
        f"- golden questions: **{n_dev} dev + {n_test} test** (stratified by source; selection on dev only)",
        f"- baseline: **{baseline}** | cost cap: context <= **{max_context_ratio}x** baseline | verdict needs a "
        "95% paired interval above 0 on the held-out test set",
        "",
        "## Step 1: candidates on DEV (used only to choose)",
        "",
        "| config | k | MRR (dev) | hit@4 | context chars | rerank pairs/q | LLM calls/q | index rows |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, e in dev.items():
        c = e.config
        lines.append(
            f"| {name} | {c.k} | {fmt_ci(bootstrap_ci(e.result.mrr), pct=False)} | "
            f"{np.mean(e.hit_at_4):.0%} | {e.context_chars:,.0f} | {c.rerank_pairs:g} | {c.llm_calls:g} | "
            f"{c.index_rows} |"
        )
    lines += [
        "",
        f"**Chosen on dev: `{winner}`**",
        "",
        "## Step 2: confirm on TEST (touched once)",
        "",
    ]
    b, w = test[baseline], test[winner]
    p_mrr = paired_bootstrap(w.result.mrr, b.result.mrr)
    p_hit = paired_bootstrap(w.result.hit5, b.result.hit5)
    ratio = w.context_chars / b.context_chars
    lines += [
        "| metric | baseline | winner | paired diff [95% CI] | p | W/L/T |",
        "|---|---|---|---|---|---|",
        f"| MRR | {np.mean(b.result.mrr):.2f} | {np.mean(w.result.mrr):.2f} | {p_mrr['diff']:+.3f} "
        f"[{p_mrr['ci_low']:+.3f}, {p_mrr['ci_high']:+.3f}] | {p_mrr['p']:.3f} | "
        f"{p_mrr['wins']}/{p_mrr['losses']}/{p_mrr['ties']} |",
        f"| hit@5 | {np.mean(b.result.hit5):.0%} | {np.mean(w.result.hit5):.0%} | {p_hit['diff']:+.2f} "
        f"[{p_hit['ci_low']:+.2f}, {p_hit['ci_high']:+.2f}] | {p_hit['p']:.3f} | "
        f"{p_hit['wins']}/{p_hit['losses']}/{p_hit['ties']} |",
        f"| context chars | {b.context_chars:,.0f} | {w.context_chars:,.0f} ({ratio:.1f}x) | | | |",
    ]
    v = (
        "NO CHANGE (winner is the baseline)"
        if winner == baseline
        else verdict(p_mrr, ratio, max_context_ratio)
    )
    lines += [
        "",
        f"## Verdict: **{v}**",
        "",
        "## All candidates on TEST (for transparency, not for selection)",
        "",
        "| config | MRR [95% CI] | hit@4 | context chars |",
        "|---|---|---|---|",
    ]
    for name, e in test.items():
        lines.append(
            f"| {name} | {fmt_ci(bootstrap_ci(e.result.mrr), pct=False)} | {np.mean(e.hit_at_4):.0%} | "
            f"{e.context_chars:,.0f} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        f"- {n_test} test questions: intervals are wide; a 'not significant' verdict means *this test cannot tell*.",
        "- The golden set mixes hand-written and synthetic questions; synthetic ones favour lexical overlap (Day 1).",
        "- Costs shown are proxies (context size, re-rank pairs, LLM calls); measure real latency and spend before shipping.",
    ]
    return "\n".join(lines), {
        "verdict": v,
        "mrr": p_mrr,
        "hit5": p_hit,
        "context_ratio": ratio,
        "winner": winner,
    }
