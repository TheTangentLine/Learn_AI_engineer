"""Corrective retrieval: grade what you retrieved, and try harder when it looks wrong.

    from common.crag import corrective_search
    res = corrective_search(index, "how do I stop hammering a failing API?", reranker, tau=0.0,
                            rewriter=lambda q: ["exponential backoff retry", "rate limit handling"])
    res.hits            # best chunks found (best-first)
    res.grade           # correct | ambiguous | incorrect  (of the FINAL result)
    [s.action for s in res.steps]   # what the controller did, for logging and cost accounting

The control loop (an *explicit* controller, not an LLM deciding; the LLM-driven version is Week 5):

    retrieve -> rerank -> grade by the best cross-encoder score
       correct   -> done (return immediately: no extra cost, no risk of making it worse)
       otherwise -> fallback ladder, each rung adds candidates:
                      1. rewritten queries (multi-query)
                      2. a different retrieval strategy (BM25-only, vector-only)
                    pool everything, RE-RANK AGAINST THE ORIGINAL QUESTION, grade again

Two design rules that matter: (1) always score against the user's *original* question, never against a
rewrite (a rewrite can drift); (2) never return something worse than what you started with.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .vectorstores import Hit


@dataclass
class Step:
    action: str  # initial | rewrite | alt_strategy
    query: str
    candidates: int  # distinct chunks pooled so far
    top_score: float  # best cross-encoder score after this step
    grade: str


@dataclass
class CorrectiveResult:
    hits: list[Hit]
    grade: str
    steps: list[Step] = field(default_factory=list)
    retrievals: int = 0  # index searches performed
    rerank_pairs: int = 0  # (query, passage) pairs scored

    @property
    def corrected(self) -> bool:
        """True if any fallback ran (the first attempt was not graded 'correct')."""
        return len(self.steps) > 1


def grade(top_score: float, tau_hi: float, tau_lo: float | None = None) -> str:
    """correct: confident the evidence is relevant. ambiguous: between thresholds. incorrect: below both."""
    tau_lo = tau_hi if tau_lo is None else tau_lo
    if top_score >= tau_hi:
        return "correct"
    return "ambiguous" if top_score >= tau_lo else "incorrect"


def corrective_search(
    index,
    query: str,
    reranker,
    tau: float,
    *,
    tau_low: float | None = None,
    rewriter: Callable[[str], list[str]] | None = None,
    k: int = 5,
    depth: int = 20,
    alpha: float = 0.5,
    max_steps: int = 3,
) -> CorrectiveResult:
    """Retrieve, grade, and correct. ``index.search(query, k, alpha=..)`` must return Hits with ``.id``/``.text``."""
    res = CorrectiveResult([], "incorrect")
    pool: dict[str, Hit] = {}
    scores: dict[str, float] = {}

    def add(hits: list[Hit]) -> None:
        for h in hits:
            pool.setdefault(h.id, h)

    def rerank_pool() -> float:
        missing = [h for h in pool.values() if h.id not in scores]
        if missing:
            for h, s in zip(
                missing, reranker.scores(query, [h.text for h in missing]), strict=True
            ):
                scores[h.id] = s
            res.rerank_pairs += len(missing)
        ordered = sorted(pool.values(), key=lambda h: -scores[h.id])
        res.hits = ordered[:k]
        return scores[ordered[0].id] if ordered else float("-inf")

    def record(action: str, q: str) -> str:
        top = rerank_pool()
        g = grade(top, tau, tau_low)
        res.steps.append(Step(action, q, len(pool), top, g))
        res.grade = g
        return g

    add(index.search(query, depth, alpha=alpha))
    res.retrievals += 1
    if record("initial", query) == "correct" or max_steps <= 1:
        return res

    rungs: list[tuple[str, str, float]] = []  # (action, query, alpha)
    if rewriter is not None:
        for q in rewriter(query):
            rungs.append(("rewrite", q, alpha))
    rungs += [("alt_strategy", query, 0.0), ("alt_strategy", query, 1.0)]  # BM25-only, vector-only

    for action, q, a in rungs:
        if len(res.steps) >= max_steps:
            break
        add(index.search(q, depth, alpha=a))
        res.retrievals += 1
        if record(action, q) == "correct":
            break
    return res
