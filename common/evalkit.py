"""Evaluation toolkit: retrieval metrics, golden sets, and the statistics to read them honestly.

    from common.evalkit import GoldQuery, ranks_of_relevant, mrr, ndcg_at_k, bootstrap_ci, paired_bootstrap

The reason this module exists: with 20-50 queries, differences of a few points are NOISE. Always
report an interval, and compare two systems with a *paired* test on the same queries.

Conventions
-----------
* A "ranking" is a list of retrieved items in order; ``is_relevant(item) -> bool`` or a graded
  gain function decides relevance.
* Per-query metrics return floats in [0, 1]; aggregate with ``bootstrap_ci`` (mean + 95% interval).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ----------------------------------------------------------------- golden sets


def norm(text: str) -> str:
    """Normalise text for phrase matching: strip markdown emphasis/code ticks, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", text)).lower()


@dataclass
class GoldQuery:
    """A query plus chunker-independent ground truth: the right lesson AND an answer phrase.

    A retrieved chunk is relevant iff it comes from ``doc`` and contains ``phrase`` (after ``norm``).
    Because relevance is defined by text rather than chunk IDs, the golden set survives re-chunking.
    """

    id: str
    query: str
    doc: str
    phrase: str
    kind: str = "manual"  # manual | synthetic
    tags: list[str] = field(default_factory=list)

    def is_relevant(self, doc: str, text: str) -> bool:
        return doc == self.doc and self.phrase in norm(text)


def save_golden(path: str | Path, queries: Iterable[GoldQuery]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(q), ensure_ascii=False) for q in queries) + "\n")


def load_golden(path: str | Path) -> list[GoldQuery]:
    return [
        GoldQuery(**json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


# ----------------------------------------------------------------- per-query metrics


def ranks_of_relevant(relevance: Sequence[bool]) -> list[int]:
    """1-based ranks at which relevant items appear."""
    return [i for i, r in enumerate(relevance, 1) if r]


def hit_at_k(relevance: Sequence[bool], k: int) -> float:
    return float(any(relevance[:k]))


def reciprocal_rank(relevance: Sequence[bool], cutoff: int | None = None) -> float:
    for i, r in enumerate(relevance[:cutoff] if cutoff else relevance, 1):
        if r:
            return 1.0 / i
    return 0.0


def recall_at_k(relevance: Sequence[bool], k: int, n_relevant: int) -> float:
    """Share of ALL relevant items found in the top-k (needs the total count of relevant items)."""
    if n_relevant <= 0:
        raise ValueError("recall needs at least one relevant item")
    return min(sum(relevance[:k]), n_relevant) / n_relevant


def precision_at_k(relevance: Sequence[bool], k: int) -> float:
    top = relevance[:k]
    return sum(top) / len(top) if top else 0.0


def dcg(gains: Sequence[float], k: int) -> float:
    return sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(gains[:k], 1))


def ndcg_at_k(gains: Sequence[float], k: int, ideal_gains: Sequence[float] | None = None) -> float:
    """Normalised DCG: rewards putting highly relevant items first. ``gains`` are graded (0,1,2...)
    in retrieved order; ``ideal_gains`` are ALL known gains (defaults to the retrieved ones)."""
    ideal = sorted(ideal_gains if ideal_gains is not None else gains, reverse=True)
    best = dcg(ideal, k)
    return dcg(gains, k) / best if best > 0 else 0.0


# ----------------------------------------------------------------- statistics


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 4000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and (1-alpha) percentile bootstrap interval: resample queries with replacement."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return (
        float(v.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, seed: int = 0
) -> dict:
    """Is system A better than system B on the SAME queries?

    Resamples queries (keeping each query's pair together) and looks at the distribution of the mean
    difference. Returns the observed mean difference, its 95% interval, and a two-sided p-value
    (twice the smaller tail mass around zero). A paired test is far more sensitive than comparing two
    separate intervals, because query difficulty cancels out.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size == 0:
        raise ValueError("no queries")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "diff": float(d.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p": float(min(1.0, p)),
        "n": int(d.size),
        "wins": int((d > 0).sum()),
        "losses": int((d < 0).sum()),
        "ties": int((d == 0).sum()),
    }


def cohens_kappa(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Agreement between two labelers beyond chance (1 = perfect, 0 = chance, <0 = worse than chance)."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("need two equal-length, non-empty label lists")
    labels = sorted(set(y_true) | set(y_pred), key=str)
    n = len(y_true)
    observed = sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / n
    expected = sum(
        (sum(t == lab for t in y_true) / n) * (sum(p == lab for p in y_pred) / n) for lab in labels
    )
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def confusion(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[tuple[Any, Any], int]:
    out: dict[tuple[Any, Any], int] = {}
    for t, p in zip(y_true, y_pred, strict=True):
        out[(t, p)] = out.get((t, p), 0) + 1
    return out


def fmt_ci(ci: tuple[float, float, float], pct: bool = True) -> str:
    m, lo, hi = ci
    return f"{m:.0%} [{lo:.0%}-{hi:.0%}]" if pct else f"{m:.2f} [{lo:.2f}-{hi:.2f}]"


# ----------------------------------------------------------------- evaluating a retriever


@dataclass
class RetrievalResult:
    """Per-query metric arrays (aligned with the golden queries) for one retriever."""

    name: str
    hit1: list[float] = field(default_factory=list)
    hit5: list[float] = field(default_factory=list)
    mrr: list[float] = field(default_factory=list)
    ndcg5: list[float] = field(default_factory=list)
    first_rank: list[int | None] = field(default_factory=list)

    def summary(self) -> dict[str, tuple[float, float, float]]:
        return {
            "hit@1": bootstrap_ci(self.hit1),
            "hit@5": bootstrap_ci(self.hit5),
            "MRR": bootstrap_ci(self.mrr),
            "nDCG@5": bootstrap_ci(self.ndcg5),
        }


def evaluate_retriever(
    name: str,
    retrieve: Callable[[str], list[tuple[str, str]]],
    golden: Sequence[GoldQuery],
    depth: int = 20,
) -> RetrievalResult:
    """``retrieve(query)`` returns ``[(doc, text), ...]`` best-first. Relevance comes from the gold."""
    res = RetrievalResult(name)
    for g in golden:
        rel = [g.is_relevant(doc, text) for doc, text in retrieve(g.query)[:depth]]
        ranks = ranks_of_relevant(rel)
        res.hit1.append(hit_at_k(rel, 1))
        res.hit5.append(hit_at_k(rel, 5))
        res.mrr.append(reciprocal_rank(rel))
        # ideal ranking = every relevant item we can see within `depth`, best first
        res.ndcg5.append(
            ndcg_at_k([float(r) for r in rel], 5, ideal_gains=[1.0] * max(1, len(ranks)))
        )
        res.first_rank.append(ranks[0] if ranks else None)
    return res
