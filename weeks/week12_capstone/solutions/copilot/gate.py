"""The relevance gate: decide, before any answer is written, whether the retrieved evidence can answer the question at all.

A product that answers everything is a product that makes things up. The gate turns "the sources do not contain this" from something a small model must notice into a decision made in code.
Two signals, both measured on the golden set (Day 2):

``CosineGate``   the best dense similarity among the retrieved sources (free: it came with the retrieval)
``RerankGate``   the best cross-encoder score of (question, source) over the top few sources (a second model, about a tenth of a second on a CPU)

``calibrate`` picks the threshold from labelled scores; ``auc`` says how well a signal separates answerable from unanswerable questions regardless of the threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    score: float
    threshold: float
    signal: str

    @property
    def reason(self) -> str:
        return f"gate: best {self.signal} {self.score:.2f} {'>=' if self.allowed else '<'} threshold {self.threshold:.2f}"


class CosineGate:
    signal = "cosine"

    def __init__(self, threshold: float):
        self.threshold = threshold

    def score(self, question: str, retrieval) -> float:
        return retrieval.top_cosine

    def decide(self, question: str, retrieval) -> GateDecision:
        s = self.score(question, retrieval) if retrieval.sources else float("-inf")
        return GateDecision(
            bool(retrieval.sources) and s >= self.threshold, s, self.threshold, self.signal
        )


class RerankGate:
    signal = "rerank"

    def __init__(self, reranker, threshold: float, top: int = 3):
        self.reranker, self.threshold, self.top = reranker, threshold, top

    def score(self, question: str, retrieval) -> float:
        texts = [s.text for s in retrieval.sources[: self.top]]
        return max(self.reranker.scores(question, texts)) if texts else float("-inf")

    def decide(self, question: str, retrieval) -> GateDecision:
        s = self.score(question, retrieval)
        return GateDecision(
            bool(retrieval.sources) and s >= self.threshold, s, self.threshold, self.signal
        )


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Probability that a random answerable question scores higher than a random unanswerable one (ties count half). 0.5 is a coin flip, 1.0 is perfect separation."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def calibrate(pos: Sequence[float], neg: Sequence[float], *, min_recall: float = 0.0) -> dict:
    """The threshold that maximises balanced accuracy (the mean of the share of answerable questions admitted and the share of unanswerable ones refused), subject to admitting at least
    ``min_recall`` of the answerable ones. Candidates are the midpoints between consecutive observed scores, so the choice is deterministic and sits between two real examples."""
    pts = sorted(set(list(pos) + list(neg)))
    if not pts:
        raise ValueError("no scores to calibrate on")
    cands = (
        [pts[0] - 1e-6]
        + [(a + b) / 2 for a, b in zip(pts, pts[1:], strict=False)]
        + [pts[-1] + 1e-6]
    )
    best = None
    for t in cands:
        recall = sum(p >= t for p in pos) / len(pos) if pos else 1.0
        refuse = sum(n < t for n in neg) / len(neg) if neg else 1.0
        if recall < min_recall:
            continue
        bal = (recall + refuse) / 2
        if best is None or bal > best["balanced_accuracy"] + 1e-12:
            best = {"threshold": t, "balanced_accuracy": bal, "recall": recall, "refusal": refuse}
    if best is None:  # no candidate met the recall floor: admit everything
        t = cands[0]
        best = {"threshold": t, "balanced_accuracy": 0.5, "recall": 1.0, "refusal": 0.0}
    best["auc"] = auc(pos, neg)
    return best


class CascadeGate:
    """Ask the cheap signal first and the expensive one only when the cheap one is unsure.

    cosine >= ``high``  -> admit without the cross-encoder
    cosine <= ``low``   -> refuse without the cross-encoder
    in between          -> the ``RerankGate`` decides

    ``choose_band`` picks (low, high) from labelled dev scores so that the cheap rule makes NO mistake on the dev examples it settles; the band is then the uncertain middle that costs a
    cross-encoder call. ``called`` counts how many questions needed it.
    """

    signal = "cascade"

    def __init__(self, rerank_gate: RerankGate, low: float, high: float):
        if low > high:
            raise ValueError("low must not exceed high")
        self.rerank_gate, self.low, self.high = rerank_gate, low, high
        self.called = 0
        self.decided = 0

    @property
    def threshold(self) -> float:
        return self.rerank_gate.threshold

    def decide(self, question: str, retrieval) -> GateDecision:
        self.decided += 1
        if not retrieval.sources:
            return GateDecision(False, float("-inf"), self.high, "cosine")
        c = retrieval.top_cosine
        if c >= self.high:
            return GateDecision(True, c, self.high, "cosine")
        if c <= self.low:
            return GateDecision(False, c, self.low, "cosine")
        self.called += 1
        return self.rerank_gate.decide(question, retrieval)

    @staticmethod
    def choose_band(pos: Sequence[float], neg: Sequence[float]) -> tuple[float, float]:
        """(low, high) from the cosines of answerable (``pos``) and out-of-scope (``neg``) dev questions: ``high`` is just above the best out-of-scope cosine (everything above it is answerable),
        ``low`` just below the worst answerable cosine (everything below it is out of scope). If the two groups do not overlap the band collapses to one threshold."""
        high = max(neg) + 1e-6 if neg else min(pos)
        low = min(pos) - 1e-6 if pos else max(neg)
        return (min(low, high), high) if low <= high else ((high + low) / 2, (high + low) / 2)
