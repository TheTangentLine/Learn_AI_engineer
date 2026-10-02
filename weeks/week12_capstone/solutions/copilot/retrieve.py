"""Retrieval for the capstone product: hybrid search (BM25 + dense), scoped by week when the question names one, merged by reciprocal rank fusion, optionally re-ranked.

    r = Retriever(index, RetrievalConfig(alpha=0.5, k=5))
    result = r.retrieve("How does the KV cache from Week 1 relate to PagedAttention in Week 11?")
    result.sources            # [Source(n=1, doc=..., heading=..., text=..., score=..., cosine=...), ...]
    result.weeks              # [1, 11]  (the question named two weeks: each was searched on its own so both are represented)

The pieces are the Week 3-4 ones (``common.rag.RagIndex.search``); what is new here is the *policy* around them: week scoping, fusion across scoped searches, a memo so the same question is
never retrieved twice (the gateway asks once to show sources and once to answer), per-stage timings, and a result object the answerers and the guards share.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field

WEEK = re.compile(r"\bweeks?\s+(\d{1,2})\b", re.I)
RRF_K = 60


@dataclass(frozen=True)
class RetrievalConfig:
    alpha: float = 0.5  # 1 = dense only, 0 = BM25 only (Week 3 Day 5)
    k: int = 5  # sources handed to the answerer
    per_scope: int = 4  # candidates taken from each week-scoped search
    scope_by_week: bool = True  # search each week the question names on its own
    rerank: bool = False  # cross-encoder over the fused candidates (Week 3 Day 5)
    rerank_depth: int = 12
    memo_size: int = 256


@dataclass(frozen=True)
class Source:
    n: int  # 1-based position: the number the answer cites as [n]
    id: str
    doc: str
    heading: str
    text: str
    score: float
    cosine: float
    week: int
    day: int

    def as_dict(self, snippet: int = 300) -> dict:
        return {
            "n": self.n,
            "id": self.id,
            "doc": self.doc,
            "heading": self.heading,
            "snippet": self.text[:snippet],
            "score": round(self.score, 4),
        }


@dataclass
class Retrieval:
    question: str
    sources: list[Source]
    weeks: list[int]
    seconds: dict[str, float] = field(default_factory=dict)
    cached: bool = False

    @property
    def top_cosine(self) -> float:
        return max((s.cosine for s in self.sources), default=0.0)


def weeks_named(question: str, limit: int = 3) -> list[int]:
    """The distinct week numbers (1-12) a question mentions, in order of appearance: ``"Week 4 and Week 7"`` -> [4, 7]."""
    seen: list[int] = []
    for m in WEEK.finditer(question):
        w = int(m.group(1))
        if 1 <= w <= 12 and w not in seen:
            seen.append(w)
    return seen[:limit]


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal rank fusion: an id's score is the sum over rankings of 1 / (k + rank). Ties keep first-seen order (deterministic)."""
    score: dict[str, float] = {}
    order: dict[str, int] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, 1):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank)
            order.setdefault(i, len(order))
    return sorted(score, key=lambda i: (-score[i], order[i]))


class Retriever:
    def __init__(self, index, config: RetrievalConfig | None = None, reranker=None):
        self.index, self.config, self.reranker = index, config or RetrievalConfig(), reranker
        self._memo: OrderedDict[str, Retrieval] = OrderedDict()
        self.hits = self.misses = 0

    def _search(self, q: str, k: int, where: dict | None = None):
        c = self.config
        return self.index.search(q, k, c.alpha, where=where)

    def retrieve(self, question: str) -> Retrieval:
        key = question.strip()
        if key in self._memo:
            self._memo.move_to_end(key)
            self.hits += 1
            r = self._memo[key]
            return Retrieval(r.question, r.sources, r.weeks, r.seconds, cached=True)
        self.misses += 1
        r = self._retrieve(key)
        self._memo[key] = r
        while len(self._memo) > self.config.memo_size:
            self._memo.popitem(last=False)
        return r

    def _retrieve(self, question: str) -> Retrieval:
        c = self.config
        t0 = time.perf_counter()
        weeks = weeks_named(question) if c.scope_by_week else []
        rankings: list[list[str]] = []
        by_id: dict[str, object] = {}
        searches: list[tuple[str, dict | None, int]] = [("global", None, max(c.k, c.per_scope))]
        searches += [(f"week{w}", {"week": w}, c.per_scope) for w in weeks]
        for _, where, k in searches:
            hits = self._search(question, k, where)
            rankings.append([h.id for h in hits])
            for h in hits:
                by_id.setdefault(h.id, h)
        fused = rrf(rankings)
        t1 = time.perf_counter()
        seconds = {"search": t1 - t0}
        if len(weeks) > 1:
            # every named week must be represented: take the best fused chunk of each week first, then fill by fused rank
            chosen: list[str] = []
            for w in weeks:
                first = next((i for i in fused if by_id[i].metadata.get("week") == w), None)
                if first and first not in chosen:
                    chosen.append(first)
            fused = chosen + [i for i in fused if i not in chosen]
        top = fused[: max(c.k, c.rerank_depth if self.reranker and c.rerank else c.k)]
        if self.reranker is not None and c.rerank and top:
            ranked = self.reranker.rerank(question, [by_id[i].text for i in top])
            top = [top[i] for i, _ in ranked]
            seconds["rerank"] = time.perf_counter() - t1
        sources = []
        for n, i in enumerate(top[: c.k], 1):
            h = by_id[i]
            m = h.metadata
            sources.append(
                Source(
                    n,
                    h.id,
                    m.get("doc", ""),
                    m.get("heading", ""),
                    h.text,
                    h.score,
                    float(m.get("cosine", 0.0)),
                    int(m.get("week", 0)),
                    int(m.get("day", 0)),
                )
            )
        return Retrieval(question, sources, weeks, seconds)
