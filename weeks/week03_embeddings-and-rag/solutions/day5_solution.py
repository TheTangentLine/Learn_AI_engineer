"""Week 3 Day 5 - Solution: hybrid search (BM25 + vectors, RRF) and cross-encoder re-ranking.

Four retrievers over the same 159 heading-aware chunks, on 29 queries in two flavours:
  * 15 PARAPHRASED questions   (vocabulary differs from the lessons -> vectors should help)
  * 14 EXACT-TERM queries      (identifiers like model_validator, o200k_base -> BM25 should help)
Ground truth is chunker-independent: right lesson AND the chunk contains the answer phrase.

  uv run python weeks/week03_embeddings-and-rag/solutions/day5_solution.py
(first run scores ~600 query-chunk pairs with a cross-encoder; results are cached in outputs/)
"""

from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from day3_solution import GOLD, by_headings, norm  # noqa: E402

from common.corpus import load_course_docs  # noqa: E402
from common.embed import get_embedder  # noqa: E402
from common.rerank import get_reranker  # noqa: E402

PINNED = ("week01", "week02")

LEXICAL_GOLD = [  # (query, lesson, phrase): someone searching for a specific term
    ("o200k_base encoding", "week01/day2", "o200k_base"),
    ("asyncio.Semaphore", "week01/day5", "asyncio.semaphore"),
    ("model_validator after", "week02/day3", "model_validator"),
    ("BootstrapFewShot compile", "week02/day6", "bootstrapfewshot"),
    ("cache_control ephemeral", "week01/day4", 'cache_control={"type": "ephemeral"}'),
    ("RetryableError vs FatalError", "week01/day5", "retryableerror"),
    ("client.messages.count_tokens", "week01/day2", "client.messages.count_tokens"),
    ("retry-after header", "week01/day5", "retry-after"),
    ("evict_batch", "week02/day5", "evict_batch"),
    ("boxed answer extraction", "week02/day2", "\\boxed"),
    ("PagedAttention", "week01/day4", "pagedattention"),
    ("MIPROv2 GEPA optimisers", "week02/day6", "miprov2"),
    ("--mock harness test", "week01/day6", "--mock"),
    ("asyncio.gather", "week02/day4", "asyncio.gather"),
]
SETS = {"paraphrased": GOLD, "exact-term": LEXICAL_GOLD}


@dataclass
class Corpus:
    texts: list[str]
    docs: list[str]
    vecs: np.ndarray
    bm25: BM25Okapi


def tokenize(text: str) -> list[str]:
    """Code-friendly tokens: keep underscores so model_validator stays one token."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def build_corpus(emb) -> Corpus:
    texts, docs = [], []
    for d in (d for d in load_course_docs() if d.short.startswith(PINNED)):
        for c in by_headings(d.text, 1200):
            texts.append(c.text)
            docs.append(d.short)
    return Corpus(texts, docs, emb.embed_documents(texts), BM25Okapi([tokenize(t) for t in texts]))


# ----------------------------------------------------------------- the retrievers (each returns a ranking)


def rank_bm25(c: Corpus, query: str) -> list[int]:
    return list(np.argsort(-c.bm25.get_scores(tokenize(query)), kind="stable"))


def rank_vector(c: Corpus, emb, query: str) -> list[int]:
    return list(np.argsort(-(c.vecs @ emb.embed_query(query)), kind="stable"))


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: score(d) = sum over rankers of 1 / (k + rank_d). Needs ranks only, so
    BM25 scores (unbounded) and cosines (0..1) never have to be put on a common scale."""
    score: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for r, doc in enumerate(ranking, 1):
            score[doc] += 1.0 / (k + r)
    return sorted(score, key=lambda d: -score[d])


def weighted_fusion(c: Corpus, emb, query: str, alpha: float = 0.5) -> list[int]:
    """The alternative to RRF: min-max normalise both score vectors, then alpha*vector + (1-alpha)*bm25."""
    bm = c.bm25.get_scores(tokenize(query))
    vec = c.vecs @ emb.embed_query(query)
    norm_ = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)  # noqa: E731
    return list(np.argsort(-(alpha * norm_(vec) + (1 - alpha) * norm_(bm)), kind="stable"))


def rerank(c: Corpus, rr, query: str, candidates: list[int], depth: int = 20) -> list[int]:
    top = candidates[:depth]
    order = rr.rerank(query, [c.texts[i] for i in top])
    return [top[i] for i, _ in order] + candidates[depth:]


# ----------------------------------------------------------------- evaluation


def gold_rank(c: Corpus, ranking: list[int], lesson: str, phrase: str) -> int | None:
    for r, i in enumerate(ranking, 1):
        if c.docs[i] == lesson and phrase in norm(c.texts[i]):
            return r
    return None


def summarise(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        "hit@1": sum(r == 1 for r in ranks) / n,
        "hit@5": sum(r is not None and r <= 5 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def main() -> None:
    emb, rr = get_embedder(), get_reranker()
    c = build_corpus(emb)
    for gold in SETS.values():  # a bad gold phrase would silently invalidate everything
        for _, lesson, phrase in gold:
            assert any(
                d == lesson and phrase in norm(t) for t, d in zip(c.texts, c.docs, strict=True)
            ), phrase
    print(
        f"corpus: {len(c.texts)} chunks | queries: {', '.join(f'{k}={len(v)}' for k, v in SETS.items())}\n"
    )

    methods = {
        "BM25 only": lambda q: rank_bm25(c, q),
        "vector only": lambda q: rank_vector(c, emb, q),
        "hybrid (RRF k=60)": lambda q: rrf([rank_bm25(c, q), rank_vector(c, emb, q)]),
        "hybrid, weighted a=0.5": lambda q: weighted_fusion(c, emb, q, 0.5),
        "hybrid + rerank top-20": lambda q: rerank(
            c, rr, q, rrf([rank_bm25(c, q), rank_vector(c, emb, q)])
        ),
    }
    t0 = time.perf_counter()
    all_ranks: dict[str, dict[str, list[int | None]]] = {m: {} for m in methods}
    for set_name, gold in SETS.items():
        for m, fn in methods.items():
            all_ranks[m][set_name] = [
                gold_rank(c, fn(q), lesson, phrase) for q, lesson, phrase in gold
            ]
    print(
        f"(evaluation took {time.perf_counter() - t0:.0f}s; reranker scored {rr.calls} new pairs)\n"
    )

    print(f"{'method':<26}" + "".join(f"| {s:^26}" for s in SETS) + "| all 29")
    print(f"{'':<26}" + ("| hit@1  hit@5   MRR        " * len(SETS)) + "| MRR")
    for m in methods:
        row = f"{m:<26}"
        every = []
        for s in SETS:
            r = summarise(all_ranks[m][s])
            row += f"| {r['hit@1']:>5.0%}  {r['hit@5']:>5.0%}  {r['mrr']:>5.2f}      "
            every += all_ranks[m][s]
        print(row + f"| {summarise(every)['mrr']:.2f}")

    print("\nRank of the answer chunk per query (lower is better, '-' = not found):")
    short = {
        "BM25 only": "bm25",
        "vector only": "vec",
        "hybrid (RRF k=60)": "rrf",
        "hybrid + rerank top-20": "+rr",
    }
    print(f"{'query':<42}" + "".join(f"{v:>6}" for v in short.values()))
    for set_name, gold in SETS.items():
        print(f"-- {set_name}")
        for j, (q, _, _) in enumerate(gold):
            cells = [all_ranks[m][set_name][j] for m in short]
            best_single = min((r for r in cells[:2] if r), default=None)
            mark = ""
            if cells[2] and (best_single is None or cells[2] < best_single):
                mark = "  <- hybrid beats both"
            if cells[3] and cells[2] and cells[3] < cells[2]:
                mark += "  <- rerank helped"
            print(f"{q[:41]:<42}" + "".join(f"{(str(r) if r else '-'):>6}" for r in cells) + mark)

    print("\nRRF k sensitivity (all 29 queries, MRR):", end=" ")
    for k in (5, 20, 60, 200):
        ranks = [
            gold_rank(c, rrf([rank_bm25(c, q), rank_vector(c, emb, q)], k), le, ph)
            for g in SETS.values()
            for q, le, ph in g
        ]
        print(f"k={k}: {summarise(ranks)['mrr']:.3f}", end="  ")
    print("\nWeighted-fusion alpha sweep (MRR):", end=" ")
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        ranks = [
            gold_rank(c, weighted_fusion(c, emb, q, a), le, ph)
            for g in SETS.values()
            for q, le, ph in g
        ]
        print(f"a={a}: {summarise(ranks)['mrr']:.3f}", end="  ")
    sample = [c.texts[i] for i in range(20)]
    t0 = time.perf_counter()
    rr._score("how do I stop retry storms", sample)
    print(
        f"\n\nRe-ranking cost: {(time.perf_counter() - t0) * 1000:.0f} ms per query for 20 candidates on CPU "
        f"(cross-encoders can't be precomputed; that's why you rerank only a shortlist)."
    )


if __name__ == "__main__":
    main()
