"""Week 3 Day 1 - Solution: semantic search in pure numpy, versus a keyword baseline.

Corpus: every prose sentence of the Week 1-2 lessons (pinned, so numbers are reproducible).
Queries are deliberately PARAPHRASED (little word overlap with the lessons) and graded at file
level: a query "hits" if the top-k sentences include one from an acceptable lesson.

  uv run python weeks/week03_embeddings-and-rag/solutions/day1_solution.py
  EMBED_PROVIDER=hash uv run python ...day1_solution.py     # no model download; NOT semantic
"""

from __future__ import annotations

import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.corpus import load_course_docs, split_sentences  # noqa: E402
from common.embed import cosine_top_k, get_embedder  # noqa: E402

PINNED_WEEKS = ("week01", "week02")

QUERIES: list[tuple[str, set[str]]] = [
    ("how do I stop my app from hammering an API that keeps failing", {"week01/day5"}),
    ("why can't a language model count the letters in a word", {"week01/day2"}),
    ("how much GPU memory does remembering a long conversation take", {"week01/day4"}),
    ("what setting makes the output more random or more deterministic", {"week01/day3"}),
    ("ways to get machine-readable JSON out of a model without parse errors", {"week02/day3"}),
    (
        "keep a chat from growing too expensive while still remembering the user's name",
        {"week02/day5", "week01/day4"},
    ),
    ("asking the same question several times and voting to be more accurate", {"week02/day2"}),
    ("send each support ticket to a specialised handler", {"week02/day4"}),
    ("how to split data so I don't fool myself when tuning a prompt", {"week02/day6"}),
    ("picking the cheapest model that is still good enough", {"week01/day6"}),
    ("reuse the start of a prompt across requests to save money", {"week01/day4"}),
    ("what should a well written prompt contain", {"week02/day1"}),
    ("can I trust the probabilities a model assigns to its words", {"week01/day2"}),
    ("running many requests at once without hitting rate limits", {"week01/day5"}),
    ("why do small changes in the prompt text change token counts and cost", {"week01/day2"}),
]


def build_corpus():
    docs = [d for d in load_course_docs() if d.short.startswith(PINNED_WEEKS)]
    sents, owner, titled = [], [], []
    for d in docs:
        for s in split_sentences(d.text):
            sents.append(s)
            owner.append(d.short)
            titled.append(f"{d.title}: {s}")  # same sentence, plus the context it came from
    return sents, owner, titled


# ----------------------------------------------------------------- the two retrievers


def semantic_ranking(emb, doc_vecs, query: str, use_prefix: bool = True) -> list[int]:
    q = emb.embed_query(query) if use_prefix else emb.embed_documents([query])[0]
    return [i for i, _ in cosine_top_k(doc_vecs, q, k=len(doc_vecs))]


class KeywordIndex:
    """TF-IDF-weighted word overlap: the 'old' lexical baseline (no semantics, no stemming)."""

    def __init__(self, sentences: list[str]):
        self.tok = [self._tokens(s) for s in sentences]
        df = Counter(w for t in self.tok for w in set(t))
        n = len(sentences)
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}

    @staticmethod
    def _tokens(s: str) -> list[str]:
        stop = {
            "the",
            "a",
            "an",
            "of",
            "to",
            "and",
            "in",
            "is",
            "it",
            "for",
            "on",
            "i",
            "do",
            "how",
            "what",
            "can",
            "my",
            "with",
            "that",
            "this",
            "you",
            "are",
            "be",
            "or",
            "so",
            "at",
        }
        return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in stop]

    def ranking(self, query: str) -> list[int]:
        q = set(self._tokens(query))
        scores = np.array([sum(self.idf.get(w, 0) for w in q & set(t)) for t in self.tok])
        return list(np.argsort(-scores, kind="stable"))


# ----------------------------------------------------------------- evaluation


def evaluate(rank_fn, owner: list[str], queries=QUERIES, k: int = 5) -> dict:
    hit1 = hitk = rr = 0.0
    misses = []
    for q, ok in queries:
        files = [owner[i] for i in rank_fn(q)]
        first = next((r for r, f in enumerate(files, 1) if f in ok), None)
        hit1 += first == 1
        hitk += first is not None and first <= k
        rr += 1 / first if first and first <= 50 else 0
        if first is None or first > k:
            misses.append((q, files[0]))
    n = len(queries)
    return {"hit@1": hit1 / n, f"hit@{k}": hitk / n, "mrr": rr / n, "misses": misses}


def show(name: str, r: dict, k: int = 5) -> None:
    print(f"{name:<34} hit@1 {r['hit@1']:.0%}   hit@{k} {r[f'hit@{k}']:.0%}   MRR {r['mrr']:.2f}")


def main() -> None:
    sents, owner, titled = build_corpus()
    emb = get_embedder()
    print(
        f"corpus: {len(sents)} sentences from {len(set(owner))} lessons | embedder: {emb.name} "
        f"({emb.dim} dims)\n"
    )

    t0 = time.perf_counter()
    doc_vecs = emb.embed_documents(sents)
    print(
        f"embedded corpus in {time.perf_counter() - t0:.1f}s -> matrix {doc_vecs.shape}, "
        f"{doc_vecs.nbytes / 1e6:.1f} MB (n x d x 4 bytes)"
    )
    emb.embed_query("warm up")
    t0 = time.perf_counter()
    for q, _ in QUERIES:
        cosine_top_k(doc_vecs, emb.embed_query(q), 5)
    per_query = (time.perf_counter() - t0) / len(QUERIES) * 1000
    t0 = time.perf_counter()
    qv = emb.embed_query(QUERIES[0][0])
    t_embed = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(200):
        cosine_top_k(doc_vecs, qv, 5)
    t_search = (time.perf_counter() - t0) / 200 * 1000
    print(
        f"per query: embed {t_embed * 1000:.1f} ms (cached after first), brute-force search over "
        f"{len(sents)} vectors {t_search:.2f} ms  [{per_query:.1f} ms avg incl. embedding]\n"
    )

    # 1) the property that makes everything simple: normalised vectors -> dot == cosine
    a, b = doc_vecs[0], doc_vecs[1]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(
        f"norm of a stored vector = {np.linalg.norm(a):.4f};  dot == cosine: {float(a @ b):.4f} vs {cos:.4f}\n"
    )

    # 2) semantic vs keyword retrieval
    kw = KeywordIndex(sents)
    print(f"{len(QUERIES)} paraphrased queries, file-level grading")
    show("keyword baseline (TF-IDF overlap)", evaluate(kw.ranking, owner))
    sem = evaluate(lambda q: semantic_ranking(emb, doc_vecs, q), owner)
    show("semantic (embeddings, numpy)", sem)
    nopre = evaluate(lambda q: semantic_ranking(emb, doc_vecs, q, use_prefix=False), owner)
    show("semantic, query WITHOUT bge prefix", nopre)
    for q, got in sem["misses"]:
        print(f"   semantic miss: {q!r} -> top hit came from {got}")

    # 2b) a sentence alone is context-poor; prepend the lesson title and re-embed
    titled_vecs = emb.embed_documents(titled)
    show(
        "semantic, title + sentence",
        evaluate(lambda q: semantic_ranking(emb, titled_vecs, q), owner),
    )
    print(
        "   -> no gain: a lesson title is too coarse to disambiguate a sentence. Week 4's 'contextual\n"
        "      retrieval' has an LLM write chunk-specific context, which is a different thing."
    )

    # 3) what the top results look like
    print("\nTop-3 for: 'how do I stop my app from hammering an API that keeps failing'")
    for i, s in cosine_top_k(doc_vecs, emb.embed_query(QUERIES[0][0]), 3):
        print(f"   {s:.3f} [{owner[i]}] {s[:100] if isinstance(s, str) else sents[i][:100]}")

    # 4) similarity is not relevance, and not negation-aware
    pairs = [
        ("The service is available.", "The service is not available."),
        ("The service is available.", "The weather is lovely today."),
        ("Revenue rose 20 percent.", "Revenue fell 20 percent."),
    ]
    print("\nSimilarity is not truth:")
    for x, y in pairs:
        vx, vy = emb.embed_documents([x, y])
        print(f"   {float(vx @ vy):.3f}  {x!r} vs {y!r}")


if __name__ == "__main__":
    main()
