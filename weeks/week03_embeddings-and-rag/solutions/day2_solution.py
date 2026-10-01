"""Week 3 Day 2 - Solution: the same corpus in five vector stores, compared.

1. Agreement with exact search (recall@10) on real queries
2. Metadata filtering: correctness and result counts, including a *selective* filter
3. Build time and per-query latency on the real corpus (808 sentences: everything is fast)
4. A SCALE experiment on 100k synthetic vectors: brute force vs HNSW, and the ef knob

uv run python weeks/week03_embeddings-and-rag/solutions/day2_solution.py
(pgvector is skipped unless Postgres is up:  docker compose -f weeks/week03_embeddings-and-rag/docker-compose.yml up -d)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from day1_solution import QUERIES, build_corpus  # noqa: E402

from common.embed import get_embedder  # noqa: E402
from common.vectorstores import HnswStore, NumpyStore, PgvectorStore, make_store  # noqa: E402

K = 10


def timed(fn, repeat: int = 1):
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    return out, (time.perf_counter() - t0) / repeat


def part_real_corpus() -> None:
    sents, owner, _ = build_corpus()
    emb = get_embedder()
    vecs = emb.embed_documents(sents)
    qvecs = [emb.embed_query(q) for q, _ in QUERIES]
    ids = [f"s{i}" for i in range(len(sents))]
    metas = [{"lesson": o, "week": int(o[4:6])} for o in owner]
    print(f"corpus: {len(sents)} sentences, {vecs.shape[1]} dims, {len(set(owner))} lessons\n")

    names = ["numpy", "hnsw", "chroma", "qdrant"] + (
        ["pgvector"] if PgvectorStore.available() else []
    )
    if "pgvector" not in names:
        print("(pgvector skipped: Postgres not reachable on localhost:55432)\n")

    stores = {}
    print(f"{'store':<9} {'build ms':>9} {'query ms':>9} {'recall@10 vs exact':>19}")
    exact = NumpyStore(vecs.shape[1])
    exact.add(ids, vecs, metas, sents)
    truth = [{h.id for h in exact.search(q, K)} for q in qvecs]
    for name in names:
        s = make_store(name, vecs.shape[1])
        _, build = timed(lambda s=s: s.add(ids, vecs, metas, sents))
        _, per_query = timed(lambda s=s: [s.search(q, K) for q in qvecs], repeat=5)
        recall = np.mean(
            [len({h.id for h in s.search(q, K)} & t) / K for q, t in zip(qvecs, truth, strict=True)]
        )
        print(
            f"{name:<9} {build * 1000:>9.1f} {per_query / len(qvecs) * 1000:>9.2f} {recall:>19.0%}"
        )
        stores[name] = s

    # --- filtering: a selective filter is where naive implementations break
    print("\nFiltered search: query = QUERIES[4] (structured output), top-10")
    q = qvecs[4]
    for label, where in [
        ("week == 2 (about half)", {"week": 2}),
        ("one lesson (very selective)", {"lesson": "week02/day6"}),
        ("nonexistent lesson", {"lesson": "week99/day9"}),
    ]:
        total = sum(1 for m in metas if all(m[k] == v for k, v in where.items()))
        row = [f"{label:<30} ({total:>3} matching rows):"]
        for name, s in stores.items():
            hits = s.search(q, K, where)
            ok = all(all(h.metadata.get(k) == v for k, v in where.items()) for h in hits)
            row.append(f"{name}={len(hits)}{'' if ok else '!!WRONG'}")
        print("  " + " ".join(row))
    print(
        "  (expected: min(10, matching rows) results per store; fewer than that = recall lost to the filter)"
    )
    for s in stores.values():
        s.close()


def synthetic(kind: str, n: int, dim: int, n_queries: int, seed: int = 0):
    """'uniform': random directions (ANN's worst case). 'clustered': 500 topics + noise, which is
    much closer to how real text embeddings are distributed (queries come from the same topics)."""
    rng = np.random.default_rng(seed)
    unit = lambda a: (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)  # noqa: E731
    if kind == "uniform":
        return unit(rng.normal(size=(n, dim))), unit(rng.normal(size=(n_queries, dim)))
    centers = unit(rng.normal(size=(500, dim)))
    make = lambda m: unit(centers[rng.integers(0, 500, m)] + 0.5 * unit(rng.normal(size=(m, dim))))  # noqa: E731
    return make(n), make(n_queries)


def part_scale(n: int = 50_000, dim: int = 384, n_queries: int = 200) -> None:
    for kind in ("uniform", "clustered"):
        scale_one(kind, n, dim, n_queries)
    print(
        "\n-> ef trades recall for speed at query time; build time is the up-front price. ANN works by exploiting\n"
        "   structure: on structureless random vectors in 384 dims almost every point is 'equally far', and\n"
        "   recall collapses. Real embeddings cluster, so measure recall on YOUR data before trusting defaults."
    )


def scale_one(kind: str, n: int, dim: int, n_queries: int) -> None:
    print(f"\n=== SCALE: {n:,} {kind} unit vectors x {dim} dims ({n * dim * 4 / 1e6:.0f} MB) ===")
    X, Q = synthetic(kind, n, dim, n_queries)
    ids = [str(i) for i in range(n)]
    exact = NumpyStore(dim)
    exact.add(ids, X, [{}] * n, [""] * n)
    truth, t_exact = timed(lambda: [{h.id for h in exact.search(q, K)} for q in Q])
    print(f"{'index':<22} {'build s':>8} {'ms/query':>9} {'recall@10':>10}")
    print(f"{'numpy (exact)':<22} {'0.0':>8} {t_exact / n_queries * 1000:>9.2f} {'100%':>10}")
    hnsw = HnswStore(dim, m=16, ef_construction=100, max_elements=n)
    _, build = timed(lambda: hnsw.add(ids, X, [{}] * n, [""] * n))
    for ef in (16, 64, 256):
        hnsw.set_ef(ef)
        res, t = timed(lambda: [{h.id for h in hnsw.search(q, K)} for q in Q])
        recall = np.mean([len(r & t_) / K for r, t_ in zip(res, truth, strict=True)])
        print(
            f"{'hnsw M=16 ef=' + str(ef):<22} {build:>8.1f} {t / n_queries * 1000:>9.2f} {recall:>10.0%}"
        )


if __name__ == "__main__":
    part_real_corpus()
    part_scale()
