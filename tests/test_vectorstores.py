"""Contract tests: every backend must behave identically through the VectorStore interface."""

from __future__ import annotations

import numpy as np
import pytest

from common.vectorstores import PgvectorStore, make_store

BACKENDS = ["numpy", "hnsw", "chroma", "qdrant", "pgvector"]
DIM, N = 32, 400


def _unit(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    X = _unit(rng.normal(size=(N, DIM)))
    meta = [{"week": int(i % 4), "kind": "even" if i % 2 == 0 else "odd"} for i in range(N)]
    return X, [f"id{i}" for i in range(N)], meta, [f"text {i}" for i in range(N)]


@pytest.fixture(params=BACKENDS)
def store(request, data):
    if request.param == "pgvector" and not PgvectorStore.available():
        pytest.skip("pgvector not running (docker compose up -d)")
    X, ids, meta, texts = data
    s = make_store(request.param, DIM)
    s.add(ids, X, meta, texts)
    yield s
    s.close()


def test_count_and_exact_self_match(store, data):
    X, ids, meta, texts = data
    assert store.count() == N
    hits = store.search(X[17], k=3)
    assert hits[0].id == "id17" and hits[0].score == pytest.approx(1.0, abs=1e-3)
    assert hits[0].text == "text 17" and hits[0].metadata["week"] == 17 % 4
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_scores_are_cosine_and_match_numpy_top1(store, data):
    X, *_ = data
    rng = np.random.default_rng(9)
    for _ in range(10):
        q = _unit(rng.normal(size=DIM))
        top = store.search(q, k=1)[0]
        exact = int(np.argmax(X @ q))
        assert top.id == f"id{exact}"
        assert top.score == pytest.approx(float(X[exact] @ q), abs=1e-3)


def test_filters_equality_and_conjunction(store, data):
    X, ids, meta, _ = data
    hits = store.search(X[0], k=10, where={"week": 2})
    assert hits and all(h.metadata["week"] == 2 for h in hits)
    both = store.search(X[0], k=10, where={"week": 2, "kind": "even"})
    assert both and all(h.metadata["week"] == 2 and h.metadata["kind"] == "even" for h in both)
    assert store.search(X[0], k=5, where={"week": 99}) == []


def test_k_larger_than_corpus_and_empty_filter(store, data):
    X, *_ = data
    assert len(store.search(X[1], k=10_000)) <= N
    assert len(store.search(X[1], k=5, where=None)) == 5


def test_filtered_search_matches_exact_numpy_on_most_queries(store, data):
    """ANN may differ slightly from exact, but recall on a filtered top-5 must be high."""
    X, ids, meta, _ = data
    exact = make_store("numpy", DIM)
    exact.add(ids, X, meta, [f"t{i}" for i in range(N)])
    rng = np.random.default_rng(5)
    overlap = []
    for _ in range(20):
        q = _unit(rng.normal(size=DIM))
        want = {h.id for h in exact.search(q, 5, {"week": 1})}
        got = {h.id for h in store.search(q, 5, {"week": 1})}
        overlap.append(len(want & got) / 5)
    assert np.mean(overlap) >= 0.9


def test_hnsw_selective_filter_degrades_instead_of_raising():
    """hnswlib raises RuntimeError when a filter leaves fewer than k reachable candidates."""
    import hnswlib

    rng = np.random.default_rng(0)
    X = _unit(rng.normal(size=(2000, 32)))
    keep = set(range(0, 2000, 400))  # 5 matching rows only
    idx = hnswlib.Index(space="cosine", dim=32)
    idx.init_index(2000, ef_construction=100, M=16)
    idx.add_items(X)
    idx.set_ef(50)
    with pytest.raises(RuntimeError):  # the raw library fails...
        idx.knn_query(X[1], k=10, filter=lambda i: i in keep)
    s = make_store("hnsw", 32, max_elements=2000)  # ...our adapter returns what exists
    s.add(
        [str(i) for i in range(2000)], X, [{"g": int(i in keep)} for i in range(2000)], [""] * 2000
    )
    assert len(s.search(X[1], k=10, where={"g": 1})) == 5
