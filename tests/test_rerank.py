from __future__ import annotations

import pytest

from common.rerank import LocalReranker, OverlapReranker


def test_overlap_reranker_orders_and_top_k():
    rr = OverlapReranker()
    docs = ["cats purr", "retry with backoff and jitter", "backoff"]
    order = rr.rerank("retry backoff jitter", docs, top_k=2)
    assert [i for i, _ in order] == [1, 2] and len(order) == 2


def test_cache_avoids_recomputation(tmp_path):
    class Counting(OverlapReranker):
        pass

    from common import rerank as r

    rr = Counting()
    rr.db = r.sqlite3.connect(tmp_path / "c.sqlite")
    rr.db.execute("CREATE TABLE IF NOT EXISTS s (k TEXT PRIMARY KEY, v REAL)")
    a = rr.scores("q words", ["alpha q", "beta words"])
    first = rr.calls
    b = rr.scores("q words", ["alpha q", "beta words", "gamma"])
    assert a == b[:2] and first == 2 and rr.calls == 3


@pytest.fixture(scope="module")
def local():
    return LocalReranker(cache_path=None)


def test_local_cross_encoder_ranks_relevant_passage_first(local):
    docs = [
        "Paris is the capital of France.",
        "Use exponential backoff with jitter to avoid retry storms.",
        "The KV cache stores keys and values for previous tokens.",
    ]
    order = local.rerank("how do I avoid retry storms when an API fails", docs)
    assert order[0][0] == 1 and order[0][1] > order[1][1]
    assert all(s == s for _, s in order), "scores must not be NaN"
