"""Tests for common/crag.py with stubbed index and reranker (no models)."""

from __future__ import annotations

from common.crag import corrective_search, grade
from common.vectorstores import Hit


class StubIndex:
    """search(query, k, alpha) -> hits looked up by (query, alpha) in a script; counts calls."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def search(self, query, k=5, alpha=0.5, **_):
        self.calls.append((query, alpha))
        return [
            Hit(i, 0.0, {"doc": "d"}, f"text-{i}") for i in self.script.get((query, alpha), [])
        ][:k]


class StubReranker:
    """score = relevance table by chunk text, recording the query each pair was scored against."""

    def __init__(self, table):
        self.table, self.queries = table, []

    def scores(self, query, passages):
        self.queries += [query] * len(passages)
        return [self.table.get(p, -10.0) for p in passages]


def test_grade_thresholds():
    assert grade(5, 1) == "correct" and grade(0.5, 1) == "incorrect"
    assert (
        grade(0.5, 1, 0.0) == "ambiguous"
        and grade(-1, 1, 0.0) == "incorrect"
        and grade(1, 1) == "correct"
    )


def test_confident_first_retrieval_returns_immediately_with_zero_extra_cost():
    idx = StubIndex({("q", 0.5): ["a", "b"]})
    rr = StubReranker({"text-a": 3.0, "text-b": 0.0})
    res = corrective_search(idx, "q", rr, tau=1.0, rewriter=lambda q: ["never used"], k=2)
    assert (
        res.grade == "correct"
        and not res.corrected
        and res.retrievals == 1
        and idx.calls == [("q", 0.5)]
    )
    assert [h.id for h in res.hits] == ["a", "b"] and len(res.steps) == 1


def test_low_confidence_triggers_rewrite_and_rescues_the_answer():
    idx = StubIndex({("q", 0.5): ["a", "b"], ("better q", 0.5): ["c"]})
    rr = StubReranker({"text-a": -3.0, "text-b": -4.0, "text-c": 2.0})
    res = corrective_search(idx, "q", rr, tau=1.0, rewriter=lambda q: ["better q"], k=2)
    assert res.corrected and res.grade == "correct" and res.hits[0].id == "c"
    assert [s.action for s in res.steps] == ["initial", "rewrite"] and res.retrievals == 2
    assert set(rr.queries) == {"q"}, (
        "every pair is scored against the ORIGINAL question, never the rewrite"
    )
    assert {h.id for h in res.hits} == {"c", "a"}, "original candidates stay in the pool"


def test_ladder_runs_in_order_and_stops_at_the_first_correct_result():
    idx = StubIndex(
        {
            ("q", 0.5): ["a"],
            ("r1", 0.5): ["b"],
            ("r2", 0.5): ["c"],
            ("q", 0.0): ["d"],
            ("q", 1.0): ["e"],
        }
    )
    rr = StubReranker({"text-a": -5, "text-b": -4, "text-c": -3, "text-d": 2.5, "text-e": 9})
    res = corrective_search(
        idx, "q", rr, tau=2.0, rewriter=lambda q: ["r1", "r2"], k=3, max_steps=9
    )
    assert [s.action for s in res.steps] == ["initial", "rewrite", "rewrite", "alt_strategy"]
    assert set(rr.queries) == {"q"}, (
        "even with several rewrites, every pair is scored against the original"
    )
    assert idx.calls[-1] == ("q", 0.0), (
        "stopped after BM25-only found a correct result; vector-only never ran"
    )
    assert res.hits[0].id == "d"


def test_max_steps_bounds_the_work_and_result_is_never_worse_than_the_start():
    idx = StubIndex({("q", 0.5): ["a", "b"], ("r1", 0.5): ["c"], ("r2", 0.5): ["d"]})
    rr = StubReranker({"text-a": -1.0, "text-b": -2.0, "text-c": -9.0, "text-d": -9.0})
    res = corrective_search(
        idx, "q", rr, tau=5.0, rewriter=lambda q: ["r1", "r2"], k=2, max_steps=3
    )
    assert len(res.steps) == 3 and res.retrievals == 3 and res.grade == "incorrect"
    assert [h.id for h in res.hits] == ["a", "b"], (
        "bad rewrites must not displace the original best chunks"
    )
    one = corrective_search(StubIndex({("q", 0.5): ["a"]}), "q", rr, tau=5.0, max_steps=1)
    assert len(one.steps) == 1 and one.retrievals == 1


def test_works_without_a_rewriter_and_with_an_empty_index():
    idx = StubIndex({("q", 0.5): ["a"], ("q", 1.0): ["z"]})
    rr = StubReranker({"text-a": -1, "text-z": 4})
    res = corrective_search(idx, "q", rr, tau=2.0)
    assert [s.action for s in res.steps] == ["initial", "alt_strategy", "alt_strategy"]
    assert res.hits[0].id == "z"
    empty = corrective_search(StubIndex({}), "q", rr, tau=2.0)
    assert empty.hits == [] and empty.grade == "incorrect"


def test_rerank_pair_accounting_counts_each_chunk_once():
    idx = StubIndex({("q", 0.5): ["a", "b"], ("r", 0.5): ["b", "c"]})
    rr = StubReranker({"text-a": -1, "text-b": -1, "text-c": -1})
    res = corrective_search(idx, "q", rr, tau=9.0, rewriter=lambda q: ["r"], k=3, max_steps=2)
    assert res.rerank_pairs == 3 and len(rr.queries) == 3
