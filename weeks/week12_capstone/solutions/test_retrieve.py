"""Tests for retrieval policy: week naming, reciprocal rank fusion, week scoping, the memo, and reranking."""

from __future__ import annotations

import pytest
from conftest import FakeIndex, FakeReranker
from copilot import retrieve as R


def test_weeks_named_finds_distinct_weeks_in_order_and_ignores_nonsense():
    assert R.weeks_named("How does Week 1 relate to Week 11?") == [1, 11]
    assert R.weeks_named("In weeks 4 and week 4 and Week 7") == [4, 7]
    assert R.weeks_named("Week 0, week 13, week 99 and a weekend") == []
    assert R.weeks_named("no week here") == []
    assert R.weeks_named("Week 1, Week 2, Week 3, Week 4", limit=3) == [1, 2, 3]
    assert R.weeks_named("WEEK 12 is the capstone") == [12]


def test_rrf_scores_by_reciprocal_rank_and_breaks_ties_by_first_appearance():
    assert R.rrf([["a", "b", "c"], ["a", "b", "c"]]) == ["a", "b", "c"]
    assert R.rrf([["a", "b"], ["b", "a"]]) == ["a", "b"]  # equal scores: first seen wins
    # by hand: a = 1/61 ; c = 1/63 + 1/61 -> c beats b (1/62) and ties nothing; c > a? c = 0.03226, a = 0.01639, b = 0.01613
    assert R.rrf([["a", "b", "c"], ["c"]]) == ["c", "a", "b"]
    assert R.rrf([]) == [] and R.rrf([[]]) == []


def test_a_plain_question_is_searched_once_globally(index):
    r = R.Retriever(index, R.RetrievalConfig(k=3))
    res = r.retrieve("What does the KV cache store?")
    assert res.weeks == [] and len(index.calls) == 1 and index.calls[0][2] is None
    assert [s.n for s in res.sources] == [1, 2, 3] and res.sources[
        0
    ].doc == "week01_x/day4_cache.md"
    assert res.top_cosine == max(s.cosine for s in res.sources)


def test_naming_a_week_adds_a_scoped_search_and_naming_two_represents_both(index):
    r = R.Retriever(index, R.RetrievalConfig(k=2))
    res = r.retrieve("How does the cache in Week 1 relate to paged blocks in Week 11?")
    assert res.weeks == [1, 11]
    assert [c[2] for c in index.calls] == [None, {"week": 1}, {"week": 11}]
    assert {s.week for s in res.sources} == {
        1,
        11,
    }  # each named week is represented even though k is tiny
    off = R.Retriever(
        FakeIndex([dict(c) for c in index.chunks]), R.RetrievalConfig(k=2, scope_by_week=False)
    )
    assert (
        off.retrieve("How does the cache in Week 1 relate to paged blocks in Week 11?").weeks == []
    )


def test_the_memo_serves_a_repeated_question_without_searching_again(index):
    r = R.Retriever(index, R.RetrievalConfig(memo_size=2))
    first = r.retrieve("What does the KV cache store?")
    n = len(index.calls)
    again = r.retrieve("  What does the KV cache store?  ")  # whitespace does not matter
    assert (
        again.cached
        and len(index.calls) == n
        and again.sources == first.sources
        and (r.hits, r.misses) == (1, 1)
    )
    r.retrieve("question two")
    r.retrieve("question three")  # evicts the oldest
    assert not r.retrieve("What does the KV cache store?").cached


def test_sources_carry_numbers_metadata_and_a_snippet(index):
    res = R.Retriever(index, R.RetrievalConfig(k=2)).retrieve("tokens price")
    s = res.sources[0]
    assert (s.n, s.doc, s.week, s.day) == (1, "week01_x/day2_tokens.md", 1, 2)
    d = s.as_dict(snippet=20)
    assert (
        d["n"] == 1
        and len(d["snippet"]) == 20
        and set(d) == {"n", "id", "doc", "heading", "snippet", "score"}
    )


def test_reranking_reorders_the_candidates_and_is_timed(index):
    rr = FakeReranker()
    cfg = R.RetrievalConfig(k=2, rerank=True, rerank_depth=4)
    res = R.Retriever(index, cfg, rr).retrieve("PagedAttention blocks memory")
    assert (
        rr.calls == 1
        and "rerank" in res.seconds
        and res.sources[0].doc == "week11_y/day2_batching.md"
    )
    assert R.Retriever(index, R.RetrievalConfig(k=2, rerank=False), rr).retrieve(
        "anything"
    ).seconds.keys() == {"search"}


def test_an_empty_index_gives_empty_sources():
    res = R.Retriever(FakeIndex([]), R.RetrievalConfig()).retrieve("anything")
    assert res.sources == [] and res.top_cosine == 0.0


@pytest.mark.parametrize("k", [1, 3, 10])
def test_k_bounds_the_number_of_sources(index, k):
    assert len(
        R.Retriever(index, R.RetrievalConfig(k=k)).retrieve("cache tokens blocks golden").sources
    ) == min(k, 4)


def test_two_named_weeks_are_both_represented_even_when_one_week_dominates_the_ranking():
    chunks = [
        {
            "id": f"a{i}",
            "doc": "week01_x/d.md",
            "week": 1,
            "day": 1,
            "heading": "h",
            "text": "cache keys values tokens stored reused " * 2,
        }
        for i in range(5)
    ] + [
        {
            "id": "b1",
            "doc": "week11_y/d.md",
            "week": 11,
            "day": 1,
            "heading": "h",
            "text": "paged blocks",
        }
    ]
    res = R.Retriever(FakeIndex(chunks), R.RetrievalConfig(k=2, per_scope=2)).retrieve(
        "cache keys values tokens stored reused paged blocks Week 1 Week 11"
    )
    assert {s.week for s in res.sources} == {
        1,
        11,
    }  # without the per-week guarantee both slots go to week 1


def test_the_memo_keeps_exactly_memo_size_questions():
    idx = FakeIndex([dict(c) for c in __import__("conftest").CHUNKS])
    r = R.Retriever(idx, R.RetrievalConfig(memo_size=2))
    r.retrieve("first question about cache")
    r.retrieve("second question about tokens")
    assert (
        r.retrieve("first question about cache").cached
        and r.retrieve("second question about tokens").cached
    )
