"""Tests for Day 5's fusion logic (pure functions, no models needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from day5_solution import Corpus, gold_rank, rrf, summarise, tokenize, weighted_fusion  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

from common.embed import HashEmbedder  # noqa: E402


def test_rrf_rewards_agreement_over_a_single_first_place():
    a = [0, 1, 2, 3]  # ranker A
    b = [1, 0, 3, 2]  # ranker B
    assert rrf([a, b])[:2] in ([0, 1], [1, 0])  # docs both rankers like float to the top
    c = [5, 6, 0, 1]  # a ranker that loves 5 but A and B never mention it high
    assert rrf([a, b, [1, 0, 2, 3]])[0] in (0, 1)
    assert rrf([c, a])[0] in (5, 0) and set(rrf([c, a])) == {0, 1, 2, 3, 5, 6}


def test_rrf_uses_ranks_not_scores_and_k_controls_steepness():
    only_first = [[7, 8, 9]]
    assert rrf(only_first) == [7, 8, 9]
    small, large = rrf([[1, 2], [2, 1]], k=1), rrf([[1, 2], [2, 1]], k=1000)
    assert set(small) == set(large) == {1, 2}


def test_rrf_ties_are_deterministic():
    assert rrf([[1, 2], [2, 1]]) == rrf([[1, 2], [2, 1]])


def test_tokenizer_keeps_identifiers_whole_and_splits_dots():
    assert tokenize("Use model_validator with asyncio.Semaphore!") == [
        "use",
        "model_validator",
        "with",
        "asyncio",
        "semaphore",
    ]
    assert tokenize("o200k_base --mock") == ["o200k_base", "mock"]


def make_corpus():
    texts = [
        "use model_validator for cross field checks",
        "cats like sleeping on sofas",
        "backoff with jitter prevents retry storms",
    ]
    emb = HashEmbedder()
    return Corpus(
        texts,
        ["w/d1", "w/d2", "w/d3"],
        emb.embed_documents(texts),
        BM25Okapi([tokenize(t) for t in texts]),
    ), emb


def test_gold_rank_requires_right_lesson_and_phrase():
    c, _ = make_corpus()
    assert gold_rank(c, [2, 0, 1], "w/d1", "model_validator") == 2
    assert gold_rank(c, [2, 0, 1], "w/d3", "model_validator") is None  # right phrase, wrong lesson
    assert gold_rank(c, [0, 1, 2], "w/d1", "not present") is None


def test_weighted_fusion_extremes_match_single_rankers():
    c, emb = make_corpus()
    q = "retry jitter backoff"
    bm_only = list(np.argsort(-c.bm25.get_scores(tokenize(q)), kind="stable"))
    assert weighted_fusion(c, emb, q, alpha=0.0) == bm_only
    vec_only = list(np.argsort(-(c.vecs @ emb.embed_query(q)), kind="stable"))
    assert weighted_fusion(c, emb, q, alpha=1.0) == vec_only
    assert weighted_fusion(c, emb, q, 0.5)[0] == 2


def test_summarise_metrics():
    r = summarise([1, None, 3, 7])
    assert r["hit@1"] == 0.25 and r["hit@5"] == 0.5
    assert r["mrr"] == pytest.approx((1 + 0 + 1 / 3 + 1 / 7) / 4)
