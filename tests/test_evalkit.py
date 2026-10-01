"""Tests for common/evalkit.py: metric maths, bootstrap statistics, kappa, golden-set I/O."""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.evalkit import (
    GoldQuery,
    bootstrap_ci,
    cohens_kappa,
    confusion,
    dcg,
    evaluate_retriever,
    hit_at_k,
    load_golden,
    ndcg_at_k,
    norm,
    paired_bootstrap,
    precision_at_k,
    ranks_of_relevant,
    recall_at_k,
    reciprocal_rank,
    save_golden,
)


def test_rank_metrics_on_known_lists():
    rel = [False, False, True, False, True]
    assert ranks_of_relevant(rel) == [3, 5]
    assert hit_at_k(rel, 2) == 0.0 and hit_at_k(rel, 3) == 1.0
    assert reciprocal_rank(rel) == pytest.approx(1 / 3) and reciprocal_rank([False] * 4) == 0.0
    assert reciprocal_rank(rel, cutoff=2) == 0.0
    assert recall_at_k(rel, 5, 4) == 0.5 and recall_at_k(rel, 3, 1) == 1.0
    assert precision_at_k(rel, 4) == 0.25 and precision_at_k([], 3) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(rel, 5, 0)


def test_ndcg_properties():
    assert ndcg_at_k([1, 0, 0], 3) == pytest.approx(1.0)  # best possible order
    assert ndcg_at_k([0, 0, 1], 3) == pytest.approx(0.5)  # relevant at rank 3: 1/log2(4) = 0.5
    assert ndcg_at_k([0, 0, 0], 3) == 0.0
    assert (
        ndcg_at_k([0, 1, 2], 3) < ndcg_at_k([2, 1, 0], 3) == pytest.approx(1.0)
    )  # graded: order matters
    assert dcg([1], 1) == pytest.approx(1.0) and dcg([3], 1) == pytest.approx(7.0)
    assert (
        ndcg_at_k([1, 1, 0], 3, ideal_gains=[1, 1, 1]) < 1.0
    )  # a relevant item we never retrieved
    assert ndcg_at_k([1, 1], 2, ideal_gains=[1, 1, 1]) == pytest.approx(
        1.0
    )  # ...unless it is beyond k


def test_bootstrap_ci_contains_mean_and_shrinks_with_data():
    rng = np.random.default_rng(0)
    small = rng.binomial(1, 0.6, 20)
    large = rng.binomial(1, 0.6, 400)
    m, lo, hi = bootstrap_ci(small)
    assert lo <= m <= hi and m == pytest.approx(small.mean())
    assert (hi - lo) > (bootstrap_ci(large)[2] - bootstrap_ci(large)[1])
    assert bootstrap_ci([1.0] * 30) == (1.0, 1.0, 1.0)  # no variance, no uncertainty
    assert all(math.isnan(x) for x in bootstrap_ci([]))
    assert bootstrap_ci(small, seed=1) == bootstrap_ci(small, seed=1)  # deterministic


def test_the_ci_coverage_is_roughly_nominal():
    """About 95% of intervals from repeated samples should contain the true mean."""
    rng = np.random.default_rng(1)
    hits = 0
    trials = 150
    for t in range(trials):
        sample = rng.binomial(1, 0.3, 40)
        _, lo, hi = bootstrap_ci(sample, n_boot=500, seed=t)
        hits += lo <= 0.3 <= hi
    assert 0.88 <= hits / trials <= 1.0


def test_paired_bootstrap_detects_real_differences_and_ignores_noise():
    a = [1.0] * 25 + [0.0] * 5
    b = [0.0] * 20 + [1.0] * 5 + [0.0] * 5
    r = paired_bootstrap(a, b)
    assert (
        r["diff"] > 0.3
        and r["p"] < 0.01
        and r["ci_low"] > 0
        and r["wins"] == 20
        and r["losses"] == 0
    )
    same = paired_bootstrap(a, a)
    assert same["diff"] == 0 and same["ties"] == 30 and same["p"] == 1.0
    rng = np.random.default_rng(3)
    x = rng.binomial(1, 0.5, 40).astype(float)
    y = rng.binomial(1, 0.5, 40).astype(float)
    assert paired_bootstrap(x, y)["p"] > 0.05  # two coin flips are not "different"
    with pytest.raises(ValueError):
        paired_bootstrap([], [])


def test_pairing_is_more_sensitive_than_overlapping_unpaired_intervals():
    """Queries differ a lot in difficulty; B is always slightly worse than A on the same query."""
    rng = np.random.default_rng(5)
    base = rng.uniform(0.2, 0.9, 40)
    a, b = base, base - 0.05
    ia, ib = bootstrap_ci(a), bootstrap_ci(b)
    assert ia[1] < ib[2] and ib[1] < ia[2], "separate intervals overlap heavily"
    assert paired_bootstrap(a, b)["p"] < 0.01, "but the paired test sees the consistent gap"


def test_cohens_kappa_and_confusion():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    assert cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)
    assert cohens_kappa(["a"] * 5, ["a"] * 5) == 1.0
    assert confusion([1, 1, 0], [1, 0, 0]) == {(1, 1): 1, (1, 0): 1, (0, 0): 1}
    with pytest.raises(ValueError):
        cohens_kappa([], [])


def test_gold_relevance_is_text_based_and_survives_rechunking():
    g = GoldQuery("q1", "how?", "week01/day5", "unbounded gather = stampede")
    assert g.is_relevant("week01/day5", "# Heading\n\n**Unbounded `gather` = stampede.** More text")
    assert not g.is_relevant("week01/day4", "unbounded gather = stampede")
    assert not g.is_relevant("week01/day5", "something else entirely")
    assert norm("A  *B*\n`c`") == "a b c"


def test_golden_jsonl_round_trip(tmp_path):
    qs = [
        GoldQuery("a", "q1", "d1", "p1", tags=["x"]),
        GoldQuery("b", "q2", "d2", "p2", kind="synthetic"),
    ]
    save_golden(tmp_path / "g.jsonl", qs)
    assert load_golden(tmp_path / "g.jsonl") == qs


def test_evaluate_retriever_end_to_end():
    gold = [
        GoldQuery("1", "alpha?", "d1", "alpha fact"),
        GoldQuery("2", "beta?", "d2", "beta fact"),
    ]
    corpus = {
        "alpha?": [("d9", "noise"), ("d1", "the alpha fact is here")],
        "beta?": [("d2", "beta fact"), ("d2", "also beta fact")],
    }
    res = evaluate_retriever("toy", lambda q: corpus[q], gold)
    assert res.first_rank == [2, 1] and res.hit1 == [0.0, 1.0] and res.hit5 == [1.0, 1.0]
    assert res.mrr == [0.5, 1.0]
    assert res.ndcg5[1] == pytest.approx(1.0)  # two relevant at ranks 1-2, ideal has two
    assert res.ndcg5[0] == pytest.approx(1 / math.log2(3))
    s = res.summary()
    assert set(s) == {"hit@1", "hit@5", "MRR", "nDCG@5"} and s["hit@5"][0] == 1.0
