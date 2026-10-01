"""Tests for common/judges.py: claim splitting, lexical support, windows, aggregation, and the real NLI model."""

from __future__ import annotations

import pytest

from common.judges import (
    Claim,
    NLIJudge,
    aggregate,
    lexical_support,
    split_claims,
    windows,
)


def test_split_claims_drops_citations_and_fragments():
    ans = "Use backoff with jitter [1]. Honour the Retry-After header. Ok. - Also cap attempts at three [2]."
    assert split_claims(ans) == [
        "Use backoff with jitter.",
        "Honour the Retry-After header.",
        "Also cap attempts at three.",
    ]
    assert split_claims("Hi. Yes.") == [] and split_claims("") == []


def test_lexical_support_ranges_and_blindness_to_swaps():
    ctx = "Heading-aware chunkers keep answers whole 93% of the time while fixed cuts keep 40%."
    assert lexical_support(
        ctx, "Heading-aware chunkers keep answers whole 93% of the time."
    ) == pytest.approx(1.0)
    assert lexical_support(ctx, "Quantum entanglement powers submarines.") < 0.2
    swapped = "Fixed cuts keep answers whole 93% of the time while heading-aware chunkers keep 40%."
    assert lexical_support(ctx, swapped) == pytest.approx(1.0), (
        "the lexical judge cannot see a swap"
    )
    assert lexical_support(ctx, "") == 0.0


def test_windows_cover_long_context_and_keep_short_ones_whole():
    short = "One sentence here. Another one."
    assert windows(short) == [short]
    long = " ".join(f"Sentence number {i} is here." for i in range(10))
    ws = windows(long)
    assert len(ws) > 1 and all(w.count("Sentence") <= 3 for w in ws)
    assert "Sentence number 9" in " ".join(ws)


def test_aggregate_is_only_as_strong_as_the_weakest_claim():
    good = Claim("a", 0.95, 0.01, 0.04)
    bad = Claim("b", 0.10, 0.85, 0.05)
    meh = Claim("c", 0.30, 0.10, 0.60)
    assert aggregate([good, good]).score == 0.95 and not aggregate([good, good]).contradicted
    r = aggregate([good, bad])
    assert r.score == 0.10 and r.contradicted
    assert (
        aggregate([good, meh]).score == 0.30 and not aggregate([good, meh]).contradicted
    )  # unsupported != contradicted
    assert aggregate([]).score == 0.0


@pytest.fixture(scope="module")
def judge():
    return NLIJudge()


def test_nli_entails_paraphrase_contradicts_opposite_and_is_neutral_on_unrelated(judge):
    ctx = "Use exponential backoff with jitter when the API returns HTTP 429."
    c, e, n = judge.classify(ctx, "Clients should back off exponentially and add jitter.")
    assert e > 0.7 and c < 0.2
    c, e, n = judge.classify(ctx, "Clients should retry immediately without waiting.")
    assert c > 0.7 and e < 0.2
    c, e, n = judge.classify(ctx, "The sky is blue.")
    assert n > 0.7


def test_nli_faithfulness_catches_a_swap_that_lexical_support_misses(judge):
    ctx = (
        "Heading-aware and semantic chunkers keep the answer's paragraph whole 93% of the time; "
        "naive fixed cuts do so only 40%."
    )
    faithful = (
        "Heading-aware chunkers keep answer paragraphs whole 93% of the time, fixed cuts only 40%."
    )
    swapped = (
        "Fixed cuts keep answer paragraphs whole 93% of the time, heading-aware chunkers only 40%."
    )
    rf, rs = judge.faithfulness(ctx, faithful), judge.faithfulness(ctx, swapped)
    assert rf.score > 0.5 > rs.score
    assert lexical_support(ctx, swapped) > 0.8  # ...while the lexical judge is fooled
    assert judge.classify(ctx, swapped) is judge.classify(ctx, swapped)  # cached
