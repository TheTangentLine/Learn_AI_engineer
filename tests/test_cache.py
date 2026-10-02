"""Tests for common/cache.py: a cache must never serve a wrong answer, so most of these test REFUSALS."""

from __future__ import annotations

import threading

import pytest

from common.cache import ResponseCache, entities, has_negation, jaccard, normalize, words


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# ----------------------------------------------------------------------------- normalisation


@pytest.mark.parametrize(
    "a,b",
    [
        ("How do I reset my password?", "how do i reset my password"),
        ("  How   do I\treset my password ?!", "How do I reset my password"),
        ("Reset my PASSWORD, please.", "reset my password please"),
        ("Ｒeset my password", "reset my password"),  # full-width letter: NFKC
    ],
)
def test_normalisation_makes_surface_variants_equal(a, b):
    assert normalize(a) == normalize(b)


def test_normalisation_keeps_ids_and_numbers_intact():
    assert normalize("Refund INV-1001, please!") == "refund inv-1001 please"
    assert normalize("Version 3.14 failed.") == "version 3.14 failed"
    assert normalize("mail bob@example.com now") == "mail bob@example.com now"
    assert normalize("error 0x5F -- help") == "error 0x5f help"
    assert normalize("INV-1001") != normalize("INV-1002")


def test_entities_ids_numbers_and_emails():
    assert entities("Refund INV-1001 for $49.00 to bob@example.com") == {
        "inv-1001",
        "49.00",
        "bob@example.com",
    }
    assert entities("no entities here") == frozenset()
    assert entities("INV-1001") == entities("inv-1001")


def test_negation_detection():
    assert has_negation("my export does not work") and has_negation("it doesn’t load")
    assert has_negation("can't sign in") and not has_negation("my export works")
    assert not has_negation("a note about nothing")


def test_jaccard_ignores_stopwords_and_is_symmetric():
    assert jaccard("reset my password", "how do I reset the password") == 1.0
    assert jaccard("a b", "b a") == jaccard("b a", "a b")
    assert jaccard("", "") == 0.0 and jaccard("export queue", "invoice refund") == 0.0
    assert words("The the A") == frozenset()


# ----------------------------------------------------------------------------- exact tier


def test_exact_hit_after_normalisation_and_miss_before():
    c = ResponseCache()
    assert c.get("How do I reset my password?") is None
    c.put("How do I reset my password?", "Use the reset link.")
    hit = c.get("  how do i RESET my password ")
    assert (
        hit and hit.value == "Use the reset link." and hit.kind == "exact" and hit.similarity == 1.0
    )
    assert c.stats["hits"] == 1 and c.stats["misses"] == 1 and c.hit_rate == 0.5


def test_scope_separates_entries_so_a_prompt_change_cannot_serve_an_old_answer():
    c = ResponseCache()
    c.put("reset password", "v1 answer", scope="tech|prompt-v1")
    assert c.get("reset password", scope="tech|prompt-v1").value == "v1 answer"
    assert c.get("reset password", scope="tech|prompt-v2") is None
    assert c.get("reset password") is None


def test_different_ids_never_share_an_exact_entry():
    c = ResponseCache()
    c.put("refund INV-1001", "refunded 1001")
    assert c.get("refund INV-1002") is None


def test_ttl_expires_entries_at_exactly_the_boundary():
    clock = Clock()
    c = ResponseCache(ttl_s=60, clock=clock)
    c.put("q", "a")
    clock.t = 59.9
    assert c.get("q").age_s == pytest.approx(59.9)
    clock.t = 60.0
    assert c.get("q") is None and c.stats["expired"] == 1 and len(c) == 0


def test_lru_eviction_keeps_the_recently_used_entry():
    c = ResponseCache(max_entries=2)
    c.put("one", "1")
    c.put("two", "2")
    assert c.get("one")  # touch one: two is now the oldest
    c.put("three", "3")
    assert c.get("two") is None and c.get("one") and c.get("three")
    assert c.stats["evictions"] == 1 and len(c) == 2


def test_overwriting_a_key_refreshes_value_and_age():
    clock = Clock()
    c = ResponseCache(ttl_s=60, clock=clock)
    c.put("q", "old")
    clock.t = 50
    c.put("q", "new")
    clock.t = 100  # 50s after the second put, 100s after the first
    assert c.get("q").value == "new"


def test_clear():
    c = ResponseCache()
    c.put("q", "a")
    c.clear()
    assert len(c) == 0 and c.get("q") is None


# ----------------------------------------------------------------------------- fuzzy tier and its refusals


def fuzzy(**kw):
    return ResponseCache(fuzzy=True, threshold=0.7, **kw)


def test_fuzzy_hit_on_a_paraphrase_reports_similarity_and_the_matched_question():
    c = fuzzy()
    c.put("how do I reset my password", "Use the reset link.")
    hit = c.get("please reset the password")
    assert (
        hit
        and hit.kind == "fuzzy"
        and hit.similarity == 1.0
        and hit.matched == "how do I reset my password"
    )
    assert c.stats["fuzzy_hits"] == 1


def test_fuzzy_is_off_by_default():
    c = ResponseCache(threshold=0.1)
    c.put("how do I reset my password", "x")
    assert c.get("please reset the password") is None


def test_fuzzy_refuses_a_different_entity_even_at_high_similarity():
    c = fuzzy()
    stored = "refund invoice INV-1001 for the duplicate monthly charge this week"
    other = "refund invoice INV-1002 for the duplicate monthly charge this week"
    c.put(stored, "refunded 1001")
    assert jaccard(stored, other) >= 0.7, (
        "similar enough that the fuzzy tier WOULD match without the entity guard"
    )
    assert c.get(other) is None and c.stats["refused_entity"] == 1
    assert c.get(stored.lower()).kind == "exact"


def test_fuzzy_refuses_when_one_side_has_an_entity_and_the_other_has_none():
    c = fuzzy()
    c.put("my export fails with error 0x5F", "see KB-12")
    assert c.get("my export fails with error") is None and c.stats["refused_entity"] == 1


def test_fuzzy_refuses_a_negation_flip():
    stored = "export works with large files fine"
    flipped = "export does not works with large files fine"
    assert jaccard(stored, flipped) >= 0.7, "plain word overlap cannot tell these apart"
    c = fuzzy()
    c.put(stored, "Great!")
    assert c.get(flipped) is None and c.stats["refused_negation"] == 1
    assert c.get("export works with large files fine now").kind == "fuzzy"  # same polarity: allowed
    c2 = fuzzy()  # and the other direction: stored negative, asked positive
    c2.put(flipped, "Try the new export queue.")
    assert c2.get(stored) is None and c2.stats["refused_negation"] == 1


def test_fuzzy_picks_the_most_similar_and_ignores_other_scopes():
    c = fuzzy()
    c.put("reset password email link", "A", scope="s")
    c.put("reset password", "B", scope="s")
    c.put("reset password", "OTHER SCOPE", scope="t")
    assert c.get("reset my password", scope="s").value == "B"


def test_threshold_is_inclusive():
    for threshold, expect_hit in ((0.5, True), (0.51, False)):
        c = ResponseCache(fuzzy=True, threshold=threshold)
        c.put("alpha beta", "x")
        assert jaccard("alpha beta", "alpha beta gamma delta") == 0.5
        assert (c.get("alpha beta gamma delta") is not None) is expect_hit


def test_a_pluggable_similarity_function_is_used():
    c = ResponseCache(fuzzy=True, threshold=0.9, similarity=lambda a, b: 0.95)
    c.put("completely different words", "x")
    assert c.get("nothing in common at all").kind == "fuzzy"
    c2 = ResponseCache(fuzzy=True, threshold=0.9, similarity=lambda a, b: 0.5)
    c2.put("same words", "x")
    assert c2.get("same words here") is None


# ----------------------------------------------------------------------------- concurrency


def test_concurrent_puts_and_gets_keep_the_bound_and_do_not_crash():
    c = ResponseCache(max_entries=50)
    errors = []

    def work(n):
        try:
            for i in range(200):
                c.put(f"question {n} {i}", "a")
                c.get(f"question {n} {i - 1}")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(c) <= 50


def test_a_fuzzy_candidate_in_another_scope_is_never_served():
    c = fuzzy()
    c.put("how do I reset my password", "tech answer", scope="tech|prompt-v1")
    assert c.get("please reset the password", scope="billing|prompt-v1") is None
    assert c.get("please reset the password", scope="tech|prompt-v1").value == "tech answer"


def test_equal_similarity_ties_go_to_the_entry_stored_first():
    c = ResponseCache(fuzzy=True, threshold=0.5)
    c.put("alpha beta gamma", "first")
    c.put("alpha beta delta", "second")  # equally similar to the query below (2 of 4 words each)
    assert jaccard("alpha beta epsilon", "alpha beta gamma") == jaccard(
        "alpha beta epsilon", "alpha beta delta"
    )
    assert c.get("alpha beta epsilon").value == "first"


def test_jaccard_divides_by_the_union_not_the_larger_set():
    assert jaccard("alpha beta", "alpha gamma") == pytest.approx(1 / 3)
    assert jaccard("alpha beta", "alpha beta gamma") == pytest.approx(2 / 3)


def test_a_fuzzy_hit_counts_as_use_for_lru_purposes():
    c = ResponseCache(max_entries=2, fuzzy=True, threshold=0.6)
    c.put("reset password email link", "A")
    c.put("export csv json formats", "B")
    assert c.get("reset password email link now").kind == "fuzzy"  # touches A
    c.put("api rate limit requests", "C")  # evicts the least recently used: B
    assert c.get("export csv json formats") is None and c.get("reset password email link")
