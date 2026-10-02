"""Tests for the answerers: quotable units, extractive selection, citation verification and repair, and the model-backed answerer with its fallback."""

from __future__ import annotations

import pytest
from conftest import ScriptedChat, make_source
from copilot import answer as A
from copilot.llm import ChatError

CHUNK = """[Week 1, Day 4: Cache > 2. Why]
## 2. Why: attention and the KV cache

To predict the next token the model attends to every previous token. The KV cache stores the key and value vectors of every earlier token.

- Bullet one explains prompt caching reuses the beginning of a prompt.
- Short.

| Tier | Holds | Cost |
|---|---|---|
| Hot cache | recent messages as text | tokens each turn |

```python
cache = {}  # code is never quoted
```

> A quoted aside that is skipped by the extractor entirely.
"""


def test_units_skips_breadcrumb_headings_fences_quotes_separators_and_short_lines():
    us = A.units(CHUNK)
    assert not any("Week 1, Day 4" in u for u in us)  # the chunker's breadcrumb line
    assert not any(
        u.startswith("#") or "code is never quoted" in u or "quoted aside" in u for u in us
    )
    assert "To predict the next token the model attends to every previous token." in us
    assert "The KV cache stores the key and value vectors of every earlier token." in us
    assert (
        "Bullet one explains prompt caching reuses the beginning of a prompt." in us
    )  # bullet marker removed
    assert "Short." not in us  # under 25 characters
    assert (
        "Hot cache; recent messages as text; tokens each turn" in us
    )  # a table row became one readable unit
    assert not any("---" in u for u in us)


def test_clean_flattens_markdown():
    assert (
        A.clean("See [the docs](http://x.y) for **bold** and `code`.")
        == "See the docs for bold and code."
    )
    assert A.clean("1. First item") == "First item" and A.clean("| a | b | c |") == "a; b; c"


def test_long_units_are_truncated_at_a_word_boundary():
    (u,) = A.units("word " * 200)
    assert len(u) <= 320 and u.endswith("…")


def test_tokens_drop_stopwords_and_stem_lightly():
    assert A.tokens("The caches are stored in tokens") == ["cach", "stor", "token"]
    assert A.stem("tokens") == "token" and A.stem("is") == "is" and A.stem("running") == "runn"


def test_the_extractive_answer_quotes_the_best_sentence_with_its_source_number():
    sources = [
        make_source(1, text=CHUNK, heading="Week 1, Day 4: Cache > 2. Why"),
        make_source(
            2,
            doc="week01_a/day2_t.md",
            text="Everything is metered in tokens: price, context limit, rate limits, speed.",
            heading="Week 1, Day 2: T > 1",
        ),
    ]
    d = A.ExtractiveAnswerer(max_units=1, neighbors=0).answer(
        "What does the KV cache store?", sources
    )
    assert d.text == "The KV cache stores the key and value vectors of every earlier token. [1]"
    assert d.cited == [1] and not d.abstained and d.mode == "extractive"


def test_neighbors_add_the_following_units_in_reading_order():
    sources = [make_source(1, text=CHUNK)]
    d = A.ExtractiveAnswerer(max_units=1, neighbors=1).answer(
        "What does the KV cache store?", sources
    )
    assert d.text.index("The KV cache stores") < d.text.index(
        "Bullet one"
    )  # the unit that follows it is quoted after it
    assert d.text.count("[1]") == 2


def test_no_sources_or_no_overlap_means_an_abstention():
    assert A.ExtractiveAnswerer().answer("anything", []).abstained
    d = A.ExtractiveAnswerer().answer(
        "zebra giraffe", [make_source(1, text="The KV cache stores keys and values for tokens.")]
    )
    assert d.abstained and d.text == A.IDK and "shares a word" in d.notes[0]


def test_a_comparison_question_quotes_both_named_weeks():
    s1 = make_source(
        1,
        doc="week01_a/day4_c.md",
        week=1,
        text="The KV cache stores keys and values of earlier tokens so attention is not recomputed.",
    )
    s2 = make_source(
        2,
        doc="week11_b/day2_p.md",
        week=11,
        text="PagedAttention stores the KV cache in blocks so memory is not wasted on empty slots.",
    )
    d = A.ExtractiveAnswerer(max_units=2, neighbors=0).answer(
        "How does the KV cache relate to PagedAttention?", [s1, s2], must_span=[1, 11]
    )
    assert d.cited == [1, 2]


def test_cited_numbers_are_distinct_and_ordered():
    assert A.cited_numbers("a [2] b [1] c [2] [10]") == [2, 1, 10] and A.cited_numbers("none") == []


def test_verify_checks_nonempty_valid_numbers_and_at_least_one_citation():
    assert A.verify("", 3) == ["empty answer"]
    assert A.verify("The cache stores keys [1].", 3) == []
    assert "do not exist" in A.verify("It does [4].", 3)[0]
    assert A.verify("It does.", 3) == ["cites no source"]
    assert A.verify(A.IDK, 3) == []  # an honest refusal needs no citation


def test_support_is_the_share_of_the_claims_words_found_in_the_source():
    src = "The KV cache stores the key and value vectors of every earlier token."
    assert A.support("The KV cache stores key and value vectors [1]", src) == 1.0
    assert A.support("The cache was invented by aliens", src) < 0.4
    assert A.support("[1]", src) == 1.0  # nothing to check


def test_repair_repoints_a_citation_to_the_source_that_supports_the_sentence():
    s1 = make_source(
        1, text="Everything is metered in tokens: price, context limit, rate limits, speed."
    )
    s2 = make_source(
        2, text="The KV cache stores the key and value vectors of every earlier token."
    )
    text, rep = A.repair_citations(
        "The KV cache stores the key and value vectors of every earlier token [1].", [s1, s2]
    )
    assert text.endswith("[2]") and rep == {
        "sentences": 1,
        "kept": 0,
        "repointed": 1,
        "unsupported": 0,
    }
    ok, rep2 = A.repair_citations("Everything is metered in tokens: price and speed [1].", [s1, s2])
    assert "[1]" in ok and "[2]" not in ok and rep2["kept"] == 1 and rep2["repointed"] == 0


def test_repair_reports_sentences_that_no_source_supports_without_rewording_them():
    s1 = make_source(1, text="The KV cache stores keys and values.")
    text, rep = A.repair_citations("The model passed every test with zero errors [1].", [s1])
    assert text == "The model passed every test with zero errors [1]." and rep["unsupported"] == 1


def sources():
    return [
        make_source(
            1, text="The KV cache stores the key and value vectors of every earlier token."
        ),
        make_source(2, text="Prompt caching reuses the beginning of a prompt."),
    ]


def test_a_verified_model_answer_is_used_and_token_usage_is_recorded():
    chat = ScriptedChat(
        ("The KV cache stores key and value vectors of earlier tokens [1].", 120, 18)
    )
    d = A.LlmAnswerer(chat).answer("What does the cache store?", sources())
    assert (
        d.mode == "llm"
        and d.cited == [1]
        and (d.prompt_tokens, d.completion_tokens) == (120, 18)
        and not d.abstained
    )
    messages = chat.seen[0]
    assert messages[0]["role"] == "system" and "ONLY the numbered sources" in messages[0]["content"]
    assert (
        '<source n="1"' in messages[1]["content"]
        and "What does the cache store?" in messages[1]["content"]
    )


def test_an_unverifiable_model_answer_falls_back_to_the_extractive_one_and_says_why():
    chat = ScriptedChat(("The cache stores keys.", 100, 10))  # cites nothing
    d = A.LlmAnswerer(chat, A.ExtractiveAnswerer()).answer(
        "What does the KV cache store?", sources()
    )
    assert (
        d.mode == "llm+fallback"
        and d.notes == ["cites no source"]
        and "[1]" in d.text
        and (d.prompt_tokens, d.completion_tokens) == (100, 10)
    )
    bad = A.LlmAnswerer(ScriptedChat(("It does [9].", 1, 1)), A.ExtractiveAnswerer()).answer(
        "What does the KV cache store?", sources()
    )
    assert bad.mode == "llm+fallback" and "do not exist" in bad.notes[0]


def test_without_a_fallback_the_unverified_text_is_returned_with_its_problems():
    d = A.LlmAnswerer(ScriptedChat(("no citation here", 1, 1))).answer("q", sources())
    assert d.mode == "llm" and d.notes == ["cites no source"] and d.text == "no citation here"


def test_a_model_that_invents_a_fact_is_caught_by_the_support_check():
    chat = ScriptedChat(("The agent passed all seven tasks with no errors at all [1].", 50, 12))
    d = A.LlmAnswerer(chat, A.ExtractiveAnswerer()).answer(
        "What does the KV cache store?", sources()
    )
    assert d.mode == "llm+fallback" and "not supported by any source" in d.notes[0]


def test_a_wrong_citation_number_is_repaired_not_rejected():
    chat = ScriptedChat(
        ("The KV cache stores the key and value vectors of every earlier token [2].", 50, 12)
    )
    d = A.LlmAnswerer(chat, A.ExtractiveAnswerer()).answer("q", sources())
    assert (
        d.mode == "llm"
        and d.cited == [1]
        and d.text.endswith("[1]")
        and "re-pointed 1" in d.notes[0]
    )


def test_a_refusal_is_accepted_as_an_abstention_and_no_sources_skips_the_model():
    chat = ScriptedChat((A.IDK, 40, 9))
    d = A.LlmAnswerer(chat).answer("q", sources())
    assert d.abstained and d.mode == "llm"
    none = ScriptedChat()
    assert A.LlmAnswerer(none).answer("q", []).abstained and none.seen == []


def test_a_model_failure_propagates_so_the_pipeline_can_report_it():
    with pytest.raises(ChatError):
        A.LlmAnswerer(ScriptedChat(ChatError("down")), A.ExtractiveAnswerer()).answer(
            "q", sources()
        )


def test_the_idf_weights_make_rare_words_decide_the_sentence():
    idf = {"hnsw": 9.0, "cach": 0.5}
    ex = A.ExtractiveAnswerer(idf, max_units=1, neighbors=0)
    src = make_source(
        1,
        text="The cache stores things in memory for later use by the system.\nHNSW is a graph index for approximate nearest neighbour search.",
    )
    assert "HNSW" in ex.answer("What is the HNSW cache?", [src]).text


def test_a_unit_reranker_can_override_the_lexical_choice_of_sentence():
    class Prefer:
        """A stand-in cross-encoder that likes sentences about 'blocks'."""

        def scores(self, query, passages):
            return [5.0 if "blocks" in p else -5.0 for p in passages]

    src = make_source(
        1,
        text="The KV cache stores keys and values for every token in the history.\nPagedAttention splits the cache into blocks that are allocated on demand.",
    )
    plain = A.ExtractiveAnswerer(max_units=1, neighbors=0).answer(
        "What does the KV cache store?", [src]
    )
    guided = A.ExtractiveAnswerer(max_units=1, neighbors=0, unit_reranker=Prefer()).answer(
        "What does the KV cache store?", [src]
    )
    assert "stores keys" in plain.text and "blocks" in guided.text


# ----------------------------------------------------------------------------- edges found by mutation checks


def test_stemming_boundaries_and_table_rows_with_two_pipes():
    assert (
        A.stem("types") == "type" and A.stem("cache") == "cache" and A.stem("aces") == "aces"
    )  # a suffix needs more than three letters in front of it
    assert (
        A.clean("| only |") == "only" and A.clean("a | b") == "a | b"
    )  # two pipes make a row; one is just a character


def test_unit_length_limits_are_inclusive():
    exactly_min = "abcdefghij abcdefghij ab."
    assert len(exactly_min) == 25 and A.units(exactly_min) == [exactly_min]
    assert A.units("abcdefghij abcdefghij a.") == []  # 24 characters: too short
    exactly_max = "x" * 320
    assert A.units(exactly_max) == [exactly_max]  # not truncated
    assert A.units("x" * 321)[0].endswith("…") and len(A.units("x" * 321)[0]) <= 320


def test_a_sentence_is_scored_by_overlap_over_the_square_root_of_its_length():
    import math

    ex = A.ExtractiveAnswerer({"cache": 1.0, "block": 1.0})
    q = {"cache", "block"}
    short = ex._score(q, "cache blocks", set())
    long = ex._score(q, "cache blocks " + " ".join(f"filler{i}" for i in range(18)), set())
    assert (
        short == pytest.approx(2 / math.sqrt(2 + 4))
        and long == pytest.approx(2 / math.sqrt(20 + 4))
        and short > long > 0
    )
    assert ex._score(q, "nothing relevant here at all", set()) == 0.0


def test_a_near_duplicate_sentence_is_not_quoted_twice():
    first = "The KV cache stores the key and value vectors of every earlier token."
    near = "The KV cache stores the key and value vectors of every prior token."  # differs by one word: a different unit, but redundant
    src = make_source(
        1, text=f"{first}\n{near}\nA second cache stores the beginning of a prompt across requests."
    )
    d = A.ExtractiveAnswerer(max_units=2, neighbors=0).answer(
        "What does the KV cache store?", [src]
    )
    assert d.text.count("KV cache stores") == 1 and "second cache" in d.text


def test_naming_two_weeks_forces_a_sentence_from_each_even_when_one_week_dominates_the_scoring():
    s1 = make_source(
        1,
        doc="week01_a/day4_c.md",
        week=1,
        text="The KV cache stores keys and values for every token.\nTokens already seen are cached, so storing them again is skipped entirely.",
    )
    s2 = make_source(
        2, doc="week11_b/day2_p.md", week=11, text="PagedAttention splits memory into blocks."
    )
    q = "How does the KV cache storing keys and values for tokens relate to PagedAttention?"
    free = A.ExtractiveAnswerer({"pagedattention": 0.2}, max_units=2, neighbors=0).answer(
        q, [s1, s2]
    )
    forced = A.ExtractiveAnswerer({"pagedattention": 0.2}, max_units=2, neighbors=0).answer(
        q, [s1, s2], must_span=[1, 11]
    )
    assert free.cited == [1] and forced.cited == [1, 2]
    one_week = A.ExtractiveAnswerer({"pagedattention": 0.2}, max_units=2, neighbors=0).answer(
        q, [s1, s2], must_span=[1]
    )
    assert one_week.cited == [1]  # a single named week does not force anything


def test_citation_zero_does_not_exist():
    assert "do not exist" in A.verify("It does [0].", 3)[0]


def test_support_exactly_at_the_threshold_is_enough_to_keep_or_to_repoint():
    s1 = make_source(1, text="alpha beta gamma and some other words")
    s2 = make_source(2, text="completely unrelated sentence content")
    kept, rep = A.repair_citations(
        "alpha beta gamma delta epsilon [1].", [s1, s2]
    )  # 3 of 5 content words = 0.6 = the threshold
    assert rep == {"sentences": 1, "kept": 1, "repointed": 0, "unsupported": 0} and "[1]" in kept
    moved, rep2 = A.repair_citations("alpha beta gamma delta epsilon [2].", [s1, s2])
    assert rep2["repointed"] == 1 and "[1]" in moved and "[2]" not in moved


def test_exactly_half_unsupported_sentences_is_tolerated_but_more_falls_back():
    src = [make_source(1, text="alpha beta gamma delta are the facts here")]
    half = ScriptedChat(("Alpha beta gamma delta [1]. Zebra giraffe okapi lemur [1].", 10, 10))
    d = A.LlmAnswerer(half, A.ExtractiveAnswerer()).answer("alpha?", src)
    assert d.mode == "llm"  # 1 of 2 is exactly 0.5: not more than half
    most = ScriptedChat(
        (
            "Alpha beta gamma delta [1]. Zebra giraffe okapi lemur [1]. Quokka numbat wombat [1].",
            10,
            10,
        )
    )
    d2 = A.LlmAnswerer(most, A.ExtractiveAnswerer()).answer("alpha?", src)
    assert d2.mode == "llm+fallback" and "not supported" in d2.notes[0]
