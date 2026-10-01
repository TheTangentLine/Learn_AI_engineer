"""Tests for Day 1: synthetic question generation and its quality filters (scripted generator)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from day1_solution import (  # noqa: E402
    META,
    clean_question,
    content_words,
    generate_synthetic,
    jaccard,
    target_sentence,
)

from common.embed import HashEmbedder  # noqa: E402
from common.rag import RagIndex, SourceDoc  # noqa: E402

PROSE = (
    "# Retries\n\nExponential backoff with jitter prevents synchronized clients from hammering a recovering "
    "server after an outage. Honour the Retry-After header whenever the provider sends one back to you.\n\n"
    "```python\nx = 1\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nShort.\n"
)


def test_target_sentence_picks_longest_prose_sentence_and_skips_code_tables_headings():
    s = target_sentence(PROSE)
    assert s is not None and s.startswith("Exponential backoff with jitter") and "```" not in s
    assert target_sentence("# Only a heading\n\n```python\nprint('x')\n```\n| a | b |") is None
    assert target_sentence("too short.") is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "Question: How should clients space out retries after failures?",
            "How should clients space out retries after failures?",
        ),
        (
            '"How should clients space out retries after failures?"\nExtra chatter',
            "How should clients space out retries after failures?",
        ),
        ("Not a question at all, just text that is long enough to pass", None),
        ("Too short?", None),
        ("", None),
        ("Q - " + "x" * 200 + "?", None),
    ],
)
def test_clean_question(raw, expected):
    assert clean_question(raw) == expected


def test_overlap_helpers_and_meta_filter():
    a = content_words("How should clients space out retries after failures")
    b = content_words("Exponential backoff spaces out client retries after failures")
    assert 0 < jaccard(a, b) < 1 and jaccard(set(), set()) == 0.0
    assert META.search("What does the passage suggest about caching?")
    assert META.search("According to the following document, why?")
    assert not META.search("Why does exponential backoff need jitter?")


class ScriptedChat:
    """Stand-in for LocalChat: returns scripted questions in order."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def __call__(self, system, user, max_new_tokens=40):
        self.calls += 1
        return (
            self.replies.pop(0)
            if self.replies
            else "How is this unanswerable question phrased exactly?"
        )


PLAIN = (
    "# Retries {i}\n\nExponential backoff with jitter prevents synchronized clients from hammering a "
    "recovering server after an outage. Honour the Retry-After header whenever the provider sends one "
    "back to you.\n\n" + "More prose about operating retry policies in production. " * 12
)


def make_index():
    idx = RagIndex(HashEmbedder())
    idx.sync([SourceDoc(f"w1/d{i}", PLAIN.format(i=i), {"week": 1}) for i in range(1, 5)])
    return idx


def test_filters_reject_bad_questions_and_funnel_counts_them():
    idx = make_index()
    assert any(len(c["text"]) >= 400 for c in idx.chunks), "fixture must be eligible for generation"
    replies = [
        "What does the passage say about retry behaviour for servers?",  # meta reference
        "Why is the sky blue during the afternoon hours today?",  # not grounded in the sentence
        "no question mark here but long enough to pass the length test",  # bad format
        "Why does exponential backoff need jitter to protect a recovering server?",  # good
    ]
    chat = ScriptedChat(replies)
    out, stats = generate_synthetic(idx, chat, n=1)
    assert [g.query for g in out] == [replies[3]] and out[0].kind == "synthetic" and chat.calls == 4
    assert (stats["meta_reference"], stats["not_grounded"], stats["bad_format"]) == (1, 1, 1)


def test_loose_mode_lets_ungrounded_questions_through_but_strict_does_not():
    idx = make_index()
    ungrounded = "Why is the sky blue during the afternoon hours today?"
    loose, _ = generate_synthetic(idx, ScriptedChat([ungrounded]), n=1, strict=False)
    strict, stats = generate_synthetic(idx, ScriptedChat([ungrounded]), n=1)
    assert [g.query for g in loose] == [ungrounded]
    assert strict == [] and stats["not_grounded"] >= 1


def test_gold_is_satisfiable_by_its_own_chunk_and_duplicates_are_dropped():
    idx = make_index()
    q = "Why does exponential backoff need jitter to protect a recovering server?"
    out, stats = generate_synthetic(idx, ScriptedChat([q, q, q, q]), n=3)
    assert len(out) == 1 and stats["duplicate"] >= 1
    assert out[0].is_relevant(
        out[0].doc, next(c["text"] for c in idx.chunks if c["doc"] == out[0].doc)
    )
