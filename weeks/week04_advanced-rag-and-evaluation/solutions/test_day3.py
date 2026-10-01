"""Tests for Day 3: MiniIndex mechanics, parent-child collapse, extractive context, generation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from day3_solution import (  # noqa: E402
    MiniIndex,
    doc_title,
    extractive_context,
    hypothetical_questions,
    llm_context,
    parent_child,
)

from common.embed import HashEmbedder  # noqa: E402

LESSON = (
    "# Week 9: Tiny GPT\n\n**Time:** 3h\n\n## Learning objectives\n"
    "- Build attention from scratch in numpy\n- Train a mini model\n\n## Body\n\nText."
)


def test_doc_title_and_extractive_context():
    assert doc_title(LESSON) == "Week 9: Tiny GPT"
    assert (
        extractive_context(LESSON, "Week 9: Tiny GPT")
        == "Week 9: Tiny GPT. Build attention from scratch in numpy"
    )
    assert (
        extractive_context("# No objectives here\n\ntext", "No objectives here")
        == "No objectives here"
    )
    assert doc_title("no heading at all") == ""


def test_miniindex_returns_display_text_not_index_text():
    emb = HashEmbedder()
    idx = MiniIndex(
        emb, ["d1", "d2"], ["shown one", "shown two"], ["alpha beta gamma", "delta epsilon zeta"]
    )
    (doc, text), *_ = idx.search("alpha gamma", 2)
    assert (doc, text) == (
        "d1",
        "shown one",
    )  # searched alpha/gamma text, returned the display text


def test_parent_child_collapses_children_to_one_parent_and_returns_the_parent_text():
    emb = HashEmbedder()
    parent_a = (
        "alpha " * 60 + "uniquetokenxyz " + "alpha " * 60
    )  # long: splits into several children
    parent_b = "beta gamma delta " * 30
    chunks = [{"doc": "d1", "text": parent_a}, {"doc": "d2", "text": parent_b}]
    idx = parent_child(emb, chunks, child_size=120)
    assert len(idx.index_texts) > len(chunks), "children are smaller and more numerous"
    results = idx.search("uniquetokenxyz", 5)
    assert [d for d, _ in results].count("d1") == 1, "many matching children, ONE parent returned"
    assert results[0] == ("d1", parent_a) and len({t for _, t in results}) == len(results)


def test_contextual_prefix_changes_what_is_searchable_but_not_what_is_returned():
    emb = HashEmbedder()
    plain = MiniIndex(
        emb, ["d1", "d2"], ["chunk one", "chunk two"], ["it works well", "it works well"]
    )
    ctx = MiniIndex(
        emb,
        ["d1", "d2"],
        ["chunk one", "chunk two"],
        ["About retries. it works well", "About caching. it works well"],
    )
    q = "retries"
    assert plain.search(q, 1)[0][0] in ("d1", "d2")
    assert ctx.search(q, 1)[0] == ("d1", "chunk one")  # the prefix made the right chunk findable


class StubChat:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def __call__(self, system, user, max_new_tokens=40):
        self.prompts.append(user)
        return self.reply


def test_llm_context_takes_first_line_and_truncates_long_inputs():
    chat = StubChat("This chunk covers retry backoff.\nExtra line")
    out = llm_context(chat, "Title", "intro " * 500, "chunk " * 500)
    assert out == "This chunk covers retry backoff."
    assert len(chat.prompts[0]) < 1600, "document and chunk are truncated to keep prompts small"
    assert llm_context(StubChat(""), "T", "i", "c") == ""


def test_hypothetical_questions_are_flattened_and_capped():
    chat = StubChat("1. What is backoff?\n2) Why add jitter?\n\n- Third?")
    out = hypothetical_questions(chat, "chunk text")
    assert out == "What is backoff? Why add jitter? Third?" and len(out) <= 300
