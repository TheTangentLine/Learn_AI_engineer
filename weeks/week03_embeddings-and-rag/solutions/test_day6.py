"""Tests for Day 6: defensive parsing of LLM output, routing, and the transformation flows."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from day5_solution import Corpus, tokenize  # noqa: E402
from day6_solution import (  # noqa: E402
    clean_lines,
    decompose,
    hybrid_rank,
    hyde,
    looks_like_identifier,
    multi_query,
    rewrite_followup,
    routed_multi_query_rank,
)
from rank_bm25 import BM25Okapi  # noqa: E402

from common.embed import HashEmbedder  # noqa: E402
from common.fake import fake_llm  # noqa: E402


def test_clean_lines_strips_numbering_bullets_quotes_and_dupes():
    raw = (
        '1. "retry with backoff"\n2) Retry With Backoff\n- jitter strategy\n\n* • okay then\n  \nx'
    )
    assert clean_lines(raw, 5) == ["retry with backoff", "jitter strategy", "okay then"]
    assert clean_lines("a\nb\nc", 2) == []  # lines of <=3 chars are noise
    assert clean_lines("alpha beta\ngamma delta\nepsilon zeta", 2) == ["alpha beta", "gamma delta"]


def test_multi_query_keeps_original_first_and_drops_echoes():
    q = "how to stop hammering a failing api"
    reply = f"1. {q}\n2. exponential backoff with jitter\n3. circuit breaker pattern"
    with fake_llm([(r"search queries", reply)]):
        out = multi_query(q, 3)
    assert out[0] == q and out.count(q) == 1 and "circuit breaker pattern" in out and len(out) == 3


def test_decompose_splits_compound_but_leaves_single_questions_alone():
    with fake_llm(
        [(r"Question: A and B\?", "What is A?\nWhat is B?"), (r"(?s).*", "Only one thing")]
    ):
        assert decompose("A and B?") == ["What is A?", "What is B?"]
        assert decompose("Just one") == ["Just one"]


def test_rewrite_followup_takes_first_line_and_falls_back_when_empty():
    with fake_llm(
        [(r"(?s).*", '"How do I limit concurrency when retrying API calls?"\nExtra chatter')]
    ):
        assert (
            rewrite_followup("How do I retry?", "and limit it?")
            == "How do I limit concurrency when retrying API calls?"
        )
    with fake_llm([(r"(?s).*", "")]):
        assert rewrite_followup("h", "follow up") == "follow up"


def test_hyde_returns_stripped_passage():
    with fake_llm([(r"documentation that answers", "  Use exponential backoff.  ")]):
        assert hyde("why retries?") == "Use exponential backoff."


@pytest.mark.parametrize(
    "q,expected",
    [
        ("model_validator after", True),
        ("asyncio.Semaphore", True),
        ("--mock harness test", True),
        ("o200k_base encoding", True),
        ("evict_batch", True),
        ("PagedAttention", True),
        ("how do I stop my app from hammering an API that keeps failing", False),
        ("what setting makes the output more random", False),
        ("retry-after header", True),
    ],
)
def test_identifier_router(q, expected):
    assert looks_like_identifier(q) is expected


def make_corpus():
    texts = [
        "use model_validator for cross field checks",
        "backoff with jitter prevents retry storms",
        "cats like sleeping on sofas",
    ]
    emb = HashEmbedder()
    return Corpus(
        texts, ["a", "b", "c"], emb.embed_documents(texts), BM25Okapi([tokenize(t) for t in texts])
    ), emb


def test_routed_multi_query_skips_the_llm_for_identifier_queries():
    c, emb = make_corpus()
    with fake_llm([(r"(?s).*", "unused")]) as fake:
        ranking = routed_multi_query_rank(c, emb, "model_validator")
        assert fake.calls == [] and ranking[0] == 0
        routed_multi_query_rank(c, emb, "how do clients avoid synchronized retry storms")
        assert len(fake.calls) == 1


def test_hybrid_rank_accepts_a_custom_dense_vector():
    c, emb = make_corpus()
    only_dense = emb.embed_documents(["cats like sleeping on sofas"])[0]
    r = hybrid_rank(c, emb, "zzzz unknown words", dense=np.asarray(only_dense), alpha=1.0)
    assert r[0] == 2
