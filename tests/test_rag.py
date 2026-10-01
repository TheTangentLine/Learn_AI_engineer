"""Tests for common/rag.py: incremental indexing, hybrid search, gate, repair, follow-ups."""

from __future__ import annotations

import json

import numpy as np
import pytest

from common.embed import HashEmbedder
from common.fake import fake_llm
from common.rag import IDK, Grounded, RagBot, RagIndex, SourceDoc, build_prompt, validate_grounded

DOCS = [
    SourceDoc(
        "retries.md",
        "# Retries\n\nUse exponential backoff with jitter when the API returns HTTP 429.\n"
        "Honour the Retry-After header.",
        {"week": 1},
    ),
    SourceDoc(
        "cache.md",
        "# KV cache\n\nThe KV cache stores keys and values for earlier tokens in GPU memory.",
        {"week": 1},
    ),
    SourceDoc(
        "schemas.md",
        "# Schemas\n\nPydantic model_validator checks fields across a whole model.",
        {"week": 2},
    ),
]


class CountingEmbedder(HashEmbedder):
    """HashEmbedder that counts how many texts it was asked to embed (cache disabled)."""

    def __init__(self):
        super().__init__(cache_path=None)
        self.embedded = 0

    def _embed(self, texts, kind):
        self.embedded += len(texts)
        return super()._embed(texts, kind)


def test_sync_is_idempotent_and_incremental(tmp_path):
    emb = CountingEmbedder()
    idx = RagIndex(emb, tmp_path / "idx")
    r1 = idx.sync(DOCS)
    assert len(r1.added) == 3 and r1.chunks_embedded == len(idx.chunks) > 0
    n_after_first = emb.embedded
    r2 = idx.sync(DOCS)
    assert len(r2.unchanged) == 3 and r2.chunks_embedded == 0 and emb.embedded == n_after_first
    edited = [
        *DOCS[:2],
        SourceDoc("schemas.md", "# Schemas\n\nNow about discriminated unions.", {"week": 2}),
    ]
    r3 = idx.sync(edited)
    assert r3.updated == ["schemas.md"] and r3.unchanged == ["retries.md", "cache.md"]
    assert emb.embedded == n_after_first + r3.chunks_embedded
    assert not any("model_validator" in c["text"] for c in idx.chunks)


def test_removed_documents_disappear_from_search(tmp_path):
    idx = RagIndex(HashEmbedder(), tmp_path / "i")
    idx.sync(DOCS)
    assert idx.search("KV cache GPU memory", 1)[0].metadata["doc"] == "cache.md"
    rep = idx.sync([d for d in DOCS if d.id != "cache.md"])
    assert rep.removed == ["cache.md"]
    assert all(h.metadata["doc"] != "cache.md" for h in idx.search("KV cache GPU memory", 5))
    assert len(idx.vecs) == len(idx.chunks)


def test_persistence_roundtrip_and_embedder_mismatch_rebuild(tmp_path):
    a = RagIndex(HashEmbedder(), tmp_path / "i")
    a.sync(DOCS)
    b = RagIndex(HashEmbedder(), tmp_path / "i")  # a fresh process loading the same directory
    assert b.manifest == a.manifest and np.allclose(b.vecs, a.vecs) and b.chunks == a.chunks
    assert b.sync(DOCS).chunks_embedded == 0
    other = HashEmbedder(dim=64)
    c = RagIndex(
        other, tmp_path / "i"
    )  # different embedder -> stored vectors unusable -> start empty
    assert c.chunks == [] and c.sync(DOCS).chunks_embedded > 0


def test_hybrid_search_combines_lexical_and_vector_signals():
    idx = RagIndex(HashEmbedder())
    idx.sync(DOCS)
    assert idx.search("model_validator", 1, alpha=0.0)[0].metadata["doc"] == "schemas.md"
    assert idx.search("Retry-After header", 1)[0].metadata["doc"] == "retries.md"
    assert idx.search("anything", 3, where={"week": 2})[0].metadata["doc"] == "schemas.md"
    assert all(h.metadata["week"] == 2 for h in idx.search("anything", 5, where={"week": 2}))
    assert idx.search("x", 3, where={"week": 99}) == []
    assert RagIndex(HashEmbedder()).search("empty index") == []


def test_chunk_ids_are_stable_across_runs():
    a, b = RagIndex(HashEmbedder()), RagIndex(HashEmbedder())
    a.sync(DOCS)
    b.sync(DOCS)
    assert [c["id"] for c in a.chunks] == [c["id"] for c in b.chunks]


def good(sid=1):
    return json.dumps(
        {"answerable": True, "answer": f"Use backoff with jitter. [{sid}]", "citations": [sid]}
    )


@pytest.fixture()
def bot():
    idx = RagIndex(HashEmbedder())
    idx.sync(DOCS)
    return RagBot(idx, tau=0.0, k=2)


def test_gate_refuses_without_llm_call(bot):
    bot.tau = 0.99
    with fake_llm([(r"(?s).*", good())]) as fake:
        a = bot.ask("completely unrelated zebra question")
    assert a.abstained and a.text == IDK and a.reason.startswith("gate") and fake.calls == []


def test_answer_has_cited_sources_support_and_timings(bot):
    with fake_llm([(r"(?s).*", good(1))]) as fake:
        a = bot.ask("exponential backoff jitter HTTP 429")
    assert not a.abstained and a.issues == [] and a.cited and len(fake.calls) == 1
    assert a.support > 0.5 and {"rewrite", "retrieve", "generate"} <= set(a.seconds)


def test_repair_attempt_states_the_problem(bot):
    bad = json.dumps({"answerable": True, "answer": "Sure [9]", "citations": [9]})
    with fake_llm([(r"previous answer was invalid", good(1)), (r"(?s).*", bad)]) as fake:
        a = bot.ask("exponential backoff jitter HTTP 429")
    assert a.issues == [] and len(fake.calls) == 2 and "not provided" in fake.calls[1].prompt


def test_followup_is_rewritten_using_history(bot):
    rewritten = "How do I use exponential backoff with jitter for HTTP 429?"
    with fake_llm([(r"Rewrite the follow-up", rewritten), (r"(?s).*", good(1))]) as fake:
        a = bot.ask("and for rate limits?", history=[("How do I retry?", "Use backoff.")])
    assert a.standalone == rewritten and a.question == "and for rate limits?"
    assert "How do I retry?" in fake.calls[0].prompt  # the rewriter saw the conversation
    with fake_llm([(r"(?s).*", good(1))]) as fake:
        bot.ask("exponential backoff jitter")  # no history -> no rewrite call
    assert len(fake.calls) == 1


def test_calibrate_separates_answerable_from_unanswerable(bot):
    cal = bot.calibrate(
        ["exponential backoff jitter HTTP 429", "KV cache keys values GPU memory"],
        ["zebra giraffe savannah", "volcano lava eruption"],
    )
    assert cal["balanced_accuracy"] == 1.0 and cal["neg"].max() < bot.tau <= cal["pos"].min()


def test_prompt_numbers_sources_and_validator_flags_bad_citation():
    idx = RagIndex(HashEmbedder())
    idx.sync(DOCS)
    hits = idx.search("backoff", 2)
    p = build_prompt("q?", hits)
    assert '<source id="1"' in p and '<source id="2"' in p and p.count(">") >= 6
    assert validate_grounded(Grounded(answerable=True, answer="x [3]", citations=[3]), 2)


def test_prompt_survives_headings_containing_gt_and_quotes():
    """Regression: heading paths like "A > B" and quotes used to break `<source ...>` parsing."""
    import re

    from common.vectorstores import Hit

    hits = [
        Hit(f"c{i}", 0.5, {"doc": "w/d", "heading": h}, f"body {i}")
        for i, h in enumerate(['Week 1 > Day "2" > <b>', "Plain"], 1)
    ]
    p = build_prompt("q?", hits)
    parsed = re.findall(r'<source id="(\d+)"[^>]*>\n(.*?)\n</source>', p, re.S)
    assert [(i, b) for i, b in parsed] == [("1", "body 1"), ("2", "body 2")]


def test_same_name_different_dimension_embedders_do_not_share_an_index(tmp_path):
    """Regression: the stored-vector compatibility key must include the dimension."""

    class Fixed(HashEmbedder):
        def __init__(self, dim):
            super().__init__(dim=dim)
            self.name = "same-name"  # identical names, different sizes

    a = RagIndex(Fixed(384), tmp_path / "i")
    a.sync(DOCS)
    b = RagIndex(Fixed(64), tmp_path / "i")
    assert b.chunks == [] and b.vecs.shape[1] == 64
    assert b.sync(DOCS).chunks_embedded > 0 and b.vecs.shape[1] == 64
