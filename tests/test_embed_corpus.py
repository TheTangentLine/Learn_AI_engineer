from __future__ import annotations

import numpy as np
import pytest

from common.corpus import load_course_docs, split_sentences
from common.embed import HashEmbedder, LocalEmbedder, cosine_top_k, normalize


def test_hash_embedder_normalised_and_deterministic():
    e = HashEmbedder()
    a = e.embed_documents(["the cat sat", "the cat sat", "quantum chromodynamics"])
    assert a.shape == (3, 384) and np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.allclose(a[0], a[1])
    q = e.embed_query("cat sat")
    assert cosine_top_k(a, q, 2)[0][0] in (0, 1)


def test_cosine_top_k_order_and_k_larger_than_corpus():
    docs = normalize(np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32))
    top = cosine_top_k(docs, normalize(np.array([[1.0, 0.0]], dtype=np.float32))[0], k=10)
    assert [i for i, _ in top] == [0, 1, 2] and top[0][1] > top[1][1] > top[2][1]


def test_cache_hits_on_second_call(tmp_path):
    e = HashEmbedder(cache_path=tmp_path / "c.sqlite")
    e.embed_documents(["a b c", "d e f"])
    assert (e.cache_hits, e.cache_misses) == (0, 2)
    v = e.embed_documents(["a b c", "d e f", "new text"])
    assert (e.cache_hits, e.cache_misses) == (2, 3) and v.shape == (3, 384)


def test_corpus_loader_and_sentence_splitter():
    docs = load_course_docs()
    assert len(docs) >= 12 and all(d.path.startswith("weeks/") for d in docs)
    assert {"week01/day2", "week02/day5"} <= {d.short for d in docs}
    sents = [s for d in docs for s in split_sentences(d.text)]
    assert len(sents) > 500
    assert not any("```" in s or s.startswith("|") for s in sents)
    got = split_sentences(
        "Short. This one is long enough to keep, surely. "
        "Another sentence that is comfortably long enough."
    )
    assert got == [
        "This one is long enough to keep, surely.",
        "Another sentence that is comfortably long enough.",
    ]


@pytest.fixture(scope="module")
def local():
    return LocalEmbedder(cache_path=None)


def test_local_model_is_semantic_not_lexical(local):
    docs = [
        "A kitten is sleeping on the sofa.",
        "The stock market fell sharply on Monday.",
        "Cats like to nap on couches.",
    ]
    d = local.embed_documents(docs)
    q = local.embed_query("a young cat resting on furniture")
    ranked = [i for i, _ in cosine_top_k(d, q, 3)]
    assert ranked[0] in (0, 2) and ranked[-1] == 1
    assert d.shape == (3, 384) and np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-4)
