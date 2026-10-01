"""Property tests for the production chunkers in common/chunking.py."""

from __future__ import annotations

import re

import pytest

from common.chunking import by_headings, fixed, recursive
from common.corpus import load_course_docs

TEXTS = [d.text for d in load_course_docs()[:4]]


def words(t: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", t.lower()))


@pytest.mark.parametrize("size,overlap", [(300, 0), (500, 80), (1200, 100)])
def test_recursive_loses_no_words_and_caps_size(size, overlap):
    for t in TEXTS:
        chunks = recursive(t, size, overlap)
        assert chunks and all(c.strip() for c in chunks)
        assert words(t) <= words(" ".join(chunks))
        assert max(len(c) for c in chunks) <= size * 1.1


def test_fixed_reconstructs_exactly():
    for t in TEXTS:
        chunks = fixed(t, 400, 50)
        assert (chunks[0] + "".join(c[50:] for c in chunks[1:])).rstrip() == t.rstrip()


def test_by_headings_prefix_and_no_loss():
    for t in TEXTS:
        chunks = by_headings(t, 1200)
        assert words(t) <= words(" ".join(c.text for c in chunks))
        assert any(c.text.startswith("[") and c.heading for c in chunks)
        assert not any(c.text.startswith("[") for c in by_headings(t, 1200, prefix=False))


def test_by_headings_on_text_without_headings_and_empty():
    assert [
        c.text for c in by_headings("just one paragraph of plain text here, long enough to keep.")
    ] == ["just one paragraph of plain text here, long enough to keep."]
    assert by_headings("") == []
