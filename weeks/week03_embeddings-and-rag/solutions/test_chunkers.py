"""Property tests for the Day 3 chunkers: no text lost, size caps respected, deterministic."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from day3_solution import by_headings, fixed, norm, recursive, semantic  # noqa: E402

from common.corpus import load_course_docs  # noqa: E402
from common.embed import HashEmbedder  # noqa: E402

DOCS = [d for d in load_course_docs() if d.short.startswith("week01")][:4]
TEXTS = [d.text for d in DOCS]


def words(t: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", t.lower()))


@pytest.mark.parametrize("size,overlap", [(300, 0), (300, 60), (800, 150)])
def test_fixed_reconstructs_the_text_exactly_and_respects_size(size, overlap):
    """Fixed cuts mid-WORD (its known flaw), so the right property is exact character reconstruction."""
    for t in TEXTS:
        chunks = fixed(t, size, overlap)
        assert all(0 < len(c) <= size for c in chunks)
        rebuilt = chunks[0] + "".join(c[overlap:] for c in chunks[1:])
        assert rebuilt.rstrip() == t.rstrip()


def test_fixed_really_does_cut_words():
    assert "requ" in fixed("requirements are great", 4, 0)[0]


@pytest.mark.parametrize("size,overlap", [(300, 0), (500, 80), (1200, 100)])
def test_recursive_loses_no_words_and_caps_size(size, overlap):
    for t in TEXTS:
        chunks = recursive(t, size, overlap)
        assert chunks and all(c.strip() for c in chunks)
        assert words(t) <= words(" ".join(chunks))
        # only a single unsplittable token may exceed the cap, and by little
        assert max(len(c) for c in chunks) <= size * 1.1, max(len(c) for c in chunks)


def test_recursive_prefers_paragraph_boundaries():
    text = "alpha beta gamma.\n\n" + "delta epsilon zeta. " * 5 + "\n\nomega psi chi."
    chunks = recursive(text, 120, 0)
    assert any(c.strip() == "omega psi chi." for c in chunks), chunks


def test_by_headings_prefix_and_no_content_loss():
    for t in TEXTS:
        chunks = by_headings(t, 1200)
        assert chunks and all(c.text.strip() for c in chunks)
        assert words(t) <= words(" ".join(c.text for c in chunks))
        assert any(c.text.startswith("[") and "]\n" in c.text for c in chunks), (
            "heading path prefix"
        )
        assert not any(c.text.startswith("[") for c in by_headings(t, 1200, prefix=False))


def test_semantic_is_deterministic_loses_nothing_and_caps_size():
    emb = HashEmbedder()
    for t in TEXTS:
        a, b = semantic(t, emb, 1200), semantic(t, emb, 1200)
        assert a == b
        assert words(t) <= words(" ".join(a))
        assert max(len(c) for c in a) <= 1200 * 1.1


def test_norm_strips_markdown_and_whitespace():
    assert norm("**Bold**  and\n`code`") == "bold and code"


def test_by_headings_never_drops_a_document_without_headings():
    """Regression: headingless text used to produce zero chunks (the whole file vanished)."""
    plain = "A long enough paragraph of plain prose without any markdown headings at all."
    assert [c.text for c in by_headings(plain)] == [plain]
    assert by_headings("") == []
