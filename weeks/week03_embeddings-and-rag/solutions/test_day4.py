"""Tests for Day 4: citation validation, groundedness proxy, retrieval gate and repair loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from day4_solution import (  # noqa: E402
    IDK,
    Grounded,
    Index,
    ask,
    build_prompt,
    support_score,
    validate_grounded,
)

from common.embed import HashEmbedder  # noqa: E402
from common.fake import fake_llm  # noqa: E402
from common.vectorstores import Hit, make_store  # noqa: E402

CHUNKS = [
    (
        "week01/day5",
        "Retries",
        "Use exponential backoff with jitter when the API returns HTTP 429 errors.",
    ),
    (
        "week01/day4",
        "KV cache",
        "The KV cache stores keys and values for every previous token in GPU memory.",
    ),
    (
        "week02/day3",
        "Schemas",
        "Pydantic models validate structured output and report precise errors.",
    ),
]


@pytest.fixture()
def index():
    emb = HashEmbedder()
    store = make_store("numpy", emb.dim)
    texts = [c[2] for c in CHUNKS]
    store.add(
        [f"c{i}" for i in range(3)],
        emb.embed_documents(texts),
        [{"doc": c[0], "heading": c[1]} for c in CHUNKS],
        texts,
    )
    return Index(emb, store, 3)


def good(sid=1, text="Use backoff with jitter. [1]"):
    return json.dumps({"answerable": True, "answer": text, "citations": [sid]})


# ------------------------------------------------------------------ validator


def test_validate_grounded_catches_each_failure_mode():
    ok = Grounded(answerable=True, answer="Use jitter [1] and backoff [2].", citations=[1, 2])
    assert validate_grounded(ok, 3) == []
    assert any(
        "not provided" in i
        for i in validate_grounded(Grounded(answerable=True, answer="x [9]", citations=[9]), 3)
    )
    assert any(
        "no citations" in i
        for i in validate_grounded(
            Grounded(answerable=True, answer="no markers here", citations=[]), 3
        )
    )
    assert any(
        "disagrees" in i
        for i in validate_grounded(Grounded(answerable=True, answer="x [1]", citations=[2]), 3)
    )
    assert any(
        "doesn't know" in i
        for i in validate_grounded(Grounded(answerable=True, answer=f"{IDK} [1]", citations=[1]), 3)
    )
    assert any(
        "not answerable but" in i
        for i in validate_grounded(Grounded(answerable=False, answer=IDK, citations=[1]), 3)
    )
    assert validate_grounded(Grounded(answerable=False, answer=IDK, citations=[]), 3) == []


def test_support_score_quote_vs_invention():
    hit = Hit("c", 1.0, {}, "Use exponential backoff with jitter on HTTP 429 errors.")
    assert support_score("Use exponential backoff with jitter. [1]", [hit]) == pytest.approx(1.0)
    assert support_score("Quantum entanglement powers the stock market. [1]", [hit]) < 0.2
    assert support_score("anything", []) == 0.0


def test_prompt_numbers_sources_and_has_no_gt_in_attributes():
    hits = [
        Hit("a", 0.9, {"doc": "week01/day5", "heading": "Retries"}, "text A"),
        Hit("b", 0.8, {"doc": "week01/day4", "heading": "KV"}, "text B"),
    ]
    p = build_prompt("why?", hits)
    assert '<source id="1"' in p and '<source id="2"' in p and "<question>why?</question>" in p
    # the real guarantee: each opening tag's attributes contain no '>' (it must end at the first '>')
    import re

    for m in re.finditer(r"<source [^\n]*", p):
        assert m.group(0).count(">") == 1


# ------------------------------------------------------------------ gate + repair flow


def test_gate_abstains_without_calling_the_llm(index):
    with fake_llm([(r"(?s).*", good())]) as fake:
        a = ask(index, "completely unrelated zebra question", k=2, tau=0.99)
    assert a.abstained and a.text == IDK and "gate" in a.reason and fake.calls == []


def test_answered_path_cites_the_source_it_used(index):
    with fake_llm([(r"(?s).*", good(1))]) as fake:
        a = ask(index, "exponential backoff jitter HTTP 429 errors", k=2, tau=0.0)
    assert not a.abstained and a.issues == [] and len(fake.calls) == 1
    assert a.cited[0].metadata["doc"] == "week01/day5" and a.support > 0.5


def test_one_repair_attempt_on_invalid_citation_then_success(index):
    bad = json.dumps({"answerable": True, "answer": "Sure [9]", "citations": [9]})
    with fake_llm([(r"previous answer was invalid", good(1)), (r"(?s).*", bad)]) as fake:
        a = ask(index, "exponential backoff jitter HTTP 429 errors", k=2, tau=0.0)
    assert a.issues == [] and len(fake.calls) == 2
    assert "not provided" in fake.calls[1].prompt, "the repair prompt must state what was wrong"


def test_persistent_invalid_output_is_surfaced_not_hidden(index):
    bad = json.dumps({"answerable": True, "answer": "Sure [9]", "citations": [9]})
    with fake_llm([(r"(?s).*", bad)]) as fake:
        a = ask(index, "exponential backoff jitter HTTP 429 errors", k=2, tau=0.0)
    assert a.issues and len(fake.calls) == 2 and a.cited == []


def test_model_can_abstain_itself(index):
    refuse = json.dumps({"answerable": False, "answer": IDK, "citations": []})
    with fake_llm([(r"(?s).*", refuse)]):
        a = ask(index, "exponential backoff jitter HTTP 429 errors", k=2, tau=0.0)
    assert a.abstained and a.issues == []


def test_prompt_survives_headings_containing_gt_and_quotes():
    """Regression: nested heading paths ("A > B") and quotes used to break `<source ...>` parsing."""
    import re

    from common.vectorstores import Hit

    hits = [
        Hit(f"c{i}", 0.5, {"doc": "w/d", "heading": h}, f"body {i}")
        for i, h in enumerate(['Week 1 > Day "2" > <b>', "Plain"], 1)
    ]
    p = build_prompt("q?", hits)
    parsed = re.findall(r'<source id="(\d+)"[^>]*>\n(.*?)\n</source>', p, re.S)
    assert [(i, b) for i, b in parsed] == [("1", "body 1"), ("2", "body 2")]
