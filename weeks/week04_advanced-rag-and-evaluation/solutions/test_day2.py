"""Tests for Day 2: the labelled set's integrity, threshold tuning, detector metrics, LLM-judge parsing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from day2_solution import best_threshold, llm_judge, metrics  # noqa: E402
from faithfulness_data import N_DEV, PAIRS, items  # noqa: E402

from common.corpus import load_course_docs  # noqa: E402
from common.evalkit import norm  # noqa: E402
from common.judges import lexical_support  # noqa: E402


def test_dataset_is_balanced_split_by_context_and_contexts_are_real_lesson_text():
    data = items()
    assert len(data) == 2 * len(PAIRS) == 36
    assert sum(d["label"] for d in data) == 18, "half faithful, half unfaithful"
    dev_pairs = {d["pair"] for d in data if d["split"] == "dev"}
    test_pairs = {d["pair"] for d in data if d["split"] == "test"}
    assert dev_pairs.isdisjoint(test_pairs) and len(dev_pairs) == N_DEV, (
        "no context appears in both splits"
    )
    assert {d["type"] for d in data if d["label"] == 0} == {
        "contradiction",
        "numeric",
        "swap",
        "fabrication",
    }
    corpus = " ".join(norm(d.text) for d in load_course_docs())
    for p in PAIRS:
        sentences = [x for x in p["context"].replace("? ", ". ").split(". ") if len(x) > 12]
        assert all(norm(x.rstrip(".")) in corpus for x in sentences), p["context"][:60]


def test_dataset_has_no_trivially_leaky_labels():
    """If unfaithful answers were just much shorter/longer, a length rule would 'solve' the benchmark."""
    good = [len(p["good"]) for p in PAIRS]
    bad = [len(p["bad"]) for p in PAIRS]
    assert abs(sum(good) / len(good) - sum(bad) / len(bad)) < 15
    assert all(p["good"] != p["bad"] for p in PAIRS)


def test_best_threshold_separates_and_handles_degenerate_inputs():
    scores, labels = [0.1, 0.2, 0.3, 0.8, 0.9], [0, 0, 0, 1, 1]
    t = best_threshold(scores, labels)
    assert 0.3 < t <= 0.8 and all(
        (s >= t) == bool(label) for s, label in zip(scores, labels, strict=True)
    )
    assert best_threshold([0.5, 0.5], [1, 0]) <= 0.5  # inseparable: any answer, but must not crash
    assert best_threshold([0.4], [1]) < 0.4  # a single example: accept everything


def test_metrics_treat_unfaithful_as_the_positive_class():
    m = metrics([1, 1, 0, 0], [1, 0, 0, 1])  # one false alarm, one missed hallucination
    assert m["recall"] == 0.5 and m["precision"] == 0.5 and m["acc"][0] == 0.5
    always_faithful = metrics([1, 1, 0, 0], [1, 1, 1, 1])
    assert always_faithful["recall"] == 0.0 and always_faithful["kappa"] == 0.0
    assert metrics([1, 0], [1, 0])["kappa"] == 1.0


class StubChat:
    def __init__(self, reply):
        self.reply = reply

    def __call__(self, system, user, max_new_tokens=4):
        return self.reply


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("No", 0),
        ("no.", 0),
        ("  NO", 0),
        ("Yes", 1),
        ("yes", 1),
        ("Maybe", 1),
        ("", 1),
        ("I think no", 1),
    ],
)
def test_llm_judge_parsing_and_its_dangerous_default(reply, expected):
    """Anything that doesn't START with 'no' counts as faithful: parsing choices silently bias a judge."""
    assert llm_judge(StubChat(reply), "ctx", "answer") == expected


def test_lexical_judge_cannot_tell_swaps_apart_but_sees_contradictions():
    """Swapped answers reuse the same words (tiny score gap); contradictions change words (bigger gap)."""

    def mean_gap(kind):
        gaps = [
            lexical_support(p["context"], p["good"]) - lexical_support(p["context"], p["bad"])
            for p in PAIRS
            if p["type"] == kind
        ]
        return sum(gaps) / len(gaps)

    assert mean_gap("swap") < 0.10
    assert mean_gap("contradiction") > 0.15
