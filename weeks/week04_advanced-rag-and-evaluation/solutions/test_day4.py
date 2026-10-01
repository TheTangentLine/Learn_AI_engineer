"""Tests for Day 4: grader calibration maths and the LLM-driven retrieval loop (scripted model)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from day4_solution import agentic_search, auc, best_tau, clean_lines, loo_taus  # noqa: E402

from common.fake import fake_llm  # noqa: E402


def test_auc_perfect_useless_and_ties():
    assert auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert auc([0, 1, 2], [3, 4, 5]) == 0.0
    assert auc([1, 1], [1, 1]) == pytest.approx(0.5)
    assert auc([1, 3], [2, 2]) == pytest.approx(0.5)
    assert auc([], [1]) != auc([], [1])  # NaN: undefined without both classes


def test_best_tau_separates_when_possible_and_handles_one_class():
    scores, hit = [-2.0, -1.0, 0.5, 1.0, 2.0], [False, False, True, True, True]
    t = best_tau(scores, hit)
    assert -1.0 < t < 0.5 and all((s >= t) == h for s, h in zip(scores, hit, strict=True))
    assert isinstance(best_tau([1.0, 2.0], [True, True]), float)  # all hits: must not crash
    assert isinstance(best_tau([1.0], [False]), float)


def test_loo_tau_never_sees_its_own_question():
    """Each held-out question gets a threshold learned from the others; one outlier can't leak into itself."""
    scores = [-2.0, -1.5, -1.0, 1.0, 1.5, 2.0, -3.0]
    hit = [False, False, False, True, True, True, True]  # the last point (-3, hit) is an outlier
    taus = loo_taus(scores, hit)
    assert len(taus) == len(scores)
    full = best_tau(scores, hit)
    assert any(abs(t - full) > 1e-9 for t in taus), (
        "LOO thresholds must differ from the all-data one"
    )


def test_clean_lines_for_rewrites():
    assert clean_lines(
        '1. "exponential backoff retries"\n- * jitter strategy for clients\n\nx', 5
    ) == ["exponential backoff retries", "jitter strategy for clients"]
    assert clean_lines("retry backoff\nRetry Backoff\nother query text", 5) == [
        "retry backoff",
        "other query text",
    ]


def script(*replies):
    seq = list(replies)
    return [
        (r"(?s).*", lambda p: seq.pop(0) if seq else '{"action": "answer", "text": "fallback"}')
    ]


def test_agent_searches_then_answers_and_feeds_results_back():
    results = {"backoff": ["use exponential backoff", "add jitter"]}
    with fake_llm(
        script(
            '{"action": "search", "query": "backoff"}',
            '{"action": "answer", "text": "Use backoff."}',
        )
    ) as f:
        out = agentic_search(lambda q: results[q], "how to retry?", max_steps=3)
    assert out["answer"] == "Use backoff." and out["searches"] == 1
    assert [t["action"] for t in out["trace"]] == ["search", "answer"]
    assert "use exponential backoff" in f.calls[1].prompt, (
        "search results must be fed to the next turn"
    )


def test_agent_surfaces_malformed_json_and_recovers():
    with fake_llm(script("I think we should search", '{"action": "answer", "text": "ok"}')) as f:
        out = agentic_search(lambda q: [], "q", max_steps=3)
    assert out["trace"][0]["error"] == "unparseable action" and out["answer"] == "ok"
    assert "not valid JSON" in f.calls[1].prompt


def test_agent_is_forced_to_answer_on_the_last_step_and_is_bounded():
    searching = '{"action": "search", "query": "again"}'
    with fake_llm(script(searching, searching, searching, searching)) as f:
        out = agentic_search(lambda q: ["r"], "q", max_steps=3)
    assert (
        len(f.calls) == 3 and "answer now" in f.calls[2].prompt
    )  # bounded; last turn is a hard prompt
    assert out["answer"] == "" and out["searches"] == 2, (
        "a model that never answers yields no answer, not a loop"
    )
