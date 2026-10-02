"""Tests for the Week 10 weekly pipeline: the cost arithmetic, the report builder, the reply cache, and a one-minute end-to-end smoke test."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import costs as K  # noqa: E402
import report as R  # noqa: E402
import run_weekly as W  # noqa: E402

# ----------------------------------------------------------------------------- costs


def test_the_price_card_converts_tokens_to_dollars_by_hand():
    card = K.PriceCard(input_per_m=2.0, output_per_m=10.0)
    assert card.per_request(1_000, 100) == pytest.approx(0.002 + 0.001) and card.per_1000(
        1_000, 100
    ) == pytest.approx(3.0)
    assert card.per_1000(0, 0) == 0 and K.PriceCard(0, 0).per_1000(10**6, 10**6) == 0


def test_a_shorter_prompt_is_cheaper_in_proportion_to_the_input_price():
    card = K.PriceCard(input_per_m=1.0, output_per_m=1.0)
    long, short = card.per_1000(593, 64), card.per_1000(85, 88)
    assert short < long and card.per_1000(593, 0) / card.per_1000(85, 0) == pytest.approx(593 / 85)


def test_self_hosted_cost_is_time_times_the_hourly_price():
    assert K.compute_cost_per_1000(1.8, 0.2) == pytest.approx(1000 * 1.8 * 0.2 / 3600)
    assert K.compute_cost_per_1000(0, 5) == 0
    assert K.training_cost(150_000, 3, 1200, 0.2) == pytest.approx(1200 * 0.2 / 3600)


def test_break_even_is_the_one_off_cost_over_the_saving_and_infinite_without_one():
    assert K.break_even_requests(1.2, 0.002) == pytest.approx(600)
    assert math.isinf(K.break_even_requests(1.0, 0)) and math.isinf(
        K.break_even_requests(1.0, -0.1)
    )


# ----------------------------------------------------------------------------- the report


def summary(exact=0.5, valid=0.9, fields=0.7, prompt=100.0, seconds=1.5):
    return {
        "valid_order": valid,
        "exact": exact,
        "exact_ci": (exact, exact - 0.1, exact + 0.1),
        "field_accuracy": fields,
        "prompt_tokens": prompt,
        "new_tokens": 80.0,
        "seconds": seconds,
    }


def fake_metrics():
    return {
        "task": "t",
        "model": "m",
        "training": "tr",
        "systems": {
            "base": {"human": summary(0.03), "synthetic": summary(0.1)},
            "tuned": {"human": summary(0.6), "synthetic": summary(0.99)},
        },
        "compare": {
            "n": 38,
            "diff_exact": 0.57,
            "exact_low": 0.4,
            "exact_high": 0.73,
            "p_exact": 0.0001,
            "wins": 22,
            "losses": 0,
            "ties": 16,
            "diff_fields": 0.4,
            "fields_low": 0.3,
            "fields_high": 0.5,
        },
        "costs": {
            "base": {"prompt_tokens": 593, "new_tokens": 64, "hosted": 0.39, "self_hosted": 0.1},
            "tuned": {"prompt_tokens": 85, "new_tokens": 88, "hosted": 0.17, "self_hosted": 0.1},
        },
        "break_even": {"one_off": 0.07, "requests": 318.0},
        "forgetting": "probes 40% -> 20%",
        "assumptions": {"card": "a card", "list": ["a1", "a2"]},
        "not_run": ["n1"],
        "limits": ["l1"],
    }


def test_the_report_contains_every_section_and_the_numbers_it_was_given():
    text = R.build_report(fake_metrics())
    for heading in (
        "## Results",
        "## What it costs",
        "## Forgetting",
        "## Assumptions",
        "## Not run",
        "## Limits",
    ):
        assert heading in text
    assert (
        "+0.57 [+0.40, +0.73]" in text
        and "better on 22, worse on 0, tied on 16" in text
        and "about 318 requests" in text
    )
    assert "| tuned | 100% " not in text and "- a1" in text and "- n1" in text and "- l1" in text


def test_the_markdown_table_has_a_header_a_separator_and_one_row_each():
    t = R.md_table(["a", "b"], [["1", "2"], ["3", "4"]]).splitlines()
    assert t == ["| a | b |", "|---|---|", "| 1 | 2 |", "| 3 | 4 |"]


# ----------------------------------------------------------------------------- the reply cache


class Boom:
    def __call__(self, *a, **k):
        raise AssertionError("generated although the reply was cached")


def test_cached_runs_generate_once_and_re_score_from_stored_replies(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "CACHE", tmp_path)
    import infer as I
    import orders as O

    calls = []

    def fake_complete(model, tok, messages, max_new_tokens=160):
        calls.append(1)
        return O.order_json(O.HUMAN_EMAILS[0][1]), 80, 88, 1.0

    monkeypatch.setattr(I, "complete", fake_complete)
    items = O.HUMAN_EMAILS[:3]
    first = W.cached_run("sys", None, None, items, I.tuned_messages, "t")
    assert len(calls) == 3 and first[0].score.exact and not first[1].score.exact, (
        "the same reply is right for email 0 and wrong for the others"
    )
    monkeypatch.setattr(I, "complete", Boom())
    again = W.cached_run("sys", None, None, items, I.tuned_messages, "t")
    assert [r.reply for r in again] == [r.reply for r in first] and [
        r.score.exact for r in again
    ] == [r.score.exact for r in first]
    assert len(list(tmp_path.glob("*.json"))) == 1 and set(
        json.loads(next(tmp_path.glob("*.json")).read_text())
    ) == {W.key(e) for e, _ in items}


def test_cache_keys_differ_per_system_and_email():
    assert W.key("a", "b") != W.key("a", "c") and W.key("x") == W.key("x")


def test_the_quick_pipeline_runs_end_to_end_and_writes_a_report(capsys):
    """A smoke test of the whole pipeline on a handful of items (data -> a few training steps -> evaluation -> cost model -> report). It does not check accuracy."""
    pytest.importorskip("transformers")
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"SmolLM2 is not available: {exc}")
    W.main(["--quick"])
    out = capsys.readouterr().out
    report = W.OUT / "w10_report_quick.md"
    assert (
        report.exists()
        and "## Results" in out
        and "## Assumptions" in out
        and "not run" in out.lower()
    )
    assert (
        "fine-tuned (LoRA, merged)" in report.read_text()
        and "assumed hosted-style card" in report.read_text()
    )
