"""Tests for common/llm_judge.py: prompts, parsing, position debiasing, calibration arithmetic, bias detectors."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from common import llm_judge as lj
from common.fake import fake_llm
from common.llm_judge import Criterion, PairwiseResult, Rubric

RUBRIC = Rubric(
    [
        Criterion("grounded", "Every number appears in the CONTEXT."),
        Criterion("polite", "The reply is respectful."),
    ]
)


# ----------------------------------------------------------------------------- pointwise


def test_rubric_prompt_system_and_schema():
    system = RUBRIC.system()
    assert (
        "grounded: Every number appears in the CONTEXT." in system
        and "polite: The reply is respectful." in system
    )
    assert "true if it is satisfied" in system and "keys are exactly the criterion names" in system
    p = RUBRIC.prompt("Hi there", context="Refund issued", question="Where is my refund?")
    assert (
        p.index("<customer_message>") < p.index("<context>") < p.index("<reply>")
        and "Refund issued" in p
    )
    assert "<context>" not in RUBRIC.prompt("Hi") and "<customer_message>" not in RUBRIC.prompt(
        "Hi"
    )
    schema = RUBRIC.schema()
    assert set(schema.model_fields) == {"grounded", "polite"} and all(
        f.is_required() for f in schema.model_fields.values()
    )
    assert RUBRIC.names() == ["grounded", "polite"]


def test_pointwise_returns_one_boolean_per_criterion_and_sends_the_reply_and_context():
    with fake_llm([(r"(?s).*", json.dumps({"grounded": True, "polite": False}))]) as f:
        v = lj.judge_pointwise(
            RUBRIC,
            "Your refund is $20.",
            context="Refund issued: $20.00",
            question="status?",
            provider="anthropic",
        )
    assert v == {"grounded": True, "polite": False} and all(isinstance(x, bool) for x in v.values())
    sent = f.calls[0]
    assert (
        "Your refund is $20." in sent.prompt
        and "Refund issued: $20.00" in sent.prompt
        and "Every number appears" in sent.system
    )


def test_a_judge_that_omits_a_criterion_or_answers_in_prose_raises_instead_of_guessing():
    with fake_llm([(r"(?s).*", json.dumps({"grounded": True}))]), pytest.raises(ValidationError):
        lj.judge_pointwise(RUBRIC, "x", provider="anthropic")
    with (
        fake_llm([(r"(?s).*", "I think it is fine.")]),
        pytest.raises((ValidationError, ValueError)),
    ):
        lj.judge_pointwise(RUBRIC, "x", provider="anthropic")


# ----------------------------------------------------------------------------- pairwise and position bias


def content_judge(prompt):  # picks whichever slot contains the word GOOD, regardless of position
    first = prompt.split("<first>")[1].split("</first>")[0]
    second = prompt.split("<second>")[1].split("</second>")[0]
    if "GOOD" in first and "GOOD" not in second:
        return '{"winner": "first"}'
    if "GOOD" in second and "GOOD" not in first:
        return '{"winner": "second"}'
    return '{"winner": "tie"}'


def test_a_content_based_judge_is_consistent_in_both_orders():
    with fake_llm([(r"(?s).*", content_judge)]) as f:
        r = lj.judge_pairwise("q", "GOOD reply", "bad reply", provider="anthropic")
        r2 = lj.judge_pairwise("q", "bad reply", "GOOD reply", provider="anthropic")
        tie = lj.judge_pairwise("q", "ok", "fine", provider="anthropic")
    assert (
        (r.winner, r.flipped) == ("A", False)
        and (r2.winner, r2.flipped) == ("B", False)
        and (tie.winner, tie.flipped) == ("tie", False)
    )
    assert len(f.calls) == 6, "two model calls per pair: both orders"
    assert "Do not prefer a reply for being longer" in f.calls[0].system


def test_a_judge_that_always_picks_the_first_slot_is_a_flip_and_reported_as_a_tie():
    with fake_llm([(r"(?s).*", '{"winner": "first"}')]):
        r = lj.judge_pairwise("q", "A text", "B text", provider="anthropic")
    assert r == PairwiseResult("tie", True, "A", "B") and r.flipped


def test_a_judge_that_always_picks_the_second_slot_is_also_position_biased():
    with fake_llm([(r"(?s).*", '{"winner": "second"}')]):
        r = lj.judge_pairwise("q", "A text", "B text", provider="anthropic")
    assert r.winner == "tie" and r.flipped and (r.first_order, r.second_order) == ("B", "A")


def test_unparseable_winner_is_a_tie_and_one_sided_ties_are_not_flips():
    with fake_llm([(r"(?s).*", '{"winner": "banana"}')]):
        assert lj.judge_pairwise("q", "a", "b", provider="anthropic") == PairwiseResult(
            "tie", False, "tie", "tie"
        )
    seq = ['{"winner": "first"}', '{"winner": "tie"}']  # A wins one order, tie in the other
    with fake_llm([(r"(?s).*", seq)]):
        r = lj.judge_pairwise("q", "a", "b", provider="anthropic")
    assert r.winner == "tie" and not r.flipped and (r.first_order, r.second_order) == ("A", "tie")


def test_position_bias_summary():
    rs = [
        PairwiseResult("A", False, "A", "A"),
        PairwiseResult("tie", True, "A", "B"),
        PairwiseResult("tie", True, "A", "B"),
        PairwiseResult("tie", True, "B", "A"),
    ]
    s = lj.position_bias(rs)
    assert s["n"] == 4 and s["flip_rate"] == 0.75 and s["first_bias"] == pytest.approx(2 / 3)
    assert lj.position_bias([]) == {"n": 0, "flip_rate": 0.0, "first_bias": 0.0}
    assert lj.position_bias([PairwiseResult("A", False, "A", "A")]) == {
        "n": 1,
        "flip_rate": 0.0,
        "first_bias": 0.0,
    }


# ----------------------------------------------------------------------------- calibration arithmetic


def labels(tp, fn, fp, tn):
    truth = [True] * (tp + fn) + [False] * (fp + tn)
    pred = [True] * tp + [False] * fn + [True] * fp + [False] * tn
    return truth, pred


def test_calibration_matches_hand_computed_values():
    r = lj.calibrate(*labels(40, 10, 20, 30))
    assert (r.n, r.accuracy) == (100, 0.7) and r.confusion == {
        "tp": 40,
        "fn": 10,
        "fp": 20,
        "tn": 30,
    }
    assert (
        r.tpr == pytest.approx(0.8)
        and r.tnr == pytest.approx(0.6)
        and r.precision == pytest.approx(40 / 60)
    )
    assert (
        r.base_rate == 0.5
        and r.judge_pass_rate == pytest.approx(0.6)
        and r.leniency == pytest.approx(0.1)
    )
    assert r.kappa == pytest.approx(0.4), "po=.7, pe=.5*.6+.5*.4=.5 -> (.7-.5)/.5"
    assert r.accuracy_ci[0] < 0.7 < r.accuracy_ci[1]
    assert "accuracy 70%" in str(r) and "kappa 0.40" in str(r) and "TNR 60%" in str(r)


def test_a_judge_that_passes_everything_has_high_accuracy_on_skewed_data_but_zero_kappa_and_zero_tnr():
    truth, pred = labels(90, 0, 10, 0)
    r = lj.calibrate(truth, pred)
    assert r.accuracy == 0.9 and r.tnr == 0.0 and r.kappa == 0.0, (
        "accuracy flatters a lenient judge; TNR and kappa do not"
    )
    perfect = lj.calibrate(*labels(30, 0, 0, 70))
    assert perfect.kappa == 1.0 and perfect.accuracy == 1.0
    inverted = lj.calibrate([True, True, False, False], [False, False, True, True])
    assert inverted.kappa == -1.0 and inverted.accuracy == 0.0


def test_calibration_validates_its_input_and_handles_one_class():
    with pytest.raises(ValueError, match="same length"):
        lj.calibrate([True], [True, False])
    with pytest.raises(ValueError, match="no items"):
        lj.calibrate([], [])
    r = lj.calibrate([True, True, True], [True, True, True])
    assert r.kappa == 0.0 and r.tpr == 1.0 and r.tnr == 0.0, (
        "no negatives: TNR is undefined and reported as 0"
    )


def test_per_criterion_reports():
    out = lj.calibrate_rubric(
        {"a": [True, False], "b": [True, True]}, {"a": [True, False], "b": [False, False]}
    )
    assert out["a"].accuracy == 1.0 and out["b"].accuracy == 0.0 and set(out) == {"a", "b"}


# ----------------------------------------------------------------------------- bias detectors


def test_verbosity_bias_is_the_within_class_length_preference_of_the_judge():
    truth = [True] * 10 + [False] * 10
    lengths = list(range(10, 20)) * 2
    liked_long = [
        i >= 5 for i in range(10)
    ] * 2  # within BOTH classes the judge passes only the long ones
    assert lj.verbosity_bias(lengths, truth, liked_long) > 0.7
    unbiased = [i % 2 == 0 for i in range(10)] * 2
    assert abs(lj.verbosity_bias(lengths, truth, unbiased)) < 0.3
    # a REAL quality-length relationship (good answers are longer) is not counted as bias when the judge is right
    lengths2 = [30] * 10 + [10] * 10
    assert lj.verbosity_bias(lengths2, truth, truth) == 0.0
    assert lj.verbosity_bias([1, 2], [True, False], [True, False]) == 0.0, "too few items per class"
    assert lj._pearson([1, 1, 1], [0, 1, 0]) == 0.0


def test_split_labelled_is_deterministic_disjoint_and_sized():
    items = list(range(50))
    dev, test = lj.split_labelled(items)
    assert (
        len(dev) == 30
        and len(test) == 20
        and set(dev).isdisjoint(test)
        and sorted(dev + test) == items
    )
    assert lj.split_labelled(items) == (dev, test) and lj.split_labelled(items, seed=1) != (
        dev,
        test,
    )
    assert lj.split_labelled(items, dev_fraction=0.5)[0].__len__() == 25


def test_the_optional_reasoning_field_comes_first_and_is_not_part_of_the_verdict():
    r = Rubric(RUBRIC.criteria, reasoning=True)
    assert (
        'preceded by "reasoning"' in r.system()
        and 'First write a one-sentence "reasoning"' in r.system()
    )
    assert list(r.schema().model_fields) == ["reasoning", "grounded", "polite"]
    assert "reasoning" not in RUBRIC.system() and list(RUBRIC.schema().model_fields) == [
        "grounded",
        "polite",
    ]
    with fake_llm(
        [
            (
                r"(?s).*",
                json.dumps({"reasoning": "The amount matches.", "grounded": True, "polite": True}),
            )
        ]
    ):
        assert lj.judge_pointwise(r, "x", provider="anthropic") == {
            "grounded": True,
            "polite": True,
        }


def test_classes_with_fewer_than_three_items_are_too_small_to_estimate_a_correlation():
    # two items per class would give a "perfect" correlation of +/-1 from pure chance
    assert (
        lj.verbosity_bias([1, 2, 5, 6], [True, True, False, False], [False, True, False, True])
        == 0.0
    )
