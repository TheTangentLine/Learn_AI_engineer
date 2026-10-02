"""Tests for Week 7 Day 2: the labelled set is what it claims to be, the split is leak-free, and the calibration machinery
detects each judge pathology (lenient, verbose-biased, position-biased) when we plant it."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import day2_solution as d2  # noqa: E402
import judge_data as jd  # noqa: E402

from common import llm_judge as lj  # noqa: E402
from common.fake import fake_llm  # noqa: E402

ITEMS = jd.build_items()


# ----------------------------------------------------------------------------- the labelled set


def test_shape_and_label_semantics():
    assert (
        len(ITEMS) == 80
        and len({i.id for i in ITEMS}) == 80
        and len({i.reply for i in ITEMS}) == 80
    )
    assert Counter(i.kind for i in ITEMS) == {
        "good": 24,
        "decoy": 8,
        **{f"defect:{n}": 8 for n in jd.NAMES},
    }
    for i in ITEMS:
        if i.kind in ("good", "decoy"):
            assert all(i.labels.values()), i.id
        else:
            failed = [n for n, ok in i.labels.items() if not ok]
            assert failed == [i.kind.split(":")[1]], (i.id, failed)
        assert set(i.labels) == set(jd.NAMES)


def test_each_defect_really_contains_its_defect_and_nothing_else_changed_for_the_other_criteria():
    by_id = {i.id: i for i in ITEMS}
    for si, s in enumerate(jd.SITUATIONS):
        base = by_id[f"s{si}-good0"].reply
        assert jd.PROMISE_ADD in by_id[f"s{si}-bad-no_promise"].reply and by_id[
            f"s{si}-bad-no_promise"
        ].reply.startswith(base)
        assert "card number" in by_id[f"s{si}-bad-no_secret_request"].reply
        assert jd.RUDE_PREFIX in by_id[f"s{si}-bad-polite"].reply
        assert by_id[f"s{si}-bad-next_step"].reply == s.cores[0], "the next step is simply missing"
        assert s.irrelevant in by_id[f"s{si}-bad-on_topic"].reply
        old, new = s.perturb
        g = by_id[f"s{si}-bad-grounded"].reply
        assert new in g and old not in g and new not in s.context + " " + s.question, (
            "the replacement really is NOT in the context"
        )
        assert old in s.context or old in s.question or old in s.cores[0]


def test_the_decoy_is_a_much_longer_but_equally_good_reply():
    by_id = {i.id: i for i in ITEMS}
    for si in range(len(jd.SITUATIONS)):
        decoy, good = by_id[f"s{si}-decoy"], by_id[f"s{si}-good0"]
        assert decoy.length > good.length + 40 and decoy.is_good and good.is_good


def test_the_rule_judge_agrees_with_the_construction_labels_which_cross_validates_them():
    truth = {n: [i.labels[n] for i in ITEMS] for n in jd.NAMES}
    pred = {n: [jd.rule_judge(i)[n] for i in ITEMS] for n in jd.NAMES}
    rs = lj.calibrate_rubric(truth, pred)
    for n in ("no_promise", "no_secret_request", "polite"):
        assert rs[n].accuracy == 1.0, n
    assert all(r.accuracy >= 0.83 for r in rs.values()) and rs["next_step"].accuracy >= 0.95


def test_rule_judge_accepts_plain_text_too_and_has_sensible_edges():
    ok = jd.rule_judge(
        "Your refund of $20.00 (RF-0007) is issued. Reply here if it has not arrived.",
        context="Refund issued: RF-0007 ($20.00)",
        question="Where is it?",
    )
    assert all(ok.values()), ok
    bad = jd.rule_judge("I guarantee $99.00. Send me your password.", context="x", question="y")
    assert not bad["grounded"] and not bad["no_promise"] and not bad["no_secret_request"]
    assert jd.rule_judge("").get("next_step") is False


# ----------------------------------------------------------------------------- the group split


def test_group_split_never_puts_a_situation_on_both_sides_and_is_deterministic():
    dev, test = d2.group_split(ITEMS)
    assert {i.situation for i in dev}.isdisjoint({i.situation for i in test}) and len(dev) + len(
        test
    ) == 80
    assert (
        (len(dev), len(test)) == (50, 30)
        or (len(dev), len(test)) == (40, 40)
        or len(dev) in (40, 50)
    )
    assert d2.group_split(ITEMS) == (dev, test) and d2.group_split(ITEMS, seed=3) != (dev, test)
    for side in (dev, test):
        assert {i.kind for i in side} >= {f"defect:{n}" for n in jd.NAMES} | {"good", "decoy"}, (
            "every defect type is on both sides"
        )


# ----------------------------------------------------------------------------- pointwise machinery with planted pathologies


def truthful(item):
    return dict(item.labels)


def lenient(item):
    return dict.fromkeys(jd.NAMES, True)


def test_a_perfect_judge_gets_kappa_one_and_a_lenient_judge_is_exposed_by_tnr_not_accuracy():
    perfect = d2.reports(truthful, ITEMS)
    assert all(r.kappa == 1.0 for r in perfect.values()) and d2.mean_kappa(perfect) == 1.0
    lax = d2.reports(lenient, ITEMS)
    for r in lax.values():
        assert (
            r.accuracy == 0.9
            and r.tnr == 0.0
            and r.kappa == 0.0
            and r.leniency == pytest.approx(0.1)
        )
    text = d2.format_reports(lax)
    assert "grounded" in text and "mean kappa 0.00" in text


def test_verbosity_bias_is_detected_when_a_judge_passes_long_replies():
    median = sorted(i.length for i in ITEMS)[len(ITEMS) // 2]

    def long_is_good(item):
        return dict.fromkeys(jd.NAMES, item.length > median)

    biased = d2.bias_summary(long_is_good, ITEMS)
    fair = d2.bias_summary(truthful, ITEMS)
    assert max(biased.values()) > 0.2 and max(abs(v) for v in fair.values()) < 0.2
    assert set(biased) == set(jd.NAMES)


def test_score_aligns_truth_and_predictions_per_criterion():
    truth, pred = d2.score(truthful, ITEMS[:5])
    assert all(len(v) == 5 for v in truth.values()) and truth == pred


# ----------------------------------------------------------------------------- pairs


def test_pairs_have_a_known_winner_balanced_positions_and_decoy_ties():
    pairs = d2.build_pairs(ITEMS)
    assert (
        len(pairs) == 8 * 7
        and d2.build_pairs(ITEMS) == pairs
        and d2.build_pairs(ITEMS, seed=1) != pairs
    )
    defect = [p for p in pairs if p["truth"] != "tie"]
    ties = [p for p in pairs if p["truth"] == "tie"]
    assert len(defect) == 48 and len(ties) == 8 and {p["kind"] for p in ties} == {"decoy-vs-good"}
    assert 0.3 < sum(p["truth"] == "A" for p in defect) / 48 < 0.7, (
        "the good reply sits in either slot"
    )
    for p in ties:
        longer = p[p["longer"].lower()]
        shorter = p["b" if p["longer"] == "A" else "a"]
        assert len(longer.split()) > len(shorter.split()) + 40
    by_id = {i.reply: i for i in ITEMS}
    for p in defect:
        winner, loser = by_id[p[p["truth"].lower()]], by_id[p["b" if p["truth"] == "A" else "a"]]
        assert winner.is_good and not loser.is_good


def oracle_pair_judge(prompt):
    """A judge that KNOWS the construction labels: picks the reply with more satisfied criteria, a tie when equal."""
    first = prompt.split("<first>")[1].split("</first>")[0].strip()
    second = prompt.split("<second>")[1].split("</second>")[0].strip()
    by_reply = {i.reply: sum(i.labels.values()) for i in ITEMS}
    a, b = by_reply[first], by_reply[second]
    return json.dumps({"winner": "first" if a > b else "second" if b > a else "tie"})


def longer_wins(prompt):
    first = prompt.split("<first>")[1].split("</first>")[0]
    second = prompt.split("<second>")[1].split("</second>")[0]
    if len(first) == len(second):
        return '{"winner": "tie"}'
    return json.dumps({"winner": "first" if len(first) > len(second) else "second"})


def pairwise(responder):
    pairs = d2.build_pairs(ITEMS)
    with fake_llm([(r"(?s).*", responder)]):
        results = d2.run_pairwise(pairs, "anthropic")
    return pairs, results, d2.pairwise_report(pairs, results)


def test_a_judge_that_sees_the_defects_is_accurate_consistent_and_ties_the_decoys():
    _, _, rep = pairwise(oracle_pair_judge)
    assert (
        rep["accuracy_when_decided"] == 1.0
        and rep["correct"] == 48
        and rep["bias"]["flip_rate"] == 0.0
    )
    assert rep["decoy_ties"] == 8 and rep["longer_wins"] == 0, (
        "equal quality: a fair judge does not prefer the long one"
    )


def test_a_first_slot_judge_produces_only_flips_and_no_verdicts():
    _, _, rep = pairwise(lambda p: '{"winner": "first"}')
    assert rep["correct"] == 0 and rep["wrong"] == 0 and rep["tie"] == 48
    assert rep["bias"]["flip_rate"] == 1.0 and rep["bias"]["first_bias"] == 1.0


def test_a_judge_that_prefers_length_wins_every_decoy_pair_and_is_wrong_on_some_defects():
    _, _, rep = pairwise(longer_wins)
    assert rep["longer_wins"] == 8, "the verbose-but-equal reply beats the good one every time"
    assert rep["bias"]["flip_rate"] == 0.0, (
        "length is a consistent preference, so it is NOT visible as position bias"
    )
    assert rep["wrong"] > 0


# ----------------------------------------------------------------------------- selecting and reporting


def test_the_rubric_variant_is_chosen_on_dev_and_reported_on_test_only(tmp_path):
    dev, test = d2.group_split(ITEMS)
    data = {
        "split": {"dev": [i.id for i in dev], "test": [i.id for i in test]},
        "pointwise": {
            "plain": {i.id: dict(i.labels) for i in ITEMS},
            "reasoning": {i.id: dict.fromkeys(jd.NAMES, True) for i in ITEMS},
        },
        "pairs": d2.build_pairs(ITEMS),
        "pairwise": [
            {"winner": "tie", "flipped": False, "first_order": "tie", "second_order": "tie"}
        ]
        * 56,
    }
    text = d2.evaluate_saved(data)
    assert (
        "chosen on dev: plain" in text
        and "plain: dev mean kappa 1.00" in text
        and "reasoning: dev mean kappa 0.00" in text
    )
    test_block = text.split("== TEST (touched once), LLM judge:")[1].split("== TEST, rule-based")[0]
    assert (
        "mean kappa 1.00" in test_block
        and f"n={len(test)}" in test_block
        and "n=80" not in test_block
    )
    assert "== pairwise" in text and "position bias" in text


def test_the_two_rubric_variants_differ_only_in_the_reasoning_field():
    assert (
        d2.RUBRICS["plain"].criteria == d2.RUBRICS["reasoning"].criteria
        and not d2.RUBRICS["plain"].reasoning
        and d2.RUBRICS["reasoning"].reasoning
    )
    assert re.search(r"grounded", d2.RUBRICS["plain"].system()) and set(d2.RUBRICS) == {
        "plain",
        "reasoning",
    }
