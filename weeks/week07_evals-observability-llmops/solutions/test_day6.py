"""Tests for Week 7 Day 6: the replayed experiment samples what it claims, the harness's statistics are validated against a
KNOWN true effect, and the feedback and drift helpers behave on hand-built data."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("opentelemetry.sdk")
sys.path.insert(0, str(Path(__file__).parent))

import day5_solution as d5  # noqa: E402
import day6_solution as d6  # noqa: E402
import scripted  # noqa: E402
from support_system.system import SupportSystem  # noqa: E402

from common import abtest as ab  # noqa: E402
from common import drift  # noqa: E402
from common.fake import fake_llm  # noqa: E402

C, T = d6.CONTROL, d6.TREATMENT


def synthetic(
    control_pass: int = 26,
    treat_pass: int = 34,
    *,
    treat_escalated: int = 4,
    control_escalated: int = 4,
    n: int = 50,
):
    """Two variants over n cases with a KNOWN effect: the first k cases pass; the first m cases escalate."""
    mk = lambda passes, esc, cost: {  # noqa: E731
        f"case-{i:02d}": d6.Outcome(i < passes, cost, 1.0, i < esc) for i in range(n)
    }
    return {
        C: mk(control_pass, control_escalated, 0.0011),
        T: mk(treat_pass, treat_escalated, 0.0006),
    }


# ----------------------------------------------------------------------------- outcomes and the true effect


def test_true_effect_is_the_population_mean_difference_by_hand():
    o = synthetic(26, 34)
    t = d6.true_effect(o[C], o[T])
    assert t["pass"] == pytest.approx(0.16) and t["cost"] == pytest.approx(0.0006 - 0.0011)
    assert t["latency"] == 0 and t["escalated"] == 0


def test_outcomes_load_from_saved_variant_files(tmp_path):
    runs = {}
    for name in ("base", "direct"):
        with fake_llm(scripted.rules()):
            r, s = d5.run_variant_traced(name, provider="anthropic")
        d5.save_variant(name, r, s, tmp_path)
        runs[name] = r
    out = d6.load_outcomes("direct", tmp_path)
    assert len(out) == 50 and all(isinstance(o, d6.Outcome) for o in out.values())
    assert {c for c, o in out.items() if o.passed} == {
        r.case_id for r in runs["direct"] if r.passed
    }
    no_model = ("human-", "off-topic-")  # the keyword router answers these without any model call
    assert all(o.cost_usd == 0 for c, o in out.items() if c.startswith(no_model))
    assert sum(o.cost_usd > 0 for o in out.values()) >= 35 and all(
        o.latency_s > 0 for o in out.values() if o.cost_usd > 0
    )
    assert {c for c in out if c.startswith("human-")} <= {
        c for c, o in out.items() if o.escalated
    }, (
        "human requests escalate by design (and so do the two pressure cases the keyword router sends to a human)"
    )
    assert (
        d6.load_outcomes("base", tmp_path)["billing-small-00"].cost_usd
        > out["billing-small-00"].cost_usd
    ), "direct is cheaper"


# ----------------------------------------------------------------------------- the replay


def test_replay_is_deterministic_and_looks_up_the_real_outcome_of_the_assigned_arm():
    o = synthetic()
    a, b = d6.replay(o, 500, seed=3), d6.replay(o, 500, seed=3)
    assert a == b and a != d6.replay(o, 500, seed=4)
    for row in a:
        assert (
            row["passed"] == o[row["arm"]][row["case"]].passed
            and row["cost"] == o[row["arm"]][row["case"]].cost_usd
        )
    assert Counter(r["arm"] for r in a).keys() == {C, T}


def test_replays_with_different_seeds_assign_users_differently():
    o = synthetic()
    arms = lambda seed: [r["arm"] for r in d6.replay(o, 300, seed=seed)]  # noqa: E731
    users = lambda seed: [r["user"] for r in d6.replay(o, 300, seed=seed)]  # noqa: E731
    assert users(1) == users(2) and arms(1) != arms(2), (
        "user 17 is not always in the same arm across repeated experiments"
    )


def test_replay_respects_weights_and_each_user_keeps_one_arm_per_experiment():
    o = synthetic()
    rows = d6.replay(o, 20_000, weights={C: 3, T: 1}, seed=0)
    assert Counter(r["arm"] for r in rows)[C] / len(rows) == pytest.approx(0.75, abs=0.01)
    assert len({r["user"] for r in rows}) == len(rows)


def test_the_logging_bug_drops_only_the_treatment_arms_escalated_conversations():
    o = synthetic()
    clean, buggy = d6.replay(o, 4000, seed=1), d6.replay(o, 4000, seed=1, drop_escalated_from=T)
    assert len(clean) - len(buggy) == sum(1 for r in clean if r["arm"] == T and r["escalated"])
    assert not any(r["arm"] == T and r["escalated"] for r in buggy)
    assert sum(1 for r in buggy if r["arm"] == C and r["escalated"]) == sum(
        1 for r in clean if r["arm"] == C and r["escalated"]
    )


def test_an_sample_ratio_check_catches_the_logging_bug_only_with_enough_users():
    o = synthetic()
    arms = {C: 1, T: 1}

    def srm_p(n):
        rows = d6.replay(o, n, seed=2, drop_escalated_from=T)
        return ab.srm_check(Counter(r["arm"] for r in rows), arms).p

    assert srm_p(300) > 0.001, "a small experiment cannot see a small bug"
    assert srm_p(20_000) < 0.001


# ----------------------------------------------------------------------------- analysis and the decision


def test_a_real_improvement_ships_and_an_identical_variant_keeps_running():
    win = d6.analyse(d6.replay(synthetic(20, 38), 1500, seed=5))
    assert win["primary"]["ci"][0] > 0 and win["decision"] == "ship"
    assert win["guardrails"]["cost"]["diff"] < 0 and win["guardrails"]["latency"][
        "diff"
    ] == pytest.approx(0)
    same = d6.analyse(d6.replay(synthetic(26, 26), 1500, seed=5))
    assert (
        same["decision"] == "keep-running"
        and same["primary"]["ci"][0] < 0 < same["primary"]["ci"][1]
    )


def test_a_harmed_guardrail_blocks_a_winning_variant_unless_within_tolerance():
    harmful = synthetic(
        20, 38, treat_escalated=20, control_escalated=4
    )  # +32 points of escalations
    res = d6.analyse(d6.replay(harmful, 3000, seed=5))
    assert res["primary"]["ci"][0] > 0 and res["decision"] == "block"
    mild = synthetic(
        20, 38, treat_escalated=5, control_escalated=4
    )  # +2 points: inside the 0.02 tolerance
    assert d6.analyse(d6.replay(mild, 3000, seed=5))["decision"] in ("ship", "keep-running")


def test_the_interval_covers_the_known_truth_and_the_planned_sample_size_delivers_its_power():
    o = synthetic(26, 34)
    need = ab.sample_size(0.52, 0.16)
    r = d6.coverage_and_power(o, 2 * need, n_experiments=400)
    assert r["truth"] == pytest.approx(0.16) and r["n_per_arm"] == need
    assert 0.91 <= r["coverage"] <= 0.99 and 0.72 <= r["power"] <= 0.88, r
    assert r["coverage"] == pytest.approx(1 - r["missed_low"] - r["missed_high"])
    assert 0 < r["missed_low"] < 0.06 and 0 < r["missed_high"] < 0.06, (
        "misses happen on both sides, about 2.5% each"
    )
    assert d6.coverage_and_power(o, 2 * need // 3, n_experiments=300)["power"] < 0.5


def test_stopping_rules_on_an_a_a_test_expose_peeking_and_the_calibrated_designs_do_not():
    r = d6.stopping_rules(synthetic(26, 34), 100, 5, 600, null=True)
    assert r["peek, p<0.05 at any look"] > 0.10 and r["fixed horizon"] < 0.08
    assert r["pocock"] < 0.085 and r["obf"] < 0.085, r


def test_with_a_real_effect_the_sequential_designs_stop_early_and_still_detect_it():
    r = d6.stopping_rules(synthetic(26, 34), 100, 5, 300, null=False)
    assert r["fixed horizon"] > 0.95, (
        "the final look (500 per arm) has ample power for +16 points; the first look (100 per arm) would not"
    )
    assert r["pocock"] > 0.85 and r["obf"] > 0.85
    assert (
        r["avg_users_pocock"] < 0.8 * r["max_users"]
        and r["avg_users_obf"] >= r["avg_users_pocock"] - 20
    ), "Pocock stops earlier, OBF waits"


# ----------------------------------------------------------------------------- implicit signals and feedback into cases


def user(t):
    return {"role": "user", "content": t}


def test_implicit_signals_by_hand():
    h = [
        user("How do I reset my password?"),
        {"role": "assistant", "content": "x"},
        user("how do i reset my password"),
        user("thanks!"),
    ]
    assert d6.implicit_signals(h) == {"turns": 3, "reasks": 1, "thanks": True}
    para = [
        user("my export keeps failing with error 0x5F"),
        user("export keeps failing with error 0x5F again"),
    ]
    assert d6.implicit_signals(para)["reasks"] == 1
    assert d6.implicit_signals([user("refund INV-1001"), user("where is the export button")]) == {
        "turns": 2,
        "reasks": 0,
        "thanks": False,
    }
    assert d6.implicit_signals([]) == {"turns": 0, "reasks": 0, "thanks": False}
    assert (
        d6.implicit_signals([user("Thank you")])["thanks"]
        and not d6.implicit_signals([user("thankless job")])["thanks"]
    )
    tools_in_history = [{"role": "assistant", "content": None, "tool_calls": []}, user("hi")]
    assert d6.implicit_signals(tools_in_history)["turns"] == 1


@pytest.fixture()
def feedback_system(tmp_path):
    with fake_llm(scripted.rules()):
        s = SupportSystem(tmp_path / "s.sqlite", provider="anthropic", triage_mode="rules")
        s.handle("a", "I was charged twice for INV-3001")
        s.feedback("a", -1, reason="unhelpful", comment="too vague")
        s.handle("b", "i was charged twice for inv-3001!")  # the same opening, normalised
        s.feedback("b", -1, reason="wrong", comment="my card is 4111 1111 1111 1111")
        s.handle("c", "How do I reset my password?")
        s.feedback("c", -1, reason="slow")
        s.handle("d", "The export is down with error 0x5F")
        s.feedback("d", 1)
    return s


def test_negative_feedback_becomes_deduplicated_candidate_cases_that_need_a_human_label(
    feedback_system,
):
    cases = d6.cases_from_feedback(feedback_system)
    assert len(cases) == 2, "the thumbs-up is not a candidate and the repeated question is merged"
    first = cases[0]
    assert first["opening"].startswith("I was charged") and sorted(first["reasons"]) == [
        "unhelpful",
        "wrong",
    ]
    assert first["conversations"] == ["a", "b"] and first["needs_label"] is True
    assert "4111" not in json.dumps(cases) and "[card number removed]" in json.dumps(
        first["comments"]
    )
    assert (
        cases[1]["opening"].startswith("How do I reset")
        and cases[1]["reasons"] == ["slow"]
        and cases[1]["comments"] == []
    )


def test_candidate_cases_are_sorted_by_how_many_people_complained_and_apply_the_redactor(
    feedback_system,
):
    cases = d6.cases_from_feedback(
        feedback_system, redact=lambda t: t.replace("INV-3001", "INV-XXXX")
    )
    assert [len(c["conversations"]) for c in cases] == [2, 1], (
        "most-complained-about first, although it sorts last alphabetically"
    )
    assert "INV-3001" not in json.dumps(cases) and "INV-XXXX" in cases[0]["opening"]
    assert (
        d6.cases_from_feedback(
            SupportSystem(feedback_system.store.path + ".other", provider="anthropic")
        )
        == []
    )


# ----------------------------------------------------------------------------- drift helpers


def test_detection_rate_rises_with_the_window_size_and_false_alarms_stay_near_alpha():
    old, new = {"a": 0.5, "b": 0.5}, {"a": 0.3, "b": 0.7}
    rates = [d6.detection_rate(old, new, n, trials=200) for n in (30, 120, 600)]
    assert rates == sorted(rates) and rates[0] < 0.4 and rates[-1] > 0.95
    assert d6.detection_rate(old, old, 600, trials=400, alpha=0.05, seed=3) == pytest.approx(
        0.05, abs=0.03
    )


def test_the_change_happens_exactly_at_the_change_point():
    assert d6.cusum_stream(0.0, 1.0, 3, 6, seed=0) == [0, 0, 0, 1, 1, 1]


def test_cusum_stream_changes_rate_at_the_change_point_and_first_alarm_is_one_based():
    s = d6.cusum_stream(0.05, 0.60, 500, 1000, seed=0)
    assert s == d6.cusum_stream(0.05, 0.60, 500, 1000, seed=0) and len(s) == 1000
    assert sum(s[:500]) / 500 < 0.12 and sum(s[500:]) / 500 > 0.5
    import math

    h = math.log(0.7 / 0.5) * 3
    assert d6.first_alarm([1, 1, 1, 1], drift.BernoulliCusum(0.5, 0.7, threshold=h)) == 3
    assert d6.first_alarm([0, 0, 0], drift.BernoulliCusum(0.5, 0.7, threshold=h)) is None


def test_the_cusum_notices_a_persistent_shift_after_the_change_and_rarely_before():
    """Calibrated for a false alarm about every 500 observations; the change comes at observation 50, so only about
    1 - exp(-50/500) = 10% of streams should alarm before it."""
    h = drift.calibrate_threshold(0.10, 0.20, arl0=500, n_sims=1000, seed=1)
    alarms = [
        d6.first_alarm(
            d6.cusum_stream(0.10, 0.25, 50, 800, seed=k), drift.BernoulliCusum(0.10, 0.20, h)
        )
        for k in range(200)
    ]
    after = [a for a in alarms if a is not None and a > 50]
    before = [a for a in alarms if a is not None and a <= 50]
    assert len(before) <= 0.20 * 200, f"too many early false alarms: {len(before)} of 200"
    assert len(after) + len(before) >= 0.97 * 200, (
        "essentially every simulated change is detected within 800 observations"
    )
    assert np.mean([a - 50 for a in after]) < 120


def test_embedding_windows_shift_when_the_mix_changes():
    try:
        near_a, near_b = d6.embedding_windows(40, 0.1, seed=0)
        far_a, far_b = d6.embedding_windows(40, 0.6, seed=0)
    except Exception as exc:  # the embedding model is not in the local cache
        pytest.skip(f"embedder unavailable: {type(exc).__name__}")
    assert near_a.shape == (40, near_a.shape[1]) and near_b.shape == near_a.shape
    assert np.linalg.norm(far_a.mean(0) - far_b.mean(0)) > np.linalg.norm(
        near_a.mean(0) - near_b.mean(0)
    )


# ----------------------------------------------------------------------------- the report, on the saved real-model runs

needs_runs = pytest.mark.skipif(
    not all((d5.OUT / f"w7d5_{n}.jsonl").exists() for n in (C, T)),
    reason="Day 5 real-model runs not generated",
)


@needs_runs
def test_real_outcomes_have_the_known_effect_the_lesson_quotes():
    o = {n: d6.load_outcomes(n) for n in (C, T)}
    t = d6.true_effect(o[C], o[T])
    assert (
        t["pass"] == pytest.approx(0.16, abs=1e-9)
        and t["cost"] < 0
        and t["latency"] < 0
        and t["escalated"] == 0
    )


@needs_runs
def test_report_runs_and_the_interval_covers_the_truth_in_replays():
    text = d6.report()
    assert "True effect in this population: pass +16.0%" in text and "decision: ship" in text
    import re

    covered = [float(x) for x in re.findall(r"contained the truth ([\d.]+)%", text)]
    assert len(covered) == 3 and all(90 <= c <= 99 for c in covered), covered
