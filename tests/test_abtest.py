"""Tests for common/abtest.py. Statistical code is verified three ways: against hand-computed values, against scipy's
independent implementations, and by SIMULATION (does a 95% interval cover 95% of the time? is the false-positive rate 5%?)."""

from __future__ import annotations

import math
from collections import Counter
from statistics import NormalDist

import numpy as np
import pytest

from common import abtest as ab

scipy_stats = pytest.importorskip("scipy.stats")

# ----------------------------------------------------------------------------- assignment


def test_assignment_is_deterministic_sticky_and_independent_of_dict_order():
    arms = {"control": 0.5, "lean": 0.5}
    first = [ab.assign("exp1", f"user{i}", arms) for i in range(200)]
    assert first == [ab.assign("exp1", f"user{i}", arms) for i in range(200)]
    assert first == [
        ab.assign("exp1", f"user{i}", {"lean": 0.5, "control": 0.5}) for i in range(200)
    ]
    assert ab.assign("exp1", 42, arms) == ab.assign("exp1", "42", arms), (
        "ids are compared as strings"
    )


def test_assignment_hits_the_requested_proportions():
    arms = {"a": 0.5, "b": 0.3, "c": 0.2}
    counts = Counter(ab.assign("props", f"u{i}", arms) for i in range(60_000))
    for k, w in arms.items():
        assert abs(counts[k] / 60_000 - w) < 0.01, counts
    assert Counter(ab.assign("x", f"u{i}", {"a": 3, "b": 1}) for i in range(20_000))[
        "a"
    ] / 20_000 == pytest.approx(0.75, abs=0.01), "weights are relative"


def test_different_experiments_are_independent_of_each_other():
    users = [f"u{i}" for i in range(20_000)]
    a = [ab.assign("exp-A", u, {"x": 1, "y": 1}) for u in users]
    b = [ab.assign("exp-B", u, {"x": 1, "y": 1}) for u in users]
    table = Counter(zip(a, b, strict=True))
    chi2 = sum((table[(i, j)] - len(users) / 4) ** 2 / (len(users) / 4) for i in "xy" for j in "xy")
    assert ab.chi2_sf(chi2, 3) > 0.001, (
        table
    )  # a shared salt would put (almost) everyone on the diagonal


def test_assignment_rejects_bad_weights_and_handles_a_single_arm():
    with pytest.raises(ValueError):
        ab.assign("e", "u", {})
    with pytest.raises(ValueError):
        ab.assign("e", "u", {"a": 0, "b": 0})
    with pytest.raises(ValueError):
        ab.assign("e", "u", {"a": -1, "b": 2})
    assert ab.assign("e", "u", {"only": 1}) == "only"
    assert Counter(ab.assign("e", f"u{i}", {"a": 1, "b": 0}) for i in range(500)) == {"a": 500}, (
        "a zero weight gets nobody"
    )


# ----------------------------------------------------------------------------- distributions


@pytest.mark.parametrize(
    "df,x", [(1, 3.841459), (2, 5.991465), (3, 7.814728), (5, 11.0705), (10, 18.307)]
)
def test_chi2_sf_matches_the_textbook_five_percent_points(df, x):
    assert ab.chi2_sf(x, df) == pytest.approx(0.05, abs=2e-4)


@pytest.mark.parametrize("df", [1, 2, 3, 4, 7, 12, 30])
@pytest.mark.parametrize("x", [0.01, 0.5, 1.0, 3.0, 9.0, 25.0, 80.0])
def test_chi2_sf_agrees_with_scipy_over_a_grid(df, x):
    assert ab.chi2_sf(x, df) == pytest.approx(scipy_stats.chi2.sf(x, df), rel=1e-9, abs=1e-12)


def test_chi2_sf_edges():
    assert ab.chi2_sf(0, 3) == 1.0 and ab.chi2_sf(-1, 3) == 1.0
    assert ab.chi2_sf(1e6, 2) < 1e-300 or ab.chi2_sf(1e6, 2) == 0.0


def test_normal_two_sided_p():
    assert ab.norm_two_sided_p(1.959964) == pytest.approx(0.05, abs=1e-6)
    assert (
        ab.norm_two_sided_p(-1.959964) == ab.norm_two_sided_p(1.959964)
        and ab.norm_two_sided_p(0) == 1.0
    )


# ----------------------------------------------------------------------------- sample ratio mismatch


def test_srm_accepts_a_split_that_is_just_noise_and_flags_a_broken_one():
    ok = ab.srm_check({"control": 5040, "lean": 4960}, {"control": 1, "lean": 1})
    assert ok.ok and ok.p > 0.1 and ok.expected == {"control": 5000, "lean": 5000}
    broken = ab.srm_check({"control": 5300, "lean": 4700}, {"control": 1, "lean": 1})
    assert not broken.ok and broken.p < 0.001
    assert broken.chi2 == pytest.approx(((300**2) / 5000) * 2)


def test_srm_agrees_with_scipy_and_handles_three_arms_and_unequal_weights():
    obs, w = {"a": 480, "b": 330, "c": 190}, {"a": 0.5, "b": 0.3, "c": 0.2}
    r = ab.srm_check(obs, w)
    ref = scipy_stats.chisquare([480, 330, 190], [500, 300, 200])
    assert r.chi2 == pytest.approx(ref.statistic) and r.p == pytest.approx(ref.pvalue, rel=1e-9)
    assert ab.srm_check({"a": 100}, {"a": 1, "b": 1}).ok is False, (
        "an arm with no users at all is a mismatch"
    )


def test_srm_alert_level_is_configurable_and_empty_data_is_an_error():
    r = ab.srm_check({"a": 5120, "b": 4880}, {"a": 1, "b": 1}, alpha=0.001)
    assert (
        r.p < 0.05
        and r.ok
        and not ab.srm_check({"a": 5120, "b": 4880}, {"a": 1, "b": 1}, alpha=0.05).ok
    )
    with pytest.raises(ValueError):
        ab.srm_check({}, {"a": 1})


def test_srm_false_alarm_rate_is_about_alpha_under_correct_assignment():
    arms = {"a": 0.5, "b": 0.5}
    alarms = 0
    for k in range(300):
        counts = Counter(ab.assign(f"srm-{k}", f"u{i}", arms) for i in range(2000))
        alarms += not ab.srm_check(counts, arms, alpha=0.05).ok
    assert alarms / 300 < 0.10, alarms  # nominal 5%


# ----------------------------------------------------------------------------- comparing rates


def test_wilson_agrees_with_scipy_and_behaves_at_the_extremes():
    for s, n in [(0, 10), (10, 10), (3, 20), (50, 100), (480, 1000), (1, 1000)]:
        ref = scipy_stats.binomtest(s, n).proportion_ci(method="wilson")
        lo, hi = ab.wilson(s, n)
        assert lo == pytest.approx(ref.low, abs=1e-9) and hi == pytest.approx(ref.high, abs=1e-9)
    assert ab.wilson(0, 0) == (0.0, 1.0)
    assert ab.wilson(0, 10)[0] == 0.0 and ab.wilson(10, 10)[1] == 1.0


def test_compare_rates_by_hand():
    r = ab.compare_rates(48, 80, 56, 70)  # control 60%, treatment 80%
    assert r["control_rate"] == 0.6 and r["treat_rate"] == 0.8 and r["diff"] == pytest.approx(0.2)
    pooled = 104 / 150
    se = math.sqrt(pooled * (1 - pooled) * (1 / 80 + 1 / 70))
    assert r["z"] == pytest.approx(0.2 / se) and r["p"] == pytest.approx(
        2 * (1 - NormalDist().cdf(0.2 / se))
    )
    l1, u1 = scipy_stats.binomtest(56, 70).proportion_ci(method="wilson")
    l2, u2 = scipy_stats.binomtest(48, 80).proportion_ci(method="wilson")
    assert r["ci"][0] == pytest.approx(0.2 - math.sqrt((0.8 - l1) ** 2 + (u2 - 0.6) ** 2))
    assert r["ci"][1] == pytest.approx(0.2 + math.sqrt((u1 - 0.8) ** 2 + (0.6 - l2) ** 2))
    assert r["ci"][0] > 0 and r["p"] < 0.05, (
        "the interval excludes zero exactly when the test is significant here"
    )


def test_compare_rates_identical_arms_and_degenerate_cases():
    r = ab.compare_rates(50, 100, 50, 100)
    assert r["diff"] == 0 and r["z"] == 0 and r["p"] == 1.0 and r["ci"][0] < 0 < r["ci"][1]
    zero = ab.compare_rates(0, 50, 0, 50)
    assert zero["p"] == 1.0 and zero["diff"] == 0
    with pytest.raises(ValueError):
        ab.compare_rates(1, 0, 1, 10)
    flipped = ab.compare_rates(60, 100, 40, 100)
    assert flipped["diff"] < 0 and flipped["z"] < 0 and flipped["ci"][1] < 0


@pytest.mark.parametrize(
    "p_control,p_treat,n", [(0.5, 0.5, 100), (0.1, 0.1, 150), (0.9, 0.85, 200)]
)
def test_the_newcombe_interval_covers_the_true_difference_about_95_percent_of_the_time(
    p_control, p_treat, n
):
    rng = np.random.default_rng(0)
    sims = 4000
    a, b = rng.binomial(n, p_control, sims), rng.binomial(n, p_treat, sims)
    hits = sum(
        (lo := ab.compare_rates(int(x), n, int(y), n)["ci"])[0] <= p_treat - p_control <= lo[1]
        for x, y in zip(a, b, strict=True)
    )
    assert 0.935 <= hits / sims <= 0.975, hits / sims


def test_bootstrap_diff_recovers_a_known_shift_and_is_reproducible():
    rng = np.random.default_rng(1)
    a, b = rng.normal(10, 2, 400), rng.normal(11, 2, 400)
    r = ab.bootstrap_diff(a, b, seed=3)
    assert r["diff"] == pytest.approx(b.mean() - a.mean()) and 0.6 < r["diff"] < 1.4
    assert r["ci"][0] < r["diff"] < r["ci"][1] and r["ci"][0] > 0
    assert ab.bootstrap_diff(a, b, seed=3) == r and ab.bootstrap_diff(a, b, seed=4)["ci"] != r["ci"]
    null = ab.bootstrap_diff(a, a[::-1], seed=0)
    assert null["diff"] == pytest.approx(0) and null["ci"][0] < 0 < null["ci"][1]
    with pytest.raises(ValueError):
        ab.bootstrap_diff([], [1.0])


def test_bootstrap_diff_interval_covers_95_percent_of_the_time():
    rng = np.random.default_rng(5)
    hits = 0
    runs = 300
    for i in range(runs):
        a, b = rng.exponential(3, 60), rng.exponential(3, 60)  # skewed, equal means
        lo, hi = ab.bootstrap_diff(a, b, n_boot=600, seed=i)["ci"]
        hits += lo <= 0 <= hi
    assert 0.90 <= hits / runs <= 0.99, hits / runs


# ----------------------------------------------------------------------------- planning: sample size and power


def test_sample_size_by_hand_and_properties():
    assert (
        ab.sample_size(0.5, 0.1) == 388
    )  # (1.96*sqrt(2*.55*.45) + .8416*sqrt(.25+.24))^2 / .1^2 = 387.3
    assert ab.sample_size(0.5, 0.05) > 3.5 * ab.sample_size(0.5, 0.1), (
        "half the effect needs about four times the users"
    )
    assert ab.sample_size(0.5, 0.1, power=0.9) > ab.sample_size(0.5, 0.1)
    assert ab.sample_size(0.5, 0.1, alpha=0.01) > ab.sample_size(0.5, 0.1)
    assert ab.sample_size(0.5, -0.1) == ab.sample_size(0.4, 0.1) or ab.sample_size(0.5, -0.1) > 0
    for bad in [(0, 0.1), (0.5, 0.0), (0.95, 0.1)]:
        with pytest.raises(ValueError):
            ab.sample_size(*bad)


@pytest.mark.parametrize("p,mde", [(0.5, 0.1), (0.2, 0.05), (0.8, -0.1)])
def test_the_sample_size_formula_delivers_the_requested_power_in_simulation(p, mde):
    n = ab.sample_size(p, mde, power=0.8)
    assert ab.empirical_power(p, p + mde, n, n_sims=30_000, seed=7) == pytest.approx(0.8, abs=0.015)
    assert ab.empirical_power(p, p + mde, n // 3, n_sims=10_000, seed=7) < 0.5, (
        "a third of the users is badly underpowered"
    )


def test_with_no_effect_the_rejection_rate_is_alpha():
    assert ab.empirical_power(0.5, 0.5, 300, n_sims=40_000, seed=3) == pytest.approx(
        0.05, abs=0.006
    )


# ----------------------------------------------------------------------------- peeking and group sequential boundaries


def test_peeking_at_every_look_inflates_false_positives_far_above_five_percent():
    r1 = ab.peeking_false_positive_rate(
        0.5, 400, looks=1, seed=0
    )  # n large enough that the z-test is not lumpy
    assert (
        r1["fixed_horizon"] == pytest.approx(0.05, abs=0.007)
        and r1["peek_every_look"] == r1["fixed_horizon"]
    )
    r10 = ab.peeking_false_positive_rate(0.5, 100, looks=10, seed=0)
    assert r10["fixed_horizon"] == pytest.approx(0.05, abs=0.007), (
        "the final-look test alone is still fine"
    )
    assert 0.17 <= r10["peek_every_look"] <= 0.23, (
        r10
    )  # stop at the first p < 0.05 over ten looks: about 19%
    r3 = ab.peeking_false_positive_rate(0.5, 100, looks=3, seed=0)
    assert 0.09 <= r3["peek_every_look"] <= 0.12, r3


@pytest.mark.parametrize("looks", [2, 5, 10])
@pytest.mark.parametrize("kind", ["pocock", "obf"])
def test_group_sequential_boundaries_hold_the_overall_false_positive_rate_at_alpha(looks, kind):
    gs = ab.GroupSequential(looks, kind=kind)
    assert gs.false_positive_rate(0.5, 200, n_sims=40_000) == pytest.approx(0.05, abs=0.007)


def test_pocock_and_obrien_fleming_boundary_shapes_and_known_values():
    pk, ob = ab.GroupSequential(5, kind="pocock"), ab.GroupSequential(5, kind="obf")
    assert len(set(pk.bounds)) == 1 and 2.38 <= pk.bounds[0] <= 2.45, (
        pk.bounds
    )  # the published 5-look value is about 2.41
    assert ob.bounds[0] > ob.bounds[1] > ob.bounds[2] > ob.bounds[3] > ob.bounds[4], (
        "O'Brien-Fleming relaxes over time"
    )
    assert 4.3 <= ob.bounds[0] <= 4.8 and 1.98 <= ob.bounds[-1] <= 2.10, ob.bounds
    assert ab.GroupSequential(1).bounds[0] == pytest.approx(1.96, abs=0.03), (
        "one look is the ordinary test"
    )
    assert ob.bounds[-1] < pk.bounds[-1], (
        "OBF spends almost nothing early, so its last look is nearly the fixed-horizon test"
    )


def test_group_sequential_decisions_and_validation():
    gs = ab.GroupSequential(3, kind="pocock")
    c = gs.boundary(1)
    assert gs.check(1, c + 0.01) == "stop-win" and gs.check(1, -c - 0.01) == "stop-lose"
    assert gs.check(1, c - 0.01) == "continue" and gs.check(3, 0.5) == "inconclusive"
    assert gs.check(2, c) == "stop-win", "the boundary itself counts"
    for bad in (0, 4):
        with pytest.raises(ValueError):
            gs.boundary(bad)
    with pytest.raises(ValueError):
        ab.GroupSequential(0)
    with pytest.raises(ValueError):
        ab.GroupSequential(3, kind="nonsense")


def test_sequential_designs_trade_power_for_early_stopping():
    n_fixed = ab.sample_size(0.5, 0.1)
    per_look = math.ceil(n_fixed / 5)
    pk = ab.GroupSequential(5, kind="pocock")
    ob = ab.GroupSequential(5, kind="obf")
    fixed_power = ab.empirical_power(0.5, 0.6, n_fixed, n_sims=20_000)
    assert pk.power(0.5, 0.6, per_look) < fixed_power, (
        "same maximum sample, but a stricter test at every look"
    )
    assert ob.power(0.5, 0.6, per_look) > pk.power(0.5, 0.6, per_look) - 0.02, (
        "OBF keeps nearly all of the fixed-horizon power"
    )
    assert pk.power(0.5, 0.6, per_look) > 0.6


# ----------------------------------------------------------------------------- the decision rule


def rate(lo, hi):
    return {"ci": (lo, hi)}


def test_decide_ships_blocks_stops_or_keeps_running():
    assert ab.decide(rate(0.02, 0.2), {}) == "ship"
    assert ab.decide(rate(-0.05, 0.2), {}) == "keep-running"
    assert ab.decide(rate(-0.2, -0.02), {}) == "stop"
    assert ab.decide(rate(0.02, 0.2), {"escalations": rate(0.01, 0.1)}) == "block", (
        "a harmed guardrail beats a winning primary"
    )
    assert ab.decide(rate(0.02, 0.2), {"escalations": rate(-0.05, 0.1)}) == "ship", (
        "an inconclusive guardrail does not block"
    )
    assert ab.decide(rate(0.02, 0.2), {"cost": rate(0.01, 0.1)}, tolerance={"cost": 0.02}) == "ship"
    assert (
        ab.decide(rate(0.02, 0.2), {"cost": rate(0.03, 0.1)}, tolerance={"cost": 0.02}) == "block"
    )
    assert ab.decide(rate(0.0, 0.2), {}) == "keep-running", (
        "an interval that only touches zero is not a win"
    )


def test_decide_does_not_stop_on_an_interval_that_only_touches_zero_from_below():
    assert ab.decide(rate(-0.2, 0.0), {}) == "keep-running"


def test_bootstrap_diff_handles_arms_of_different_sizes():
    rng = np.random.default_rng(11)
    a, b = rng.normal(5, 1, 120), rng.normal(5.5, 1, 700)
    r = ab.bootstrap_diff(a, b, seed=1)
    assert (
        r["ci"][0] < 0.5 < r["ci"][1]
        and r["control_mean"] == pytest.approx(a.mean())
        and r["treat_mean"] == pytest.approx(b.mean())
    )
    width_small = r["ci"][1] - r["ci"][0]
    big = ab.bootstrap_diff(rng.normal(5, 1, 700), rng.normal(5.5, 1, 700), seed=1)
    assert big["ci"][1] - big["ci"][0] < width_small, (
        "more data in the small arm narrows the interval"
    )
