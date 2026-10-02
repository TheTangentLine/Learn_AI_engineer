"""Tests for common/drift.py: distribution distances by hand, the change detector by simulation (false-alarm interval and delay)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from common import drift

scipy_stats = pytest.importorskip("scipy.stats")

# ----------------------------------------------------------------------------- PSI


def test_psi_by_hand_symmetry_and_zero_for_identical():
    e, a = {"x": 50, "y": 50}, {"x": 40, "y": 60}
    expected = (0.4 - 0.5) * math.log(0.4 / 0.5) + (0.6 - 0.5) * math.log(0.6 / 0.5)
    assert drift.psi(e, a) == pytest.approx(expected) and expected == pytest.approx(
        0.040546, abs=1e-6
    )
    assert drift.psi(a, e) == pytest.approx(drift.psi(e, a)), "PSI is symmetric"
    assert drift.psi(e, e) == 0.0
    assert drift.psi({"x": 5, "y": 5}, {"x": 500, "y": 500}) == pytest.approx(0.0), (
        "counts are normalised to proportions"
    )


def test_psi_with_a_new_or_vanished_category_is_large_but_finite():
    v = drift.psi({"billing": 60, "technical": 40}, {"billing": 50, "technical": 30, "other": 20})
    assert math.isfinite(v) and v > 0.25
    gone = drift.psi({"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 1})
    assert math.isfinite(gone) and gone > 0.25
    assert drift.psi({"a": 1}, {"b": 1}) > 10, "completely disjoint distributions"


def test_psi_grows_with_the_shift_and_bands_follow_the_rule_of_thumb():
    base = {"a": 50, "b": 50}
    values = [drift.psi(base, {"a": 50 - s, "b": 50 + s}) for s in (0, 5, 10, 20, 30)]
    assert values == sorted(values) and values[0] == 0
    assert [drift.psi_band(v) for v in (0.0, 0.0999, 0.1, 0.2499, 0.25, 3)] == [
        "stable",
        "stable",
        "moderate",
        "moderate",
        "major",
        "major",
    ]


def test_psi_rejects_an_empty_distribution():
    with pytest.raises(ValueError):
        drift.psi({}, {"a": 1})
    with pytest.raises(ValueError):
        drift.psi({"a": 0}, {"a": 1})


# ----------------------------------------------------------------------------- chi-square homogeneity


def test_chi2_homogeneity_by_hand_and_against_scipy():
    a, b = {"x": 30, "y": 70}, {"x": 50, "y": 50}
    r = drift.chi2_homogeneity(a, b)
    # cells: (30-40)^2/40 + (70-60)^2/60 + (50-40)^2/40 + (50-60)^2/60 = 2.5 + 1.6667 + 2.5 + 1.6667
    assert r["chi2"] == pytest.approx(8.3333, abs=1e-3) and r["df"] == 1 and r["n"] == (100, 100)
    ref = scipy_stats.chi2_contingency([[30, 70], [50, 50]], correction=False)
    assert r["chi2"] == pytest.approx(ref.statistic) and r["p"] == pytest.approx(
        ref.pvalue, rel=1e-6
    )


def test_chi2_homogeneity_with_several_categories_and_unseen_ones():
    a, b = (
        {"billing": 120, "technical": 80, "human": 20, "other": 10},
        {"billing": 90, "technical": 110, "human": 15, "other": 40, "new": 0},
    )
    r = drift.chi2_homogeneity(a, b)
    ref = scipy_stats.chi2_contingency([[120, 80, 20, 10], [90, 110, 15, 40]], correction=False)
    assert r["df"] == 3, "the all-zero category is dropped, not counted as a degree of freedom"
    assert r["chi2"] == pytest.approx(ref.statistic) and r["p"] == pytest.approx(
        ref.pvalue, rel=1e-6
    )
    same = drift.chi2_homogeneity({"a": 40, "b": 60}, {"a": 400, "b": 600})
    assert same["chi2"] == pytest.approx(0, abs=1e-9) and same["p"] == pytest.approx(1)
    with pytest.raises(ValueError):
        drift.chi2_homogeneity({"a": 5}, {"a": 7})


def test_chi2_false_alarm_rate_under_no_drift_is_about_alpha():
    rng = np.random.default_rng(0)
    alarms = 0
    for _ in range(500):
        a = dict(zip("abcd", rng.multinomial(400, [0.4, 0.3, 0.2, 0.1]), strict=True))
        b = dict(zip("abcd", rng.multinomial(400, [0.4, 0.3, 0.2, 0.1]), strict=True))
        alarms += drift.chi2_homogeneity(a, b)["p"] < 0.05
    assert 0.02 <= alarms / 500 <= 0.09, alarms / 500


# ----------------------------------------------------------------------------- embedding centroid test


def blob(rng, centre, n, dim=16, spread=1.0):
    return rng.normal(0, spread, (n, dim)) + centre


def test_centroid_test_is_quiet_without_a_shift_and_loud_with_one():
    rng = np.random.default_rng(0)
    same = drift.centroid_shift_test(blob(rng, 0, 80), blob(rng, 0, 80), n_perm=500, seed=1)
    moved = drift.centroid_shift_test(blob(rng, 0, 80), blob(rng, 0.6, 80), n_perm=500, seed=1)
    assert same["p"] > 0.05 and moved["p"] < 0.01 and moved["distance"] > same["distance"]
    assert moved["p"] >= 1 / 501, "the p-value is never exactly zero"


def test_centroid_test_false_alarm_rate_is_about_alpha_and_is_reproducible():
    rng = np.random.default_rng(3)
    alarms = sum(
        drift.centroid_shift_test(blob(rng, 0, 40), blob(rng, 0, 40), n_perm=200, seed=i)["p"]
        < 0.05
        for i in range(120)
    )
    assert alarms / 120 <= 0.12, alarms
    a, b = blob(rng, 0, 30), blob(rng, 0.3, 30)
    assert drift.centroid_shift_test(a, b, n_perm=300, seed=5) == drift.centroid_shift_test(
        a, b, n_perm=300, seed=5
    )
    with pytest.raises(ValueError):
        drift.centroid_shift_test(np.zeros((0, 3)), np.zeros((2, 3)))


def test_centroid_test_detects_a_change_in_the_mix_of_topics():
    """Two topic clusters; the new sample has more of one of them: category counts can miss this, the centroid moves."""
    rng = np.random.default_rng(2)
    t1, t2 = np.zeros(16), np.ones(16) * 2
    old = np.vstack([blob(rng, t1, 70, spread=0.8), blob(rng, t2, 30, spread=0.8)])
    new = np.vstack([blob(rng, t1, 40, spread=0.8), blob(rng, t2, 60, spread=0.8)])
    assert drift.centroid_shift_test(old, new, n_perm=400, seed=0)["p"] < 0.01


# ----------------------------------------------------------------------------- CUSUM


def test_cusum_accumulates_log_likelihood_ratios_and_resets_at_zero_by_hand():
    c = drift.BernoulliCusum(0.5, 0.7, threshold=1.0)
    hit, miss = math.log(0.7 / 0.5), math.log(0.3 / 0.5)
    assert hit == pytest.approx(0.33647, abs=1e-5) and miss == pytest.approx(-0.51083, abs=1e-5)
    assert not c.update(0) and c.score == 0.0, "a miss at zero stays at zero"
    assert not c.update(1) and c.score == pytest.approx(hit)
    assert not c.update(1) and c.score == pytest.approx(2 * hit)
    assert not c.update(0) and c.score == pytest.approx(max(0.0, 2 * hit + miss))
    for _ in range(3):
        c.update(1)
    assert c.update(1) and c.score >= 1.0 and c.n == 8
    c.reset()
    assert c.score == 0.0 and c.n == 0


def test_cusum_detects_a_decrease_too_and_validates_its_arguments():
    c = drift.BernoulliCusum(0.5, 0.3, threshold=1.5)
    assert [c.update(0) for _ in range(6)].count(True) >= 1, (
        "a run of failures where the rate should be 50% is an alarm"
    )
    for bad in [(0.5, 0.5), (0, 0.5), (0.5, 1.0)]:
        with pytest.raises(ValueError):
            drift.BernoulliCusum(*bad, threshold=1.0)


def test_calibrated_threshold_delivers_the_requested_false_alarm_interval():
    h = drift.calibrate_threshold(0.10, 0.20, arl0=300, n_sims=1500, seed=1)
    independent = drift.average_run_length(0.10, 0.10, 0.20, h, n_sims=3000, max_len=6000, seed=99)
    assert independent == pytest.approx(300, rel=0.2), (h, independent)
    h_longer = drift.calibrate_threshold(0.10, 0.20, arl0=900, n_sims=800, seed=1)
    assert h_longer > h, "fewer false alarms need a higher threshold"


def test_detection_delay_is_far_shorter_than_the_false_alarm_interval_and_shrinks_with_the_shift():
    h = drift.calibrate_threshold(0.10, 0.20, arl0=300, n_sims=1500, seed=1)
    delays = {
        p: drift.average_run_length(p, 0.10, 0.20, h, n_sims=1500, max_len=3000, seed=4)
        for p in (0.15, 0.20, 0.30)
    }
    assert delays[0.30] < delays[0.20] < delays[0.15] < 300 * 0.8
    assert delays[0.20] < 80, delays


def test_a_cusum_beats_testing_each_fixed_window_for_a_small_persistent_shift():
    """Same false-alarm budget: a per-window z-test on 100-observation windows versus CUSUM, for a 10% -> 15% shift."""
    rng = np.random.default_rng(0)
    h = drift.calibrate_threshold(0.10, 0.15, arl0=500, n_sims=1500, seed=2)
    cusum_delay = drift.average_run_length(0.15, 0.10, 0.15, h, n_sims=1500, max_len=4000, seed=5)
    z_crit = 2.576  # a window test with a similar false-alarm interval (one-sided 0.5% per window of 100 = every ~20,000 obs)
    windows = 0
    for _ in range(400):
        w = 0
        while True:
            w += 1
            x = rng.binomial(100, 0.15)
            if (x / 100 - 0.10) / math.sqrt(0.10 * 0.90 / 100) > z_crit:
                break
            if w > 200:
                break
        windows += w * 100
    window_delay = windows / 400
    assert cusum_delay < window_delay, (cusum_delay, window_delay)


def test_identical_samples_have_a_p_value_of_exactly_one():
    a = np.random.default_rng(0).normal(size=(20, 5))
    assert drift.centroid_shift_test(a, a.copy(), n_perm=100, seed=0)["p"] == 1.0, (
        "ties count as 'as extreme'"
    )


def test_cusum_alarms_exactly_when_the_score_reaches_the_threshold():
    exact = math.log(0.7 / 0.5)
    assert drift.BernoulliCusum(0.5, 0.7, threshold=exact).update(1) is True
    assert drift.BernoulliCusum(0.5, 0.7, threshold=exact + 1e-9).update(1) is False


def test_run_length_is_the_one_based_time_of_the_first_alarm_and_censored_at_the_stream_length():
    hit = math.log(0.7 / 0.5)
    always_hit = np.zeros((2, 10))  # every uniform is 0 < p_true: every observation is a 1
    never_hit = np.ones((2, 10))
    got = drift._run_lengths(
        0.5, 0.5, 0.7, threshold=2.5 * hit, uniforms=np.vstack([always_hit[:1], never_hit[:1]])
    )
    assert list(got) == [3, 10], (
        "alarm on the 3rd observation (score 3*hit >= 2.5*hit); the calm stream is censored at its length"
    )


def test_calibration_simulates_much_longer_streams_than_the_target_interval(monkeypatch):
    seen = []
    original = drift._run_lengths

    def spy(*args, **kwargs):
        seen.append(kwargs["max_len"])
        return original(*args, **kwargs)

    monkeypatch.setattr(drift, "_run_lengths", spy)
    drift.calibrate_threshold(0.1, 0.2, arl0=100, n_sims=200, seed=0)
    assert min(seen) >= 10 * 100, "runs censored near the target interval would bias the estimate"


def test_streamed_simulation_matches_an_explicit_matrix_and_is_memory_bounded():
    # the same numbers, supplied as one matrix or produced in blocks of 512 columns, give the same first-alarm times
    blocks = np.hstack(list(drift._uniform_blocks(50, 700, seed=3, block=512)))
    assert blocks.shape == (50, 700)
    assert (
        drift._run_lengths(0.5, 0.5, 0.7, 2.0, blocks)
        == drift._run_lengths(0.5, 0.5, 0.7, 2.0, n_sims=50, max_len=700, seed=3)
    ).all()
    assert [b.shape for b in drift._uniform_blocks(7, 1300, seed=0, block=512)] == [
        (7, 512),
        (7, 512),
        (7, 276),
    ]


def test_a_very_long_false_alarm_interval_calibrates_quickly_without_a_huge_matrix():
    import time

    t0 = time.perf_counter()
    h = drift.calibrate_threshold(0.1, 0.2, arl0=3000, n_sims=300, seed=0, tol=0.05)
    assert h > drift.calibrate_threshold(0.1, 0.2, arl0=300, n_sims=300, seed=0, tol=0.05)
    assert time.perf_counter() - t0 < 30


def test_with_one_item_per_group_every_relabelling_is_exactly_as_extreme():
    r = drift.centroid_shift_test(np.array([[0.0]]), np.array([[1.0]]), n_perm=50, seed=0)
    assert r["distance"] == 1.0 and r["p"] == 1.0, (
        "swapping the two labels changes nothing, so no evidence of a shift"
    )
