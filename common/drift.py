"""Drift detection for LLM applications: has the INPUT changed, has the OUTPUT changed, and how fast would we notice?

    from common import drift

    drift.psi({"billing": 60, "technical": 40}, {"billing": 35, "technical": 45, "other": 20})   # one number per feature
    drift.chi2_homogeneity(last_month_counts, this_week_counts)                                   # a p-value
    drift.centroid_shift_test(old_embeddings, new_embeddings, n_perm=2000)                        # did the TOPICS move?
    cusum = drift.BernoulliCusum(p0=0.10, p1=0.20, threshold=drift.calibrate_threshold(0.10, 0.20, arl0=500))
    for escalated in stream: if cusum.update(escalated): alert()                                  # a rate that crept up

Three kinds of drift, three questions:
  * INPUT drift   what people ask changed (a launch, a new customer segment, an outage): compare distributions of the
                  route, the length, the topic (embeddings), the language.
  * OUTPUT drift  what the system does changed (a provider updated the model; a prompt edit): compare rates (pass,
                  escalate, refuse, cost per conversation) and distributions of tool calls.
  * LABEL drift   what "good" means changed: only humans can see it; audit a sample.

A detector is judged by two numbers (both measured here by simulation, never assumed): the FALSE-ALARM interval when
nothing changed (ARL0, the average number of observations until a false alert) and the DELAY after a real change.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .abtest import chi2_sf

# ----------------------------------------------------------------------------- distribution comparison


def _proportions(counts: Mapping[str, float], keys: Sequence[str], eps: float) -> np.ndarray:
    total = float(sum(counts.get(k, 0) for k in keys))
    if total <= 0:
        raise ValueError("a distribution needs at least one observation")
    return np.array([max(counts.get(k, 0) / total, eps) for k in keys])


def psi(expected: Mapping[str, float], actual: Mapping[str, float], eps: float = 1e-4) -> float:
    """Population Stability Index: sum over categories of (a - e) * ln(a / e), on proportions floored at ``eps`` so that a
    category that appears or vanishes gives a large finite number instead of infinity. Symmetric in its arguments."""
    keys = sorted(set(expected) | set(actual))
    e, a = _proportions(expected, keys, eps), _proportions(actual, keys, eps)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_band(value: float) -> str:
    """The industry rule of thumb (a convention from credit scoring, not a law): below 0.1 stable, 0.1 to 0.25 moderate,
    above 0.25 major. Calibrate on your own history."""
    return "stable" if value < 0.1 else ("moderate" if value < 0.25 else "major")


def chi2_homogeneity(a: Mapping[str, int], b: Mapping[str, int]) -> dict:
    """Pearson chi-square test that two samples of category counts come from the same distribution (a 2 x k table).
    Categories with no observations in either sample are ignored."""
    keys = [k for k in sorted(set(a) | set(b)) if a.get(k, 0) + b.get(k, 0) > 0]
    na, nb = sum(a.get(k, 0) for k in keys), sum(b.get(k, 0) for k in keys)
    if na == 0 or nb == 0 or len(keys) < 2:
        raise ValueError("need two non-empty samples and at least two categories")
    chi2 = 0.0
    for k in keys:
        total = a.get(k, 0) + b.get(k, 0)
        for observed, n in ((a.get(k, 0), na), (b.get(k, 0), nb)):
            expected = total * n / (na + nb)
            chi2 += (observed - expected) ** 2 / expected
    df = len(keys) - 1
    return {"chi2": chi2, "df": df, "p": chi2_sf(chi2, df), "n": (na, nb)}


def centroid_shift_test(a: np.ndarray, b: np.ndarray, *, n_perm: int = 2000, seed: int = 0) -> dict:
    """Permutation test on the distance between the mean embeddings of two samples (rows = items). Shuffling the group
    labels shows how far apart two samples of the SAME distribution typically land; the p-value is how often that is as
    far as what we observed. Detects a shift in WHAT is being asked, which category counts cannot."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("both samples need rows")
    observed = float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))
    pool = np.vstack([a, b])
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pool))
        d = np.linalg.norm(pool[idx[: len(a)]].mean(axis=0) - pool[idx[len(a) :]].mean(axis=0))
        exceed += d >= observed
    return {"distance": observed, "p": (exceed + 1) / (n_perm + 1)}


# ----------------------------------------------------------------------------- change detection on a rate


@dataclass
class BernoulliCusum:
    """CUSUM on a stream of 0/1 outcomes (escalated? thumbs-down? guard fired?). It accumulates the log-likelihood ratio of
    "the rate is now p1" against "the rate is still p0" and resets at zero, so a long calm stretch cannot hide a recent
    change. Works for increases (p1 > p0) and decreases (p1 < p0)."""

    p0: float
    p1: float
    threshold: float
    score: float = 0.0
    n: int = 0

    def __post_init__(self) -> None:
        if not (0 < self.p0 < 1 and 0 < self.p1 < 1) or self.p0 == self.p1:
            raise ValueError("p0 and p1 must differ and lie strictly between 0 and 1")
        self._hit = math.log(self.p1 / self.p0)
        self._miss = math.log((1 - self.p1) / (1 - self.p0))

    def update(self, outcome: int | bool) -> bool:
        """Feed one observation; True means 'the rate has changed' (the score reached the threshold)."""
        self.n += 1
        self.score = max(0.0, self.score + (self._hit if outcome else self._miss))
        return self.score >= self.threshold

    def reset(self) -> None:
        self.score, self.n = 0.0, 0


def _uniform_blocks(n_sims: int, max_len: int, seed: int, block: int = 512):
    """Random numbers for ``n_sims`` streams of ``max_len`` observations, produced ``block`` columns at a time so memory stays
    O(n_sims * block) however long the streams are. The same seed always gives the same streams."""
    rng = np.random.default_rng(seed)
    done = 0
    while done < max_len:
        width = min(block, max_len - done)
        yield rng.random((n_sims, width))
        done += width


def _run_lengths(
    p_true: float,
    p0: float,
    p1: float,
    threshold: float,
    uniforms: np.ndarray | None = None,
    *,
    n_sims: int = 0,
    max_len: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """First alarm time (1-based; the stream length if none) for each simulated stream. Pass an explicit ``uniforms`` matrix
    (rows = streams) or ``n_sims``/``max_len``/``seed`` to stream the random numbers. Only streams that have not alarmed yet
    are advanced, so long calibrations stay fast."""
    hit, miss = math.log(p1 / p0), math.log((1 - p1) / (1 - p0))
    blocks = [uniforms] if uniforms is not None else _uniform_blocks(n_sims, max_len, seed)
    first = score = alive = None
    t0 = 0
    for u in blocks:
        if first is None:
            n_rows = u.shape[0]
            total = u.shape[1] if uniforms is not None else max_len
            first, score, alive = np.full(n_rows, total), np.zeros(n_rows), np.arange(n_rows)
        for j in range(u.shape[1]):
            x = u[alive, j] < p_true
            score[alive] = np.maximum(0.0, score[alive] + np.where(x, hit, miss))
            fired = score[alive] >= threshold
            first[alive[fired]] = t0 + j + 1
            alive = alive[~fired]
            if alive.size == 0:
                return first
        t0 += u.shape[1]
    return first


def average_run_length(
    p_true: float,
    p0: float,
    p1: float,
    threshold: float,
    *,
    n_sims: int = 2000,
    max_len: int = 5000,
    seed: int = 0,
) -> float:
    """Average number of observations until the alarm when the true rate is ``p_true`` (censored at ``max_len``)."""
    return float(
        _run_lengths(p_true, p0, p1, threshold, n_sims=n_sims, max_len=max_len, seed=seed).mean()
    )


def calibrate_threshold(
    p0: float, p1: float, arl0: float, *, n_sims: int = 2000, seed: int = 0, tol: float = 0.01
) -> float:
    """The CUSUM threshold whose false-alarm run length (true rate = p0) is about ``arl0`` observations, found by bisection on
    a simulation with fixed random numbers (so the search is monotone). Calibrate by simulation: textbook tables assume
    a fixed p0 and p1 that are rarely yours. Streams are simulated up to 10 x ``arl0`` observations, so fewer than 0.01% of
    them are censored (a censored run reads as shorter than it is and would bias the threshold downwards)."""
    max_len = int(arl0 * 10)
    lo, hi = 0.0, 40.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        arl = _run_lengths(p0, p0, p1, mid, n_sims=n_sims, max_len=max_len, seed=seed).mean()
        lo, hi = (mid, hi) if arl < arl0 else (lo, mid)
    return (lo + hi) / 2
