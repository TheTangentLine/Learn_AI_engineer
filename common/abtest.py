"""A/B testing for LLM features: assignment, sample-ratio checks, rate comparisons, sample sizes, and sequential monitoring.

    from common import abtest

    arm = abtest.assign("prompt-v2", user_id, {"control": 0.5, "lean": 0.5})      # sticky, no state to store
    abtest.srm_check({"control": 5210, "lean": 4790}, {"control": .5, "lean": .5})  # is the split what we asked for?
    abtest.compare_rates(control_ok, control_n, lean_ok, lean_n)                   # diff, Newcombe interval, z, p
    abtest.sample_size(p_base=0.52, mde=0.10)                                      # users per arm, fixed horizon
    seq = abtest.GroupSequential(looks=5, kind="pocock")                           # peek at the data without cheating
    seq.check(look=2, z=2.1)

Why each piece exists (each is tested, several by simulation):
  * ASSIGNMENT is a hash of (experiment, unit), so a user always sees the same arm, nothing is stored, and two
    experiments are independent of each other.
  * A SAMPLE RATIO MISMATCH (the arms are not the size you asked for) means the experiment is broken (a bug that drops
    users from one arm), and every result from it is suspect: check it first.
  * PEEKING: testing at several looks and stopping at the first p < 0.05 inflates the false-positive rate far above 5%.
    ``peeking_false_positive_rate`` measures it; ``GroupSequential`` boundaries are CALIBRATED BY SIMULATION so the overall
    rate is 5% with the looks you planned in advance.
  * SAMPLE SIZE: the formula is checked against an empirical power simulation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from statistics import NormalDist

import numpy as np

_N = NormalDist()

# ----------------------------------------------------------------------------- assignment


def _unit_interval(experiment: str, unit: str) -> float:
    digest = hashlib.sha256(f"{experiment}\x00{unit}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign(experiment: str, unit: str, variants: Mapping[str, float]) -> str:
    """The arm of ``unit`` in ``experiment``. Deterministic and sticky; arms are ordered by NAME so the dict's insertion
    order cannot change who gets what. Weights are relative (they are normalised)."""
    if not variants:
        raise ValueError("no variants")
    if any(w < 0 for w in variants.values()) or sum(variants.values()) <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    total = sum(variants.values())
    u, acc = _unit_interval(experiment, str(unit)), 0.0
    names = sorted(variants)
    for name in names:
        acc += variants[name] / total
        if u < acc:
            return name
    return names[-1]  # floating-point slack at the very top of the range


# ----------------------------------------------------------------------------- distributions


def chi2_sf(x: float, df: int) -> float:
    """P(chi-square with ``df`` degrees of freedom >= x): the regularised upper incomplete gamma Q(df/2, x/2)."""
    if x <= 0:
        return 1.0
    a, z = df / 2.0, x / 2.0
    gln = math.lgamma(a)
    if z < a + 1.0:  # series for P, then 1 - P
        term = total = 1.0 / a
        n = a
        for _ in range(500):
            n += 1.0
            term *= z / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - gln))
    tiny = 1e-300  # continued fraction for Q (modified Lentz)
    b = z + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return min(1.0, math.exp(-z + a * math.log(z) - gln) * h)


def norm_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


# ----------------------------------------------------------------------------- sample ratio mismatch


@dataclass(frozen=True)
class SRM:
    chi2: float
    p: float
    ok: bool  # True = the split is consistent with what was asked for
    expected: dict[str, float]


def srm_check(
    observed: Mapping[str, int], weights: Mapping[str, float], alpha: float = 0.001
) -> SRM:
    """Chi-square goodness of fit of the observed arm sizes to the intended split. The conventional alarm level is strict
    (0.001) because a real SRM is not subtle and a false alarm only costs a look at the pipeline."""
    names = sorted(weights)
    total_w, n = sum(weights.values()), sum(observed.get(k, 0) for k in names)
    if n == 0:
        raise ValueError("no observations")
    expected = {k: n * weights[k] / total_w for k in names}
    chi2 = sum((observed.get(k, 0) - e) ** 2 / e for k, e in expected.items() if e > 0)
    p = chi2_sf(chi2, len(names) - 1)
    return SRM(chi2, p, p >= alpha, expected)


# ----------------------------------------------------------------------------- comparing rates


def wilson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval: behaves at small n and at rates near 0 or 1, where the textbook interval does not."""
    if n == 0:
        return (0.0, 1.0)
    z, p = _N.inv_cdf(1 - alpha / 2), successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def compare_rates(
    control_successes: int,
    control_n: int,
    treat_successes: int,
    treat_n: int,
    alpha: float = 0.05,
) -> dict:
    """Treatment minus control: the difference, a Newcombe (hybrid score) interval, and a pooled two-sided z-test."""
    if control_n <= 0 or treat_n <= 0:
        raise ValueError("both arms need observations")
    p1, p2 = treat_successes / treat_n, control_successes / control_n
    l1, u1 = wilson(treat_successes, treat_n, alpha)
    l2, u2 = wilson(control_successes, control_n, alpha)
    diff = p1 - p2
    low = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    high = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    pooled = (control_successes + treat_successes) / (control_n + treat_n)
    se = math.sqrt(pooled * (1 - pooled) * (1 / control_n + 1 / treat_n))
    z = diff / se if se > 0 else 0.0
    return {
        "control_rate": p2,
        "treat_rate": p1,
        "diff": diff,
        "ci": (low, high),
        "z": z,
        "p": norm_two_sided_p(z) if se > 0 else 1.0,
        "n": (control_n, treat_n),
    }


def bootstrap_diff(
    control: Sequence[float],
    treat: Sequence[float],
    *,
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Unpaired bootstrap of the difference in means (cost, latency, tokens...). Independent groups: resample each arm."""
    a, b = np.asarray(control, dtype=float), np.asarray(treat, dtype=float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("both arms need observations")
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    diffs = b[ib].mean(axis=1) - a[ia].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "control_mean": float(a.mean()),
        "treat_mean": float(b.mean()),
        "diff": float(b.mean() - a.mean()),
        "ci": (float(lo), float(hi)),
    }


# ----------------------------------------------------------------------------- planning


def sample_size(p_base: float, mde: float, *, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users PER ARM for a fixed-horizon two-sided test of two proportions to detect an absolute lift ``mde``."""
    p1, p2 = p_base, p_base + mde
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("both rates must be strictly between 0 and 1")
    if mde == 0:
        raise ValueError("mde must be non-zero")
    za, zb = _N.inv_cdf(1 - alpha / 2), _N.inv_cdf(power)
    pbar = (p1 + p2) / 2
    num = (
        za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return math.ceil(num / mde**2)


def _z_paths(
    a_succ: np.ndarray, a_n: np.ndarray, b_succ: np.ndarray, b_n: np.ndarray
) -> np.ndarray:
    """Pooled two-proportion z at each look, vectorised over simulated experiments (rows) and looks (columns)."""
    p1, p2 = b_succ / b_n, a_succ / a_n
    pooled = (a_succ + b_succ) / (a_n + b_n)
    se = np.sqrt(pooled * (1 - pooled) * (1 / a_n + 1 / b_n))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (p1 - p2) / se
    return np.nan_to_num(z)


def simulate_looks(
    p_a: float, p_b: float, n_per_look: int, looks: int, n_sims: int, seed: int = 0
) -> np.ndarray:
    """z-statistics at each of ``looks`` equally spaced interim analyses, for ``n_sims`` simulated experiments."""
    rng = np.random.default_rng(seed)
    a = rng.binomial(n_per_look, p_a, (n_sims, looks)).cumsum(axis=1)
    b = rng.binomial(n_per_look, p_b, (n_sims, looks)).cumsum(axis=1)
    n = n_per_look * np.arange(1, looks + 1)
    return _z_paths(a, n, b, n)


def peeking_false_positive_rate(
    p: float, n_per_look: int, looks: int, n_sims: int = 20_000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """An A/A test (no real difference). How often does each stopping rule declare a winner?"""
    z = simulate_looks(p, p, n_per_look, looks, n_sims, seed)
    crit = _N.inv_cdf(1 - alpha / 2)
    return {
        "fixed_horizon": float((np.abs(z[:, -1]) > crit).mean()),
        "peek_every_look": float((np.abs(z) > crit).any(axis=1).mean()),
    }


def empirical_power(
    p_a: float, p_b: float, n_per_arm: int, n_sims: int = 20_000, alpha: float = 0.05, seed: int = 0
) -> float:
    z = simulate_looks(p_a, p_b, n_per_arm, 1, n_sims, seed)[:, 0]
    return float((np.abs(z) > _N.inv_cdf(1 - alpha / 2)).mean())


# ----------------------------------------------------------------------------- group sequential monitoring


@lru_cache(maxsize=64)
def _boundaries(looks: int, alpha: float, kind: str, n_paths: int, seed: int) -> tuple[float, ...]:
    """Critical |z| at each look, calibrated by simulating a standard Brownian motion observed at ``looks`` equally spaced
    information times: Pocock = one constant boundary, O'Brien-Fleming = c * sqrt(looks / k) (strict early, near 1.96 at
    the end). The constant ``c`` is the (1 - alpha) quantile of the maximum standardised statistic, so the overall
    false-positive rate over ALL planned looks is alpha."""
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((n_paths, looks))
    k = np.arange(1, looks + 1)
    z = np.abs(steps.cumsum(axis=1)) / np.sqrt(k)
    if kind == "pocock":
        c = float(np.quantile(z.max(axis=1), 1 - alpha))
        return tuple([c] * looks)
    if kind == "obf":
        c = float(np.quantile((z * np.sqrt(k / looks)).max(axis=1), 1 - alpha))
        return tuple(float(c * math.sqrt(looks / i)) for i in k)
    raise ValueError("kind must be 'pocock' or 'obf'")


class GroupSequential:
    """Pre-planned interim analyses with boundaries that keep the overall false-positive rate at ``alpha``.
    Looks must be planned in advance and roughly equally spaced in sample size; this is a stopping rule for
    EFFICACY only (no futility stopping)."""

    def __init__(
        self,
        looks: int,
        alpha: float = 0.05,
        kind: str = "pocock",
        n_paths: int = 200_000,
        seed: int = 0,
    ):
        if looks < 1:
            raise ValueError("looks must be >= 1")
        self.looks, self.alpha, self.kind = looks, alpha, kind
        self.bounds = _boundaries(looks, alpha, kind, n_paths, seed)

    def boundary(self, look: int) -> float:
        """The |z| needed to stop at look number ``look`` (1-based)."""
        if not 1 <= look <= self.looks:
            raise ValueError(f"look must be between 1 and {self.looks}")
        return self.bounds[look - 1]

    def check(self, look: int, z: float) -> str:
        """'stop-win' / 'stop-lose' when the boundary is crossed (the sign of z says which way), else 'continue'
        (and 'inconclusive' after the last look)."""
        if abs(z) >= self.boundary(look):
            return "stop-win" if z > 0 else "stop-lose"
        return "continue" if look < self.looks else "inconclusive"

    def false_positive_rate(
        self, p: float, n_per_look: int, n_sims: int = 20_000, seed: int = 1
    ) -> float:
        z = np.abs(simulate_looks(p, p, n_per_look, self.looks, n_sims, seed))
        return float((z >= np.array(self.bounds)).any(axis=1).mean())

    def power(
        self, p_a: float, p_b: float, n_per_look: int, n_sims: int = 20_000, seed: int = 2
    ) -> float:
        z = np.abs(simulate_looks(p_a, p_b, n_per_look, self.looks, n_sims, seed))
        return float((z >= np.array(self.bounds)).any(axis=1).mean())


# ----------------------------------------------------------------------------- a decision rule


def decide(
    primary: dict, guardrails: Mapping[str, dict], *, tolerance: Mapping[str, float] | None = None
) -> str:
    """Combine a primary rate comparison (``compare_rates`` output, higher is better) with guardrail comparisons whose
    ``diff`` is "higher is worse" (cost, latency, escalation rate).
      ship            primary interval entirely above zero AND no guardrail interval entirely above its tolerance
      block           a guardrail interval entirely above its tolerance (harm), whatever the primary says
      stop            primary interval entirely below zero
      keep-running    otherwise
    ``tolerance[name]`` is how much worse a guardrail may be before it counts (default 0)."""
    tolerance = tolerance or {}
    harmed = [n for n, g in guardrails.items() if g["ci"][0] > tolerance.get(n, 0.0)]
    if harmed:
        return "block"
    low, high = primary["ci"]
    if low > 0:
        return "ship"
    if high < 0:
        return "stop"
    return "keep-running"
