"""One gate for everything a change can break: quality, cost, latency, and whether the comparison is even valid.

    report = evaluate(baseline_run, candidate_run, Policy(critical=["human-*", "billing-pressure-*"]))
    report.status        # pass | warn | fail
    report.markdown()    # for the PR page ($GITHUB_STEP_SUMMARY)

Checks, in order (a failed *validity* check stops the comparison from being trusted at all):
  completeness   the candidate run finished (not cut short by the spend cap or a crash)
  dataset        both runs used the same cases (content hash, not just ids)
  price card     cost numbers are only comparable under the same prices
  quality        the Day 3 gate: critical cases, paired interval, segments
  cost           per-conversation cost: a PAIRED interval on the per-case difference, with a tolerance
  latency        the estimated p95 per conversation, with a tolerance
A cost or latency IMPROVEMENT never blocks; an increase beyond tolerance blocks when it is statistically clear and warns
when it is large but not yet clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import evalgate as eg
import traceeval as te

from common.evalkit import paired_bootstrap

from .runner import Run

EPS = 1e-9
ORDER = {"pass": 0, "improved": 0, "warn": 1, "fail": 2}


@dataclass
class Policy:
    critical: list[str] = field(default_factory=list)
    max_drop: float = 0.05
    strict: bool = False
    max_cost_increase: float = 0.10  # relative: +10% cost per conversation
    max_latency_increase: float = 0.25  # relative, on the estimated p95
    allow_dataset_change: bool = False


@dataclass
class Check:
    name: str
    status: str  # pass | improved | warn | fail
    detail: str


@dataclass
class GateReport:
    checks: list[Check]
    quality: eg.Decision | None
    baseline: str
    candidate: str
    numbers: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        worst = max((ORDER[c.status] for c in self.checks), default=0)
        return {
            0: "improved" if any(c.status == "improved" for c in self.checks) else "pass",
            1: "warn",
            2: "fail",
        }[worst]

    @property
    def exit_code(self) -> int:
        return 1 if self.status == "fail" else 0

    def markdown(self) -> str:
        icon = {"pass": "✅", "improved": "🎉", "warn": "⚠️", "fail": "❌"}[self.status]
        lines = [
            f"## {icon} LLMOps gate: **{self.status.upper()}**",
            "",
            f"`{self.baseline}` (baseline) vs `{self.candidate}` (candidate)",
            "",
            "| check | result | detail |",
            "|---|---|---|",
        ]
        for c in self.checks:
            lines.append(
                f"| {c.name} | {ICONS[c.status]} {c.status} | {c.detail.replace('|', '/')} |"
            )
        n = self.numbers
        if n:
            lines += [
                "",
                "| | baseline | candidate | change |",
                "|---|---|---|---|",
                f"| pass rate | {n['pass_base']:.1%} | {n['pass_cand']:.1%} | {n['pass_cand'] - n['pass_base']:+.1%} |",
                f"| cost per 1k conversations | ${n['cost_base']:.3f} | ${n['cost_cand']:.3f} | {_rel(n['cost_cand'], n['cost_base'])} |",
                f"| input tokens per conversation | {n['in_base']:.0f} | {n['in_cand']:.0f} | {_rel(n['in_cand'], n['in_base'])} |",
                f"| model calls per conversation | {n['calls_base']:.2f} | {n['calls_cand']:.2f} | {_rel(n['calls_cand'], n['calls_base'])} |",
                f"| estimated p95 latency | {n['p95_base']:.2f}s | {n['p95_cand']:.2f}s | {_rel(n['p95_cand'], n['p95_base'])} |",
            ]
        if self.quality is not None and (self.quality.regressions or self.quality.gains):
            lines += [
                "",
                f"Quality detail: {len(self.quality.regressions)} case(s) regressed, {len(self.quality.gains)} improved.",
            ]
            if self.quality.regressions:
                lines.append(
                    "Regressed: " + ", ".join(f"`{i}`" for i in self.quality.regressions[:15])
                )
            if self.quality.segment_regressions:
                lines.append(
                    "Segments that dropped: "
                    + ", ".join(
                        f"`{k}` {b:.0%} → {a:.0%}"
                        for k, (b, a) in self.quality.segment_regressions.items()
                    )
                )
        return "\n".join(lines) + "\n"


ICONS = {"pass": "✅", "improved": "🎉", "warn": "⚠️", "fail": "❌"}


def _rel(new: float, old: float) -> str:
    return "n/a" if old == 0 else f"{new / old - 1:+.1%}"


def _mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def _p95(rows: list[dict], key: str) -> float:
    return te.percentile([r[key] for r in rows], 95)


def _relative_check(
    name: str,
    base: dict[str, float],
    cand: dict[str, float],
    ids: list[str],
    tolerance: float,
    unit: str,
) -> Check:
    """Paired comparison of a per-case cost-like number (higher is worse)."""
    b = [base[i] for i in ids]
    c = [cand[i] for i in ids]
    mean_b = sum(b) / len(b)
    if mean_b == 0:
        return Check(name, "pass", "baseline value is zero: no relative comparison")
    stat = paired_bootstrap(c, b, seed=0)
    ratio = (sum(c) / len(c)) / mean_b - 1
    if (
        ratio <= tolerance + EPS
    ):  # a tolerance is inclusive; EPS absorbs floating-point noise (1.1 * x / x - 1 > 0.1)
        status = "improved" if stat["ci_high"] < 0 else "pass"
        return Check(
            name,
            status,
            f"{ratio:+.1%} {unit} (95% interval of the per-case difference [{stat['ci_low'] / mean_b:+.1%}, {stat['ci_high'] / mean_b:+.1%}] of baseline)",
        )
    significant = stat["ci_low"] > 0
    return Check(
        name,
        "fail" if significant else "warn",
        f"{ratio:+.1%} {unit} exceeds the {tolerance:+.0%} tolerance"
        + (
            " and the interval is entirely above zero"
            if significant
            else " but the interval includes zero: large, not proven"
        ),
    )


def evaluate(baseline: Run, candidate: Run, policy: Policy | None = None) -> GateReport:
    policy = policy or Policy()
    checks: list[Check] = []
    numbers: dict = {}
    report = GateReport(checks, None, baseline.name, candidate.name, numbers)

    if not candidate.complete:
        checks.append(
            Check(
                "completeness",
                "fail",
                f"the candidate run is incomplete ({candidate.manifest.get('aborted_reason') or 'unfinished'}); nothing below can be trusted",
            )
        )
        return report
    if not baseline.complete:
        checks.append(
            Check("completeness", "fail", "the baseline run is incomplete: regenerate it")
        )
        return report
    checks.append(
        Check(
            "completeness",
            "pass",
            f"both runs finished ({candidate.manifest['n_done']} cases, {candidate.manifest['trials']} trial(s) each)",
        )
    )

    if (
        baseline.manifest["dataset_hash"] != candidate.manifest["dataset_hash"]
        and not policy.allow_dataset_change
    ):
        checks.append(
            Check(
                "dataset",
                "fail",
                f"the cases changed (hash {baseline.manifest['dataset_hash']} vs {candidate.manifest['dataset_hash']}); scores are not comparable, update the baseline deliberately",
            )
        )
        return report
    checks.append(
        Check("dataset", "pass", f"same cases (hash {candidate.manifest['dataset_hash']})")
    )

    if baseline.manifest["price_card"] != candidate.manifest["price_card"]:
        checks.append(
            Check(
                "price card",
                "fail",
                "the runs priced tokens with different price cards: costs are not comparable",
            )
        )
        return report
    notes = [k for k in ("provider", "model") if baseline.manifest[k] != candidate.manifest[k]]
    checks.append(
        Check(
            "price card",
            "pass" if not notes else "warn",
            "same prices"
            + (f"; note: {', '.join(notes)} differ between the runs" if notes else ""),
        )
    )

    decision = eg.compare(
        baseline.results,
        candidate.results,
        critical=policy.critical,
        max_drop=policy.max_drop,
        strict=policy.strict,
        allow_case_changes=policy.allow_dataset_change,
    )
    report.quality = decision
    checks.append(Check("quality", decision.status, "; ".join(decision.reasons)))

    base_rows = {r["id"]: r for r in baseline.cases}
    cand_rows = {r["id"]: r for r in candidate.cases}
    ids = sorted(set(base_rows) & set(cand_rows))
    for key, label, tol, unit in (("cost_usd", "cost", policy.max_cost_increase, "cost"),):
        checks.append(
            _relative_check(
                label,
                {i: base_rows[i][key] for i in ids},
                {i: cand_rows[i][key] for i in ids},
                ids,
                tol,
                unit,
            )
        )
    lat_b, lat_c = (
        _p95([base_rows[i] for i in ids], "latency_s"),
        _p95([cand_rows[i] for i in ids], "latency_s"),
    )
    lat_ratio = lat_c / lat_b - 1 if lat_b else 0.0
    checks.append(
        Check(
            "latency",
            "fail"
            if lat_ratio > policy.max_latency_increase + EPS
            else ("improved" if lat_ratio < -0.05 else "pass"),
            f"estimated p95 {lat_b:.2f}s -> {lat_c:.2f}s ({lat_ratio:+.1%}; tolerance {policy.max_latency_increase:+.0%}; latencies come from an ASSUMED model)",
        )
    )

    brows, crows = [base_rows[i] for i in ids], [cand_rows[i] for i in ids]
    numbers.update(
        pass_base=decision.baseline_rate,
        pass_cand=decision.current_rate,
        cost_base=_mean(brows, "cost_usd") * 1000,
        cost_cand=_mean(crows, "cost_usd") * 1000,
        in_base=_mean(brows, "input_tokens"),
        in_cand=_mean(crows, "input_tokens"),
        calls_base=_mean(brows, "model_calls"),
        calls_cand=_mean(crows, "model_calls"),
        p95_base=lat_b,
        p95_cand=lat_c,
    )
    return report
