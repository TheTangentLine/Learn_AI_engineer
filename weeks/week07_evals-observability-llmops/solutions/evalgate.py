"""An evaluation GATE for CI: compare a candidate run against a stored baseline and decide pass / warn / fail.

    python evalgate.py compare --baseline evals/baseline.json --current out/current.json --critical "billing-pressure-*,human-*"
    python evalgate.py baseline --from out/current.json --to evals/baseline.json --force        # accept a new baseline, explicitly

What a good gate does (each rule below has a test):
  1. Refuses to compare DIFFERENT suites. If cases were added or removed, a pass rate is not comparable: say so, fail.
  2. Treats CRITICAL cases as a hard floor: a safety case that passed on the baseline and fails now blocks the merge,
     even if the overall score went up. (Averages hide the failures that matter.)
  3. Uses a PAIRED interval (same cases, so difficulty cancels). A drop whose interval is entirely below zero is a
     significant regression: fail.
  4. Small suites have wide intervals, so a large POINT drop that is not yet significant is a WARNING (or a failure
     with --strict): "probably worse, not proven; look before merging".
  5. Watches SEGMENTS (the case-id prefix, e.g. ``billing-small``): losing most of one kind of case is a regression even
     when the overall average barely moves (6 cases lost + 3 gained elsewhere = a small, noisy net change).
  6. Says exactly which cases changed, in both directions, so a reviewer can act without re-running anything.
  7. Never rewrites the baseline implicitly: moving the goalposts requires an explicit command and a reason.

Per-case values are the FRACTION of trials that passed (1.0 / 0.0 for a single deterministic trial), so the same gate
handles stochastic agents run several times.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[2]))

from common.evalkit import paired_bootstrap  # noqa: E402


@dataclass
class SuiteResult:
    variant: str
    cases: dict[str, float]  # case id -> fraction of trials passed
    meta: dict = field(default_factory=dict)  # model, date, commit, trials ...

    @property
    def rate(self) -> float:
        return sum(self.cases.values()) / len(self.cases) if self.cases else 0.0


def save(result: SuiteResult, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")


def load(path: str | Path) -> SuiteResult:
    d = json.loads(Path(path).read_text())
    return SuiteResult(
        d["variant"], {k: float(v) for k, v in d["cases"].items()}, d.get("meta", {})
    )


@dataclass
class Decision:
    status: str  # pass | improved | warn | fail
    reasons: list[str]
    baseline_rate: float
    current_rate: float
    diff: float = 0.0
    ci: tuple[float, float] = (0.0, 0.0)
    wins: int = 0
    losses: int = 0
    regressions: list[str] = field(default_factory=list)
    gains: list[str] = field(default_factory=list)
    critical_regressions: list[str] = field(default_factory=list)
    segment_regressions: dict[str, tuple[float, float]] = field(
        default_factory=dict
    )  # segment -> (before, after)
    n: int = 0

    @property
    def exit_code(self) -> int:
        return 1 if self.status == "fail" else 0


def is_critical(case_id: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(case_id, p) for p in patterns)


def segment_of(case_id: str) -> str:
    """'billing-small-03' -> 'billing-small' (everything before the trailing -NN)."""
    head, _, tail = case_id.rpartition("-")
    return head if head and tail.isdigit() else case_id


def segment_rates(result: SuiteResult, ids: list[str]) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for i in ids:
        by.setdefault(segment_of(i), []).append(result.cases[i])
    return {k: sum(v) / len(v) for k, v in by.items()}


def compare(
    baseline: SuiteResult,
    current: SuiteResult,
    *,
    critical: list[str] | None = None,
    max_drop: float = 0.05,
    segment_drop: float = 0.5,
    min_segment_cases: int = 3,
    strict: bool = False,
    allow_case_changes: bool = False,
    seed: int = 0,
) -> Decision:
    critical = critical or []
    base_ids, cur_ids = set(baseline.cases), set(current.cases)
    if base_ids != cur_ids and not allow_case_changes:
        added, removed = sorted(cur_ids - base_ids), sorted(base_ids - cur_ids)
        return Decision(
            "fail",
            [
                f"the suite changed: {len(added)} case(s) added {added[:3]}, {len(removed)} removed {removed[:3]}. Scores are not comparable; update the baseline deliberately."
            ],
            baseline.rate,
            current.rate,
            n=len(cur_ids),
        )
    ids = sorted(base_ids & cur_ids)
    if not ids:
        return Decision("fail", ["no cases in common"], baseline.rate, current.rate)
    before = [baseline.cases[i] for i in ids]
    after = [current.cases[i] for i in ids]
    stat = paired_bootstrap(after, before, seed=seed)
    regressions = [i for i in ids if current.cases[i] < baseline.cases[i]]
    gains = [i for i in ids if current.cases[i] > baseline.cases[i]]
    critical_reg = [i for i in regressions if is_critical(i, critical) and baseline.cases[i] >= 1.0]
    d = Decision(
        "pass",
        [],
        sum(before) / len(ids),
        sum(after) / len(ids),
        stat["diff"],
        (stat["ci_low"], stat["ci_high"]),
        stat["wins"],
        stat["losses"],
        regressions,
        gains,
        critical_reg,
        n=len(ids),
    )
    reasons = d.reasons
    if critical_reg:
        reasons.append(f"critical case(s) regressed: {critical_reg}")
        d.status = "fail"
    if stat["ci_high"] < 0:
        reasons.append(
            f"significant drop: {stat['diff']:+.1%} with a 95% interval [{stat['ci_low']:+.1%}, {stat['ci_high']:+.1%}] entirely below zero"
        )
        d.status = "fail"
    elif stat["diff"] <= -max_drop and d.status != "fail":
        reasons.append(
            f"large drop ({stat['diff']:+.1%} <= -{max_drop:.0%}) but the interval [{stat['ci_low']:+.1%}, {stat['ci_high']:+.1%}] includes zero at n={len(ids)}: probably worse, not proven"
        )
        d.status = "fail" if strict else "warn"
    seg_before, seg_after = segment_rates(baseline, ids), segment_rates(current, ids)
    sizes = {k: sum(1 for i in ids if segment_of(i) == k) for k in seg_before}
    for k in sorted(seg_before):
        if sizes[k] >= min_segment_cases and seg_before[k] - seg_after[k] >= segment_drop:
            d.segment_regressions[k] = (seg_before[k], seg_after[k])
    if d.segment_regressions and d.status != "fail":
        listing = ", ".join(
            f"{k} {b:.0%} -> {a:.0%}" for k, (b, a) in d.segment_regressions.items()
        )
        reasons.append(f"segment regression (>= {segment_drop:.0%} points lost): {listing}")
        d.status = "fail" if strict else "warn"
    if d.status == "pass" and stat["ci_low"] > 0:
        d.status = "improved"
        reasons.append(
            f"significant improvement: {stat['diff']:+.1%} [{stat['ci_low']:+.1%}, {stat['ci_high']:+.1%}]"
        )
    if not reasons:
        reasons.append("no significant change")
    return d


def render_markdown(d: Decision, baseline: SuiteResult, current: SuiteResult) -> str:
    icon = {"pass": "✅", "improved": "🎉", "warn": "⚠️", "fail": "❌"}[d.status]
    lines = [
        f"## {icon} Eval gate: **{d.status.upper()}**",
        "",
        f"| | baseline (`{baseline.variant}`) | candidate (`{current.variant}`) |",
        "|---|---|---|",
        f"| pass rate | {d.baseline_rate:.1%} | {d.current_rate:.1%} |",
        f"| paired diff (n={d.n}) | | **{d.diff:+.1%}** [{d.ci[0]:+.1%}, {d.ci[1]:+.1%}] |",
        f"| cases better / worse | | {d.wins} / {d.losses} |",
        "",
    ]
    lines += [f"- {r}" for r in d.reasons]
    if d.regressions:
        lines += [
            "",
            f"**Regressed ({len(d.regressions)}):** "
            + ", ".join(f"`{i}`" for i in d.regressions[:20])
            + (" ..." if len(d.regressions) > 20 else ""),
        ]
    if d.gains:
        lines += [
            "",
            f"**Improved ({len(d.gains)}):** "
            + ", ".join(f"`{i}`" for i in d.gains[:20])
            + (" ..." if len(d.gains) > 20 else ""),
        ]
    if d.segment_regressions:
        lines += [
            "",
            "**Segments that dropped:** "
            + ", ".join(f"`{k}` {b:.0%} → {a:.0%}" for k, (b, a) in d.segment_regressions.items()),
        ]
    if d.critical_regressions:
        lines += ["", "**CRITICAL:** " + ", ".join(f"`{i}`" for i in d.critical_regressions)]
    return "\n".join(lines) + "\n"


def accept_baseline(
    source: str | Path, target: str | Path, *, force: bool, reason: str = ""
) -> str:
    """Promote a result to the new baseline. Overwriting an existing baseline needs --force AND a reason (it is a decision)."""
    tgt = Path(target)
    if tgt.exists() and not force:
        raise SystemExit(
            f"{tgt} exists; pass --force and --reason to replace the baseline deliberately"
        )
    if tgt.exists() and not reason.strip():
        raise SystemExit("replacing a baseline needs a --reason (it is recorded in the file)")
    result = load(source)
    if reason:
        result.meta = {**result.meta, "accepted_because": reason}
    save(result, tgt)
    return f"baseline written to {tgt} ({len(result.cases)} cases, pass rate {result.rate:.1%})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evalgate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--baseline", required=True)
    c.add_argument("--current", required=True)
    c.add_argument("--critical", default="", help="comma-separated case-id patterns (fnmatch)")
    c.add_argument("--max-drop", type=float, default=0.05)
    c.add_argument("--segment-drop", type=float, default=0.5)
    c.add_argument("--strict", action="store_true")
    c.add_argument("--allow-case-changes", action="store_true")
    c.add_argument(
        "--summary-file",
        default=None,
        help="append the markdown here (use $GITHUB_STEP_SUMMARY in Actions)",
    )
    b = sub.add_parser("baseline")
    b.add_argument("--from", dest="source", required=True)
    b.add_argument("--to", dest="target", required=True)
    b.add_argument("--force", action="store_true")
    b.add_argument("--reason", default="")
    args = ap.parse_args(argv)
    if args.cmd == "baseline":
        print(accept_baseline(args.source, args.target, force=args.force, reason=args.reason))
        return 0
    base, cur = load(args.baseline), load(args.current)
    d = compare(
        base,
        cur,
        critical=[p.strip() for p in args.critical.split(",") if p.strip()],
        max_drop=args.max_drop,
        segment_drop=args.segment_drop,
        strict=args.strict,
        allow_case_changes=args.allow_case_changes,
    )
    md = render_markdown(d, base, cur)
    print(md)
    if args.summary_file:
        with open(args.summary_file, "a") as f:
            f.write(md)
    return d.exit_code


if __name__ == "__main__":
    sys.exit(main())
