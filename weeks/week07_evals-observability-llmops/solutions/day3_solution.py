"""Week 7 Day 3 - Solution: turn evaluation into a CI gate.

Three layers, cheapest first:
  1. UNIT       fast offline tests with scripted models (logic regressions: guards, router, tools)  -> scripts/check.sh
  2. CRITICAL   a small must-pass set, asserted with plain pytest (the safety cases)                   -> test_day3.py
  3. STATISTICAL  the full suite compared with a stored baseline by a paired test                    -> evalgate.py

This file builds ``SuiteResult``s from the Day 1 runs and shows the gate's decision for several candidate "pull requests".

  uv run python weeks/week07_evals-observability-llmops/solutions/day3_solution.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import day1_solution as d1  # noqa: E402
import evalcases as ec  # noqa: E402
import evalgate as eg  # noqa: E402

OUT = ROOT / "outputs"
CRITICAL = [
    "billing-pressure-*",
    "human-*",
    "off-topic-*",
]  # the safety/scope cases: a regression here blocks the merge


def suite_from_day1(variant: str, outputs: Path = OUT) -> eg.SuiteResult:
    rows = [
        json.loads(line)
        for line in (outputs / f"w7d1_{variant}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return eg.SuiteResult(
        variant,
        {r["case_id"]: 1.0 if r["passed"] else 0.0 for r in rows},
        {"model": "Qwen2.5-0.5B (local)", "source": f"w7d1_{variant}.jsonl", "trials": 1},
    )


def suite_from_runs(runs, variant: str, **meta) -> eg.SuiteResult:
    """Day 1 ``CaseRun`` list -> the gate's input. One deterministic trial per case: 1.0 or 0.0."""
    return eg.SuiteResult(variant, {r.case_id: 1.0 if r.passed else 0.0 for r in runs}, meta)


def run_suite(variant: str, *, provider: str, model: str | None = None) -> eg.SuiteResult:
    """What the CI job's "run the eval suite" step does: all 50 cases against the chosen provider."""
    runs = d1.run_variant(
        ec.build_cases(lambda: None),
        variant,
        provider=provider,
        model=model,
        **d1.VARIANTS[variant],
    )
    return suite_from_runs(runs, variant, provider=provider, model=model or "default", trials=1)


CANDIDATES = [
    ("same code (re-run)", "rules+prefetch"),
    ("revert the prefetch change", "rules"),
    ("someone replaces the keyword router with the model again", "baseline"),
    ("add triage few-shot to the model router", "fewshot"),
    ("force the lookup tool on every first step", "rules+smart"),
]


def demo() -> None:
    base = suite_from_day1("rules+prefetch")
    print(f"baseline: rules+prefetch, {len(base.cases)} cases, pass rate {base.rate:.1%}\n")
    for label, variant in CANDIDATES:
        cur = suite_from_day1(variant)
        d = eg.compare(base, cur, critical=CRITICAL)
        print(f"### PR: {label}  (candidate = {variant})")
        print(eg.render_markdown(d, base, cur))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser(
        "run",
        help="run the whole suite once and write a SuiteResult json (the CI job's first step)",
    )
    r.add_argument("--variant", default="rules+prefetch", choices=sorted(d1.VARIANTS))
    r.add_argument("--provider", required=True)
    r.add_argument("--model", default=None)
    r.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        result = run_suite(args.variant, provider=args.provider, model=args.model)
        eg.save(result, args.out)
        print(
            f"{args.variant}: {len(result.cases)} cases, pass rate {result.rate:.1%} -> {args.out}"
        )
    else:
        demo()


if __name__ == "__main__":
    main()
