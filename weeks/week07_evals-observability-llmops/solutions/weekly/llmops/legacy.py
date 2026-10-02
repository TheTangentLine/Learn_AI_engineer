"""Import the Day 5 saved runs (real local-model results) as run directories, so the gate and dashboard can be shown on real data
without re-running 400 conversations.  ``python -m llmops.legacy outputs/ runs/`` converts every Day 5 variant it finds."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import costlab as cl
import day5_solution as d5
import evalcases as ec
import evalgate as eg

from . import runner


def import_day5(variant: str, out: Path = d5.OUT) -> runner.Run:
    runs, spans = d5.load_variant(variant, out)
    by_case = d5.per_case(spans)
    cases = {c.id: c for c in ec.build_cases(lambda: None)}
    rows = [runner._case_row(cases[r.case_id], [r], [by_case[r.case_id]], cl.CHEAP) for r in runs]
    manifest = {
        "version": runner.MANIFEST_VERSION,
        "name": variant,
        "variant": variant,
        "options": d5.VARIANTS[variant],
        "provider": "local",
        "model": "local-qwen",
        "trials": 1,
        "dataset_hash": runner.dataset_hash(list(cases.values())),
        "n_cases": len(cases),
        "n_done": len(rows),
        "price_card": asdict(cl.CHEAP),
        "latency_model": asdict(cl.DEFAULT_LATENCY),
        "spend_usd": sum(r["cost_usd"] for r in rows),
        "budget_usd": None,
        "complete": len(rows) == len(cases),
        "aborted_reason": "",
        "created_at": (out / f"w7d5_{variant}.jsonl").stat().st_mtime,
        "note": "imported from the Day 5 real-model run",
    }
    result = eg.SuiteResult(
        variant,
        {r["id"]: r["pass_fraction"] for r in rows},
        {"provider": "local", "model": "local-qwen", "trials": 1},
    )
    return runner.Run(manifest, result, rows, spans)


def main(argv: list[str]) -> int:
    src, dst = Path(argv[0]), Path(argv[1])
    for variant in d5.VARIANTS:
        if (src / f"w7d5_{variant}.jsonl").exists():
            run = import_day5(variant, src)
            runner.save_run(run, dst / variant.replace("+", "_"))
            print(f"{variant}: {run.results.rate:.0%} -> {dst / variant.replace('+', '_')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
