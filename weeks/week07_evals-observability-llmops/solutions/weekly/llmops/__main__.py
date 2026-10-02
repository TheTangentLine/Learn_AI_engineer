"""Command line: ``python -m llmops run | gate | baseline | dashboard | monitor`` (run from this folder's parent)."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

import day1_solution as d1
import day5_solution as d5
import scripted

from common import tracing
from common.fake import fake_llm

from . import dashboard, gate, monitor, runner

ALL_VARIANTS = {**{f"d1:{k}": v for k, v in d1.VARIANTS.items()}, **d5.VARIANTS}


@contextlib.contextmanager
def provider_context(provider: str, faults: dict):
    """local: the OpenAI-compatible local model server; scripted: a deterministic fake (offline drills); else: a hosted API."""
    if provider == "local":
        from common import llm
        from common.local_server import LocalOpenAIServer

        with LocalOpenAIServer() as srv:
            os.environ["OLLAMA_BASE_URL"] = srv.url
            llm._ollama.cache_clear()
            yield "ollama", "local-qwen"
    elif provider == "scripted":
        with fake_llm(scripted.rules(**faults)):
            yield "anthropic", None
    else:
        yield provider, None


def parse_faults(items: list[str]) -> dict:
    return {k: True for k in items}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="llmops")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the suite and write a run directory")
    r.add_argument("--variant", default="base", choices=sorted(ALL_VARIANTS))
    r.add_argument(
        "--provider", default="local", choices=["local", "scripted", "anthropic", "openai"]
    )
    r.add_argument(
        "--fault",
        action="append",
        default=[],
        help="scripted provider only: inject a named fault (skip_lookup, liar, ...)",
    )
    r.add_argument("--trials", type=int, default=1)
    r.add_argument(
        "--budget",
        type=float,
        default=None,
        help="USD cap on the simulated spend (REQUIRED for hosted providers)",
    )
    r.add_argument("--name", default=None)
    r.add_argument("--note", default="")
    r.add_argument("--out", required=True)

    g = sub.add_parser("gate", help="compare a candidate run directory with a baseline")
    g.add_argument("--baseline", required=True)
    g.add_argument("--candidate", required=True)
    g.add_argument("--critical", default="billing-pressure-*,human-*,off-topic-*")
    g.add_argument("--max-drop", type=float, default=0.05)
    g.add_argument("--max-cost-increase", type=float, default=0.10)
    g.add_argument("--max-latency-increase", type=float, default=0.25)
    g.add_argument("--strict", action="store_true")
    g.add_argument("--allow-dataset-change", action="store_true")
    g.add_argument("--summary-file", default=None)

    b = sub.add_parser("baseline", help="promote a run directory to be the baseline")
    b.add_argument("--from", dest="source", required=True)
    b.add_argument("--to", dest="target", required=True)
    b.add_argument("--force", action="store_true")
    b.add_argument("--reason", default="")

    d = sub.add_parser("dashboard", help="write the HTML dashboard")
    d.add_argument("--runs", nargs="+", required=True)
    d.add_argument(
        "--baseline", default=None, help="the name of the baseline run (default: the first)"
    )
    d.add_argument("--out", required=True)

    m = sub.add_parser("monitor", help="check production spans against a reference period")
    m.add_argument("--spans", required=True)
    m.add_argument("--reference", required=True)
    m.add_argument("--arl0", type=float, default=500.0)
    m.add_argument(
        "--sims",
        type=int,
        default=1500,
        help="simulated streams used to calibrate the alert thresholds",
    )

    args = ap.parse_args(argv)

    if args.cmd == "run":
        options = dict(ALL_VARIANTS[args.variant])
        name = args.name or args.variant.replace(":", "-")
        if args.fault and args.provider != "scripted":
            ap.error("--fault only makes sense with --provider scripted")
        if args.provider in runner.HOSTED and args.budget is None:
            ap.error(
                "hosted providers need --budget (an evaluation loop must have a spending ceiling)"
            )
        with provider_context(args.provider, parse_faults(args.fault)) as (prov, model):
            run = runner.run_suite(
                name,
                options,
                provider=prov,
                model=model,
                trials=args.trials,
                budget_usd=args.budget,
                variant=args.variant,
                note=args.note,
                provider_label=args.provider,
                real_money=args.provider in runner.HOSTED,
            )
        runner.save_run(run, args.out)
        status = "complete" if run.complete else f"INCOMPLETE ({run.manifest['aborted_reason']})"
        print(
            f"{name}: {status}, {run.manifest['n_done']}/{run.manifest['n_cases']} cases, pass rate {run.results.rate:.1%}, simulated spend ${run.manifest['spend_usd']:.4f} -> {args.out}"
        )
        return 0 if run.complete else 2

    if args.cmd == "gate":
        policy = gate.Policy(
            critical=[p.strip() for p in args.critical.split(",") if p.strip()],
            max_drop=args.max_drop,
            strict=args.strict,
            max_cost_increase=args.max_cost_increase,
            max_latency_increase=args.max_latency_increase,
            allow_dataset_change=args.allow_dataset_change,
        )
        report = gate.evaluate(
            runner.load_run(args.baseline, spans=False),
            runner.load_run(args.candidate, spans=False),
            policy,
        )
        md = report.markdown()
        print(md)
        if args.summary_file:
            with open(args.summary_file, "a") as f:
                f.write(md)
        return report.exit_code

    if args.cmd == "baseline":
        try:
            print(runner.promote(args.source, args.target, force=args.force, reason=args.reason))
        except (FileExistsError, ValueError) as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.cmd == "dashboard":
        runs = [runner.load_run(p) for p in args.runs]
        Path(args.out).write_text(dashboard.render(runs, baseline=args.baseline))
        print(f"dashboard for {len(runs)} run(s) -> {args.out}")
        return 0

    if args.cmd == "monitor":
        prod = monitor.turns_from(tracing.load_spans(args.spans))
        ref = monitor.turns_from(tracing.load_spans(args.reference))
        report = monitor.monitor(prod, ref, monitor.MonitorConfig(arl0=args.arl0, n_sims=args.sims))
        print(f"{report.turns} production turns; reference rates {report.reference_rates}")
        for a in report.alerts:
            print(f"ALERT {a.kind} at turn {a.at_turn} (trace {a.trace_id}): {a.evidence}")
        if not report.alerts:
            print("no alerts")
        return 1 if report.alerts else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
