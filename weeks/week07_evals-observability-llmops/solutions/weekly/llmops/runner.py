"""Run the evaluation suite and write a RUN DIRECTORY that everything else reads.

    runs/<name>/
      manifest.json   what was run and whether it finished: variant, options, provider, model, trials, dataset hash,
                      price card, latency model, spend, complete / aborted_reason, created_at
      results.json    the Day 3 ``SuiteResult``: case id -> fraction of trials passed
      cases.jsonl     per case: kind, split, pass fraction, failures, tokens, model calls, cost, estimated latency
      spans.jsonl     every span of every trial (Day 4): the raw evidence behind the numbers

A run that stops early (spend cap, crash) is saved with ``complete: false`` and the gate refuses to compare it.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import costlab as cl
import day1_solution as d1
import evalcases as ec
import evalgate as eg
import traceeval as te

from common import tracing

HOSTED = {"anthropic", "openai"}  # providers that cost real money
MANIFEST_VERSION = 1


class BudgetExceeded(RuntimeError):
    pass


def dataset_hash(cases) -> str:
    """A fingerprint of WHAT is being asked: ids, splits, kinds, expected routes and the opening message of every case. A
    reworded case changes it even if the id stays, which a bare id-set check (Day 3) would miss."""
    rows = sorted(
        (c.id, c.split, c.kind, c.route, c.scenario.user_factory().opening) for c in cases
    )
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


@dataclass
class Run:
    manifest: dict
    results: eg.SuiteResult
    cases: list[dict]
    spans: list = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return bool(self.manifest.get("complete"))

    @property
    def name(self) -> str:
        return self.manifest["name"]


def _case_row(case, trials: list, spans_by_trial: list[list], price: cl.PriceCard) -> dict:
    n = len(trials)
    calls = [cl.chat_calls(sp) for sp in spans_by_trial]
    return {
        "id": case.id,
        "kind": case.kind,
        "split": case.split,
        "pass_fraction": sum(t.passed for t in trials) / n,
        "failures": sorted({f for t in trials for f in t.failures}),  # a passing trial has none
        "trials": n,
        "model_calls": sum(len(c) for c in calls) / n,
        "input_tokens": sum(i for c in calls for i, _ in c) / n,
        "output_tokens": sum(o for c in calls for _, o in c) / n,
        "cost_usd": sum(cl.trace_cost(sp, price) for sp in spans_by_trial) / n,
        "latency_s": sum(cl.trace_latency(sp) for sp in spans_by_trial) / n,
        "defects": sorted(
            {v.code for sp in spans_by_trial for v in te.check_trace(sp) if v.code in te.DEFECTS}
        ),
    }


def run_suite(
    name: str,
    options: dict,
    *,
    provider: str,
    model: str | None = None,
    trials: int = 1,
    budget_usd: float | None = None,
    price: cl.PriceCard = cl.CHEAP,
    cases=None,
    clock: Callable[[], float] = time.time,
    variant: str = "custom",
    note: str = "",
    provider_label: str | None = None,
    real_money: bool | None = None,
) -> Run:
    """Run every case ``trials`` times against the support system built from ``options`` and return the ``Run``.

    ``budget_usd`` caps the SIMULATED spend (``price`` applied to the real token counts). A hosted provider REQUIRES a cap:
    an evaluation loop with no ceiling is how a bill gets surprising. ``real_money`` overrides the guess made from the provider
    name (a scripted fake pretends to be a provider but costs nothing); ``provider_label`` is the name recorded in the manifest."""
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if (real_money if real_money is not None else provider in HOSTED) and budget_usd is None:
        raise ValueError(f"provider {provider!r} costs real money: pass a budget_usd")
    cases = cases if cases is not None else ec.build_cases(lambda: None)
    rows, fractions, all_spans, spend = [], {}, [], 0.0
    aborted = ""
    with tracing.capture() as rec:
        for case in cases:
            trial_runs, per_trial_spans = [], []
            for trial in range(trials):
                before = len(rec.spans)
                attrs = {
                    "app.eval.case_id": case.id,
                    "app.eval.trial": trial,
                    "app.eval.split": case.split,
                }
                with tracing.span("eval.case", attrs) as sp:
                    (r,) = d1.run_variant([case], name, provider=provider, model=model, **options)
                    sp.set_attribute("app.eval.passed", r.passed)
                trial_runs.append(r)
                per_trial_spans.append(rec.spans[before:])
                spend += cl.trace_cost(per_trial_spans[-1], price)
                if budget_usd is not None and spend > budget_usd:
                    aborted = f"budget: spent ${spend:.4f} of ${budget_usd:.4f} after {len(rows)} complete cases"
                    break
            if aborted:
                break
            row = _case_row(case, trial_runs, per_trial_spans, price)
            rows.append(row)
            fractions[case.id] = row["pass_fraction"]
        all_spans = rec.spans
    manifest = {
        "version": MANIFEST_VERSION,
        "name": name,
        "variant": variant,
        "options": options,
        "provider": provider_label or provider,
        "model": model or "default",
        "trials": trials,
        "dataset_hash": dataset_hash(cases),
        "n_cases": len(cases),
        "n_done": len(rows),
        "price_card": asdict(price),
        "latency_model": asdict(cl.DEFAULT_LATENCY),
        "spend_usd": spend,
        "budget_usd": budget_usd,
        "complete": not aborted and len(rows) == len(cases),
        "aborted_reason": aborted,
        "created_at": clock(),
        "note": note,
    }
    return Run(
        manifest,
        eg.SuiteResult(
            name,
            fractions,
            {"provider": manifest["provider"], "model": manifest["model"], "trials": trials},
        ),
        rows,
        all_spans,
    )


# ----------------------------------------------------------------------------- files


def save_run(run: Run, directory: str | Path) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(run.manifest, indent=2, sort_keys=True) + "\n")
    eg.save(run.results, d / "results.json")
    (d / "cases.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in run.cases))
    (d / "spans.jsonl").write_text(
        "".join(json.dumps(asdict(s), sort_keys=True) + "\n" for s in run.spans)
    )
    return d


def load_run(directory: str | Path, *, spans: bool = True) -> Run:
    d = Path(directory)
    for required in ("manifest.json", "results.json", "cases.jsonl"):
        if not (d / required).exists():
            raise FileNotFoundError(f"{d} is not a run directory: missing {required}")
    manifest = json.loads((d / "manifest.json").read_text())
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"unsupported run format version {manifest.get('version')!r}")
    cases = [
        json.loads(line) for line in (d / "cases.jsonl").read_text().splitlines() if line.strip()
    ]
    loaded = tracing.load_spans(d / "spans.jsonl") if spans and (d / "spans.jsonl").exists() else []
    return Run(manifest, eg.load(d / "results.json"), cases, loaded)


def promote(
    source: str | Path, target: str | Path, *, force: bool = False, reason: str = ""
) -> str:
    """Make ``source`` the baseline at ``target``. Replacing an existing baseline needs ``force`` AND a reason (recorded in the
    manifest): moving the goalposts is a decision, never a side effect. Only a COMPLETE run can become a baseline."""
    src, tgt = Path(source), Path(target)
    run = load_run(src)
    if not run.complete:
        raise ValueError("an incomplete run cannot be a baseline")
    if tgt.exists():
        if not force:
            raise FileExistsError(
                f"{tgt} exists: pass force=True and a reason to replace the baseline deliberately"
            )
        if not reason.strip():
            raise ValueError("replacing a baseline needs a reason (it is recorded in the manifest)")
    if reason:
        run.manifest = {**run.manifest, "accepted_because": reason.strip()}
    save_run(run, tgt)
    return f"baseline {run.name} written to {tgt} ({run.manifest['n_done']} cases)"
