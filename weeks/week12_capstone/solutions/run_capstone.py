"""Week 12 weekly challenge: run the whole capstone end to end and write the report and the portfolio README.

  uv run python weeks/week12_capstone/solutions/run_capstone.py [--quick] [--no-llm] [--load] [--report-only]

  1. quality     dev and test splits; the chosen system, and (unless --no-llm) the verified model-backed system, each run once on test
  2. floors      always abstain; dump the top BM25 chunk
  3. security    input guard, obedient-model attacks, the poisoned-document table
  4. the gate    five pretend pull requests against the stored baseline
  5. latency     per stage, cold questions; cost
  6. load        (--load) the service process under concurrent users
  7. documents   ``outputs/w12_report.md`` (targets against outcomes) and ``outputs/w12_PORTFOLIO.md`` (the template, filled in)

Results are cached in ``outputs/w12_results.json``; ``--report-only`` re-renders the documents from the cache.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import capstone_report as CR
import day2_solution as D2
import day4_solution as D4
import day6_solution as D6
from lab import OUT, ROOT, Lab

sys.path.append(
    str(ROOT / "weeks/week11_inference-serving-deployment/solutions")
)  # appended: Week 11 has modules with the same names as this week's scripts

import llamacpp as LC  # noqa: E402
import loadgen as LG  # noqa: E402
from copilot import answer as A  # noqa: E402
from copilot import evaluate as E  # noqa: E402
from copilot import gate as GT  # noqa: E402
from copilot import golden as G  # noqa: E402
from copilot import guard as CG  # noqa: E402
from copilot import retrieve as R  # noqa: E402
from copilot import security as S  # noqa: E402
from copilot.llm import OpenAIChat  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = OUT / "w12_results.json"


def rate_pair(rep: dict, kinds: tuple[str, ...]) -> tuple[float, float, float]:
    n = sum(rep[f"{k}_n"] for k in kinds)
    passed = sum(round(rep[k][0] * rep[f"{k}_n"]) for k in kinds)
    return E.rate(passed, n)


def jsonable(x):
    if isinstance(x, tuple):
        return list(x)
    raise TypeError(type(x))


def demo_transcript(cp) -> list[dict]:
    from demo import SCRIPT

    out = []
    for label, q in SCRIPT:
        r = cp.ask(q)
        out.append(
            {
                "label": label,
                "question": q,
                "answer": r.answer,
                "mode": r.mode,
                "abstained": r.abstained,
                "blocked": r.blocked,
                "sources": [
                    f"{s.doc.split('/')[-1]} > {s.heading.split('>')[-1].strip()}"
                    for s in r.sources[:3]
                ],
            }
        )
    return out


def run(argv: list[str]) -> dict:
    quick = "--quick" in argv
    lab = Lab()
    items = lab.items
    stamp = f"{time.time():.0f}"
    res: dict = {"chunks": len(lab.index.chunks)}

    print("1. quality", flush=True)
    cp = lab.copilot()
    dev, test = (
        E.report(E.run_golden(cp, items, split="dev")),
        E.report(E.run_golden(cp, items, split="test")),
    )
    res["dev"] = {"overall": dev["overall"], "n": dev["n"]}
    res["test"] = {
        "overall": test["overall"],
        "n": test["n"],
        "answerable": rate_pair(test, ("single", "multi")),
    }
    systems = {"extractive (shipped)": test}
    all_runs = E.run_golden(cp, items)
    rep_all = E.report(all_runs)
    res["all"] = {
        "out_of_scope": rep_all["out_of_scope"],
        "retrieval_hit5": rep_all["retrieved_any"],
    }
    if "--no-llm" not in argv:
        with LC.LlamaServer(LC.ensure_chat_gguf(), parallel=1, ctx=8192) as srv:
            chat = OpenAIChat(srv.url, max_tokens=160)
            llm = lab.copilot(answerer=A.LlmAnswerer(chat, A.ExtractiveAnswerer(lab.idf)))
            systems["model + verification + fallback"] = E.report(
                E.run_golden(llm, items, split="test")
            )
    res["systems"] = {
        k: {
            kk: vv
            for kk, vv in v.items()
            if kk
            in ("overall", "single", "multi", "out_of_scope", "adversarial", "single_n", "multi_n")
        }
        for k, v in systems.items()
    }

    print("2. floors", flush=True)
    testitems = [i for i in items if i.split == "test"]
    abstain = [G.score(i, G.Outcome(i.id, A.IDK, True))["passed"] for i in testitems]
    r1 = R.Retriever(lab.index, R.RetrievalConfig(alpha=0.0, scope_by_week=False, k=1))
    dump = []
    for i in testitems:
        src = r1.retrieve(i.question).sources
        dump.append(
            G.score(
                i,
                G.Outcome(
                    i.id,
                    src[0].text if src else "",
                    False,
                    [s.doc for s in src],
                    [s.doc for s in src],
                ),
            )["passed"]
        )
    res["floors"] = {"abstain": sum(abstain) / len(abstain), "dump": sum(dump) / len(dump)}

    print("2b. retrieval variants, gate signals, demo transcript", flush=True)
    answerable_items = [i for i in items if i.answerable]
    variants = {}
    for name in ("BM25 only", "dense only", "hybrid", "hybrid + week scope"):
        per = D2.retrieval_metrics(lab, D2.VARIANTS[name], answerable_items)
        variants[name] = {
            "hit5": E.rate(
                sum(1 for v in per.values() if v["first"] and v["first"] <= 5), len(per)
            ),
            "both": sum(v["all"] for v in per.values() if v["multi"]),
            "multi": sum(v["multi"] for v in per.values()),
        }
    res["retrieval"] = variants
    r5 = lab.retriever()
    cos, ce = ([], []), ([], [])
    gate_rr = GT.RerankGate(lab.reranker, 0)
    for it in items:
        if it.kind == "adversarial":
            continue
        ret = r5.retrieve(it.question)
        side = 0 if it.answerable else 1
        cos[side].append(ret.top_cosine)
        ce[side].append(gate_rr.score(it.question, ret))
    res["gate"] = {
        "auc_cosine": GT.auc(*cos),
        "auc_rerank": GT.auc(*ce),
        "threshold": lab.calibrated_gate()[1]["threshold"],
    }
    res["demo"] = demo_transcript(cp)

    print("3. security", flush=True)
    guard = CG.InputGuard()
    legit = [i.question for i in items if i.answerable] + S.BENIGN_SECURITY
    leaks = D4.direct_attacks(lab)
    tmp = tempfile.mkdtemp(prefix="w12-final-")
    try:
        table = D4.poisoning(lab, D4.poisoned_lab_index(lab, tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    keys = list(leaks.values())
    res["security"] = {
        "direct_none": keys[0],
        "direct_input": keys[1],
        "direct_leaks": keys[2],
        "overblocked": sum(not guard.check(q).allowed for q in legit),
        "legit": len(legit),
        "poison_table": table,
        "poison_through": table["extractive answerer"]["quarantine + output guard"],
    }

    print("4. the gate", flush=True)
    base, prs = D4.gate_demo(lab, table)
    res["gate_prs"] = {
        name: {
            "passed": d.passed,
            "why": "; ".join(d.reasons)
            if d.reasons
            else f"{d.wins} newly pass, {d.losses} newly fail",
            "refactor": name.startswith("refactor"),
        }
        for name, (_, d) in prs.items()
    }
    res["gate_refactor_passes"] = res["gate_prs"]["refactor: no behaviour change"]["passed"]

    print("5. latency and cost", flush=True)
    answerable = [i for i in items if i.answerable]
    stages, totals = D6.stage_latency(lab, lab.copilot(), answerable, 12 if quick else 40, stamp)
    mean_s = sum(totals) / len(totals) / 1000
    res["latency"] = {
        "p50": stages["TOTAL"]["p50"] / 1000,
        "p95": stages["TOTAL"]["p95"] / 1000,
        "stages_ms": stages,
    }
    res["cost"] = {
        "seconds": mean_s,
        "per_1000": 1000 * mean_s * D6.VM_PER_HOUR / 3600,
        "llm_prompt_tokens": 1152,
    }

    if "--load" in argv:
        print("6. load", flush=True)
        from day5_solution import H, Service

        qlist = [i.question for i in items if i.kind in ("single", "multi")]
        counter = {"i": 0}

        def make(i: int) -> dict:
            counter["i"] += 1
            return {
                "question": D6.cold(qlist[i % len(qlist)], f"{stamp}-F{counter['i']}"),
                "k": 5,
                "max_tokens": 200,
            }

        closed = []
        with Service() as svc:
            for users in (1, 2, 4) if quick else (1, 2, 4, 8):
                results, wall = asyncio.run(
                    LG.closed_loop(
                        svc.url,
                        make,
                        concurrency=users,
                        n_requests=12 if quick else 40,
                        headers=H,
                        path="/v1/ask",
                    )
                )
                s = LG.summarize(results, wall)
                closed.append(
                    {
                        "users": users,
                        "rps": s.requests_per_second,
                        "lat50": s.latency["p50"],
                        "lat95": s.latency["p95"],
                        "errors": sum(s.errors.values()),
                    }
                )
        res["load_closed"] = closed

    print("7. tests", flush=True)
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(HERE)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    res["tests"] = (
        int(last.split(" passed")[0].split()[-1])
        if " passed" in last and out.returncode == 0
        else 0
    )
    return res


def write_documents(res: dict) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "w12_report.md").write_text(CR.render_report(res))
    template = (HERE.parent / "templates" / "portfolio_readme_template.md").read_text()
    (OUT / "w12_PORTFOLIO.md").write_text(CR.render_portfolio(res, template))
    print(f"documents written: {OUT / 'w12_report.md'} and {OUT / 'w12_PORTFOLIO.md'}")


def main(argv: list[str]) -> None:
    if "--report-only" in argv:
        res = json.loads(RESULTS.read_text())
        if (
            "--refresh-demo" in argv
        ):  # re-run only the demo transcript (a cheap change) and update the cache
            res["demo"] = demo_transcript(Lab(load_reranker=True).copilot())
            RESULTS.write_text(json.dumps(res, indent=1, default=jsonable))
        for section in ("dev", "test"):
            res[section]["overall"] = tuple(res[section]["overall"])
        res["test"]["answerable"] = tuple(res["test"]["answerable"])
        for k in ("out_of_scope", "retrieval_hit5"):
            res["all"][k] = tuple(res["all"][k])
        for v in res["retrieval"].values():
            v["hit5"] = tuple(v["hit5"])
        for v in res["systems"].values():
            for kk in ("overall", "single", "multi", "out_of_scope", "adversarial"):
                v[kk] = tuple(v[kk])
    else:
        res = run(argv)
        RESULTS.write_text(json.dumps(res, indent=1, default=jsonable))
    write_documents(res)
    met = sum(t["met"] for t in CR.targets(res))
    print(f"targets met: {met} of 8")


if __name__ == "__main__":
    main(sys.argv)
