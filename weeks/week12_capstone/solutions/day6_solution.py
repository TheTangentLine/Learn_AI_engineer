"""Week 12 Day 6 - Solution: observability, cost tuning and a load test.

1. WHERE THE TIME GOES   per-stage latency of the product on questions no cache has seen (cold), and one trace
2. COST                  seconds and tokens per question for the extractive product and the model-backed one, priced with an ASSUMED card
3. TUNING                three changes measured on DEV (a cascade gate, a cheaper gate, a response cache); the chosen design is then applied once to TEST
4. LOAD                  the service process under concurrent users and under fixed arrival rates (p50/p95, errors)
5. DASHBOARD             ``outputs/w12_dashboard.html``

  uv run python weeks/week12_capstone/solutions/day6_solution.py [--quick]
"""

from __future__ import annotations

import asyncio
import random
import sys
import time

from day5_solution import ADMIN, H, Service  # noqa: F401
from lab import OUT, ROOT, Lab

sys.path.append(
    str(ROOT / "weeks/week11_inference-serving-deployment/solutions")
)  # appended: Week 11 has modules with the same names as this week's scripts

import llamacpp as LC  # noqa: E402
import loadgen as LG  # noqa: E402
from copilot import answer as A  # noqa: E402
from copilot import dashboard as DASH  # noqa: E402
from copilot import evaluate as E  # noqa: E402
from copilot import gate as GT  # noqa: E402
from copilot.llm import OpenAIChat  # noqa: E402

from common import tracing  # noqa: E402
from common.cache import ResponseCache  # noqa: E402

VM_PER_HOUR = 0.20  # ASSUMED price of a small CPU machine
CARD = (0.50, 1.50)  # ASSUMED hosted-style $ per million input / output tokens


def cold(q: str, tag: str) -> str:
    """The same question with a unique suffix: no embedding, re-ranker or response cache can have seen it."""
    return f"{q} (run {tag})"


def stage_latency(lab: Lab, cp, answerable, n: int, stamp: str) -> tuple[dict, list[float]]:
    """Per-stage latency (ms, p50 and p95) over ``n`` cold questions, and the list of total milliseconds per question."""
    results = []
    for k, it in enumerate(answerable * 2):
        if k >= n:
            break
        results.append(cp.ask(cold(it.question, f"{stamp}-a{k}")))
    stage: dict[str, list[float]] = {}
    for r in results:
        for k, v in r.timings.items():
            stage.setdefault(k, []).append(v * 1000)
    totals = [r.seconds * 1000 for r in results]
    out = {k: {"p50": E.percentile(v, 50), "p95": E.percentile(v, 95)} for k, v in stage.items()}
    out["TOTAL"] = {"p50": E.percentile(totals, 50), "p95": E.percentile(totals, 95)}
    return out, totals


def main(argv: list[str]) -> None:
    quick = "--quick" in argv
    lab = Lab()
    answerable = [i for i in lab.items if i.answerable]
    stamp = f"{time.time():.0f}"
    metrics: dict = {
        "title": "Course Copilot",
        "subtitle": f"Weeks 1-11 lessons as the corpus; {len(lab.index.chunks)} chunks; extractive answerer; rerank gate; CPU (Apple M2)",
    }

    print("1. WHERE THE TIME GOES (cold questions, one process, CPU)")
    cp = lab.copilot()
    metrics["stages_ms"], totals = stage_latency(lab, cp, answerable, 12 if quick else 40, stamp)
    for k, v in metrics["stages_ms"].items():
        print(f"   {k:<13} p50 {v['p50']:>7.1f} ms   p95 {v['p95']:>7.1f} ms")
    with tracing.capture() as rec:
        cp.ask(cold(answerable[3].question, f"{stamp}-trace"))
    print("   one trace:")
    for line in tracing.render_tree(rec.spans).splitlines():
        print("     " + line)

    print(
        f"\n2. COST PER 1,000 QUESTIONS (assumed prices: machine ${VM_PER_HOUR:.2f}/hour; hosted-style card ${CARD[0]:.2f} / ${CARD[1]:.2f} per million input / output tokens)"
    )
    mean_s = sum(totals) / len(totals) / 1000
    cost_rows = [
        {
            "name": "extractive (no model)",
            "seconds": mean_s,
            "tokens": 0,
            "per_1000": 1000 * mean_s * VM_PER_HOUR / 3600,
        }
    ]
    print(
        f"   extractive: {mean_s:.3f} s per question -> ${cost_rows[0]['per_1000']:.4f} per 1,000 questions of machine time, 0 tokens"
    )
    gguf = LC.ensure_chat_gguf()
    with LC.LlamaServer(gguf, parallel=1, ctx=8192) as srv:
        chat = OpenAIChat(srv.url, max_tokens=160)
        llm_cp = lab.copilot(answerer=A.LlmAnswerer(chat, A.ExtractiveAnswerer(lab.idf)))
        m = 8 if quick else 20
        runs = [
            llm_cp.ask(cold(it.question, f"{stamp}-m{k}")) for k, it in enumerate(answerable[:m])
        ]
    sec = sum(r.seconds for r in runs) / len(runs)
    ptok = sum(r.prompt_tokens for r in runs) / len(runs)
    ctok = sum(r.completion_tokens for r in runs) / len(runs)
    machine = 1000 * sec * VM_PER_HOUR / 3600
    hosted = 1000 * (ptok * CARD[0] + ctok * CARD[1]) / 1e6
    cost_rows.append(
        {
            "name": "0.5B model + verification (self-hosted)",
            "seconds": sec,
            "tokens": int(ptok + ctok),
            "per_1000": machine,
        }
    )
    print(
        f"   model-backed: {sec:.3f} s, {ptok:,.0f} prompt + {ctok:,.0f} completion tokens per question -> ${machine:.4f} of machine time, or ${hosted:.3f} if the same tokens were bought at the assumed card"
    )
    metrics["cost"] = cost_rows
    metrics["cost_note"] = (
        "Machine time at an assumed $0.20/hour at 100% utilisation; token prices are assumptions; no hosted model was run."
    )

    print("\n3. TUNING (choose on DEV, apply once to TEST)")
    retr = lab.retriever()
    pos, neg = [], []
    for it in lab.items:
        if it.split != "dev" or it.kind == "adversarial":
            continue
        ret = retr.retrieve(it.question)
        (pos if it.answerable else neg).append(ret.top_cosine)
    low, high = GT.CascadeGate.choose_band(pos, neg)
    full, cal = lab.calibrated_gate()
    print(
        f"   cascade band chosen on dev cosines: refuse <= {low:.3f}, admit >= {high:.3f}, ask the cross-encoder in between"
    )
    dev_items = [i for i in lab.items if i.split == "dev" and i.kind != "adversarial"]
    cascade = GT.CascadeGate(full, low, high)
    agree = 0
    for it in dev_items:
        ret = retr.retrieve(it.question)
        agree += cascade.decide(it.question, ret).allowed == full.decide(it.question, ret).allowed
    print(
        f"   on dev the cascade made the same admit/refuse decision as the full gate on {agree} of {len(dev_items)} questions and needed the cross-encoder for {cascade.called} of them ({cascade.called / len(dev_items):.0%})"
    )
    for name, gate in (
        ("full cross-encoder gate", full),
        ("cascade gate", GT.CascadeGate(full, low, high)),
        ("cross-encoder over 1 source", GT.RerankGate(lab.reranker, cal["threshold"], top=1)),
    ):
        c2 = lab.copilot(gate=gate)
        t = []
        for k, it in enumerate(lab.items):
            if it.split != "dev" or it.kind == "adversarial":
                continue
            t0 = time.perf_counter()
            c2.ask(cold(it.question, f"{stamp}-g{name[:4]}{k}"))
            t.append(time.perf_counter() - t0)
        rep = E.report(E.run_golden(c2, lab.items, split="dev"))
        print(
            f"   {name:<30} dev pass {E.fmt(rep['overall'])}   cold latency p50 {E.percentile(t, 50) * 1000:>5.0f} ms  p95 {E.percentile(t, 95) * 1000:>5.0f} ms"
        )
    chosen = lab.copilot(gate=GT.CascadeGate(full, low, high))
    rep_t = E.report(E.run_golden(chosen, lab.items, split="test"))
    base_t = E.report(E.run_golden(lab.copilot(), lab.items, split="test"))
    print(
        f"   TEST, applied once: full gate {E.fmt(base_t['overall'])}, cascade {E.fmt(rep_t['overall'])}"
    )
    metrics["pass_by_kind"] = {
        k: {
            "n": rep_t[f"{k}_n"],
            "passed": round(rep_t[k][0] * rep_t[f"{k}_n"]),
            "rate": rep_t[k],
            "interval": f"[{rep_t[k][1]:.0%}, {rep_t[k][2]:.0%}]",
        }
        for k in ("single", "multi", "out_of_scope", "adversarial")
    }
    metrics["split"] = "test"

    rng = random.Random(0)
    qs = [i.question for i in lab.items if i.kind in ("single", "multi", "out_of_scope")]
    weights = [1 / (r + 1) ** 1.1 for r in range(len(qs))]
    traffic = rng.choices(qs, weights, k=300)
    cached = lab.copilot(cache=ResponseCache())
    for q in traffic:
        cached.ask(q)
    print(
        f"   response cache on 300 requests whose repeats follow a Zipf distribution over the {len(qs)} golden questions: hit rate {cached.cache.hit_rate:.0%} (a hit skips retrieval, the gate and the answer; blocked and errored requests are never cached)"
    )

    print(
        "\n4. LOAD (the service process; every question made unique so the caches do not flatter it)"
    )
    qlist = [i.question for i in lab.items if i.kind in ("single", "multi")]
    counter = {"i": 0}

    def make(i: int) -> dict:
        counter["i"] += 1
        q = qlist[i % len(qlist)]
        return {"question": cold(q, f"{stamp}-L{counter['i']}"), "k": 5, "max_tokens": 200}

    with Service() as svc:
        closed, opened = [], []
        for users in (1, 2, 4) if quick else (1, 2, 4, 8):
            res, wall = asyncio.run(
                LG.closed_loop(
                    svc.url,
                    make,
                    concurrency=users,
                    n_requests=12 if quick else 40,
                    headers=H,
                    path="/v1/ask",
                )
            )
            s = LG.summarize(res, wall)
            closed.append(
                {
                    "users": users,
                    "rps": s.requests_per_second,
                    "lat50": s.latency["p50"],
                    "lat95": s.latency["p95"],
                    "errors": sum(s.errors.values()),
                }
            )
            print(
                f"   {users} users: {s.requests_per_second:.2f} req/s, latency p50 {s.latency['p50']:.2f} s p95 {s.latency['p95']:.2f} s, first token p50 {s.ttft['p50'] * 1000:.0f} ms, errors {sum(s.errors.values())}"
            )
        for rate in (1.0, 2.0) if quick else (1.0, 2.0, 4.0):
            res, wall = asyncio.run(
                LG.open_loop(
                    svc.url, make, rate=rate, duration=8 if quick else 20, headers=H, path="/v1/ask"
                )
            )
            s = LG.summarize(res, wall)
            opened.append(
                {
                    "rate": rate,
                    "n": s.n,
                    "lat50": s.latency["p50"],
                    "lat95": s.latency["p95"],
                    "errors": sum(s.errors.values()),
                }
            )
            print(
                f"   {rate:g} arrivals/s for {8 if quick else 20} s: {s.n} sent, latency p50 {s.latency['p50']:.2f} s p95 {s.latency['p95']:.2f} s, errors {dict(s.errors) or 0}"
            )
    print(
        "   the same closed-loop test with the cascade gate (COPILOT_GATE=cascade): the cross-encoder runs only for the uncertain middle"
    )
    cascade_rows = []
    with Service(
        extra_env={"COPILOT_GATE": "cascade", "COPILOT_CASCADE_BAND": f"{low},{high}"}
    ) as svc:
        for users in (1, 8) if quick else (1, 4, 8):
            res, wall = asyncio.run(
                LG.closed_loop(
                    svc.url,
                    make,
                    concurrency=users,
                    n_requests=12 if quick else 40,
                    headers=H,
                    path="/v1/ask",
                )
            )
            s = LG.summarize(res, wall)
            cascade_rows.append(
                {
                    "users": users,
                    "rps": s.requests_per_second,
                    "lat50": s.latency["p50"],
                    "lat95": s.latency["p95"],
                    "errors": sum(s.errors.values()),
                }
            )
            print(
                f"   cascade, {users} users: {s.requests_per_second:.2f} req/s, latency p50 {s.latency['p50']:.2f} s p95 {s.latency['p95']:.2f} s, errors {sum(s.errors.values())}"
            )
    metrics["load_cascade"] = cascade_rows
    metrics["load_closed"], metrics["load_open"] = closed, opened
    metrics["notes"] = [
        "One process serves one question at a time (the pipeline runs inside the request handler): throughput does not rise with users; latency does.",
        "All prices are assumptions supplied as inputs.",
    ]
    OUT.mkdir(exist_ok=True)
    path = OUT / "w12_dashboard.html"
    path.write_text(DASH.render(metrics))
    print(f"\n5. DASHBOARD written to {path} ({path.stat().st_size / 1000:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv)
