"""Render the Week 11 weekly report (Markdown) from the measured results. Pure: a dict in, text out, so it is tested without a server. Every number in the report comes from the
results dict; the assumptions and the things that were not run are written here once and cannot drift from the code."""

from __future__ import annotations

import math

import evalrag as E

VM_DOLLARS_PER_HOUR = (
    0.20  # an ASSUMED price for a small CPU virtual machine (an input, not looked up)
)


def ms(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 1000:,.0f} ms"


def secs(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f} s"


def quality_table(res: dict) -> str:
    lines = [
        "| configuration | split | in scope: right lesson retrieved | in scope: abstained (a miss) | in scope: valid citation | out of scope: nothing retrieved | out of scope: abstained (correct) | out of scope: answered anyway |",
        "|---|---|---|---|---|---|---|---|",
    ]
    labels = {
        "baseline": "baseline (any BM25 match)",
        "floor": f"relevance floor {res['floor']:.1f}",
        "abstain": f"floor {res['floor']:.1f} + abstain without calling the model",
    }
    for name in [n for n in ("baseline", "floor", "abstain") if n in res["quality"]]:
        outcomes = res["quality"][name]["outcomes"]
        label = labels[name]
        for split in (None, "dev", "test"):
            s = E.summarize(outcomes, split)
            lines.append(
                f"| {label} | {split or 'all'} (n={s['n_in']}+{s['n_out']}) | {E.fmt(s['hit'])} | {E.fmt(s['abstained_in'])} | "
                f"{E.fmt(s['cited_in'])} | {E.fmt(s['no_sources_out'])} | {E.fmt(s['abstained_out'])} | {E.fmt(s['answered_out'])} |"
            )
    return "\n".join(lines)


def retrieval_section(res: dict) -> str:
    r = res["retrieval"]
    return (
        f"Offline, no model: BM25 over {r['chunks']} chunks of {r['docs']} lessons. The right lesson is in the top 4 for **{E.fmt(r['hit_all'])}** of the in-scope questions "
        f"(rank 1 for {r['rank1']} of {r['n_in']}). The scores that would have to separate in-scope from out-of-scope questions overlap: in-scope top scores run "
        f"{r['in_min']:.1f} to {r['in_max']:.1f} (median {r['in_median']:.1f}); out-of-scope run {r['out_min']:.1f} to {r['out_max']:.1f} (median {r['out_median']:.1f}). "
        f"The floor chosen on the dev half is **{res['floor']:.1f}** (the highest value that keeps at least 90% of the dev in-scope questions retrieving something); "
        f"on the held-out test half it keeps {r['test_in_kept']} of {r['test_in']} in-scope questions and removes the sources of {r['test_out_dropped']} of {r['test_out']} out-of-scope ones."
    )


def closed_table(rows: list[dict]) -> str:
    lines = [
        "| users | requests | errors | req/s | tokens/s | TTFT p50 | TTFT p95 | latency p50 | latency p95 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['users']} | {r['n']} | {sum(r['errors'].values())} | {r['rps']:.2f} | {r['throughput']:.0f} | {ms(r['ttft50'])} | {ms(r['ttft95'])} | {secs(r['lat50'])} | {secs(r['lat95'])} |"
        )
    return "\n".join(lines)


def open_table(rows: list[dict]) -> str:
    lines = [
        "| arrival rate (req/s) | sent | ok | errors | TTFT p50 | TTFT p95 | latency p50 | latency p95 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        err = ", ".join(f"{k}: {v}" for k, v in r["errors"].items()) or "0"
        lines.append(
            f"| {r['rate']} | {r['n']} | {r['ok']} | {err} | {ms(r['ttft50'])} | {ms(r['ttft95'])} | {secs(r['lat50'])} | {secs(r['lat95'])} |"
        )
    return "\n".join(lines)


def render(res: dict) -> str:
    labels_short = {"baseline": "baseline", "floor": "floor", "abstain": "floor + abstain"}
    ex = res["extraction"]
    sat = max(res["closed"], key=lambda r: r["rps"])
    per_1000 = VM_DOLLARS_PER_HOUR / 3600 / sat["rps"] * 1000 if sat["rps"] else float("nan")
    knee = res.get("knee")
    stats = "\n".join(
        f"| {s['name']} | {s['memory']} | {s['cpu']} |" for s in res.get("container_stats", [])
    )
    return f"""# Week 11 report: a RAG API and a fine-tuned model, deployed and load-tested

**What runs.** Three containers from `docker compose` on one machine: the Week 10 fine-tuned order extractor (SmolLM2-135M, Q8_0 GGUF) and Qwen2.5-0.5B-Instruct (Q8_0) in two
llama.cpp servers (CPU, inside Docker's Linux VM), and the FastAPI gateway (API keys, rate limits, bounded queue, SSE, Prometheus metrics, `/v1/ask` RAG with citations).
The gateway image is {res["image_mb"]:.0f} MB and holds no model weights; the corpus is this repository's Week 1-11 lessons ({res["retrieval"]["docs"]} files).
Start-up: images built and containers started in {res["up_seconds"]:.0f} s, ready (`/readyz` for both models) {res["ready_seconds"]:.1f} s later.

## 1. Does the fine-tuned model still work behind the gateway?
The 38 hand-written order emails through `POST /v1/chat/completions` with `model="order-extractor"` (greedy): **exact match {E.fmt(ex["exact"])}**, field accuracy {ex["field_accuracy"]:.0%}, valid order objects {ex["valid_order"]:.0%}.
The Week 10 and Day 1 numbers for this model outside Docker are 74% exact, 89% field accuracy, 92% valid: {"the deployment reproduces them" if abs(ex["exact"][0] - 0.74) < 0.03 else "the deployment differs from them; see Limits"}.

## 2. Does `/v1/ask` answer from the sources and abstain when it should? ({res["n_in"]} in-scope and {res["n_out"]} out-of-scope questions)
{retrieval_section(res)}

{quality_table(res)}

Requests that ended in an error (an in-band stream error after a 200 status, or an HTTP error), out of {res["n_in"] + res["n_out"]} questions: {", ".join(f"{labels_short[n]} **{E.summarize(res['quality'][n]['outcomes'])['errors']}**" for n in res["quality"])}. Errored questions count as failures in every rate above (not as abstentions).

Rates carry Wilson 95% intervals; the dev and test halves are small, so differences of a few questions are noise. The floor was chosen on dev only. Answers are judged by code:
*valid citation* means at least one `[n]` that refers to a retrieved source and none that do not; *abstained* means the exact refusal sentence.

## 3. Load (the final configuration, `/v1/ask`, 100 new tokens at most)
**Closed loop** (a fixed number of users, each sending its next question when the last answer finished; {res["per_level"]} requests per level):

{closed_table(res["closed"])}

**Open loop** (Poisson arrivals for {res["open_duration"]:.0f} s at a fixed rate, whether or not earlier requests have finished):

{open_table(res["open"])}

{"The service stops keeping up (p95 more than 3x the lightest load, or errors) at about **" + str(knee) + " requests per second**." if knee else "Within the rates tried, p95 never exceeded 3x its lightest-load value and nothing failed: the knee is above the highest rate tried, so no capacity limit was found."}

## 4. Resources
| container | memory | CPU at the time of the reading |
|---|---|---|
{stats}

## 5. Cost (assumed prices; an input, not a quote)
At the best closed-loop throughput ({sat["rps"]:.2f} requests per second) a machine priced at an assumed **${VM_DOLLARS_PER_HOUR}/hour** costs about **${per_1000:.3f} per 1,000 answered questions** at 100% utilisation
(real traffic is bursty: divide by your utilisation). This ignores operations time, which Day 3 showed is usually the larger term.

## Not run
- Any cloud deployment (Fly.io, Render, Modal): the files in `solutions/deploy/` are sketches that were never deployed
- A GPU, vLLM, Ollama, a hosted frontier model (no key was used): its answer quality, and what a larger model does with the citation instruction, are unknown
- A browser session on the Streamlit page (its behaviour is covered by headless AppTest tests)
- Multi-replica deployments: rate limits are per process

## Limits
- The questions were written by the person who wrote the lessons, with the lessons in view: retrieval will look better than it would for strangers' wording
- {res["n_in"]} + {res["n_out"]} questions give wide intervals; one run, greedy decoding
- A 0.5B-parameter chat model: how a larger model follows the citation and refusal instructions was not measured
- **Answer correctness was not graded.** The rates above are about retrieval, citation format and abstention, all judged by code. Reading the answers by hand: several are right, some are
  confidently wrong or degenerate (a repeating loop), and some copy a table cell from a retrieved lesson verbatim as the whole answer, including once a cell that described this system's own earlier output
  (the corpus is the course, which quotes the system). A faithfulness judge (Week 4 Day 2) is the next measurement.
- The load test ran against the CPU containers of one laptop: absolute latencies describe this machine, the shape (saturation, knee) is what transfers
- Closed-loop and open-loop numbers use the same question mix, so retrieval cost is included but cache effects are not controlled
"""
