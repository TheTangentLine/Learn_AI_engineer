"""Week 11 Day 2 - Solution: continuous batching, measured on a real server, modelled, and the memory manager behind it.

1. REAL      llama-server (continuous batching; `-np N` slots) under a closed-loop load at increasing concurrency: throughput, time to first token, latency
2. MODEL     a step-cost model fitted to the real measurements, and its prediction for the same load; static against continuous batching on the same workload
3. MEMORY    what a KV cache costs per token (Qwen2.5-0.5B and Llama-3-8B shapes), and how many concurrent requests fit with contiguous reservation against paged blocks

  uv run python weeks/week11_inference-serving-deployment/solutions/day2_solution.py [--model Q8_0] [--n 48]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.append(str(ROOT / "weeks/week10_fine-tuning/solutions"))
sys.path.append(str(ROOT / "weeks/week09_transformers-from-scratch/solutions"))

import chatfmt as C  # noqa: E402
import llamacpp as L  # noqa: E402
import loadgen as G  # noqa: E402
import paged_kv as P  # noqa: E402
import scheduler as S  # noqa: E402


def requests_from_dataset(n: int) -> list[dict]:
    """Chat bodies built from the Week 10 synthetic test emails (a realistic mix of lengths); greedy, capped at 200 new tokens."""
    recs = C.read_jsonl(ROOT / "outputs/w10_data/test.jsonl")
    return [{"messages": r["messages"][:-1], "max_tokens": 200, "temperature": 0} for r in recs[:n]]


def measure(model: Path, parallel: int, concurrency: int, bodies: list[dict]) -> G.Summary:
    with L.LlamaServer(model, parallel=parallel, ctx=1024 * parallel) as srv:
        # warm-up so the first request does not pay for graph and cache set-up
        srv.chat(bodies[0]["messages"], max_tokens=16)
        results, wall = asyncio.run(
            G.closed_loop(
                srv.url,
                lambda i: bodies[i % len(bodies)],
                concurrency=concurrency,
                n_requests=len(bodies),
            )
        )
    return G.summarize(results, wall)


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def main(argv: list[str]) -> None:
    fmt = argv[argv.index("--model") + 1] if "--model" in argv else "Q8_0"
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 48
    model = L.GGUF_DIR / f"{L.STEM}-{fmt}.gguf"
    bodies = requests_from_dataset(n)

    if (
        "--fit" in argv
    ):  # reuse the constants of an earlier real run (milliseconds): skips the server measurements
        base, per = (float(x) for x in argv[argv.index("--fit") + 1].split(","))
        print(f"(step model supplied: {base * 1000:.2f} ms + {per * 1000:.3f} ms per sequence)")
    else:
        print(
            f"1. REAL SERVER: {model.name}, {n} extraction requests per cell, closed loop (a user sends its next request when the last one finishes)"
        )
        print(
            f"   {'slots':>6}{'users':>7}{'tokens/s':>10}{'req/s':>8}{'TTFT p50':>10}{'TTFT p95':>10}{'latency p50':>13}{'latency p95':>13}{'per-user tok/s':>16}{'errors':>8}"
        )
        grid = [(1, 1), (1, 4), (2, 2), (4, 4), (4, 8), (8, 8), (8, 16)]
        rows = {}
        for slots, users in grid:
            s = measure(model, slots, users, bodies)
            rows[(slots, users)] = s
            print(
                f"   {slots:>6}{users:>7}{s.throughput:>10.0f}{s.requests_per_second:>8.1f}{s.ttft['p50'] * 1000:>9.0f}ms{s.ttft['p95'] * 1000:>9.0f}ms{s.latency['p50']:>12.2f}s{s.latency['p95']:>12.2f}s{s.per_user_tps:>16.0f}{sum(s.errors.values()):>8}"
            )
            print(f"   done slots={slots} users={users}", file=sys.stderr, flush=True)

        single = rows[(1, 1)]
        base_step = 1.0 / (single.throughput) if single.throughput else 0.0
        points = [
            (slots, 1.0 / (s.throughput / users))
            for (slots, users), s in rows.items()
            if slots == users and s.throughput
        ]
        base, per = S.fit_step_model([(b, b / rows[(b, b)].throughput) for b, _ in points])
        print(
            f"\n2. FITTED STEP MODEL from the runs with as many users as slots: a decoding step costs {base * 1000:.2f} ms + {per * 1000:.3f} ms per sequence in the batch"
        )
        print(
            f"   (one sequence alone: {base_step * 1000:.2f} ms per token = {single.throughput:.0f} tokens/s)"
        )
        print(f"   {'batch':>6}{'measured tok/s':>16}{'model tok/s':>13}")
        for b, _ in points:
            print(f"   {b:>6}{rows[(b, b)].throughput:>16.0f}{b / (base + per * b):>13.0f}")
    m = S.StepModel(base, per, 0.0)
    w = S.poisson_workload(300, 2.5, prompt=(60, 120), output=(40, 110), seed=0, long_tail=0.1)
    print(
        "\n   the same step model, a Poisson workload of 300 requests (2.5 per second, 10% of them 4x longer), static against continuous batching:"
    )
    print(
        f"   {'batch':>6}{'policy':>12}{'tok/s':>8}{'latency p50':>13}{'latency p95':>13}{'TTFT p95':>10}{'slot use':>10}"
    )
    for b in (1, 4, 8):
        for sim in (S.simulate_static, S.simulate_continuous):
            r = sim(w, m, b)
            print(
                f"   {b:>6}{r.policy:>12}{r.throughput:>8.0f}{r.latency(50):>12.1f}s{r.latency(95):>12.1f}s{r.ttft(95):>9.1f}s{r.utilisation:>10.0%}"
            )

    print("\n3. KV-CACHE MEMORY")
    for name, (layers, kvh, hd) in {
        "SmolLM2-135M (this model)": (30, 3, 64),
        "Qwen2.5-0.5B": (24, 2, 64),
        "Llama-3-8B": (32, 8, 128),
    }.items():
        print(
            f"   {name:<28}{kv_bytes_per_token(layers, kvh, hd):>9,} bytes per token in fp16 ({kv_bytes_per_token(layers, kvh, hd) * 1024 / 1e6:.1f} MB per 1,024 tokens)"
        )
    lengths = [
        len(
            C.encode_example(
                b["messages"] + [{"role": "assistant", "content": "x"}],
                __import__("transformers").AutoTokenizer.from_pretrained(
                    "HuggingFaceTB/SmolLM2-135M-Instruct"
                ),
            ).input_ids
        )
        + 90
        for b in bodies
    ]
    print(
        f"\n   request lengths in this workload (prompt + about 90 generated tokens): median {sorted(lengths)[len(lengths) // 2]}, max {max(lengths)}"
    )
    per_tok = kv_bytes_per_token(32, 8, 128)
    budget = 24e9 - 16e9  # a 24 GB GPU holding Llama-3-8B in bf16 leaves about 8 GB for the cache
    for max_len in (2048, 8192):
        n_contig = int(budget // (per_tok * max_len))
        avg = sum(lengths) / len(lengths)
        n_paged = int(
            budget // (per_tok * (avg + 8))
        )  # a half block of waste on average with 16-token blocks
        print(
            f"   Llama-3-8B on a 24 GB GPU (8 GB for the cache): contiguous reservation of {max_len} tokens per request fits {n_contig} requests; paged blocks (average {avg:.0f} tokens) fit about {n_paged}"
        )
    print(
        f"   wasted slots with contiguous 2,048-token buffers: {P.contiguous_waste(lengths, 2048):.0%}; with 16-token blocks: {P.paged_waste(lengths, 16):.1%}"
    )


if __name__ == "__main__":
    main(sys.argv)
