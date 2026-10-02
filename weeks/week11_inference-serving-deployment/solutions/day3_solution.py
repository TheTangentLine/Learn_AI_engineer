"""Week 11 Day 3 - Solution: GPU memory and cost arithmetic, speculative decoding on real models, and an API-versus-self-hosting break-even calculator.

1. MEMORY     will it fit? weights + KV cache against a GPU's memory, for three real model shapes (published configs, entered by hand) and several precisions
2. SPECULATE  REAL: SmolLM2-360M-Instruct (target) with SmolLM2-135M-Instruct (draft) in the Week 9 decoder: exactness, acceptance rate, tokens per target pass,
              measured wall-clock against plain cached decoding, and the formula's prediction
3. MONEY      break-even API against self-hosted with ASSUMED prices (every price and GPU spec below is an input you can change; none was looked up)

  uv run python weeks/week11_inference-serving-deployment/solutions/day3_solution.py [--tokens 64] [--prompts 6] [--only memory,spec,verify,money]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.append(str(ROOT / "weeks/week10_fine-tuning/solutions"))
sys.path.append(str(ROOT / "weeks/week09_transformers-from-scratch/solutions"))
sys.path.append(str(ROOT / "weeks/week09_transformers-from-scratch/solutions/weekly/fastgen"))

import economics as E  # noqa: E402
import speculative as SP  # noqa: E402

# Published configs, entered by hand (parameters, layers, KV heads, head dimension).
MODELS = {
    "Llama-3-8B": E.Spec(params=8_030_261_248, layers=32, kv_heads=8, head_dim=128),
    "Qwen2.5-7B": E.Spec(params=7_615_616_512, layers=28, kv_heads=4, head_dim=128),
    "Llama-3-70B": E.Spec(params=70_553_706_496, layers=80, kv_heads=8, head_dim=128),
}

# ASSUMPTIONS (inputs, not facts): a hosted-style price card, a rented-GPU price, a memory bandwidth and the share of it a real engine reaches.
API = E.ApiPrice(input_per_m=0.50, output_per_m=1.50)
GPU_PER_HOUR = 1.00
GPU_GB = 24
GPU_BANDWIDTH_GBS = 900.0
BANDWIDTH_EFFICIENCY = 0.5


def section_memory() -> None:
    print(
        f"1. WILL IT FIT? one {GPU_GB} GB GPU (and 80 GB), 1.5 GB reserved for activations and runtime (an assumption), context 4,096"
    )
    print(f"   {'model':<14}{'weights':>10}{'GPU':>8}{'KV/seq':>9}{'sequences that fit':>22}")
    for name, spec in MODELS.items():
        for bits, wb in (("bf16", 2), ("int8", 1), ("4-bit", 0.5)):
            for gb in (GPU_GB, 80):
                f = E.fit_on_gpu(spec, gpu_gb=gb, context=4096, weight_bytes=wb)
                verdict = f"{f.max_concurrent}" if f.fits else "does not fit"
                print(
                    f"   {name:<14}{f.weights_gb:>7.1f} GB{gb:>5} GB{f.kv_per_sequence_gb * 1000:>7.0f}MB{verdict:>22}   ({bits})"
                )
    spec = MODELS["Llama-3-8B"]
    print(
        f"\n   decode ceiling of Llama-3-8B bf16 (memory-bound, {GPU_BANDWIDTH_GBS:.0f} GB/s assumed, context 1,024): tokens/s ceiling by batch"
    )
    for b in (1, 8, 32, 64):
        c = E.decode_tokens_per_second_ceiling(
            spec, bandwidth_gbs=GPU_BANDWIDTH_GBS, context=1024, batch=b
        )
        print(
            f"   batch {b:>3}: {c:>7.0f} tokens/s total, {c / b:>5.0f} per user (a ceiling; real engines reach perhaps {BANDWIDTH_EFFICIENCY:.0%})"
        )


def section_speculative(n_tokens: int, n_prompts: int) -> None:
    import torch
    from generate import generate
    from infer import load_base

    print(
        f"\n2. SPECULATIVE DECODING (real models, CPU float32): target SmolLM2-360M-Instruct, draft SmolLM2-135M-Instruct, greedy, {n_tokens} new tokens, {n_prompts} prompts"
    )
    target, tok = load_base("HuggingFaceTB/SmolLM2-360M-Instruct")
    draft, _ = load_base("HuggingFaceTB/SmolLM2-135M-Instruct")
    questions = [
        "Explain what a hash table is in two sentences.",
        "Write a Python function that returns the n-th Fibonacci number.",
        "List three differences between TCP and UDP.",
        "Summarise why the sky is blue.",
        "What does a load balancer do?",
        "Give me a short checklist for reviewing a pull request.",
    ][:n_prompts]
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False
        )
        for q in questions
    ]
    ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    torch.manual_seed(0)

    # the speed of one step of each model (a single token with a warm cache), to get the draft/target cost ratio
    def step_seconds(m) -> float:
        t0 = time.perf_counter()
        generate(m, ids[0], 24, mode="cache", temperature=0.0)
        return (time.perf_counter() - t0) / 24

    step_seconds(target)
    step_seconds(draft)  # warm up
    t_step, d_step = step_seconds(target), step_seconds(draft)
    ratio = d_step / t_step
    print(
        f"   one decoding step: target {t_step * 1000:.1f} ms, draft {d_step * 1000:.1f} ms -> cost ratio c = {ratio:.2f}"
    )

    base_tok_s = []
    for p in ids:
        t0 = time.perf_counter()
        ref = generate(target, p, n_tokens, mode="cache", temperature=0.0).tokens
        base_tok_s.append(len(ref) / (time.perf_counter() - t0))
    base_rate = sum(base_tok_s) / len(base_tok_s)
    print(f"   plain cached decoding of the target: {base_rate:.1f} tokens/s")
    print(
        f"   {'k':>3}{'accept rate':>13}{'tokens / target pass':>22}{'measured tok/s':>16}{'measured speedup':>18}{'formula speedup':>17}{'exact':>7}"
    )
    for k in (2, 4, 6, 8):
        accepted = drafted = toks = passes = 0
        seconds = 0.0
        exact = True
        for p in ids:
            ref = generate(target, p, n_tokens, mode="cache", temperature=0.0).tokens
            st = SP.speculative_generate(target, draft, p, n_tokens, k=k)
            exact &= st.tokens == ref
            accepted += st.accepted
            drafted += st.drafted
            toks += len(st.tokens)
            passes += st.target_forwards
            seconds += st.seconds
        alpha = accepted / drafted
        rate = toks / seconds
        print(
            f"   {k:>3}{alpha:>13.0%}{toks / passes:>22.2f}{rate:>16.1f}{rate / base_rate:>17.2f}x{SP.expected_speedup(alpha, k, ratio):>16.2f}x{'yes' if exact else 'NO':>7}"
        )


def section_verify_cost() -> None:
    """Why speculation can lose: the formula assumes verifying k+1 tokens costs one step. That holds when a step is memory-bound (a big model on a GPU), not here."""
    import torch
    from infer import load_base
    from kvcache import KVCache, forward_cached

    print(
        "\n2b. WHAT DOES VERIFYING k+1 TOKENS COST? one forward pass of the target over n new tokens with a 100-token cache (CPU, float32)"
    )
    target, _ = load_base("HuggingFaceTB/SmolLM2-360M-Instruct")
    print(f"   {'new tokens':>11}{'ms':>9}{'x one token':>13}")
    one = None
    for n in (1, 3, 5, 9):
        best = float("inf")
        for _ in range(5):
            cache = KVCache.for_model(target, 1, 256)
            with torch.no_grad():
                forward_cached(target, torch.randint(0, 1000, (1, 100)), cache)
                t0 = time.perf_counter()
                forward_cached(target, torch.randint(0, 1000, (1, n)), cache)
                best = min(best, time.perf_counter() - t0)
        one = one or best
        print(f"   {n:>11}{best * 1000:>9.1f}{best / one:>12.2f}x")


def section_money() -> None:
    print("\n3. API OR SELF-HOSTED? (every price and GPU figure is an assumed input)")
    print(
        f"   API card: ${API.input_per_m} / ${API.output_per_m} per million input / output tokens; GPU: ${GPU_PER_HOUR}/hour, {GPU_GB} GB, "
        f"{GPU_BANDWIDTH_GBS:.0f} GB/s x {BANDWIDTH_EFFICIENCY:.0%} efficiency"
    )
    spec = MODELS["Llama-3-8B"]
    batch = 32
    tps = (
        E.decode_tokens_per_second_ceiling(
            spec, bandwidth_gbs=GPU_BANDWIDTH_GBS, context=1024, batch=batch
        )
        * BANDWIDTH_EFFICIENCY
    )
    sh = E.SelfHost(
        gpu_per_hour=GPU_PER_HOUR,
        tokens_per_second=tps,
        prefill_tokens_per_second=8 * tps,
        utilisation=0.5,
        min_gpus=1,
        ops_per_month=0.0,
    )
    print(
        f"   self-hosted Llama-3-8B, batch {batch}: {tps:.0f} output tokens/s (ceiling x efficiency), prefill assumed 8x that, 50% utilisation"
    )
    print(
        f"\n   {'output tokens / month':>22}{'API $':>12}{'self-hosted $':>16}{'GPUs':>6}{'cheaper':>13}"
    )
    for out_m in (1, 10, 100, 1_000, 10_000, 100_000):
        c = E.compare(out_m * 3e6, out_m * 1e6, API, sh)
        print(f"   {out_m:>20}M{c.api:>12,.0f}{c.selfhost:>16,.0f}{c.gpus:>6}{c.cheaper:>13}")
    be = E.break_even_tokens_per_month(API, sh)
    print(
        f"\n   break-even: {'never in range' if be is None else f'{be / 1e6:,.0f}M output tokens per month (3 input per output)'} with no operations cost"
    )
    ops = E.SelfHost(**{**sh.__dict__, "ops_per_month": 2000.0})
    be2 = E.break_even_tokens_per_month(API, ops)
    print(
        f"   adding $2,000 per month of engineering and monitoring: {'never in range' if be2 is None else f'{be2 / 1e6:,.0f}M output tokens per month'}"
    )
    print(
        "\n   sensitivity of the break-even (millions of output tokens per month) to one assumption at a time:"
    )
    sens = E.sensitivity(
        ops,
        API,
        {
            "gpu_per_hour": [0.5, 1, 2],
            "tokens_per_second": [0.5, 1, 2],
            "utilisation": [0.5, 1, 2],
            "ops_per_month": [0, 1, 2],
        },
    )
    for name, rows in sens.items():
        cells = "  ".join(
            f"x{f:g}: {'never' if v is None else f'{v / 1e6:,.0f}M'}" for f, v in rows
        )
        print(f"   {name:<20}{cells}")

    print(
        "\n   the Week 10 order extractor (60 prompt + 70 new tokens per request) on this laptop's measured 275 output tokens/s (Day 2, 8 slots),"
    )
    print(
        "   priced as if it ran on a $0.20/hour CPU VM of the same speed (an assumption: a VM may be slower than an M2):"
    )
    small = E.SelfHost(
        gpu_per_hour=0.20,
        tokens_per_second=275,
        prefill_tokens_per_second=6600,
        utilisation=0.5,
        min_gpus=1,
    )
    for req_m in (0.1, 1, 10, 100):
        n = req_m * 1e6
        c = E.compare(n * 60, n * 70, API, small)
        print(
            f"   {req_m:>6.1f}M requests / month: API ${c.api:>9,.0f}, self-hosted ${c.selfhost:>9,.0f} ({c.gpus} machine{'s' if c.gpus > 1 else ''}) -> {c.cheaper}"
        )


def main(argv: list[str]) -> None:
    n_tokens = int(argv[argv.index("--tokens") + 1]) if "--tokens" in argv else 64
    n_prompts = int(argv[argv.index("--prompts") + 1]) if "--prompts" in argv else 6
    only = argv[argv.index("--only") + 1].split(",") if "--only" in argv else None
    for name, fn in (
        ("memory", section_memory),
        ("spec", lambda: section_speculative(n_tokens, n_prompts)),
        ("verify", section_verify_cost),
        ("money", section_money),
    ):
        if only is None or name in only:
            fn()


if __name__ == "__main__":
    main(sys.argv)
