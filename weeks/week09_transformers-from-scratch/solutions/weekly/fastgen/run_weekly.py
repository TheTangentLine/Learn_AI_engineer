"""Week 9 weekly challenge: add a KV cache and top-p sampling to the mini-GPT and benchmark the generation speed-up.

uv run python weeks/week09_transformers-from-scratch/solutions/weekly/fastgen/run_weekly.py [--no-qwen] [--minigpt PATH]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOLUTIONS = HERE.parents[1]
for p in (str(HERE), str(SOLUTIONS), str(ROOT)):
    sys.path.insert(0, p)

import bench  # noqa: E402
import blocks as B  # noqa: E402
import torch  # noqa: E402
from generate import generate  # noqa: E402
from kvcache import KVCache, forward_cached  # noqa: E402


def fmt_modes(res: dict[str, dict]) -> str:
    base = res["nocache"]["total_s"]
    lines = [
        f"   {'mode':<10}{'prefill ms':>11}{'ms / new token':>16}{'tokens/s':>10}{'total s':>9}{'speed-up':>10}   same tokens"
    ]
    for m, r in res.items():
        lines.append(
            f"   {m:<10}{1000 * r['prefill_s']:>11.1f}{r['ms_per_token']:>16.2f}{r['tokens_per_second']:>10.1f}{r['total_s']:>9.2f}{base / r['total_s']:>9.1f}x   {r['same_tokens']}"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    torch.set_num_threads(4)
    model = bench.bench_model()
    print(
        f"model: {model.num_parameters():,} parameters, {model.cfg.n_layers} layers, width {model.cfg.d_model}, {model.cfg.n_heads} heads / {model.cfg.n_kv_heads} kv (random weights: speed does not depend on them)\n"
    )

    print("1. CORRECTNESS: the cached forward pass against the uncached one, token by token")
    ids = torch.randint(
        0, model.cfg.vocab_size, (1, 48), generator=torch.Generator().manual_seed(0)
    )
    with torch.no_grad():
        full = model(ids)
        cache = KVCache.for_model(model, 1, 64)
        parts = [forward_cached(model, ids[:, :20], cache)] + [
            forward_cached(model, ids[:, i : i + 1], cache) for i in range(20, 48)
        ]
    print(
        f"   largest logit difference over 48 positions (20 prefilled, 28 decoded one by one): {(full - torch.cat(parts, 1)).abs().max().item():.1e}\n"
    )

    print("2. SPEED: 64-token prompt, generate N new tokens greedily (best of 2)")
    for n in (32, 128, 256):
        print(f"   N = {n}")
        print(fmt_modes(bench.compare_modes(model, 64, n)))

    print("\n3. COST OF ONE DECODE STEP as the context grows (milliseconds, median of 5 steps)")
    print(f"   {'context':>8}{'no cache':>10}{'cache':>8}")
    for r in bench.step_latency_by_position(model, 64, 256, every=48):
        print(f"   {r['context']:>8}{r['nocache_ms']:>10.2f}{r['cache_ms']:>8.2f}")

    print(
        "\n4. GROUPED GQA DECODING (single-token step, 256 cached tokens): the H/G query heads of a group share one key/value head"
    )
    gv = bench.grouped_vs_repeated(model)
    print(
        f"   grouped (no copies of K/V): {gv['grouped']:.2f} ms/step    repeated K/V heads: {gv['repeated']:.2f} ms/step"
    )

    print("\n5. CACHE MEMORY (float32) after 200 tokens")
    mem = bench.cache_memory(model, 200)
    print(
        f"   valid {mem['valid_bytes']:,} bytes = formula {mem['formula']:,}; preallocated for the full length: {mem['reserved_bytes']:,}"
    )

    if "--no-qwen" not in argv:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = "Qwen/Qwen2.5-0.5B-Instruct"
        tok = AutoTokenizer.from_pretrained(name)
        hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
        qwen = B.Decoder(B.config_from_hf(hf.config)).eval()
        B.load_hf_state_dict(qwen, hf.state_dict())
        del hf
        prompt = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()
        print(
            "\n6. THE REAL QWEN2.5-0.5B in the from-scratch decoder: 48 greedy tokens after a 5-token prompt"
        )
        res = {m: generate(qwen, prompt, 48, mode=m) for m in ("nocache", "cache")}
        for m, g in res.items():
            print(
                f"   {m:<8} {g.tokens_per_second:>6.1f} tokens/s decoding ({g.total_s:.1f} s total)"
            )
        print(
            f"   identical tokens: {res['nocache'].tokens == res['cache'].tokens}; text: {tok.decode(res['cache'].tokens)!r}"
        )
        gq = bench.grouped_vs_repeated(qwen, prompt_len=200, steps=20, repeats=2)
        print(
            f"   grouped GQA step {gq['grouped']:.1f} ms against repeated heads {gq['repeated']:.1f} ms (14 query heads over 2 kv heads)"
        )

    ckpt = (
        Path(argv[argv.index("--minigpt") + 1])
        if "--minigpt" in argv
        else ROOT / "outputs/minigpt.pt"
    )
    if ckpt.exists():
        from bpe import BPE

        state = torch.load(ckpt, weights_only=False)
        mini = B.Decoder(state["cfg"]).eval()
        mini.load_state_dict(state["state"])
        tok = BPE.load(ckpt.with_suffix(".bpe.json"))
        prompts = [
            tok.encode(p)
            for p in (
                "## Learning objectives\n",
                "def ",
                "The attention weights",
                "import ",
                "- Explain ",
                "class ",
                "## ",
                "The retry limit is",
            )
        ]
        print(
            "\n7. SAMPLING: quality against diversity on the Day 5 mini-GPT (8 prompts x 3 seeds, 60 tokens each)"
        )
        print(f"   {'setting':<26}{'self-perplexity':>17}{'distinct bigrams':>18}{'loops':>7}")
        for label, kw in [
            ("greedy", dict(temperature=0.0)),
            ("T=1.0, no filter", dict(temperature=1.0)),
            ("T=1.0, top-k 40", dict(temperature=1.0, top_k=40)),
            ("T=1.0, top-p 0.9", dict(temperature=1.0, top_p=0.9)),
            ("T=1.0, top-p 0.5", dict(temperature=1.0, top_p=0.5)),
            ("T=0.7, no filter", dict(temperature=0.7)),
            ("T=1.5, no filter", dict(temperature=1.5)),
            ("T=1.5, top-p 0.9", dict(temperature=1.5, top_p=0.9)),
        ]:
            samples = [
                s
                for seed in range(3)
                for s in bench.sample_batch(
                    mini,
                    prompts,
                    60,
                    top_k=kw.get("top_k"),
                    top_p=kw.get("top_p"),
                    temperature=kw["temperature"],
                    seed=100 * seed,
                )
            ]
            m = bench.sampling_metrics(samples)
            print(
                f"   {label:<26}{m['perplexity']:>17.1f}{m['distinct2']:>18.2f}{m['loop_rate']:>7.0%}"
            )
        print(
            "   (self-perplexity = how surprised the model is by the tokens it chose, at temperature 1 and unfiltered. For comparison, real validation text scores about 17 (loss 2.8 nats))"
        )
        print(
            "\n   one sample, T=1.0, top-p 0.9:\n"
            + tok.decode(
                prompts[0]
                + bench.sample_batch(
                    mini, prompts[:1], 60, temperature=1.0, top_k=None, top_p=0.9, seed=7
                )[0]["tokens"]
            )
        )


if __name__ == "__main__":
    main(sys.argv)
