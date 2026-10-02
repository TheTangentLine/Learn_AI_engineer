"""Week 9 Day 6 - Solution: modern architecture in numbers and in code.

1. CONFIGS   read three real configs (Llama-3-8B, Mixtral-8x7B, Qwen2.5-0.5B); a closed-form parameter count checked against the library's own count
2. MEMORY    weights and KV cache at 4k / 32k / 128k tokens, and what grouped-query attention saves
3. MOE       a sparse mixture-of-experts layer, checked against Hugging Face's Mixtral block; what the load-balancing loss does to routing
4. FLASH     the tiled online-softmax attention algorithm: exact, and the largest tensor it needs

  uv run python weeks/week09_transformers-from-scratch/solutions/day6_solution.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import arch as A  # noqa: E402
import attention as T  # noqa: E402

GB = 1e9

# ----------------------------------------------------------------------------- 1. the three configs


def real_configs() -> dict[str, object]:
    """Llama-3-8B and Mixtral-8x7B written out from memory of their published config.json files; Qwen2.5-0.5B read from the local Hugging Face cache.
    The parameter totals they give (8.03B, 46.7B, 0.49B) are the published ones, which is the check that the numbers are right."""
    from transformers import AutoConfig, LlamaConfig, MixtralConfig

    return {
        "Llama-3-8B": LlamaConfig(
            vocab_size=128256,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            tie_word_embeddings=False,
        ),
        "Mixtral-8x7B": MixtralConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_local_experts=8,
            num_experts_per_tok=2,
            tie_word_embeddings=False,
        ),
        "Qwen2.5-0.5B": AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"),
    }


# ----------------------------------------------------------------------------- 3. mixture of experts


def make_clusters(n_clusters: int, per_cluster: int, d: int, seed: int = 0):
    """Tokens from well-separated clusters; the target is a different fixed linear map per cluster: a task where specialised experts should help."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_clusters, d, generator=g) * 3
    maps = torch.randn(n_clusters, d, d, generator=g) / d**0.5
    x = torch.cat(
        [centers[c] + 0.3 * torch.randn(per_cluster, d, generator=g) for c in range(n_clusters)]
    )
    y = torch.cat(
        [(x[c * per_cluster : (c + 1) * per_cluster] @ maps[c]) for c in range(n_clusters)]
    )
    labels = torch.arange(n_clusters).repeat_interleave(per_cluster)
    return x, y, labels


def train_moe(
    balance_coef: float, *, steps: int = 400, n_experts: int = 8, top_k: int = 2, seed: int = 0
) -> dict:
    torch.manual_seed(seed)
    d = 16
    x, y, labels = make_clusters(4, 256, d, seed=1)
    moe = A.MoEMLP(d, 32, n_experts, top_k)
    opt = torch.optim.Adam(moe.parameters(), lr=3e-3)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (128,))
        out, bal = moe(x[idx][None])
        loss = F.mse_loss(out[0], y[idx]) + balance_coef * bal
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out, bal = moe(x[None])
        mse = F.mse_loss(out[0], y).item()
        load = moe.expert_load(x[None])
    return {
        "mse": mse,
        "balance": bal.item(),
        "load": load,
        "max_over_uniform": (load.max() * n_experts).item(),
        "dead": int((load < 0.01 / n_experts * 10).sum()),
    }


# ----------------------------------------------------------------------------- 4. FlashAttention's algorithm


def flash_report(t: int = 2048, d: int = 64, heads: int = 8, block: int = 64) -> dict:
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, heads, t, d) for _ in range(3))
    t0 = time.perf_counter()
    ref, _ = T.scaled_dot_product_attention(q, k, v, causal=True)
    t_std = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = A.tiled_attention(q, k, v, block=block)
    t_tiled = time.perf_counter() - t0
    big = (q * 60, k * 60, v)  # scores of the order 60*60*sqrt(64): a naive exp(score) overflows
    stable = A.tiled_attention(*big, block=block)
    stable_ref, _ = T.scaled_dot_product_attention(*big, causal=True)
    return {
        "gap": (out - ref).abs().max().item(),
        "gap_extreme": (stable - stable_ref).abs().max().item(),
        "finite": bool(torch.isfinite(stable).all()),
        "full_elems": A.largest_score_tensor(t, t, block=None),
        "tiled_elems": A.largest_score_tensor(t, t, block=block),
        "ms_standard": 1000 * t_std,
        "ms_tiled": 1000 * t_tiled,
    }


def main(argv: list[str]) -> None:
    cfgs = real_configs()
    specs = {n: A.spec_from_hf(c, n) for n, c in cfgs.items()}

    print(
        "1. THREE REAL CONFIGS: closed-form parameter count against the library's count (built on the 'meta' device, no memory used)"
    )
    print(
        f"   {'model':<14}{'layers':>7}{'width':>7}{'heads/kv':>10}{'ffn':>7}{'experts':>14}{'vocab':>8}{'formula':>17}{'library':>17}{'active/token':>16}"
    )
    for n, s in specs.items():
        cp = A.count_params(s)
        lib = A.hf_reference_count(cfgs[n])
        ex = f"{s.n_experts} (top {s.experts_per_token})" if s.n_experts else "-"
        print(
            f"   {n:<14}{s.n_layers:>7}{s.d_model:>7}{f'{s.n_heads}/{s.n_kv_heads}':>10}{s.d_ff:>7}{ex:>14}{s.vocab_size:>8}{cp['total']:>17,}{lib:>17,}{cp['active']:>16,}   {'ok' if cp['total'] == lib else 'MISMATCH'}"
        )
    print("\n   where the parameters are:")
    for n, s in specs.items():
        cp = A.count_params(s)
        print(
            f"   {n:<14}"
            + "  ".join(
                f"{k} {cp[k] / cp['total']:.0%}"
                for k in ("embedding", "attention", "mlp", "output_head")
            )
        )

    print("\n2. MEMORY (bf16 = 2 bytes; one sequence unless stated)")
    print(
        f"   {'model':<14}{'weights bf16':>14}{'int4':>8}   KV cache at 4k / 32k / 128k tokens (GQA, as built)   same model with full multi-head attention"
    )
    for n, s in specs.items():
        mha = A.replace(s, n_kv_heads=s.n_heads)
        row = " / ".join(f"{A.kv_cache_bytes(s, t) / GB:.2f}" for t in (4096, 32768, 131072))
        row_mha = " / ".join(f"{A.kv_cache_bytes(mha, t) / GB:.2f}" for t in (4096, 32768, 131072))
        print(
            f"   {n:<14}{A.weight_bytes(s) / GB:>11.1f} GB{A.weight_bytes(s, 0.5) / GB:>6.1f} GB   {row:<44} GB   {row_mha} GB"
        )
    s = specs["Llama-3-8B"]
    print(
        f"\n   serving Llama-3-8B to 16 users at 32k tokens each: weights {A.weight_bytes(s) / GB:.1f} GB + KV cache {A.kv_cache_bytes(s, 32768, batch=16) / GB:.1f} GB = {(A.weight_bytes(s) + A.kv_cache_bytes(s, 32768, batch=16)) / GB:.1f} GB"
    )
    print(
        "   decoding one token is memory-bound: the weights (active ones) and the cache are read once per token. Upper bound on tokens/second for one sequence at 4k context,"
    )
    for hw, bw in (
        ("Apple M2 (assumed 100 GB/s)", 100e9),
        ("one H100 (assumed 3.35 TB/s)", 3.35e12),
    ):
        parts = []
        for n, sp in specs.items():
            read = A.count_params(sp)["active"] * 2 + A.kv_cache_bytes(sp, 4096)
            parts.append(f"{n} {bw / read:,.0f}")
        print(f"   {hw}: " + ", ".join(parts))

    print("\n3. MIXTURE OF EXPERTS")
    print(
        "   (checked against Hugging Face's MixtralSparseMoeBlock with copied weights in the tests: largest difference 5e-7)"
    )
    print(
        "   a router that picks experts to minimise the task loss alone, against the same with a load-balancing loss (4 clusters of tokens, 8 experts, top 2):"
    )
    print(
        f"   {'balance coefficient':>20}{'task MSE':>10}{'balance loss':>14}{'busiest expert (x uniform)':>28}{'idle experts':>14}"
    )
    for coef in (0.0, 0.01, 0.1):
        r = train_moe(coef)
        print(
            f"   {coef:>20}{r['mse']:>10.4f}{r['balance']:>14.2f}{r['max_over_uniform']:>28.2f}{r['dead']:>14}"
        )
    print(
        "   (balance loss: 1.0 = tokens spread perfectly evenly, 8.0 = every token goes to one expert)"
    )

    print("\n4. FLASHATTENTION'S ALGORITHM in plain PyTorch (8 heads, T = 2048, block 64)")
    r = flash_report()
    print(
        f"   largest difference from standard attention: {r['gap']:.1e}; with scores around 10^4: {r['gap_extreme']:.1e} (finite: {r['finite']})"
    )
    print(
        f"   largest score tensor per head: {r['full_elems']:,} elements (standard) against {r['tiled_elems']:,} (tiled): {r['full_elems'] / r['tiled_elems']:.0f}x smaller"
    )
    print(
        f"   CPU time: standard {r['ms_standard']:.0f} ms, tiled {r['ms_tiled']:.0f} ms (the real kernel's speed-up comes from keeping tiles in on-chip GPU memory; not measurable here)"
    )
    _ = nn  # keep the import used by type-checkers in some environments


if __name__ == "__main__":
    main(sys.argv)
