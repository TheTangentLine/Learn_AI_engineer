"""Benchmarks for generation: tokens per second with and without a KV cache, the cost of each step as the context grows, what preallocation and grouped
GQA decoding save, and how sampling settings trade quality against diversity. Everything here is MEASURED on this machine (a CPU, 4 threads, float32);
speed-ups on other hardware will differ, so the shapes and ratios are the lesson, not the absolute numbers."""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

import torch
from generate import generate
from kvcache import KVCache, forward_cached
from sampling import filtered_probs

SOLUTIONS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOLUTIONS))

import blocks as B  # noqa: E402


def bench_model(
    d_model: int = 256,
    n_layers: int = 6,
    n_heads: int = 8,
    n_kv_heads: int = 2,
    vocab: int = 1024,
    max_len: int = 640,
) -> B.Decoder:
    torch.manual_seed(0)
    return B.Decoder(
        B.Config(
            vocab_size=vocab,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_len,
        )
    ).eval()


def compare_modes(
    model: B.Decoder,
    prompt_len: int,
    new_tokens: int,
    modes=("nocache", "concat", "cache"),
    repeats: int = 2,
) -> dict[str, dict]:
    """Decode speed and total time per mode, best of ``repeats``; also whether all modes produced the same tokens (greedy)."""
    prompt = torch.randint(
        0, model.cfg.vocab_size, (prompt_len,), generator=torch.Generator().manual_seed(1)
    ).tolist()
    out, tokens = {}, {}
    for mode in modes:
        best = min(
            (generate(model, prompt, new_tokens, mode=mode) for _ in range(repeats)),
            key=lambda g: g.total_s,
        )
        out[mode] = {
            "tokens_per_second": best.tokens_per_second,
            "total_s": best.total_s,
            "prefill_s": best.prefill_s,
            "ms_per_token": 1000 * statistics.fmean(best.step_s),
        }
        tokens[mode] = best.tokens
    ref = tokens[modes[0]]
    for mode in modes:
        out[mode]["same_tokens"] = tokens[mode] == ref
    return out


def step_latency_by_position(
    model: B.Decoder, prompt_len: int, new_tokens: int, every: int = 32
) -> list[dict]:
    """Milliseconds for ONE decode step at different context lengths, with and without the cache (median of a small window)."""
    prompt = torch.randint(
        0, model.cfg.vocab_size, (prompt_len,), generator=torch.Generator().manual_seed(2)
    ).tolist()
    res = {m: generate(model, prompt, new_tokens, mode=m).step_s for m in ("nocache", "cache")}
    rows = []
    for i in range(0, new_tokens - 1 - 4, every):
        rows.append(
            {
                "context": prompt_len + i + 1,
                "nocache_ms": 1000 * statistics.median(res["nocache"][i : i + 5]),
                "cache_ms": 1000 * statistics.median(res["cache"][i : i + 5]),
            }
        )
    return rows


def grouped_vs_repeated(
    model: B.Decoder, prompt_len: int = 256, steps: int = 40, repeats: int = 3
) -> dict[str, float]:
    """Milliseconds per decode step with the grouped single-token path against materialising the repeated K/V heads."""
    prompt = torch.randint(
        0, model.cfg.vocab_size, (1, prompt_len), generator=torch.Generator().manual_seed(3)
    )
    out = {}
    for grouped in (True, False):
        best = float("inf")
        for _ in range(repeats):
            cache = KVCache.for_model(model, 1, prompt_len + steps + 1)
            with torch.no_grad():
                forward_cached(model, prompt, cache, grouped=grouped)
                tok = prompt[:, -1:]
                import time

                t0 = time.perf_counter()
                for _ in range(steps):
                    forward_cached(model, tok, cache, grouped=grouped)
                best = min(best, (time.perf_counter() - t0) / steps)
        out["grouped" if grouped else "repeated"] = 1000 * best
    return out


def cache_memory(model: B.Decoder, tokens: int) -> dict[str, int]:
    cache = KVCache.for_model(model, 1, tokens + 10)
    with torch.no_grad():
        forward_cached(model, torch.randint(0, model.cfg.vocab_size, (1, tokens)), cache)
    cfg = model.cfg
    formula = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * tokens * 4
    return {
        "valid_bytes": cache.nbytes(),
        "reserved_bytes": cache.reserved_bytes(),
        "formula": formula,
    }


# ----------------------------------------------------------------------------- sampling quality against diversity


@torch.no_grad()
def sample_batch(
    model: B.Decoder,
    prompts: list[list[int]],
    length: int,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    seed: int = 0,
) -> list[dict]:
    """One continuation per prompt, with the model's own log-probability (at temperature 1, unfiltered) of each token it chose."""
    out = []
    for i, p in enumerate(prompts):
        gen = torch.Generator().manual_seed(seed + i)
        cache = KVCache.for_model(model, 1, len(p) + length)
        logits = forward_cached(model, torch.tensor([p]), cache)[:, -1]
        toks, lps = [], []
        for _ in range(length):
            probs = filtered_probs(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            nxt = (
                probs.argmax(-1)
                if temperature == 0
                else torch.multinomial(probs, 1, generator=gen)[:, 0]
            )
            lps.append(torch.log_softmax(logits.float(), -1)[0, nxt[0]].item())
            toks.append(int(nxt))
            logits = forward_cached(model, nxt[:, None], cache)[:, -1]
        out.append({"tokens": toks, "logprobs": lps})
    return out


def sampling_metrics(samples: list[dict]) -> dict[str, float]:
    """mean_nll: the model's surprise at its own choices (low = safe, typical text; high = it wandered into unlikely tokens).
    distinct2: unique bigrams over all bigrams across the samples (low = repetitive). loop: share of samples whose last 20 tokens contain a repeated 4-gram."""
    nll = statistics.fmean(-lp for s in samples for lp in s["logprobs"])
    bigrams = [(a, b) for s in samples for a, b in zip(s["tokens"], s["tokens"][1:], strict=False)]
    loops = 0
    for s in samples:
        tail = s["tokens"][-20:]
        grams = [tuple(tail[i : i + 4]) for i in range(len(tail) - 3)]
        loops += len(grams) != len(set(grams))
    return {
        "mean_nll": nll,
        "distinct2": len(set(bigrams)) / max(1, len(bigrams)),
        "loop_rate": loops / len(samples),
        "perplexity": math.exp(nll),
    }
