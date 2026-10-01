"""Week 1 Day 4 - Solution: KV-cache calculator + conversation cost + prompt-caching benchmark.

Part A (no key)   KV-cache memory formula, verified against a real model's cache, plus a
                  measured speed-up of generating with vs. without the cache.
Part B (no key)   Simulate a long chat: how the bill grows when you resend the whole history,
                  and how much prompt caching and summarisation save.
Part C (API key)  Benchmark real prompt caching on Claude: same long document, 4 questions,
                  with and without ``cache_prompt=True``. Skipped without ANTHROPIC_API_KEY.

Run:  uv run python weeks/week01_how-llms-work/solutions/day4_solution.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # make `common` importable

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

LOCAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# ----------------------------------------------------------------- Part A: KV cache


@dataclass
class ModelShape:
    name: str
    layers: int
    kv_heads: int  # grouped-query attention: far fewer KV heads than query heads
    head_dim: int
    params_b: float  # billions of parameters, for the weights-vs-cache comparison


def kv_cache_bytes(shape: ModelShape, seq_len: int, batch: int = 1, bytes_per_elem: int = 2) -> int:
    """2 (K and V) x layers x kv_heads x head_dim x tokens x bytes x batch."""
    return 2 * shape.layers * shape.kv_heads * shape.head_dim * seq_len * bytes_per_elem * batch


# Published configs of well-known open models (see each model's config.json on Hugging Face).
SHAPES = [
    ModelShape("Qwen2.5-0.5B", 24, 2, 64, 0.5),
    ModelShape("Llama-3.1-8B", 32, 8, 128, 8.0),
    ModelShape("Llama-3.1-70B", 80, 8, 128, 70.0),
]


def gib(n: float) -> float:
    return n / 2**30


def part_a(model, tok) -> None:
    print("=" * 84 + "\nPART A - KV cache: memory formula, verification, speed\n" + "=" * 84)

    # 1) Verify the formula against the real cache of a real model.
    cfg = AutoConfig.from_pretrained(LOCAL_MODEL)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    shape = ModelShape("Qwen2.5-0.5B (from config)", cfg.num_hidden_layers,
                       cfg.num_key_value_heads, head_dim, 0.5)
    ids = tok("hello " * 200, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        cache = model(input_ids=ids, use_cache=True).past_key_values
    measured = sum(
        layer.keys.numel() * layer.keys.element_size()
        + layer.values.numel() * layer.values.element_size()
        for layer in cache.layers
    )
    predicted = kv_cache_bytes(shape, ids.shape[1], bytes_per_elem=4)  # model loaded in fp32
    print(f"Measured cache for {ids.shape[1]} tokens (fp32): {measured:,} bytes")
    print(f"Formula prediction                       : {predicted:,} bytes  "
          f"{'MATCH' if measured == predicted else 'MISMATCH'}")
    assert measured == predicted

    # 2) What does it cost at scale? (fp16 cache, 1 sequence vs. a busy server)
    print(f"\n{'model':16} {'weights fp16':>13} | KV cache fp16: {'32k ctx x1':>11} {'128k ctx x1':>12} "
          f"{'32k ctx x32 users':>19}")
    for s in SHAPES:
        weights = s.params_b * 1e9 * 2
        row = [kv_cache_bytes(s, n, b) for n, b in ((32_768, 1), (131_072, 1), (32_768, 32))]
        print(f"{s.name:16} {gib(weights):>10.1f} GiB | {gib(row[0]):>21.2f} GiB "
              f"{gib(row[1]):>9.2f} GiB {gib(row[2]):>16.2f} GiB")
    print("-> Long contexts x many concurrent users make the *cache*, not the weights, the "
          "bottleneck.\n   (This is why GQA, cache quantisation and PagedAttention exist - Weeks 9 and 11.)")

    # 3) Why the cache exists: generation with vs. without it.
    def generate(use_cache: bool, n_new: int = 30) -> float:
        x = ids.clone()
        t0, past, nxt = time.perf_counter(), None, ids
        with torch.no_grad():
            for _ in range(n_new):
                out = model(input_ids=nxt if use_cache else x,
                            past_key_values=past if use_cache else None, use_cache=use_cache)
                token = out.logits[0, -1].argmax().view(1, 1)
                if use_cache:
                    past, nxt = out.past_key_values, token
                x = torch.cat([x, token], dim=1)
        return time.perf_counter() - t0

    with_cache, without = generate(True), generate(False)
    print(f"\nGenerating 30 tokens after a {ids.shape[1]}-token prompt on CPU: "
          f"with cache {with_cache:.1f}s | without {without:.1f}s | speed-up x{without / with_cache:.1f}")
    print("Without the cache every new token re-reads the entire prompt; the gap widens with length.")


# ----------------------------------------------------------------- Part B: the bill


def conversation_cost(turns: int, new_tokens_per_turn: int, reply_tokens: int, *,
                      in_price: float, out_price: float, cache_read_mult: float = 1.0,
                      history_cap: int | None = None) -> float:
    """USD cost of a chat where each turn resends the (possibly capped) history.

    cache_read_mult=0.1 models prompt caching (cached prefix read at 10% of the input price).
    history_cap models a sliding window / summarisation that bounds the resent history.
    """
    total, history = 0.0, 0
    for _ in range(turns):
        resent = history if history_cap is None else min(history, history_cap)
        fresh = new_tokens_per_turn
        total += (resent * cache_read_mult + fresh) * in_price / 1e6
        total += reply_tokens * out_price / 1e6
        history += new_tokens_per_turn + reply_tokens
    return total


def part_b() -> None:
    print("\n" + "=" * 84 + "\nPART B - the chat bill (claude-opus-5 prices: $5 in / $25 out per MTok)\n" + "=" * 84)
    kw = {"new_tokens_per_turn": 300, "reply_tokens": 400, "in_price": 5.0, "out_price": 25.0}
    print(f"{'turns':>6} {'naive':>9} {'cached':>9} {'window 4k':>10} {'cached+window':>14}   (USD)")
    for turns in (5, 20, 50, 100):
        naive = conversation_cost(turns, **kw)
        cached = conversation_cost(turns, cache_read_mult=0.1, **kw)
        window = conversation_cost(turns, history_cap=4000, **kw)
        both = conversation_cost(turns, cache_read_mult=0.1, history_cap=4000, **kw)
        print(f"{turns:>6} {naive:>9.2f} {cached:>9.2f} {window:>10.2f} {both:>14.2f}")
    crossover = next(
        t for t in range(1, 5000)
        if conversation_cost(t, history_cap=4000, **kw)
        < conversation_cost(t, cache_read_mult=0.1, **kw)
    )
    print(f"A 4k window alone only beats caching alone from turn {crossover} "
          "(caching still pays ~10% for the whole history, the window stops paying for it).")
    print("-> Naive cost grows ~quadratically with turns (you re-pay for every old token every turn).")
    print("   Caching cuts the repeated part ~90%; bounding the history stops the growth entirely.")


# ----------------------------------------------------------------- Part C: real prompt caching


def build_document(n_sections: int = 60) -> str:
    """A long, deterministic document (no timestamps/UUIDs - those would break the cache)."""
    sections = []
    for i in range(1, n_sections + 1):
        sections.append(
            f"## Policy {i}: Equipment handling rule {i}\n"
            f"Rule {i}.1: Item class {i % 7} must be inspected every {i % 5 + 2} days by a qualified "
            f"technician. Rule {i}.2: Incidents involving class {i % 7} items are reported to office "
            f"{chr(65 + i % 26)}{i} within {i % 4 + 1} hours. Rule {i}.3: Spare parts code SP-{1000 + i} "
            f"is stocked at depot {i % 9}. Exceptions require written approval from manager M{i % 11}."
        )
    return "\n\n".join(sections)


QUESTIONS = [
    "Within how many hours must incidents with class 3 items be reported (see Policy 10)?",
    "Which depot stocks spare parts code SP-1020?",
    "How often must class 5 items from Policy 12 be inspected?",
    "Who approves exceptions in Policy 22?",
]


def part_c() -> None:
    print("\n" + "=" * 84 + "\nPART C - real prompt caching on Claude\n" + "=" * 84)
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Skipped: set ANTHROPIC_API_KEY in .env to run this part.")
        return
    from common.llm import complete

    system = "Answer using only this handbook. Be brief.\n\n" + build_document()

    def run(cache: bool):
        rows = []
        for q in QUESTIONS:
            r = complete(q, system=system, provider="anthropic", max_tokens=200, cache_prompt=cache)
            rows.append(r)
        return rows

    for label, cache in (("NO caching", False), ("WITH caching", True)):
        rows = run(cache)
        print(f"\n{label}")
        print(f"{'call':>4} {'fresh_in':>9} {'cache_write':>12} {'cache_read':>11} {'latency':>8} {'cost$':>9}")
        for i, r in enumerate(rows, 1):
            u = r.usage
            print(f"{i:>4} {u.input_tokens:>9} {u.cache_write_tokens:>12} {u.cache_read_tokens:>11} "
                  f"{r.latency_s:>7.2f}s {r.cost_usd:>9.5f}")
        print(f"{'sum':>4} {'':>9} {'':>12} {'':>11} {sum(r.latency_s for r in rows):>7.2f}s "
              f"{sum(r.cost_usd for r in rows):>9.5f}")
    print("\nExpect: call 1 pays a ~1.25x cache *write*; calls 2-4 read the prefix at ~0.1x and are "
          "faster.\nIf cache_read stays 0, something in the prefix changes between calls "
          "(timestamps, IDs, reordered JSON)\nor the prefix is below the model's minimum cacheable size.")


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, dtype=torch.float32).eval()
    part_a(model, tok)
    part_b()
    part_c()
