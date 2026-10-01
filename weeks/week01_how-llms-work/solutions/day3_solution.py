"""Week 1 Day 3 - Solution: sampling from scratch + diversity vs. temperature.

Part 1  Temperature / top-k / top-p implemented in numpy, shown on a toy distribution.
Part 2  The same functions applied to the *real* next-token logits of a local model.
Part 3  Challenge: sample N completions per temperature, measure diversity, plot it.

Run:  uv run python weeks/week01_how-llms-work/solutions/day3_solution.py
Output: a table in the terminal and outputs/week1_day3_diversity.png
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = Path(__file__).resolve().parents[3] / "outputs"


# ----------------------------------------------------------------- Part 1: the math


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """softmax(logits / T). T<1 sharpens, T>1 flattens, T->0 approaches argmax."""
    if temperature <= 0:  # greedy: all the mass on the best token
        p = np.zeros_like(logits, dtype=np.float64)
        p[np.argmax(logits)] = 1.0
        return p
    z = logits.astype(np.float64) / temperature
    z -= z.max()  # subtract max for numerical stability (doesn't change the result)
    e = np.exp(z)
    return e / e.sum()


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Keep only the k most likely tokens, renormalise."""
    if k <= 0 or k >= len(probs):
        return probs
    cutoff = np.sort(probs)[-k]
    out = np.where(probs >= cutoff, probs, 0.0)
    return out / out.sum()


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling: keep the smallest set of tokens whose cumulative mass reaches p."""
    if p >= 1.0:
        return probs
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])
    # keep tokens up to and including the one that crosses p
    keep = order[: int(np.searchsorted(cum, p) + 1)]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()


def sample(logits, temperature=1.0, top_k=0, top_p=1.0, rng=None) -> int:
    """Full pipeline in the order APIs apply it: temperature -> top-k -> top-p -> draw."""
    rng = rng or np.random.default_rng()
    probs = softmax(np.asarray(logits), temperature)
    probs = top_p_filter(top_k_filter(probs, top_k), top_p)
    return int(rng.choice(len(probs), p=probs))


def entropy_bits(probs: np.ndarray) -> float:
    nz = probs[probs > 0]
    return max(0.0, float(-(nz * np.log2(nz)).sum()))


def part1_toy() -> None:
    print("=" * 78 + "\nPART 1 - the math on a toy distribution\n" + "=" * 78)
    words = ["The", "A", "My", "Our", "Unicorn"]
    logits = np.array([3.0, 2.4, 1.2, 0.8, -4.0])
    print(f"{'T':>5} | " + " ".join(f"{w:>8}" for w in words) + " | entropy(bits)")
    for t in (0.2, 0.7, 1.0, 1.5, 3.0):
        p = softmax(logits, t)
        print(f"{t:>5} | " + " ".join(f"{x:>8.3%}" for x in p) + f" | {entropy_bits(p):.2f}")
    p = softmax(logits, 1.0)
    print(
        "\nAt T=1.0, top_p=0.9 keeps:",
        [w for w, x in zip(words, top_p_filter(p, 0.9), strict=True) if x > 0],
    )
    print(
        "At T=1.0, top_k=2   keeps:",
        [w for w, x in zip(words, top_k_filter(p, 2), strict=True) if x > 0],
    )
    # Sanity check: sampling frequencies match the probabilities
    rng = np.random.default_rng(0)
    draws = Counter(sample(logits, 1.0, rng=rng) for _ in range(20000))
    print(
        "Empirical freq at T=1 :", {words[i]: round(c / 20000, 3) for i, c in sorted(draws.items())}
    )
    print("Theoretical           :", {w: round(float(x), 3) for w, x in zip(words, p, strict=True)})


# ----------------------------------------------------------------- Part 2: real logits


@torch.no_grad()
def next_token_logits(model, tok, prompt: str) -> np.ndarray:
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    return model(input_ids=ids).logits[0, -1].float().numpy()


def part2_real(model, tok) -> None:
    print("\n" + "=" * 78 + f"\nPART 2 - real next-token distribution ({LOCAL_MODEL})\n" + "=" * 78)
    prompt = "The quick brown fox jumps over the"
    logits = next_token_logits(model, tok, prompt)
    for t in (0.1, 1.0, 1.5):
        p = softmax(logits, t)
        top = np.argsort(p)[::-1][:5]
        shown = ", ".join(f"{tok.decode([int(i)])!r} {p[i]:.1%}" for i in top)
        print(f"T={t:<4} top5: {shown}   | entropy {entropy_bits(p):.2f} bits")
    p = softmax(logits, 1.5)  # a hot distribution has a long tail worth cutting
    for pp in (0.99, 0.9, 0.6):
        kept = int((top_p_filter(p, pp) > 0).sum())
        print(f"T=1.5, top_p={pp}: keeps {kept:,} of {len(p):,} vocabulary tokens")
    print("Note how a high T makes 'lazy' lose mass to alternatives, and top_p cuts the long tail.")


# ----------------------------------------------------------------- Part 3: challenge


def diversity_experiment(model, tok, n_samples: int = 20, max_new: int = 10):
    """For each temperature, sample n completions and measure how different they are."""
    prompt_msgs = [
        {
            "role": "user",
            "content": "Invent a name for a new coffee shop. Reply with the name only.",
        }
    ]
    ids = tok.apply_chat_template(
        prompt_msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    n_prompt = ids["input_ids"].shape[1]
    results = []
    for t in (0.0, 0.3, 0.7, 1.0, 1.3, 1.7):
        torch.manual_seed(0)
        gen_kwargs = {"do_sample": t > 0, "max_new_tokens": max_new, "top_k": 0, "top_p": 1.0}
        if t > 0:
            gen_kwargs.update(temperature=t, num_return_sequences=n_samples)
        out = model.generate(**ids, **gen_kwargs, pad_token_id=tok.eos_token_id)
        texts = [tok.decode(o[n_prompt:], skip_special_tokens=True).strip() for o in out]
        texts = texts * (n_samples // len(texts))  # greedy returns 1 sequence; repeat it
        counts = Counter(texts)
        probs = np.array(list(counts.values())) / len(texts)
        results.append(
            {
                "temperature": t,
                "distinct": len(counts),
                "distinct_ratio": len(counts) / len(texts),
                "entropy_bits": entropy_bits(probs),
                "examples": list(counts)[:4],
            }
        )
    return results


def plot(results) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    ts = [r["temperature"] for r in results]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ts, [r["distinct_ratio"] for r in results], "o-", color="tab:blue")
    ax1.set_xlabel("temperature")
    ax1.set_ylabel("distinct outputs / samples", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(ts, [r["entropy_bits"] for r in results], "s--", color="tab:orange")
    ax2.set_ylabel("entropy of outputs (bits)", color="tab:orange")
    ax1.set_title("Output diversity vs. temperature (20 samples each)")
    fig.tight_layout()
    path = OUT_DIR / "week1_day3_diversity.png"
    fig.savefig(path, dpi=120)
    return path


def part3_challenge(model, tok) -> None:
    print("\n" + "=" * 78 + "\nPART 3 - diversity vs. temperature (20 samples each)\n" + "=" * 78)
    results = diversity_experiment(model, tok)
    print(f"{'T':>4} {'distinct':>9} {'ratio':>6} {'entropy':>8}  examples")
    for r in results:
        print(
            f"{r['temperature']:>4} {r['distinct']:>9} {r['distinct_ratio']:>6.2f} "
            f"{r['entropy_bits']:>8.2f}  {r['examples']}"
        )
    print(f"\nPlot saved to {plot(results)}")
    assert results[0]["distinct"] == 1, "T=0 (greedy) must always return the same output"
    assert results[-1]["distinct"] > results[1]["distinct"], "diversity should grow with T"


if __name__ == "__main__":
    part1_toy()
    tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, dtype=torch.float32).eval()
    part2_real(model, tok)
    part3_challenge(model, tok)
    # math.isclose-style self-check that our softmax is a distribution
    assert math.isclose(softmax(np.array([1.0, 2.0, 3.0]), 0.7).sum(), 1.0)
