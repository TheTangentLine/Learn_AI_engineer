"""Week 9 Day 3 - Solution: attention, tested and then watched learning.

1. SCALE      why the 1/sqrt(d): without it, softmax saturates as d grows (entropy and max weight measured)
2. ORDER      attention is blind to order: shuffle the tokens (no positions) and the outputs just shuffle; add the causal mask and that breaks
3. LEARN      a one-layer causal attention model learns "copy the token from 3 positions back"; the attention matrix shows HOW. Without positional
              embeddings it cannot (it has no way to know which token is 3 back)
4. COST       the T x T score matrix: time and memory against sequence length

  uv run python weeks/week09_transformers-from-scratch/solutions/day3_solution.py
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))

import attention as A  # noqa: E402

# ----------------------------------------------------------------------------- 1. the scale


def attention_entropy(
    d: int, *, scaled: bool, t: int = 32, trials: int = 20, seed: int = 0
) -> tuple[float, float]:
    """Mean entropy (in nats; the maximum is ln T) and mean largest weight of attention rows for random unit-variance queries and keys."""
    g = torch.Generator().manual_seed(seed)
    ents, tops = [], []
    for _ in range(trials):
        q, k = torch.randn(t, d, generator=g), torch.randn(t, d, generator=g)
        scores = q @ k.T / (math.sqrt(d) if scaled else 1.0)
        w = A.softmax(scores)
        ents.append(-(w * torch.log(w.clamp_min(1e-30))).sum(-1).mean().item())
        tops.append(w.max(-1).values.mean().item())
    return statistics.fmean(ents), statistics.fmean(tops)


# ----------------------------------------------------------------------------- 2. order


def permutation_gap(causal: bool, seed: int = 0) -> float:
    """Largest difference between 'attend then shuffle the outputs' and 'shuffle the inputs then attend'. ~0 means the layer cannot see order."""
    torch.manual_seed(seed)
    mha = A.MultiHeadAttention(32, 4, causal=causal)
    x = torch.randn(1, 8, 32)
    perm = torch.randperm(8)
    with torch.no_grad():
        return (mha(x)[:, perm] - mha(x[:, perm])).abs().max().item()


# ----------------------------------------------------------------------------- 3. learning to copy from 3 positions back


def copy_task(batch: int, t: int, vocab: int, offset: int, generator: torch.Generator):
    """Random token sequences; the target at position i is the INPUT token at position i - offset (ignored for i < offset)."""
    x = torch.randint(0, vocab, (batch, t), generator=generator)
    y = torch.full_like(x, -100)  # -100 is ignored by cross_entropy
    y[:, offset:] = x[:, :-offset]
    return x, y


class AttnLM(nn.Module):
    def __init__(
        self, vocab: int, d_model: int, max_len: int, *, positions: bool = True, n_heads: int = 1
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model) if positions else None
        self.attn = A.MultiHeadAttention(d_model, n_heads, causal=True)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, ids: torch.Tensor, return_weights: bool = False):
        x = self.tok(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1]))
        y, w = self.attn(x, return_weights=True)
        logits = self.head(y)
        return (logits, w) if return_weights else logits


def train_copy(
    *,
    positions: bool,
    steps: int = 400,
    offset: int = 3,
    t: int = 16,
    vocab: int = 12,
    seed: int = 0,
    lr: float = 3e-3,
) -> dict:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    model = AttnLM(vocab, 32, t, positions=positions)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        x, y = copy_task(64, t, vocab, offset, g)
        loss = nn.functional.cross_entropy(
            model(x).reshape(-1, vocab), y.reshape(-1), ignore_index=-100
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    gx = torch.Generator().manual_seed(999)
    x, y = copy_task(512, t, vocab, offset, gx)
    with torch.no_grad():
        logits, w = model(x, return_weights=True)
    pred = logits.argmax(-1)
    keep = y != -100
    return {
        "model": model,
        "losses": losses,
        "accuracy": (pred[keep] == y[keep]).float().mean().item(),
        "weights": w.mean(dim=(0, 1)),  # averaged over examples and heads: (T, T)
        "chance_loss": math.log(vocab),
        "offset": offset,
    }


def heatmap(w: torch.Tensor) -> str:
    """An attention matrix as text: rows are the query position, columns the key position; darker = more weight."""
    shades = " ░▒▓█"
    lines = ["      key: " + "".join(f"{j % 10}" for j in range(w.shape[1]))]
    for i in range(w.shape[0]):
        lines.append(
            f"query {i:>2}:  "
            + "".join(
                shades[min(len(shades) - 1, int(float(v) * (len(shades) - 1) + 0.5))] for v in w[i]
            )
        )
    return "\n".join(lines)


def attended_offsets(w: torch.Tensor, start: int) -> list[int]:
    """For each query position >= start, how far back its largest weight sits (0 = itself)."""
    return [int(i - w[i, : i + 1].argmax()) for i in range(start, w.shape[0])]


# ----------------------------------------------------------------------------- 4. cost


def attention_cost(
    lengths: list[int], *, heads: int = 8, d_head: int = 64, repeats: int = 3
) -> list[dict]:
    torch.manual_seed(0)
    rows = []
    for t in lengths:
        q, k, v = (torch.randn(1, heads, t, d_head) for _ in range(3))
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            with torch.no_grad():
                A.scaled_dot_product_attention(q, k, v, causal=True)
            times.append(time.perf_counter() - t0)
        rows.append(
            {"T": t, "ms": 1000 * statistics.median(times), "score_mb": heads * t * t * 4 / 1e6}
        )
    return rows


def main(argv: list[str]) -> None:
    print(
        "1. WHY 1/sqrt(d): attention rows over 32 positions; the maximum possible entropy is ln 32 = 3.47 nats"
    )
    print(
        f"   {'d':>5}{'entropy (scaled)':>18}{'max weight':>12}{'entropy (unscaled)':>21}{'max weight':>12}"
    )
    for d in (4, 16, 64, 256, 1024):
        (e1, m1), (e0, m0) = attention_entropy(d, scaled=True), attention_entropy(d, scaled=False)
        print(f"   {d:>5}{e1:>18.2f}{m1:>12.2f}{e0:>21.2f}{m0:>12.2f}")

    print("\n2. ORDER: max |attend-then-shuffle  -  shuffle-then-attend|")
    print(
        f"   no mask (every token sees every token): {permutation_gap(False):.1e}   (the layer cannot tell the order)"
    )
    print(
        f"   causal mask                           : {permutation_gap(True):.1e}   (the mask itself encodes order)"
    )

    print(
        "\n3. LEARNING 'copy the token from 3 positions back' (T=16, vocabulary 12, chance loss ln 12 = 2.48)"
    )
    for positions in (True, False):
        r = train_copy(positions=positions)
        tail = sum(r["losses"][-20:]) / 20
        print(
            f"   {'with' if positions else 'without'} positional embeddings: final loss {tail:.3f}, accuracy {r['accuracy']:.1%}"
        )
        if positions:
            print(
                f"   offsets the attention looks back (query positions 3..15): {attended_offsets(r['weights'], 3)}"
            )
            print(heatmap(r["weights"]))

    print("\n4. COST of the T x T score matrix (8 heads, 64 dims each, causal, CPU, one sequence)")
    print(f"   {'T':>6}{'ms':>9}{'score matrix (MB)':>20}")
    rows = attention_cost([128, 256, 512, 1024, 2048])
    for r in rows:
        print(f"   {r['T']:>6}{r['ms']:>9.1f}{r['score_mb']:>20.1f}")
    print(
        "   time ratio per doubling of T: "
        + ", ".join(f"{b['ms'] / a['ms']:.1f}x" for a, b in zip(rows, rows[1:], strict=False))
    )


if __name__ == "__main__":
    main(sys.argv)
