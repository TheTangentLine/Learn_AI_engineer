"""Attention from scratch: scaled dot-product attention, the causal mask, multi-head attention, and positional encodings.

    out, weights = scaled_dot_product_attention(q, k, v, causal=True)      # q, k, v: (..., T, d)
    mha = MultiHeadAttention(d_model=64, n_heads=4)                         # (B, T, 64) -> (B, T, 64)

Everything is written with matrix multiplications, a softmax and a mask. Shapes are in the comments: when attention code is wrong, it is almost
always a dimension in the wrong place.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Row-stable softmax: subtracting the maximum changes nothing mathematically and keeps exp() from overflowing. A row that is entirely
    -inf (every position masked) gives zeros, not NaN: there is nothing to attend to, so the output is zero."""
    m = x.amax(dim=dim, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    e = torch.exp(x - m)
    return e / e.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)


def causal_mask(t: int, device: torch.device | None = None) -> torch.Tensor:
    """(T, T) boolean, True where attention is ALLOWED: position i may look at positions j <= i (the lower triangle)."""
    return torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """softmax(Q K^T / sqrt(d)) V.

    q: (..., Tq, d)   k: (..., Tk, d)   v: (..., Tk, dv)   mask: broadcastable to (..., Tq, Tk), True = may attend
    returns (output (..., Tq, dv), weights (..., Tq, Tk)); every row of the weights sums to 1 (or is all zero if the whole row is masked).
    """
    d = q.shape[-1]
    scores = (
        q @ k.transpose(-2, -1) / math.sqrt(d)
    )  # (..., Tq, Tk): how well each query matches each key
    allowed = None
    if causal:
        allowed = causal_mask(scores.shape[-1], scores.device)[
            -scores.shape[-2] :
        ]  # the last Tq rows: right for a query block at the END
    if mask is not None:
        allowed = mask if allowed is None else (allowed & mask)
    if allowed is not None:
        scores = scores.masked_fill(~allowed, float("-inf"))
    weights = softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    """H heads that each attend with d_model/H dimensions, concatenated and mixed by an output projection.

    x: (B, T, d_model) -> (B, T, d_model). The three projections are one fused Linear (a single matrix multiplication), split afterwards.
    """

    def __init__(self, d_model: int, n_heads: int, *, causal: bool = True, bias: bool = True):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model, self.n_heads, self.d_head, self.causal = (
            d_model,
            n_heads,
            d_model // n_heads,
            causal,
        )
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_head).transpose(
            1, 2
        )  # (B, T, D) -> (B, T, H, d) -> (B, H, T, d)

    def forward(
        self, x: torch.Tensor, *, mask: torch.Tensor | None = None, return_weights: bool = False
    ):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # three (B, T, D)
        o, w = scaled_dot_product_attention(
            self.split_heads(q),
            self.split_heads(k),
            self.split_heads(v),
            mask=mask,
            causal=self.causal,
        )
        merged = o.transpose(1, 2).reshape(
            b, t, self.d_model
        )  # (B, H, T, d) -> (B, T, H, d) -> (B, T, D)
        y = self.out(merged)
        return (y, w) if return_weights else y


def key_padding_mask(lengths: torch.Tensor, t: int) -> torch.Tensor:
    """(B, 1, 1, T) True where the key position is a real token (< length), False for padding: broadcasts against (B, H, Tq, Tk) scores."""
    return (torch.arange(t, device=lengths.device)[None, :] < lengths[:, None])[:, None, None, :]


# ----------------------------------------------------------------------------- positions


def sinusoidal_positions(t: int, d_model: int, base: float = 10000.0) -> torch.Tensor:
    """The original Transformer's fixed positional encoding: pair i of the vector oscillates with wavelength 2*pi*base^(2i/d). Shape (T, d)."""
    if d_model % 2:
        raise ValueError("d_model must be even")
    pos = torch.arange(t, dtype=torch.float64)[:, None]
    freq = base ** (-torch.arange(0, d_model, 2, dtype=torch.float64) / d_model)
    pe = torch.zeros(t, d_model, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe.float()


class TokenAndPositionEmbedding(nn.Module):
    """token id -> vector, plus a learned vector per position. Attention itself is blind to order: without positions, shuffling the tokens
    only shuffles the outputs the same way."""

    def __init__(self, vocab: int, d_model: int, max_len: int):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d_model), nn.Embedding(max_len, d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.tok(ids) + self.pos(
            torch.arange(ids.shape[1], device=ids.device)
        )  # (B, T) -> (B, T, D)
