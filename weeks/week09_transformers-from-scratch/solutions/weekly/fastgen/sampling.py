"""Sampling from logits: temperature, top-k, top-p (nucleus), and a function that returns the FILTERED DISTRIBUTION so each rule can be tested exactly,
without sampling noise.

    probs = filtered_probs(logits, temperature=0.8, top_k=50, top_p=0.9)
    token = sample(logits, temperature=0.8, top_p=0.9, generator=g)

Order of operations (the common one): divide logits by the temperature, softmax, keep the top-k, keep the smallest set of tokens whose cumulative
probability reaches top_p, renormalise. Temperature 0 is greedy.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def filtered_probs(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """logits (..., V) -> probabilities (..., V) after the filters; tokens that were removed have probability exactly 0."""
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")
    if temperature == 0:  # greedy: all the probability on the (first) largest logit
        out = torch.zeros_like(logits, dtype=torch.float32)
        out.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        return out
    probs = F.softmax(logits.float() / temperature, dim=-1)
    if top_k is not None and top_k < probs.shape[-1]:
        kth = torch.topk(probs, top_k, dim=-1).values[..., -1:]
        probs = probs.masked_fill(probs < kth, 0.0)  # ties at the k-th value are all kept
    if top_p is not None and top_p < 1.0:
        sorted_p, order = torch.sort(probs, dim=-1, descending=True)
        cumulative = sorted_p.cumsum(-1)
        # keep a token while the mass BEFORE it is still below top_p: the smallest prefix whose total reaches top_p, and always at least one token
        keep_sorted = (cumulative - sorted_p) < top_p
        keep = torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)
        probs = probs.masked_fill(~keep, 0.0)
    return probs / probs.sum(-1, keepdim=True)


def sample(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logits (B, V) -> token ids (B,)."""
    probs = filtered_probs(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    if temperature == 0:
        return probs.argmax(-1)
    return torch.multinomial(probs, 1, generator=generator)[..., 0]
