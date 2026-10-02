"""A KV cache for the Day 4 decoder, and the cached forward pass.

Generation produces one token at a time. Without a cache, every step re-runs the whole prefix through every layer: the cost of step n is proportional
to n, and a sequence of N tokens costs O(N^2). With a cache, the keys and values of past tokens are stored once per layer; a step runs ONE token
through the network and attends over the stored keys: O(N) per step in attention, O(1) in everything else.

    cache = KVCache.for_model(model, batch=1, max_len=512)
    logits = forward_cached(model, prompt_ids, cache)              # PREFILL: the whole prompt in one pass; fills the cache
    logits = forward_cached(model, next_token[:, None], cache)     # DECODE: one token; attends over the cache

The model's own modules are used unchanged (weights, norms, MLPs); only the attention is re-written to read and write the cache, and it is verified
against the uncached forward pass, token by token.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

SOLUTIONS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOLUTIONS))

import blocks as B  # noqa: E402
from attention import scaled_dot_product_attention  # noqa: E402


@dataclass
class LayerCache:
    """Keys and values of one layer: preallocated (B, G, max_len, d) tensors written in place, with ``length`` valid positions."""

    k: torch.Tensor
    v: torch.Tensor
    length: int = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = k.shape[2]
        if self.length + t > self.k.shape[2]:
            raise ValueError(
                f"the cache holds {self.k.shape[2]} positions; {self.length} used and {t} more would not fit"
            )
        self.k[:, :, self.length : self.length + t] = k
        self.v[:, :, self.length : self.length + t] = v
        self.length += t
        return self.k[:, :, : self.length], self.v[:, :, : self.length]  # views, no copy


class ConcatLayerCache:
    """The naive alternative: grow the tensors with torch.cat on every step (copies the whole cache each time). Kept to measure what preallocation saves."""

    def __init__(self, k: torch.Tensor, v: torch.Tensor):
        self.k, self.v, self.length = k[:, :, :0], v[:, :, :0], 0

    def append(self, k: torch.Tensor, v: torch.Tensor):
        self.k, self.v = torch.cat([self.k, k], dim=2), torch.cat([self.v, v], dim=2)
        self.length = self.k.shape[2]
        return self.k, self.v


class KVCache:
    def __init__(self, layers: list):
        self.layers = layers

    @classmethod
    def for_model(
        cls,
        model: B.Decoder,
        batch: int = 1,
        max_len: int | None = None,
        *,
        preallocate: bool = True,
        dtype=None,
    ) -> KVCache:
        cfg = model.cfg
        max_len = max_len or cfg.max_seq_len
        dtype = dtype or next(model.parameters()).dtype
        shape = (batch, cfg.n_kv_heads, max_len, cfg.head_dim)
        device = next(model.parameters()).device
        make = (
            (lambda k, v: LayerCache(k, v))
            if preallocate
            else (lambda k, v: ConcatLayerCache(k, v))
        )
        return cls(
            [
                make(
                    torch.zeros(shape, dtype=dtype, device=device),
                    torch.zeros(shape, dtype=dtype, device=device),
                )
                for _ in range(cfg.n_layers)
            ]
        )

    @property
    def length(self) -> int:
        return self.layers[0].length

    def nbytes(self) -> int:
        """Bytes of the tensors holding VALID positions (not the unused tail of a preallocated buffer)."""
        return sum(
            2 * layer.k[:, :, : layer.length].numel() * layer.k.element_size()
            for layer in self.layers
        )

    def reserved_bytes(self) -> int:
        return sum(2 * layer.k.numel() * layer.k.element_size() for layer in self.layers)

    def reset(self) -> None:
        for layer in self.layers:
            layer.length = 0


def cached_attention(
    attn: B.GQAttention,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer: LayerCache,
    *,
    grouped: bool = True,
) -> torch.Tensor:
    """GQAttention.forward with a cache: project the NEW tokens, rotate them by their absolute positions, append their keys and values, attend over
    everything cached. ``grouped`` computes a single-token step without materialising the repeated key/value heads (see below)."""
    b, t, _ = x.shape
    h, g, d = attn.cfg.n_heads, attn.cfg.n_kv_heads, attn.cfg.head_dim
    q = attn.q_proj(x).view(b, t, h, d).transpose(1, 2)  # (B, H, T, d)
    k = attn.k_proj(x).view(b, t, g, d).transpose(1, 2)  # (B, G, T, d)
    v = attn.v_proj(x).view(b, t, g, d).transpose(1, 2)
    q, k = B.apply_rope(q, cos, sin), B.apply_rope(k, cos, sin)
    k_all, v_all = layer.append(k, v)  # (B, G, L, d): everything so far, including these tokens
    if t == 1 and grouped and g != h:
        # the H/G query heads of a group read the SAME keys: treat them as a block of queries against one kv head, instead of copying the kv heads H/G times
        rep = h // g
        qg = q.view(b, g, rep, d)  # head index = group * rep + j
        out, _ = scaled_dot_product_attention(
            qg, k_all, v_all
        )  # (B, G, rep, d); one query position sees all L cached keys, so no mask
        out = out.reshape(b, h, 1, d)
    else:
        if g != h:
            k_all, v_all = (
                k_all.repeat_interleave(h // g, dim=1),
                v_all.repeat_interleave(h // g, dim=1),
            )
        out, _ = scaled_dot_product_attention(
            q, k_all, v_all, causal=True
        )  # the new queries sit at the END of the cached keys
    return attn.o_proj(out.transpose(1, 2).reshape(b, t, h * d))


def forward_cached(
    model: B.Decoder, ids: torch.Tensor, cache: KVCache, *, grouped: bool = True
) -> torch.Tensor:
    """Run ``ids`` (B, T) through the model as the NEXT T positions of the sequence the cache holds. Returns logits (B, T, vocab)."""
    start = cache.length
    if start + ids.shape[1] > model.cfg.max_seq_len:
        raise ValueError("the sequence would exceed the model's maximum length")
    positions = torch.arange(start, start + ids.shape[1], device=ids.device)
    cos, sin = B.rope_tables(model.cfg.head_dim, positions, model.cfg.rope_theta)
    x = model.embed_tokens(ids)
    for block, layer in zip(model.layers, cache.layers, strict=True):
        x = x + cached_attention(
            block.self_attn, block.input_layernorm(x), cos, sin, layer, grouped=grouped
        )
        x = x + block.mlp(block.post_attention_layernorm(x))
    return model.lm_head(model.norm(x))
