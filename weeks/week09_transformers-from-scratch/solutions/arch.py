"""Modern architecture, in numbers and in code.

ModelSpec / count_params / kv_cache_bytes     read a Hugging Face config, count parameters and cache memory in closed form (checked against the library)
MoEMLP                                        a sparse mixture-of-experts feed-forward layer (top-k routing, load-balancing loss)
tiled_attention                               FlashAttention's algorithm (online softmax over blocks) in plain PyTorch: exact, never builds the T x T matrix
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

# ----------------------------------------------------------------------------- reading a config


@dataclass(frozen=True)
class ModelSpec:
    """The fields of a decoder-only config that decide its size. ``n_experts == 0`` means a dense feed-forward layer."""

    name: str
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int  # the feed-forward width (per expert for an MoE)
    head_dim: int = 0  # 0 = d_model // n_heads
    tie_embeddings: bool = False
    qkv_bias: bool = False
    n_experts: int = 0
    experts_per_token: int = 0
    mlp_matrices: int = 3  # 3 for a gated (SwiGLU) MLP, 2 for the classic up/down MLP

    @property
    def hd(self) -> int:
        return self.head_dim or self.d_model // self.n_heads


def spec_from_hf(config, name: str = "") -> ModelSpec:
    """Build a ModelSpec from a ``transformers`` config object (Llama, Mistral, Qwen2, Mixtral, ...): the same field names everywhere."""
    experts = getattr(config, "num_local_experts", 0) or getattr(config, "num_experts", 0) or 0
    return ModelSpec(
        name=name or getattr(config, "model_type", "model"),
        vocab_size=config.vocab_size,
        d_model=config.hidden_size,
        n_layers=config.num_hidden_layers,
        n_heads=config.num_attention_heads,
        n_kv_heads=getattr(config, "num_key_value_heads", None) or config.num_attention_heads,
        d_ff=config.intermediate_size,
        head_dim=getattr(config, "head_dim", None) or 0,
        tie_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        qkv_bias=config.model_type in ("qwen2",) or bool(getattr(config, "attention_bias", False)),
        n_experts=experts,
        experts_per_token=getattr(config, "num_experts_per_tok", 0) if experts else 0,
    )


# ----------------------------------------------------------------------------- counting


def count_params(s: ModelSpec) -> dict[str, int]:
    """Parameters by component, with 'total' and 'active' (those used for ONE token: all of a dense model, only the routed experts of an MoE)."""
    d, hd = s.d_model, s.hd
    attn = d * s.n_heads * hd + 2 * d * s.n_kv_heads * hd + s.n_heads * hd * d
    attn_bias = s.n_heads * hd + 2 * s.n_kv_heads * hd if s.qkv_bias else 0
    expert = s.mlp_matrices * d * s.d_ff
    router = d * s.n_experts if s.n_experts else 0
    mlp_total = expert * max(1, s.n_experts) + router
    mlp_active = expert * (s.experts_per_token if s.n_experts else 1) + router
    norms = 2 * d
    layer_total, layer_active = (
        attn + attn_bias + mlp_total + norms,
        attn + attn_bias + mlp_active + norms,
    )
    embed = s.vocab_size * d
    head = 0 if s.tie_embeddings else embed
    return {
        "embedding": embed,
        "attention": s.n_layers * (attn + attn_bias),
        "mlp": s.n_layers * mlp_total,
        "norms": s.n_layers * norms + d,
        "output_head": head,
        "total": embed + head + s.n_layers * layer_total + d,
        "active": embed + head + s.n_layers * layer_active + d,
    }


def kv_cache_bytes(s: ModelSpec, seq_len: int, *, batch: int = 1, dtype_bytes: int = 2) -> int:
    """Keys and values for every layer, kv head and position: 2 * layers * kv_heads * head_dim * seq * batch * bytes."""
    return 2 * s.n_layers * s.n_kv_heads * s.hd * seq_len * batch * dtype_bytes


def weight_bytes(s: ModelSpec, dtype_bytes: int = 2) -> int:
    return count_params(s)["total"] * dtype_bytes


def decode_flops_per_token(s: ModelSpec, context: int) -> int:
    """Multiply-adds x2 for ONE generated token: 2 * active parameters (matrix-vector products) + attention over ``context`` cached positions."""
    return 2 * count_params(s)["active"] + 2 * 2 * s.n_layers * s.n_heads * s.hd * context


def hf_reference_count(config) -> int:
    """The library's own count, on the 'meta' device: the model is built without allocating any memory, so a 47-billion-parameter config is free."""
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------- mixture of experts


class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj, self.up_proj = (
            nn.Linear(d_model, d_ff, bias=False),
            nn.Linear(d_model, d_ff, bias=False),
        )
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEMLP(nn.Module):
    """Sparse feed-forward: a router scores every expert for every token, the top-k experts process the token, and their outputs are mixed with the
    router's (re-normalised) weights. Total parameters grow with the number of experts; the compute per token only with k.

    x: (B, T, D) -> (y (B, T, D), balance_loss scalar). ``balance_loss`` is 1.0 when tokens spread perfectly evenly and n_experts when every token
    goes to one expert; add a small multiple of it to the training loss so the router does not collapse onto a few experts.
    """

    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("need 1 <= top_k <= n_experts")
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d_model, d_ff) for _ in range(n_experts))

    def route(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = F.softmax(self.router(flat), dim=-1, dtype=torch.float32)  # (N, E)
        weights, chosen = torch.topk(probs, self.top_k, dim=-1)  # (N, k)
        return probs, weights / weights.sum(-1, keepdim=True), chosen

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        probs, weights, chosen = self.route(flat)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            token_idx, slot = torch.where(
                chosen == e
            )  # which tokens picked expert e, and in which of their k slots
            if token_idx.numel():
                out.index_add_(
                    0,
                    token_idx,
                    expert(flat[token_idx]) * weights[token_idx, slot, None].to(out.dtype),
                )
        # load-balancing loss: E * sum_e (fraction of tokens routed to e) * (mean router probability of e)
        frac = F.one_hot(chosen, self.n_experts).float().sum(1).mean(0) / self.top_k
        balance = self.n_experts * (frac * probs.mean(0)).sum()
        return out.reshape(b, t, d), balance

    def expert_load(self, x: torch.Tensor) -> torch.Tensor:
        """Share of the routed (token, slot) pairs that each expert receives; uniform = 1/E."""
        _, _, chosen = self.route(x.reshape(-1, x.shape[-1]))
        return F.one_hot(chosen, self.n_experts).float().sum((0, 1)) / chosen.numel()


# ----------------------------------------------------------------------------- FlashAttention's algorithm


def tiled_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, block: int = 64, causal: bool = True
) -> torch.Tensor:
    """Exactly softmax(Q K^T / sqrt(d)) V, computed one block of keys and values at a time with an ONLINE softmax, so the (Tq x Tk) score matrix is
    never held in memory; the largest intermediate is (Tq x block). This is the algorithm of FlashAttention; the real kernel also keeps each tile in
    on-chip memory (the actual speed-up, which this plain-PyTorch version does not have).

    For every query row keep a running maximum m, a running denominator l = sum exp(s - m) and a running output o = sum exp(s - m) v. When a new
    block brings a larger maximum, the old l and o are multiplied by exp(m_old - m_new): the softmax is rescaled, never recomputed.
    q: (..., Tq, d)   k, v: (..., Tk, d)
    """
    tq, tk, d = q.shape[-2], k.shape[-2], q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    m = torch.full(q.shape[:-1] + (1,), float("-inf"), dtype=q.dtype)
    l = torch.zeros(q.shape[:-1] + (1,), dtype=q.dtype)  # noqa: E741
    o = torch.zeros_like(q)
    q_pos = torch.arange(tk - tq, tk)[
        :, None
    ]  # query block at the END of the keys, as in the KV-cache case
    for start in range(0, tk, block):
        kb, vb = k[..., start : start + block, :], v[..., start : start + block, :]
        s = (q @ kb.transpose(-2, -1)) * scale  # (..., Tq, block)
        if causal:
            s = s.masked_fill(
                torch.arange(start, start + kb.shape[-2])[None, :] > q_pos, float("-inf")
            )
        m_new = torch.maximum(m, s.amax(-1, keepdim=True))
        m_safe = torch.where(
            torch.isfinite(m_new), m_new, torch.zeros_like(m_new)
        )  # a row with nothing visible yet
        p = torch.exp(s - m_safe)
        rescale = torch.exp(torch.where(torch.isfinite(m), m, m_safe) - m_safe)
        l = l * rescale + p.sum(-1, keepdim=True)  # noqa: E741
        o = o * rescale + p @ vb
        m = m_new
    return o / l.clamp_min(torch.finfo(q.dtype).tiny)


def largest_score_tensor(tq: int, tk: int, *, block: int | None) -> int:
    """Elements in the biggest attention-score tensor alive at once, per head: the whole matrix, or one block of columns."""
    return tq * tk if block is None else tq * min(block, tk)


__all__ = [
    "Expert",
    "ModelSpec",
    "MoEMLP",
    "count_params",
    "decode_flops_per_token",
    "hf_reference_count",
    "kv_cache_bytes",
    "largest_score_tensor",
    "replace",
    "spec_from_hf",
    "tiled_attention",
    "weight_bytes",
]
