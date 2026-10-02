"""A modern decoder-only transformer from scratch: RMSNorm, rotary position embeddings, grouped-query attention, SwiGLU, pre-norm residual blocks.

This is the recipe of Llama, Mistral and Qwen. ``Decoder(Config(...))`` is verified against Hugging Face's ``Qwen2ForCausalLM`` (random weights copied
across, and the real Qwen2.5-0.5B weights): the same logits to float32 precision, so it is not "like" a real model, it IS one.

    cfg = Config(vocab_size=1000, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128)
    model = Decoder(cfg)
    logits = model(ids)                      # (B, T) -> (B, T, vocab)

Shapes in the comments: B batch, T time, D model width, H query heads, G key/value heads, d head width (D = H * d).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from attention import scaled_dot_product_attention
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int | None = (
        None  # None = same as n_heads (plain multi-head attention); fewer = grouped-query attention
    )
    d_ff: int = 0  # 0 = the usual 8/3 * d_model rounded up to a multiple of 8 (SwiGLU keeps the parameter count of a 4x GELU MLP)
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    qkv_bias: bool = False  # Qwen uses a bias on the q, k, v projections; Llama does not
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("the head width must be even (rotary embeddings rotate pairs)")
        if not self.d_ff:
            self.d_ff = 8 * ((int(8 * self.d_model / 3) + 7) // 8)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# ----------------------------------------------------------------------------- normalisation


class RMSNorm(nn.Module):
    """x / rms(x) * weight, with rms(x) = sqrt(mean(x^2) + eps). LayerNorm without the mean subtraction and the bias: cheaper, and as good in practice."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(dim)), eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()  # the statistics in float32 even when the model runs in half precision
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class LayerNorm(nn.Module):
    """(x - mean) / sqrt(var + eps) * weight + bias, over the last dimension (the original Transformer / GPT-2 normalisation)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight, self.bias, self.eps = (
            nn.Parameter(torch.ones(dim)),
            nn.Parameter(torch.zeros(dim)),
            eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)  # biased variance, as in nn.LayerNorm
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


# ----------------------------------------------------------------------------- rotary position embeddings


def rope_tables(
    head_dim: int, positions: torch.Tensor, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin for every position: shape (..., T, head_dim). Pair i of a head rotates by position * theta^(-2i/d) radians."""
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device) / head_dim)
    )  # (d/2,)
    angles = positions.float()[..., None] * inv_freq  # (..., T, d/2)
    emb = torch.cat(
        [angles, angles], dim=-1
    )  # (..., T, d): the same angle for both members of a pair
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) -> (-x2, x1) where x1, x2 are the two halves of the last dimension: a 90-degree turn of every (x1[i], x2[i]) pair."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each (x[i], x[i + d/2]) pair of a query or key by its position's angle. x: (B, H, T, d); cos, sin: (T, d) or (B, T, d)."""
    if cos.dim() == 3:  # (B, T, d) -> (B, 1, T, d) to broadcast over heads
        cos, sin = cos[:, None], sin[:, None]
    return x * cos + rotate_half(x) * sin


# ----------------------------------------------------------------------------- the blocks


class SwiGLU(nn.Module):
    """down( silu(gate(x)) * up(x) ): a gated MLP. The gate decides, per hidden unit, how much of the 'up' signal to let through."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GQAttention(nn.Module):
    """Causal self-attention with H query heads and G <= H key/value heads (each K/V head is shared by H/G query heads). G = H is multi-head
    attention, G = 1 is multi-query attention. The saving is in the KV cache: it stores G heads, not H (Day 6)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, h, g = cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
        self.q_proj = nn.Linear(cfg.d_model, h * d, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(cfg.d_model, g * d, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(cfg.d_model, g * d, bias=cfg.qkv_bias)
        self.o_proj = nn.Linear(h * d, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h, g, d = self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim
        q = self.q_proj(x).view(b, t, h, d).transpose(1, 2)  # (B, H, T, d)
        k = self.k_proj(x).view(b, t, g, d).transpose(1, 2)  # (B, G, T, d)
        v = self.v_proj(x).view(b, t, g, d).transpose(1, 2)  # (B, G, T, d)
        q, k = (
            apply_rope(q, cos, sin),
            apply_rope(k, cos, sin),
        )  # positions enter HERE, on queries and keys only (not values)
        if (
            g != h
        ):  # each key/value head serves a group of H/G query heads: [kv0, kv0, kv1, kv1, ...]
            k, v = k.repeat_interleave(h // g, dim=1), v.repeat_interleave(h // g, dim=1)
        out, _ = scaled_dot_product_attention(q, k, v, causal=True)  # (B, H, T, d)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, h * d))


class Block(nn.Module):
    """x + attention(norm(x)), then x + mlp(norm(x)): the pre-norm residual block. The 'residual stream' x is only ever ADDED to."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.self_attn = GQAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Decoder(nn.Module):
    """Token embeddings -> N blocks -> final norm -> output projection (tied to the embedding table by default)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def hidden_states(
        self, ids: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Everything up to (and including) the final norm: (B, T) -> (B, T, D). ``forward`` is ``lm_head(hidden_states(...))``; fine-tuning uses the
        split to apply the (vocabulary-sized) output projection only at the positions that have a label."""
        b, t = ids.shape
        if positions is None:
            positions = torch.arange(t, device=ids.device)
        cos, sin = rope_tables(self.cfg.head_dim, positions, self.cfg.rope_theta)
        x = self.embed_tokens(ids)  # (B, T, D)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)

    def forward(self, ids: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        return self.lm_head(self.hidden_states(ids, positions))  # (B, T, vocab)

    def num_parameters(self) -> int:
        return sum(
            p.numel() for p in self.parameters()
        )  # parameters() de-duplicates the tied table


# ----------------------------------------------------------------------------- loading Hugging Face weights


def config_from_hf(hf_config) -> Config:
    return Config(
        vocab_size=hf_config.vocab_size,
        d_model=hf_config.hidden_size,
        n_layers=hf_config.num_hidden_layers,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        d_ff=hf_config.intermediate_size,
        max_seq_len=hf_config.max_position_embeddings,
        rope_theta=_rope_theta(hf_config),
        rms_eps=hf_config.rms_norm_eps,
        qkv_bias=hf_config.model_type == "qwen2"
        or bool(
            getattr(hf_config, "attention_bias", False)
        ),  # Qwen2 always has q/k/v biases; Llama only if attention_bias
        tie_embeddings=bool(hf_config.tie_word_embeddings),
    )


def _rope_theta(hf_config) -> float:
    if hasattr(hf_config, "rope_theta") and hf_config.rope_theta is not None:
        return float(hf_config.rope_theta)
    return float(hf_config.rope_parameters["rope_theta"])


def load_hf_state_dict(model: Decoder, state: dict[str, torch.Tensor]) -> list[str]:
    """Copy a Hugging Face Qwen2/Llama-style state dict into ``model`` (the names are identical apart from the leading ``model.``).
    Returns the names that were copied; raises if anything in the model is left untouched or any shape differs."""
    own = model.state_dict()
    copied: list[str] = []
    with torch.no_grad():
        for name, tensor in state.items():
            key = name.removeprefix("model.")
            if key in own:
                if own[key].shape != tensor.shape:
                    raise ValueError(
                        f"{name}: shape {tuple(tensor.shape)} but the model expects {tuple(own[key].shape)}"
                    )
                own[key].copy_(tensor)
                copied.append(key)
    missing = set(own) - set(copied)
    if model.cfg.tie_embeddings:
        missing.discard("lm_head.weight")  # tied to the embedding table
    if missing:
        raise ValueError(f"weights not found in the checkpoint: {sorted(missing)[:5]}")
    return copied


def count_parameters(cfg: Config) -> int:
    """The closed-form parameter count of ``Decoder(cfg)`` (counting a tied output table once)."""
    d, h, g, hd = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    attn = d * h * hd + 2 * d * g * hd + h * hd * d + (h * hd + 2 * g * hd if cfg.qkv_bias else 0)
    mlp = 3 * d * cfg.d_ff
    per_layer = attn + mlp + 2 * d
    total = cfg.vocab_size * d + cfg.n_layers * per_layer + d
    return total + (0 if cfg.tie_embeddings else cfg.vocab_size * d)
