"""LoRA (low-rank adaptation) from scratch, for the Week 9 decoder.

Fine-tuning a layer means changing its weight W (out x in). LoRA freezes W and learns the CHANGE as a product of two thin matrices:

    y = x W^T  +  (alpha / r) * (x A^T) B^T         A: (r, in)   B: (out, r)   r << in, out

B starts at zero, so the adapted model begins exactly equal to the base model; only A and B are trained. For a 576 x 576 projection with r = 16 that is
18,432 numbers instead of 331,776. After training the update can be MERGED into W (W + (alpha/r) B A), so inference has no extra cost.

    add_lora(model, r=16, alpha=32)        # freeze everything, wrap the chosen Linear layers
    state = lora_state_dict(model)         # only the adapters (tens of MB at most)
    merge_lora(model)                      # fold them into the weights and unwrap
"""

from __future__ import annotations

import math

import torch
from torch import nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r < 1:
            raise ValueError("the rank must be at least 1")
        self.base, self.r, self.alpha = base, r, alpha
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(
            self.lora_A, a=math.sqrt(5)
        )  # PEFT's default: uniform(-1/sqrt(in), 1/sqrt(in))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def delta_weight(self) -> torch.Tensor:
        """The change LoRA makes to the weight matrix: (alpha/r) * B A, shape (out, in)."""
        return self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        return out + self.scale * ((self.dropout(x) @ self.lora_A.T) @ self.lora_B.T)

    def merged_linear(self) -> nn.Linear:
        lin = nn.Linear(self.in_features, self.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(self.base.weight + self.delta_weight())
            if self.base.bias is not None:
                lin.bias.copy_(self.base.bias)
        return lin


def _parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    *path, leaf = name.split(".")
    mod = model
    for p in path:
        mod = getattr(mod, p)
    return mod, leaf


def add_lora(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    dropout: float = 0.0,
    layers: range | None = None,
) -> list[str]:
    """Freeze every parameter, then wrap each ``nn.Linear`` whose name ends in one of ``targets`` (optionally only in some layer indices). Returns the
    wrapped names. The output projection of the model (the tied embedding table) is never wrapped."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, mod in list(model.named_modules()):
        if (
            not isinstance(mod, nn.Linear)
            or name.split(".")[-1] not in targets
            or name.endswith("lm_head")
        ):
            continue
        if layers is not None and not any(f"layers.{i}." in name for i in layers):
            continue
        parent, leaf = _parent(model, name)
        setattr(parent, leaf, LoRALinear(mod, r, alpha, dropout))
        wrapped.append(name)
    if not wrapped:
        raise ValueError(f"no Linear layer matched {targets}")
    return wrapped


def lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter matrices: a few megabytes, whatever the size of the base model."""
    return {
        f"{n}.{k}": getattr(m, k).detach().clone()
        for n, m in lora_modules(model).items()
        for k in ("lora_A", "lora_B")
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    mods = lora_modules(model)
    expected = {f"{n}.{k}" for n in mods for k in ("lora_A", "lora_B")}
    if set(state) != expected:
        raise ValueError(
            f"adapter mismatch: missing {sorted(expected - set(state))[:3]}, unexpected {sorted(set(state) - expected)[:3]}"
        )
    with torch.no_grad():
        for n, m in mods.items():
            m.lora_A.copy_(state[f"{n}.lora_A"])
            m.lora_B.copy_(state[f"{n}.lora_B"])


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every adapter into its weight and put a plain nn.Linear back: the model is then an ordinary one with the fine-tuning baked in."""
    for name, m in lora_modules(model).items():
        parent, leaf = _parent(model, name)
        setattr(parent, leaf, m.merged_linear())
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "frozen": total - trainable}


def lora_parameter_count(in_features: int, out_features: int, r: int) -> int:
    return r * (in_features + out_features)
