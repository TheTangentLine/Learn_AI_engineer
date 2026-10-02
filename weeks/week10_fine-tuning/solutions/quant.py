"""Block-wise weight quantisation, from scratch: NF4 (QLoRA's data type), a uniform 4-bit grid, and GGUF's Q8_0 and Q4_0.

All of them work the same way: cut the weights into blocks, store one scale per block (the block's largest magnitude) and, for every weight, the index of
the nearest value in a small table (a "codebook"). Reconstruction multiplies the table value by the block's scale. They differ in the table and the block size:

    NF4    16 values placed at the quantiles of a normal distribution (pretrained weights are roughly normal): fine steps near 0 where most weights are
    int4   16 evenly spaced values
    Q8_0   GGUF: blocks of 32, an int8 per weight, a float16 scale            = 8.5 bits per weight
    Q4_0   GGUF: blocks of 32, 4 bits per weight, a float16 scale             = 4.5 bits per weight

``fake_quantize`` quantises and immediately reconstructs, so the effect of the rounding on a model's outputs can be measured in plain PyTorch.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# the 16 NF4 values published with QLoRA (Dettmers et al., 2023): quantiles of N(0, 1) scaled to [-1, 1], with an exact zero
NF4_PUBLISHED = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


def nf4_codebook(offset: float = 0.9677083) -> torch.Tensor:
    """Rebuild the NF4 table from its definition: 8 positive and 7 negative quantiles of a standard normal, 0 in the middle, scaled so the extremes are +-1."""
    from scipy.stats import norm

    pos = norm.ppf(torch.linspace(offset, 0.5, 9).numpy()[:-1])
    neg = -norm.ppf(torch.linspace(offset, 0.5, 8).numpy()[:-1])
    values = torch.tensor(sorted([*pos.tolist(), 0.0, *neg.tolist()]))
    return values / values.abs().max()


def uniform_codebook(bits: int = 4) -> torch.Tensor:
    return torch.linspace(-1, 1, 2**bits)


def quantize_blockwise(
    w: torch.Tensor, codebook: torch.Tensor, block: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """w (any shape) -> (indices uint8 of shape (n_blocks, block), scales float32 of shape (n_blocks,)). The tensor is flattened and zero-padded to a
    whole number of blocks; the scale of a block is its largest absolute value, so the block's extremes land exactly on +-1 of the codebook."""
    flat = w.detach().float().reshape(-1)
    pad = (-flat.numel()) % block
    blocks = torch.cat([flat, flat.new_zeros(pad)]).view(-1, block)
    scales = blocks.abs().amax(dim=1).clamp_min(1e-12)
    normed = blocks / scales[:, None]
    idx = (normed[..., None] - codebook.to(normed)).abs().argmin(dim=-1)
    return idx.to(torch.uint8), scales


def dequantize_blockwise(
    idx: torch.Tensor, scales: torch.Tensor, codebook: torch.Tensor, shape: torch.Size
) -> torch.Tensor:
    flat = (codebook.to(scales)[idx.long()] * scales[:, None]).reshape(-1)
    return flat[: math.prod(shape)].view(shape)


# ----------------------------------------------------------------------------- GGUF's formats (block of 32)


def q8_0(w: torch.Tensor) -> torch.Tensor:
    """x -> round(x / d) as int8 with d = absmax / 127 per block of 32, d stored as float16. Returns the reconstruction."""
    flat = w.detach().float().reshape(-1)
    pad = (-flat.numel()) % 32
    b = torch.cat([flat, flat.new_zeros(pad)]).view(-1, 32)
    d = (b.abs().amax(1) / 127.0).half().float().clamp_min(1e-12)
    q = torch.round(b / d[:, None]).clamp(-127, 127)
    return (q * d[:, None]).reshape(-1)[: flat.numel()].view(w.shape)


def q4_0(w: torch.Tensor) -> torch.Tensor:
    """GGUF Q4_0: per block of 32, d = (the signed value of largest magnitude) / -8, q = clamp(round(x / d) + 8, 0, 15), x ~ (q - 8) * d."""
    flat = w.detach().float().reshape(-1)
    pad = (-flat.numel()) % 32
    b = torch.cat([flat, flat.new_zeros(pad)]).view(-1, 32)
    idx = b.abs().argmax(1)
    signed_max = b.gather(1, idx[:, None])[:, 0]
    d = (signed_max / -8.0).half().float()
    inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp(torch.round(b * inv[:, None]) + 8, 0, 15)
    return ((q - 8) * d[:, None]).reshape(-1)[: flat.numel()].view(w.shape)


BITS_PER_WEIGHT = {
    "fp32": 32.0,
    "fp16": 16.0,
    "q8_0": 8 + 16 / 32,
    "q4_0": 4 + 16 / 32,
    "nf4": 4 + 32 / 64,
    "int4": 4 + 32 / 64,
}


def fake_quantize(w: torch.Tensor, fmt: str) -> torch.Tensor:
    """Quantise and reconstruct ``w`` in format ``fmt`` (nf4, int4, q8_0, q4_0)."""
    if fmt == "q8_0":
        return q8_0(w)
    if fmt == "q4_0":
        return q4_0(w)
    if fmt in ("nf4", "int4"):
        cb = nf4_codebook() if fmt == "nf4" else uniform_codebook(4)
        idx, scales = quantize_blockwise(w, cb, 64)
        return dequantize_blockwise(idx, scales, cb, w.shape)
    raise ValueError(f"unknown format {fmt!r}")


def relative_error(w: torch.Tensor, wq: torch.Tensor) -> float:
    """RMS of the reconstruction error divided by the RMS of the weights."""
    return ((w - wq).pow(2).mean().sqrt() / w.pow(2).mean().sqrt()).item()


@torch.no_grad()
def quantize_model_(
    model: nn.Module, fmt: str, skip: tuple[str, ...] = ("embed_tokens", "lm_head", "norm")
) -> int:
    """Replace the weight of every Linear layer (not the embedding table, the output matrix or the norms) with its quantised reconstruction, in place.
    Returns the number of weights touched."""
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and not any(s in name for s in skip):
            m.weight.copy_(fake_quantize(m.weight, fmt))
            n += m.weight.numel()
    return n
