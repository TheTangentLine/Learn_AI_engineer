"""Week 9 Day 4 - Solution: the transformer block, verified against a real model.

1. PARTS      RMSNorm, LayerNorm, RoPE and SwiGLU against Hugging Face's / PyTorch's own implementations
2. WHOLE      a random-weight decoder (multi-head, grouped-query, multi-query; with and without q/k/v bias) against Qwen2ForCausalLM / LlamaForCausalLM
3. REAL       the actual Qwen2.5-0.5B weights loaded into this from-scratch decoder: logits and 16 greedy tokens against Hugging Face
4. PROPERTIES rotary embeddings turn positions into rotations: relative position only, norm preserved
5. WHY        residual connections and pre-norm: gradients and activations through a 24-layer stack with each ingredient removed

  uv run python weeks/week09_transformers-from-scratch/solutions/day4_solution.py [--no-real]
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import blocks as B  # noqa: E402

# ----------------------------------------------------------------------------- 1. parts


def parts_vs_reference(seed: int = 0) -> dict[str, float]:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Config,
        Qwen2MLP,
        Qwen2RMSNorm,
        apply_rotary_pos_emb,
    )

    torch.manual_seed(seed)
    out = {}
    x = torch.randn(2, 5, 64)
    ref = Qwen2RMSNorm(64, eps=1e-6)
    mine = B.RMSNorm(64, 1e-6)
    with torch.no_grad():
        ref.weight.copy_(torch.randn(64))
        mine.weight.copy_(ref.weight)
    out["RMSNorm vs Qwen2RMSNorm"] = (mine(x) - ref(x)).abs().max().item()

    ln, ref_ln = B.LayerNorm(64), nn.LayerNorm(64)
    with torch.no_grad():
        ref_ln.weight.copy_(torch.randn(64))
        ref_ln.bias.copy_(torch.randn(64))
        ln.weight.copy_(ref_ln.weight)
        ln.bias.copy_(ref_ln.bias)
    out["LayerNorm vs nn.LayerNorm"] = (ln(x) - ref_ln(x)).abs().max().item()

    cfg = Qwen2Config(hidden_size=64, intermediate_size=160, hidden_act="silu")
    ref_mlp, mlp = Qwen2MLP(cfg), B.SwiGLU(64, 160)
    mlp.load_state_dict(ref_mlp.state_dict())
    out["SwiGLU vs Qwen2MLP"] = (mlp(x) - ref_mlp(x)).abs().max().item()

    q, k = torch.randn(2, 4, 9, 16), torch.randn(2, 4, 9, 16)
    cos, sin = B.rope_tables(16, torch.arange(9), 10000.0)
    rq, rk = apply_rotary_pos_emb(q, k, cos[None], sin[None])
    out["RoPE vs apply_rotary_pos_emb"] = max(
        (B.apply_rope(q, cos, sin) - rq).abs().max().item(),
        (B.apply_rope(k, cos, sin) - rk).abs().max().item(),
    )
    return out


# ----------------------------------------------------------------------------- 2. whole models


def hf_model(
    kind: str,
    *,
    vocab: int,
    d: int,
    layers: int,
    heads: int,
    kv: int,
    ff: int,
    tie: bool,
    seed: int,
):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    common = dict(
        vocab_size=vocab,
        hidden_size=d,
        intermediate_size=ff,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        tie_word_embeddings=tie,
    )
    if kind == "qwen2":
        return Qwen2ForCausalLM(Qwen2Config(**common)).eval()
    return LlamaForCausalLM(LlamaConfig(**common, attention_bias=False)).eval()


def whole_model_gap(kind: str, *, heads: int, kv: int, tie: bool = True, seed: int = 0) -> float:
    hf = hf_model(kind, vocab=150, d=64, layers=3, heads=heads, kv=kv, ff=160, tie=tie, seed=seed)
    cfg = B.config_from_hf(hf.config)
    cfg.qkv_bias = kind == "qwen2"
    mine = B.Decoder(cfg).eval()
    B.load_hf_state_dict(mine, hf.state_dict())
    ids = torch.randint(0, 150, (3, 21), generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        return (mine(ids) - hf(ids).logits).abs().max().item()


# ----------------------------------------------------------------------------- 3. the real model


def real_qwen(prompt: str = "The capital of France is", new_tokens: int = 16) -> dict:
    """Load Qwen2.5-0.5B into the from-scratch decoder. Hugging Face runs first and is freed before ours loads (8 GB of RAM is tight)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    ids = tok(prompt, return_tensors="pt").input_ids
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    with torch.no_grad():
        ref_logits = hf(ids).logits
        # the checkpoint's generation_config applies a repetition penalty (and sampling defaults) even with do_sample=False: switch them off
        out = hf.generate(
            ids,
            max_new_tokens=new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            top_k=None,
            top_p=None,
            temperature=None,
        )
        ref_gen = out[0, ids.shape[1] :].tolist()
    state, hf_cfg, n_hf = (
        {k: v.clone() for k, v in hf.state_dict().items()},
        hf.config,
        sum(p.numel() for p in hf.parameters()),
    )
    del hf
    gc.collect()
    cfg = B.config_from_hf(hf_cfg)
    mine = B.Decoder(cfg).eval()
    B.load_hf_state_dict(mine, state)
    del state
    gc.collect()
    with torch.no_grad():
        logits = mine(ids)
        gen = ids
        for _ in range(
            new_tokens
        ):  # greedy decoding, recomputing the whole prefix each step (the KV cache is Day 7)
            gen = torch.cat([gen, mine(gen)[:, -1:].argmax(-1)], dim=1)
    return {
        "prompt": prompt,
        "max_logit_gap": (logits - ref_logits).abs().max().item(),
        "argmax_equal": bool((logits.argmax(-1) == ref_logits.argmax(-1)).all()),
        "greedy_equal": gen[0, ids.shape[1] :].tolist() == ref_gen,
        "text": tok.decode(gen[0, ids.shape[1] :]),
        "params": mine.num_parameters(),
        "params_hf": n_hf,
        "formula": B.count_parameters(cfg),
        "cfg": cfg,
    }


# ----------------------------------------------------------------------------- 4. RoPE properties


def rope_relative_gap(head_dim: int = 32, trials: int = 50, seed: int = 0) -> float:
    """<rope(q, m), rope(k, n)> should depend only on m - n. Largest spread of that dot product over (m, n) pairs with the same offset."""
    g = torch.Generator().manual_seed(seed)
    worst = 0.0
    for _ in range(trials):
        q, k = (
            torch.randn(1, 1, 1, head_dim, generator=g),
            torch.randn(1, 1, 1, head_dim, generator=g),
        )
        offset = int(torch.randint(0, 40, (1,), generator=g))
        dots = []
        for m in (offset, offset + 7, offset + 100, offset + 1000):
            cm, sm = B.rope_tables(head_dim, torch.tensor([m]))
            cn, sn = B.rope_tables(head_dim, torch.tensor([m - offset]))
            dots.append((B.apply_rope(q, cm, sm) * B.apply_rope(k, cn, sn)).sum().item())
        worst = max(worst, max(dots) - min(dots))
    return worst


def rope_norm_change(head_dim: int = 32, seed: int = 0) -> float:
    torch.manual_seed(seed)
    x = torch.randn(1, 1, 64, head_dim)
    cos, sin = B.rope_tables(head_dim, torch.arange(64))
    return (B.apply_rope(x, cos, sin).norm(dim=-1) - x.norm(dim=-1)).abs().max().item()


# ----------------------------------------------------------------------------- 5. why residuals and pre-norm


class PlainBlock(nn.Module):
    """No residual connection: x -> mlp(norm(attn(norm(x)))). The same sub-layers, stacked directly."""

    def __init__(self, cfg: B.Config):
        super().__init__()
        self.b = B.Block(cfg)

    def forward(self, x, cos, sin):
        x = self.b.self_attn(self.b.input_layernorm(x), cos, sin)
        return self.b.mlp(self.b.post_attention_layernorm(x))


class PostNormBlock(nn.Module):
    """The original Transformer arrangement: norm(x + sublayer(x))."""

    def __init__(self, cfg: B.Config):
        super().__init__()
        self.b = B.Block(cfg)

    def forward(self, x, cos, sin):
        x = self.b.input_layernorm(x + self.b.self_attn(x, cos, sin))
        return self.b.post_attention_layernorm(x + self.b.mlp(x))


def depth_experiment(block_cls, n_layers: int = 24, seed: int = 0, init_std: float = 0.02) -> dict:
    """Stack ``n_layers`` blocks, run a random batch, and measure (a) the size of the gradient reaching the first and the last block and
    (b) the RMS of the activations after each block."""
    torch.manual_seed(seed)
    cfg = B.Config(vocab_size=50, d_model=64, n_layers=n_layers, n_heads=4, d_ff=128)
    blocks = nn.ModuleList(block_cls(cfg) for _ in range(n_layers))
    for m in blocks.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=init_std if block_cls is not B.Block else 0.02)
    x = torch.randn(4, 16, 64, requires_grad=False)
    cos, sin = B.rope_tables(cfg.head_dim, torch.arange(16))
    rms = []
    h = x
    for blk in blocks:
        h = blk(h, cos, sin)
        rms.append(h.pow(2).mean().sqrt().item())
    h.pow(2).mean().backward()

    def gnorm(blk):
        return sum(p.grad.norm().item() ** 2 for p in blk.parameters() if p.grad is not None) ** 0.5

    return {"grad_first": gnorm(blocks[0]), "grad_last": gnorm(blocks[-1]), "rms": rms}


def main(argv: list[str]) -> None:
    print("1. PARTS against the reference implementations (largest absolute difference)")
    for k, v in parts_vs_reference().items():
        print(f"   {k:<32}{v:.1e}")

    print(
        "\n2. WHOLE random-weight decoders against Hugging Face (logits, largest absolute difference)"
    )
    for label, kind, heads, kv, tie in [
        ("Qwen2, multi-head (8 heads)", "qwen2", 8, 8, True),
        ("Qwen2, grouped-query (8 heads, 2 kv)", "qwen2", 8, 2, True),
        ("Qwen2, multi-query (8 heads, 1 kv)", "qwen2", 8, 1, True),
        ("Llama, grouped-query, untied output", "llama", 8, 4, False),
    ]:
        print(f"   {label:<42}{whole_model_gap(kind, heads=heads, kv=kv, tie=tie):.1e}")

    if "--no-real" not in argv:
        r = real_qwen()
        print("\n3. REAL Qwen2.5-0.5B weights in the from-scratch decoder")
        print(
            f"   prompt {r['prompt']!r}: logits differ by at most {r['max_logit_gap']:.1e}; argmax identical at every position: {r['argmax_equal']}"
        )
        print(
            f"   16 greedy tokens identical to Hugging Face: {r['greedy_equal']}   -> {r['text']!r}"
        )
        print(
            f"   parameters: {r['params']:,} (Hugging Face {r['params_hf']:,}; closed-form {r['formula']:,})"
        )

    print("\n4. ROTARY EMBEDDINGS")
    print(
        f"   spread of <rope(q,m), rope(k,n)> over pairs with the same m - n: {rope_relative_gap():.1e}   (it depends only on the offset)"
    )
    print(
        f"   largest change in a vector's length after rotation:         {rope_norm_change():.1e}   (a rotation preserves length)"
    )

    print(
        "\n5. WHY RESIDUALS AND PRE-NORM: 24 blocks, random weights, gradient of the mean squared output"
    )
    print(
        f"   {'arrangement':<26}{'grad at block 1':>17}{'grad at block 24':>18}{'ratio first/last':>18}{'activation RMS after 1 / 12 / 24':>36}"
    )
    for name, cls in [
        ("pre-norm + residual", B.Block),
        ("post-norm + residual", PostNormBlock),
        ("no residual", PlainBlock),
    ]:
        r = depth_experiment(cls)
        print(
            f"   {name:<26}{r['grad_first']:>17.2e}{r['grad_last']:>18.2e}{r['grad_first'] / max(r['grad_last'], 1e-30):>18.2e}   {r['rms'][0]:.4f} / {r['rms'][11]:.4f} / {r['rms'][23]:.4f}"
        )


if __name__ == "__main__":
    main(sys.argv)
