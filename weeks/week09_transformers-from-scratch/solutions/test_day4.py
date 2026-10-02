"""Tests for Week 9 Day 4: every part against Hugging Face / PyTorch, the whole decoder against Qwen2 and Llama, the real Qwen2.5-0.5B weights, and the
properties (causality, relative positions, residual identity, parameter counts) that equivalence alone would not pin down."""

from __future__ import annotations

import glob
import math
import random
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent))

import blocks as B  # noqa: E402
import day4_solution as d4  # noqa: E402

# ----------------------------------------------------------------------------- configuration


def test_config_defaults_and_validation():
    c = B.Config(vocab_size=100, d_model=96, n_layers=2, n_heads=4)
    assert (
        c.n_kv_heads == 4 and c.head_dim == 24 and c.d_ff % 8 == 0 and abs(c.d_ff - 8 * 96 / 3) < 8
    )
    with pytest.raises(ValueError, match="divisible by n_heads"):
        B.Config(vocab_size=1, d_model=30, n_layers=1, n_heads=4)
    with pytest.raises(ValueError, match="multiple of n_kv_heads"):
        B.Config(vocab_size=1, d_model=32, n_layers=1, n_heads=4, n_kv_heads=3)
    with pytest.raises(ValueError, match="even"):
        B.Config(vocab_size=1, d_model=12, n_layers=1, n_heads=4)


# ----------------------------------------------------------------------------- normalisation


def test_rmsnorm_is_the_formula_and_has_unit_rms_at_init():
    x = torch.randn(4, 7, 32) * 5 + 3
    y = B.RMSNorm(32)(x)
    manual = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(y, manual, atol=1e-6) and torch.allclose(
        y.pow(2).mean(-1).sqrt(), torch.ones(4, 7), atol=1e-4
    )
    assert not torch.allclose(y.mean(-1), torch.zeros(4, 7), atol=1e-3), (
        "RMSNorm does not subtract the mean (LayerNorm does)"
    )


def test_rmsnorm_keeps_the_input_dtype_and_computes_statistics_in_float32():
    x = (torch.randn(2, 3, 16) * 100).to(torch.bfloat16)
    y = B.RMSNorm(16).to(torch.bfloat16)(x)
    assert y.dtype == torch.bfloat16 and torch.isfinite(y).all()
    big = torch.full((1, 1, 16), 3e4, dtype=torch.float16)
    out = B.RMSNorm(16).to(torch.float16)(big)
    assert torch.allclose(out.float(), torch.ones(1, 1, 16), atol=1e-2), (
        "x**2 would overflow float16 if the statistics were computed in it"
    )


def test_layernorm_matches_pytorch():
    ours, ref = B.LayerNorm(32), nn.LayerNorm(32)
    with torch.no_grad():
        ref.weight.copy_(torch.randn(32))
        ref.bias.copy_(torch.randn(32))
        ours.weight.copy_(ref.weight)
        ours.bias.copy_(ref.bias)
    x = torch.randn(3, 5, 32) * 4 + 2
    assert torch.allclose(ours(x), ref(x), atol=1e-5)
    assert torch.allclose(B.LayerNorm(32)(x).mean(-1), torch.zeros(3, 5), atol=1e-5)


# ----------------------------------------------------------------------------- rotary embeddings


def test_rope_at_position_zero_is_the_identity_and_rotation_preserves_length():
    cos, sin = B.rope_tables(16, torch.arange(12))
    assert torch.allclose(cos[0], torch.ones(16)) and torch.allclose(sin[0], torch.zeros(16))
    x = torch.randn(2, 3, 12, 16)
    assert torch.allclose(B.apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    assert torch.equal(B.apply_rope(x[:, :, :1], cos[:1], sin[:1]), x[:, :, :1])


def test_rotate_half_twice_negates():
    x = torch.randn(3, 8)
    assert torch.equal(B.rotate_half(B.rotate_half(x)), -x)


def test_rope_dot_products_depend_only_on_the_offset():
    assert d4.rope_relative_gap() < 1e-3
    assert d4.rope_norm_change() < 1e-5


def test_rope_changes_with_position_and_the_first_pair_has_the_fastest_frequency():
    cos, sin = B.rope_tables(8, torch.tensor([0, 1, 2]), theta=10000.0)
    assert cos[1, 0].item() == pytest.approx(math.cos(1.0), abs=1e-6), (
        "pair 0 rotates by 1 radian per position"
    )
    assert cos[1, 3].item() == pytest.approx(math.cos(10000.0 ** (-6 / 8)), abs=1e-6)
    assert not torch.allclose(cos[1], cos[2])


def test_rope_with_batched_positions():
    pos = torch.tensor([[0, 1, 2], [5, 6, 7]])
    cos, sin = B.rope_tables(8, pos)
    assert cos.shape == (2, 3, 8)
    x = torch.randn(2, 4, 3, 8)
    out = B.apply_rope(x, cos, sin)
    assert torch.allclose(out[0], B.apply_rope(x[:1], cos[0], sin[0])[0])


# ----------------------------------------------------------------------------- parts against references


def test_every_part_matches_its_reference():
    gaps = d4.parts_vs_reference()
    assert (
        gaps["RMSNorm vs Qwen2RMSNorm"] == 0
        and gaps["SwiGLU vs Qwen2MLP"] == 0
        and gaps["RoPE vs apply_rotary_pos_emb"] == 0
    )
    assert gaps["LayerNorm vs nn.LayerNorm"] < 1e-5


def test_swiglu_is_gate_times_up_through_down():
    m = B.SwiGLU(8, 12)
    x = torch.randn(2, 3, 8)
    assert torch.allclose(m(x), m.down_proj(F.silu(m.gate_proj(x)) * m.up_proj(x)))
    assert sum(p.numel() for p in m.parameters()) == 3 * 8 * 12 and m.gate_proj.bias is None


# ----------------------------------------------------------------------------- attention and blocks


def test_grouped_query_attention_equals_multi_head_attention_with_the_key_value_weights_repeated():
    torch.manual_seed(0)
    gq = B.GQAttention(
        B.Config(vocab_size=1, d_model=32, n_layers=1, n_heads=8, n_kv_heads=2, qkv_bias=True)
    )
    mh = B.GQAttention(
        B.Config(vocab_size=1, d_model=32, n_layers=1, n_heads=8, n_kv_heads=8, qkv_bias=True)
    )
    d = 4
    with torch.no_grad():
        mh.q_proj.load_state_dict(gq.q_proj.state_dict())
        mh.o_proj.load_state_dict(gq.o_proj.state_dict())
        for name in ("k_proj", "v_proj"):
            w = getattr(gq, name).weight.view(2, d, 32).repeat_interleave(4, dim=0).reshape(32, 32)
            b = getattr(gq, name).bias.view(2, d).repeat_interleave(4, dim=0).reshape(32)
            getattr(mh, name).weight.copy_(w)
            getattr(mh, name).bias.copy_(b)
    x = torch.randn(2, 9, 32)
    cos, sin = B.rope_tables(d, torch.arange(9))
    assert torch.allclose(gq(x, cos, sin), mh(x, cos, sin), atol=1e-5)
    assert gq.k_proj.weight.shape == (8, 32) and mh.k_proj.weight.shape == (32, 32), (
        "the saving is in the K and V projections (and the cache)"
    )


def test_query_heads_in_a_group_share_one_key_value_head():
    """Isolate query head h (silence every other head in the output projection), then change kv head g: the output may move only if g == h // 4."""
    torch.manual_seed(1)
    cfg = B.Config(vocab_size=1, d_model=32, n_layers=1, n_heads=8, n_kv_heads=2)
    d = cfg.head_dim
    x = torch.randn(1, 6, 32)
    cos, sin = B.rope_tables(d, torch.arange(6))
    for h in range(8):
        for g in range(2):
            a = B.GQAttention(cfg)
            with torch.no_grad():
                mask = torch.zeros(32)
                mask[h * d : (h + 1) * d] = 1
                a.o_proj.weight.mul_(mask[None, :])
            before = a(x, cos, sin)
            with torch.no_grad():
                a.v_proj.weight[g * d : (g + 1) * d] += 1.0
            moved = not torch.allclose(a(x, cos, sin), before, atol=1e-6)
            assert moved == (g == h // 4), (h, g)


def test_a_block_with_silent_sublayers_is_the_identity():
    cfg = B.Config(vocab_size=1, d_model=16, n_layers=1, n_heads=2)
    blk = B.Block(cfg)
    with torch.no_grad():
        blk.self_attn.o_proj.weight.zero_()
        blk.mlp.down_proj.weight.zero_()
    x = torch.randn(2, 5, 16)
    cos, sin = B.rope_tables(8, torch.arange(5))
    assert torch.equal(blk(x, cos, sin), x), "the residual path carries x through unchanged"


def test_the_decoder_is_causal():
    torch.manual_seed(2)
    m = B.Decoder(B.Config(vocab_size=50, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2)).eval()
    ids = torch.randint(0, 50, (1, 12))
    other = ids.clone()
    other[:, 7:] = torch.randint(0, 50, (1, 5))
    with torch.no_grad():
        a, b = m(ids), m(other)
    assert torch.equal(a[:, :7], b[:, :7]) and not torch.allclose(a[:, 7:], b[:, 7:])


def test_shifting_every_position_by_a_constant_changes_nothing_because_rope_is_relative():
    torch.manual_seed(3)
    m = B.Decoder(B.Config(vocab_size=50, d_model=32, n_layers=2, n_heads=4)).eval()
    ids = torch.randint(0, 50, (2, 10))
    with torch.no_grad():
        assert torch.allclose(m(ids), m(ids, positions=torch.arange(10) + 500), atol=1e-3)


def test_forward_is_the_output_projection_of_the_hidden_states():
    torch.manual_seed(5)
    m = B.Decoder(B.Config(vocab_size=40, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2)).eval()
    ids = torch.randint(0, 40, (2, 9))
    with torch.no_grad():
        h = m.hidden_states(ids)
        assert h.shape == (2, 9, 32) and torch.equal(m.lm_head(h), m(ids))


def test_a_fresh_model_predicts_roughly_uniformly():
    torch.manual_seed(4)
    m = B.Decoder(B.Config(vocab_size=256, d_model=64, n_layers=2, n_heads=4)).eval()
    ids = torch.randint(0, 256, (8, 32))
    with torch.no_grad():
        loss = F.cross_entropy(m(ids)[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1)).item()
    assert loss == pytest.approx(math.log(256), abs=0.1), (
        "the starting loss must be ln(vocabulary), as on Day 1"
    )


# ----------------------------------------------------------------------------- parameters


@pytest.mark.parametrize("seed", range(12))
def test_the_closed_form_parameter_count_matches_the_model(seed):
    rng = random.Random(seed)
    heads = rng.choice([2, 4, 8])
    kv = rng.choice([k for k in (1, 2, 4, 8) if heads % k == 0])
    cfg = B.Config(
        vocab_size=rng.randint(50, 300),
        d_model=heads * rng.choice([8, 16]),
        n_layers=rng.randint(1, 4),
        n_heads=heads,
        n_kv_heads=kv,
        d_ff=rng.choice([0, 48, 100]),
        qkv_bias=rng.random() < 0.5,
        tie_embeddings=rng.random() < 0.5,
    )
    assert B.Decoder(cfg).num_parameters() == B.count_parameters(cfg)


def test_tied_embeddings_share_storage_and_untied_ones_do_not():
    tied = B.Decoder(B.Config(vocab_size=40, d_model=16, n_layers=1, n_heads=2))
    untied = B.Decoder(
        B.Config(vocab_size=40, d_model=16, n_layers=1, n_heads=2, tie_embeddings=False)
    )
    assert (
        tied.lm_head.weight is tied.embed_tokens.weight
        and untied.lm_head.weight is not untied.embed_tokens.weight
    )
    assert untied.num_parameters() - tied.num_parameters() == 40 * 16


# ----------------------------------------------------------------------------- whole-model equivalence with Hugging Face


@pytest.mark.parametrize(
    ("kind", "heads", "kv", "tie"),
    [
        ("qwen2", 8, 8, True),
        ("qwen2", 8, 2, True),
        ("qwen2", 8, 1, True),
        ("llama", 8, 4, False),
        ("llama", 4, 4, True),
    ],
)
def test_a_random_decoder_gives_hugging_faces_logits(kind, heads, kv, tie):
    assert d4.whole_model_gap(kind, heads=heads, kv=kv, tie=tie) < 1e-5


def test_the_loader_reports_what_it_copied_and_rejects_mismatches():
    hf = d4.hf_model("qwen2", vocab=60, d=32, layers=2, heads=4, kv=2, ff=64, tie=True, seed=0)
    cfg = B.config_from_hf(hf.config)
    m = B.Decoder(cfg)
    copied = B.load_hf_state_dict(m, hf.state_dict())
    assert "layers.0.self_attn.q_proj.weight" in copied and "embed_tokens.weight" in copied
    bad = {k: v for k, v in hf.state_dict().items() if "layers.1" not in k}
    with pytest.raises(ValueError, match="not found"):
        B.load_hf_state_dict(B.Decoder(cfg), bad)
    wrong = dict(hf.state_dict())
    wrong["model.layers.0.mlp.up_proj.weight"] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="shape"):
        B.load_hf_state_dict(B.Decoder(cfg), wrong)


def test_a_tied_model_loads_from_a_checkpoint_that_has_no_output_matrix():
    """Qwen2.5-0.5B's safetensors file has no lm_head.weight: the output layer IS the embedding table."""
    hf = d4.hf_model("qwen2", vocab=60, d=32, layers=2, heads=4, kv=2, ff=64, tie=True, seed=0)
    state = {k: v for k, v in hf.state_dict().items() if k != "lm_head.weight"}
    m = B.Decoder(B.config_from_hf(hf.config)).eval()
    B.load_hf_state_dict(m, state)
    ids = torch.randint(0, 60, (1, 7))
    with torch.no_grad():
        assert (m(ids) - hf(ids).logits).abs().max() < 1e-5
    untied = B.Decoder(
        B.Config(vocab_size=60, d_model=32, n_layers=2, n_heads=4, tie_embeddings=False)
    )
    with pytest.raises(ValueError, match="lm_head.weight"):
        B.load_hf_state_dict(
            untied, {k: v for k, v in untied.state_dict().items() if k != "lm_head.weight"}
        )


def test_a_mutation_in_the_loader_would_be_caught_by_the_equivalence_check():
    """Equivalence must be able to fail: perturb one weight after loading and the logits must move."""
    hf = d4.hf_model("qwen2", vocab=60, d=32, layers=2, heads=4, kv=2, ff=64, tie=True, seed=0)
    m = B.Decoder(B.config_from_hf(hf.config)).eval()
    B.load_hf_state_dict(m, hf.state_dict())
    ids = torch.randint(0, 60, (1, 9))
    with torch.no_grad():
        assert (m(ids) - hf(ids).logits).abs().max() < 1e-5
        m.layers[1].mlp.up_proj.weight[0, 0] += 0.5
        assert (m(ids) - hf(ids).logits).abs().max() > 1e-4


# ----------------------------------------------------------------------------- the real model


@pytest.mark.skipif(
    not glob.glob(
        str(
            Path.home()
            / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/model.safetensors"
        )
    ),
    reason="Qwen2.5-0.5B is not in the local cache",
)
def test_the_real_qwen_weights_run_in_the_from_scratch_decoder_and_generate_the_same_tokens():
    r = d4.real_qwen(new_tokens=8)
    assert r["max_logit_gap"] < 1e-3 and r["argmax_equal"] and r["greedy_equal"]
    assert r["params"] == r["params_hf"] == r["formula"] == 494_032_768
    assert r["cfg"].n_heads == 14 and r["cfg"].n_kv_heads == 2 and r["cfg"].qkv_bias


# ----------------------------------------------------------------------------- the depth experiment


def test_residual_connections_keep_gradients_even_across_layers_and_removing_them_does_not():
    pre = d4.depth_experiment(B.Block, n_layers=24)
    post = d4.depth_experiment(d4.PostNormBlock, n_layers=24)
    plain = d4.depth_experiment(d4.PlainBlock, n_layers=24)
    assert 0.5 < pre["grad_first"] / pre["grad_last"] < 2.0
    assert post["grad_first"] / post["grad_last"] < 1e-5, (
        "post-norm: the early layers get almost no gradient at initialisation"
    )
    assert plain["grad_first"] / plain["grad_last"] > 100, (
        "no residual: the gradient is wildly uneven across layers"
    )
    assert max(plain["rms"]) < 0.05, (
        "without a residual the signal is whatever the last sub-layer produces, which is small at this initialisation"
    )
