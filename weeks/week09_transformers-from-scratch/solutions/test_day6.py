"""Tests for Week 9 Day 6: parameter and cache arithmetic against the library, the MoE layer against Mixtral's, and tiled attention against standard attention."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent))

import arch as A  # noqa: E402
import attention as T  # noqa: E402
import day6_solution as d6  # noqa: E402

# ----------------------------------------------------------------------------- parameter counts


@pytest.fixture(scope="module")
def configs():
    return d6.real_configs()


def test_the_three_real_configs_give_the_published_parameter_totals(configs):
    totals = {n: A.count_params(A.spec_from_hf(c, n)) for n, c in configs.items()}
    assert totals["Llama-3-8B"]["total"] == 8_030_261_248
    assert totals["Mixtral-8x7B"]["total"] == 46_702_792_704
    assert totals["Qwen2.5-0.5B"]["total"] == 494_032_768
    assert totals["Mixtral-8x7B"]["active"] == 12_879_925_248, (
        "two of eight experts per token: the '12.9B active' of the model card"
    )
    assert totals["Llama-3-8B"]["active"] == totals["Llama-3-8B"]["total"]


def test_the_closed_form_count_equals_the_librarys_for_the_real_configs(configs):
    for n, c in configs.items():
        assert A.count_params(A.spec_from_hf(c, n))["total"] == A.hf_reference_count(c), n


def _tiny(kind: str, **kw):
    from transformers import AutoModelForCausalLM, LlamaConfig, MixtralConfig, Qwen2Config

    base = {
        "vocab_size": 90,
        "hidden_size": 48,
        "intermediate_size": 80,
        "num_hidden_layers": 3,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
    }
    base.update(kw)
    cfg = {"llama": LlamaConfig, "qwen2": Qwen2Config, "mixtral": MixtralConfig}[kind](**base)
    return cfg, AutoModelForCausalLM.from_config(cfg)


@pytest.mark.parametrize(
    ("kind", "extra"),
    [
        ("llama", {"tie_word_embeddings": False}),
        ("llama", {"tie_word_embeddings": True, "num_key_value_heads": 6}),
        ("llama", {"tie_word_embeddings": False, "num_key_value_heads": 1}),
        ("llama", {"head_dim": 16, "tie_word_embeddings": False}),
        ("qwen2", {"tie_word_embeddings": True}),
        (
            "mixtral",
            {"num_local_experts": 4, "num_experts_per_tok": 2, "tie_word_embeddings": False},
        ),
        (
            "mixtral",
            {"num_local_experts": 5, "num_experts_per_tok": 1, "tie_word_embeddings": False},
        ),
    ],
)
def test_the_formula_matches_real_instantiated_models_of_several_shapes(kind, extra):
    cfg, model = _tiny(kind, **extra)
    assert A.count_params(A.spec_from_hf(cfg))["total"] == sum(
        p.numel() for p in model.parameters()
    )


def test_counts_by_component_add_up():
    s = A.spec_from_hf(_tiny("llama", tie_word_embeddings=False)[0])
    c = A.count_params(s)
    assert c["embedding"] + c["attention"] + c["mlp"] + c["norms"] + c["output_head"] == c["total"]
    tied = A.count_params(A.replace(s, tie_embeddings=True))
    assert tied["output_head"] == 0 and c["total"] - tied["total"] == s.vocab_size * s.d_model


# ----------------------------------------------------------------------------- memory


def test_kv_cache_bytes_equal_the_size_of_the_real_cache_of_a_real_model():
    cfg, model = _tiny("llama", tie_word_embeddings=False)
    model.eval()
    ids = torch.randint(0, 90, (2, 13))
    with torch.no_grad():
        out = model(ids, use_cache=True)
    cache = out.past_key_values
    layers = cache.layers if hasattr(cache, "layers") else cache
    actual = (
        sum(
            layer.keys.numel() + layer.values.numel()
            if hasattr(layer, "keys")
            else layer[0].numel() + layer[1].numel()
            for layer in layers
        )
        * 4
    )  # noqa: E741
    assert actual == A.kv_cache_bytes(A.spec_from_hf(cfg), 13, batch=2, dtype_bytes=4)


def test_cache_memory_is_linear_in_length_batch_and_kv_heads():
    s = A.spec_from_hf(_tiny("llama")[0])
    base = A.kv_cache_bytes(s, 100)
    assert A.kv_cache_bytes(s, 300) == 3 * base and A.kv_cache_bytes(s, 100, batch=4) == 4 * base
    assert A.kv_cache_bytes(A.replace(s, n_kv_heads=s.n_heads), 100) == base * (
        s.n_heads // s.n_kv_heads
    ), "grouped-query attention shrinks the cache by heads / kv_heads"
    assert A.kv_cache_bytes(s, 100, dtype_bytes=1) * 2 == base


def test_known_cache_sizes_of_the_real_models(configs):
    llama = A.spec_from_hf(configs["Llama-3-8B"])
    assert A.kv_cache_bytes(llama, 8192) == 2 * 32 * 8 * 128 * 8192 * 2 == 1_073_741_824, (
        "exactly 1 GiB for 8k tokens in bf16"
    )
    qwen = A.spec_from_hf(configs["Qwen2.5-0.5B"])
    assert A.kv_cache_bytes(qwen, 1) == 2 * 24 * 2 * 64 * 2


def test_weight_bytes_scale_with_precision_and_decode_flops_with_context(configs):
    s = A.spec_from_hf(configs["Qwen2.5-0.5B"])
    assert A.weight_bytes(s, 2) == 2 * 494_032_768 and A.weight_bytes(s, 0.5) == 494_032_768 / 2
    assert A.decode_flops_per_token(s, 8000) > A.decode_flops_per_token(s, 100) > 2 * 494_032_768


def test_a_config_with_an_explicit_head_dim_is_read_correctly():
    cfg, model = _tiny("llama", head_dim=16, tie_word_embeddings=False)
    s = A.spec_from_hf(cfg)
    assert s.hd == 16 and s.n_heads * s.hd != s.d_model


# ----------------------------------------------------------------------------- mixture of experts


def copy_mixtral_block(hf_block, mine: A.MoEMLP, ff: int) -> None:
    with torch.no_grad():
        mine.router.weight.copy_(hf_block.gate.weight)
        for e, expert in enumerate(mine.experts):
            gate_up = hf_block.experts.gate_up_proj[e]
            expert.gate_proj.weight.copy_(gate_up[:ff])
            expert.up_proj.weight.copy_(gate_up[ff:])
            expert.down_proj.weight.copy_(hf_block.experts.down_proj[e])


@pytest.mark.parametrize(("experts", "top_k"), [(6, 2), (4, 1), (8, 3), (2, 2)])
def test_the_moe_layer_matches_mixtrals_sparse_block_with_copied_weights(experts, top_k):
    from transformers.models.mixtral.modeling_mixtral import MixtralConfig, MixtralSparseMoeBlock

    cfg = MixtralConfig(
        hidden_size=32,
        intermediate_size=48,
        num_local_experts=experts,
        num_experts_per_tok=top_k,
        num_hidden_layers=1,
        num_attention_heads=4,
        vocab_size=50,
    )
    torch.manual_seed(experts)
    hf = MixtralSparseMoeBlock(cfg).eval()
    for p in hf.parameters():
        torch.nn.init.normal_(p, std=0.2)
    mine = A.MoEMLP(32, 48, experts, top_k)
    copy_mixtral_block(hf, mine, 48)
    x = torch.randn(3, 9, 32)
    with torch.no_grad():
        assert (mine(x)[0] - hf(x)).abs().max() < 1e-5


def test_the_moe_output_is_the_weighted_sum_of_the_chosen_experts_computed_token_by_token():
    torch.manual_seed(1)
    m = A.MoEMLP(16, 24, 5, 2)
    x = torch.randn(2, 4, 16)
    y, _ = m(x)
    flat = x.reshape(-1, 16)
    for i in range(8):
        probs = F.softmax(m.router(flat[i]), -1)
        top = torch.topk(probs, 2)
        w = top.values / top.values.sum()
        want = sum(w[j] * m.experts[int(top.indices[j])](flat[i]) for j in range(2))
        assert torch.allclose(y.reshape(-1, 16)[i], want, atol=1e-5)


def test_one_expert_is_a_dense_mlp_and_all_experts_is_a_softmax_mixture():
    torch.manual_seed(2)
    one = A.MoEMLP(8, 12, 1, 1)
    x = torch.randn(1, 5, 8)
    assert torch.allclose(one(x)[0], one.experts[0](x), atol=1e-6)
    allk = A.MoEMLP(8, 12, 4, 4)
    probs = F.softmax(allk.router(x.reshape(-1, 8)), -1)
    want = sum(probs[:, e : e + 1] * allk.experts[e](x.reshape(-1, 8)) for e in range(4))
    assert torch.allclose(allk(x)[0].reshape(-1, 8), want, atol=1e-5)


def test_routing_weights_sum_to_one_and_loads_sum_to_one():
    m = A.MoEMLP(8, 12, 6, 2)
    flat = torch.randn(40, 8)
    _, w, chosen = m.route(flat)
    assert (
        torch.allclose(w.sum(-1), torch.ones(40))
        and chosen.shape == (40, 2)
        and (chosen[:, 0] != chosen[:, 1]).all()
    )
    assert m.expert_load(flat[None]).sum().item() == pytest.approx(1.0)


def test_the_balance_loss_is_one_for_uniform_probabilities_and_E_for_a_collapsed_router():
    m = A.MoEMLP(8, 12, 4, 1)
    with torch.no_grad():
        m.router.weight.zero_()
    assert m(torch.randn(2, 10, 8))[1].item() == pytest.approx(1.0, abs=1e-5)
    with torch.no_grad():
        m.router.weight.zero_()
        m.router.weight[0] = 50.0  # every token (positive input) goes to expert 0
    x = torch.rand(2, 10, 8) + 0.5
    assert m(x)[1].item() == pytest.approx(4.0, abs=0.05)


def test_the_balance_loss_is_one_for_uniform_routing_whatever_k():
    for k in (1, 2, 3):
        m = A.MoEMLP(8, 12, 6, k)
        with torch.no_grad():
            m.router.weight.zero_()
        assert m(torch.randn(2, 10, 8))[1].item() == pytest.approx(1.0, abs=1e-5), k


def test_gradients_reach_the_router_and_the_experts_that_were_used():
    torch.manual_seed(3)
    m = A.MoEMLP(8, 12, 4, 2)
    y, bal = m(torch.randn(2, 8, 8))
    (y.sum() + bal).backward()
    assert m.router.weight.grad.abs().sum() > 0
    assert (
        sum(
            1
            for e in m.experts
            if e.down_proj.weight.grad is not None and e.down_proj.weight.grad.abs().sum() > 0
        )
        >= 2
    )


def test_moe_rejects_an_impossible_top_k():
    with pytest.raises(ValueError):
        A.MoEMLP(8, 12, 4, 5)
    with pytest.raises(ValueError):
        A.MoEMLP(8, 12, 4, 0)


def test_a_load_balancing_loss_evens_out_the_routing_without_hurting_the_task():
    none, mid, strong = d6.train_moe(0.0), d6.train_moe(0.01), d6.train_moe(0.1)
    assert none["max_over_uniform"] > 2.0 and none["dead"] >= 1, (
        "without it, some experts take most of the tokens and some get none"
    )
    assert strong["max_over_uniform"] < 1.3 and strong["dead"] == 0
    assert strong["max_over_uniform"] < mid["max_over_uniform"] < none["max_over_uniform"]
    assert strong["mse"] < none["mse"] * 1.2


# ----------------------------------------------------------------------------- tiled attention


@pytest.mark.parametrize("block", [1, 5, 16, 33, 64, 500])
@pytest.mark.parametrize("causal", [True, False])
def test_tiled_attention_is_exact_for_every_block_size(block, causal):
    torch.manual_seed(block)
    q, k, v = (torch.randn(2, 3, 40, 8) for _ in range(3))
    ref, _ = T.scaled_dot_product_attention(q, k, v, causal=causal)
    assert torch.allclose(A.tiled_attention(q, k, v, block=block, causal=causal), ref, atol=1e-5)


def test_a_block_of_new_queries_at_the_end_of_a_longer_sequence_is_handled():
    torch.manual_seed(5)
    q, k, v = (torch.randn(1, 2, 30, 8) for _ in range(3))
    full, _ = T.scaled_dot_product_attention(q, k, v, causal=True)
    assert torch.allclose(
        A.tiled_attention(q[:, :, -4:], k, v, block=7), full[:, :, -4:], atol=1e-5
    )
    assert torch.allclose(
        A.tiled_attention(q[:, :, -1:], k, v, block=7), full[:, :, -1:], atol=1e-5
    )


def test_the_online_softmax_stays_finite_and_exact_with_enormous_scores():
    torch.manual_seed(6)
    q, k, v = (
        torch.randn(1, 2, 20, 16) * 80,
        torch.randn(1, 2, 20, 16) * 80,
        torch.randn(1, 2, 20, 16),
    )
    out = A.tiled_attention(q, k, v, block=6)
    ref, _ = T.scaled_dot_product_attention(q, k, v, causal=True)
    assert torch.isfinite(out).all() and torch.allclose(out, ref, atol=1e-4)
    naive = torch.exp(q @ k.transpose(-2, -1) / math.sqrt(16))
    assert torch.isinf(naive).any(), "a softmax without the running maximum would overflow here"


def test_tiled_attention_has_the_same_gradients_as_standard_attention():
    torch.manual_seed(7)
    base = [torch.randn(1, 2, 12, 8, dtype=torch.float64) for _ in range(3)]
    a = [t.clone().requires_grad_() for t in base]
    b = [t.clone().requires_grad_() for t in base]
    T.scaled_dot_product_attention(*a, causal=True)[0].pow(2).sum().backward()
    A.tiled_attention(*b, block=5).pow(2).sum().backward()
    for x, y in zip(a, b, strict=True):
        assert torch.allclose(x.grad, y.grad, atol=1e-8)


def test_the_largest_tensor_shrinks_with_the_block_size():
    assert A.largest_score_tensor(4096, 4096, block=None) == 4096 * 4096
    assert A.largest_score_tensor(4096, 4096, block=64) == 4096 * 64
    assert A.largest_score_tensor(10, 20, block=64) == 10 * 20


def test_the_flash_report_pins_exactness_and_the_memory_ratio():
    r = d6.flash_report(t=256, block=32)
    assert (
        r["gap"] < 1e-5
        and r["gap_extreme"] < 1e-4
        and r["finite"]
        and r["full_elems"] // r["tiled_elems"] == 8
    )
