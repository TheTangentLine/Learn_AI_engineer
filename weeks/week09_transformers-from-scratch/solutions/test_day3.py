"""Tests for Week 9 Day 3: attention against PyTorch's own implementations, the properties attention must have, and the experiments."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent))

import attention as A  # noqa: E402
import day3_solution as d3  # noqa: E402

# ----------------------------------------------------------------------------- softmax and masks


def test_softmax_matches_torch_and_rows_sum_to_one():
    x = torch.randn(4, 7, 9)
    assert torch.allclose(A.softmax(x), F.softmax(x, -1), atol=1e-6)
    assert torch.allclose(A.softmax(x).sum(-1), torch.ones(4, 7))


def test_softmax_survives_huge_scores_where_a_naive_one_overflows():
    x = torch.tensor([[1000.0, 1000.0, -1000.0]])
    assert torch.allclose(A.softmax(x), torch.tensor([[0.5, 0.5, 0.0]]))
    assert torch.isinf(torch.exp(x)).any()


def test_a_fully_masked_row_gives_zeros_not_nan():
    x = torch.tensor([[float("-inf")] * 3, [0.0, 0.0, float("-inf")]])
    out = A.softmax(x)
    assert (
        not torch.isnan(out).any()
        and out[0].tolist() == [0, 0, 0]
        and out[1].tolist() == pytest.approx([0.5, 0.5, 0.0])
    )


def test_the_causal_mask_is_the_lower_triangle():
    m = A.causal_mask(4)
    assert m.dtype == torch.bool and m.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


# ----------------------------------------------------------------------------- scaled dot-product attention


@pytest.mark.parametrize("causal", [False, True])
def test_matches_torchs_scaled_dot_product_attention(causal):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 11, 16) for _ in range(3))
    mine, _ = A.scaled_dot_product_attention(q, k, v, causal=causal)
    assert torch.allclose(
        mine, F.scaled_dot_product_attention(q, k, v, is_causal=causal), atol=1e-6
    )


def test_weights_are_a_distribution_and_causal_weights_have_no_future():
    torch.manual_seed(1)
    q, k, v = (torch.randn(5, 8, 4) for _ in range(3))
    _, w = A.scaled_dot_product_attention(q, k, v, causal=True)
    assert torch.allclose(w.sum(-1), torch.ones(5, 8)) and torch.triu(w, 1).abs().max() == 0


def test_scaling_matters_a_hand_computed_case():
    q = torch.tensor([[[2.0, 0.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    v = torch.tensor([[[10.0], [20.0]]])
    out, w = A.scaled_dot_product_attention(q, k, v)
    expected = F.softmax(torch.tensor([2.0, 0.0]) / math.sqrt(2), -1)
    assert torch.allclose(w[0, 0], expected) and out.item() == pytest.approx(
        (expected * torch.tensor([10.0, 20.0])).sum().item(), abs=1e-5
    )


def test_a_query_block_at_the_end_equals_the_last_rows_of_full_attention():
    """The property a KV cache relies on (Day 7): the last rows of causal attention need only those queries and ALL keys and values."""
    torch.manual_seed(2)
    q, k, v = (torch.randn(1, 2, 10, 8) for _ in range(3))
    full, _ = A.scaled_dot_product_attention(q, k, v, causal=True)
    tail, _ = A.scaled_dot_product_attention(q[:, :, -3:], k, v, causal=True)
    assert torch.allclose(tail, full[:, :, -3:], atol=1e-6)
    one, _ = A.scaled_dot_product_attention(q[:, :, -1:], k, v, causal=True)
    assert torch.allclose(one, full[:, :, -1:], atol=1e-6)


def test_an_explicit_mask_is_combined_with_the_causal_one():
    torch.manual_seed(3)
    q, k, v = (torch.randn(1, 6, 4) for _ in range(3))
    block = torch.ones(6, 6, dtype=torch.bool)
    block[:, 0] = False
    _, w = A.scaled_dot_product_attention(q, k, v, mask=block, causal=True)
    assert w[0, :, 0].abs().max() == 0 and w[0, 0].abs().sum() == 0, (
        "row 0 may only see column 0, which is blocked: it attends to nothing"
    )
    assert torch.triu(w, 1).abs().max() == 0


def test_gradients_are_correct_in_float64():
    torch.manual_seed(4)
    q, k, v = (torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True) for _ in range(3))
    assert torch.autograd.gradcheck(
        lambda a, b, c: A.scaled_dot_product_attention(a, b, c, causal=True)[0], (q, k, v)
    )


def test_key_padding_mask_gives_padding_zero_weight_and_real_outputs_ignore_it():
    torch.manual_seed(5)
    lengths = torch.tensor([5, 3])
    mask = A.key_padding_mask(lengths, 5)
    assert mask.shape == (2, 1, 1, 5) and mask[1, 0, 0].tolist() == [True, True, True, False, False]
    q, k, v = (torch.randn(2, 2, 5, 4) for _ in range(3))
    out, w = A.scaled_dot_product_attention(q, k, v, mask=mask)
    assert w[1, :, :, 3:].abs().max() == 0
    k2, v2 = k.clone(), v.clone()
    k2[1, :, 3:], v2[1, :, 3:] = 99.0, -99.0  # change what is in the padding
    out2, _ = A.scaled_dot_product_attention(q, k2, v2, mask=mask)
    assert torch.allclose(out[1, :, :3], out2[1, :, :3])


# ----------------------------------------------------------------------------- multi-head attention


@pytest.mark.parametrize(
    ("d_model", "heads", "t"), [(32, 4, 7), (64, 8, 1), (16, 1, 5), (48, 6, 9)]
)
def test_multihead_attention_matches_torchs_with_the_same_weights(d_model, heads, t):
    torch.manual_seed(d_model)
    mine = A.MultiHeadAttention(d_model, heads)
    ref = torch.nn.MultiheadAttention(d_model, heads, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(mine.qkv.weight)
        ref.in_proj_bias.copy_(mine.qkv.bias)
        ref.out_proj.weight.copy_(mine.out.weight)
        ref.out_proj.bias.copy_(mine.out.bias)
    x = torch.randn(3, t, d_model)
    expected, _ = ref(
        x, x, x, attn_mask=torch.triu(torch.ones(t, t, dtype=torch.bool), 1), need_weights=False
    )
    assert torch.allclose(mine(x), expected, atol=1e-5)


def test_shapes_and_head_splitting():
    m = A.MultiHeadAttention(32, 4)
    x = torch.randn(2, 6, 32)
    y, w = m(x, return_weights=True)
    assert (
        y.shape == (2, 6, 32) and w.shape == (2, 4, 6, 6) and m.split_heads(x).shape == (2, 4, 6, 8)
    )
    assert sum(p.numel() for p in m.parameters()) == 32 * 96 + 96 + 32 * 32 + 32


def test_the_model_width_must_divide_into_heads():
    with pytest.raises(ValueError, match="divisible"):
        A.MultiHeadAttention(30, 4)


def test_causality_changing_a_future_token_changes_nothing_before_it():
    torch.manual_seed(6)
    m = A.MultiHeadAttention(32, 4)
    x = torch.randn(1, 10, 32)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(1, 4, 32)
    a, b = m(x), m(x2)
    assert torch.equal(a[:, :6], b[:, :6]) and not torch.allclose(a[:, 6:], b[:, 6:])


def test_without_a_mask_attention_cannot_see_order_and_with_one_it_can():
    assert d3.permutation_gap(causal=False) < 1e-6
    assert d3.permutation_gap(causal=True) > 0.01


def test_attention_sees_each_head_separately_not_one_mixed_head():
    torch.manual_seed(7)
    m = A.MultiHeadAttention(16, 2)
    _, w = m(torch.randn(1, 5, 16), return_weights=True)
    assert not torch.allclose(w[:, 0], w[:, 1])


# ----------------------------------------------------------------------------- positions


def test_sinusoidal_positions_have_the_documented_form():
    pe = A.sinusoidal_positions(50, 16)
    assert pe.shape == (50, 16) and pe[0, 0::2].abs().max() == 0 and (pe[0, 1::2] == 1).all()
    assert pe.abs().max() <= 1.0 + 1e-6
    assert pe[1, 0] == pytest.approx(math.sin(1.0), abs=1e-6), "the fastest pair has frequency 1"
    with pytest.raises(ValueError):
        A.sinusoidal_positions(5, 15)


def test_a_shift_in_position_is_a_rotation_that_does_not_depend_on_where_you_are():
    """The property that motivated sinusoids: PE(p + k) is a fixed linear map of PE(p), whatever p is."""
    pe = A.sinusoidal_positions(200, 8).double()
    k = 7
    for pair in range(4):
        w = 10000.0 ** (-2 * pair / 8)
        rot = torch.tensor(
            [[math.cos(w * k), math.sin(w * k)], [-math.sin(w * k), math.cos(w * k)]],
            dtype=torch.float64,
        )
        for p in (0, 13, 91):
            sin_cos = pe[p, 2 * pair : 2 * pair + 2]
            assert torch.allclose(rot @ sin_cos, pe[p + k, 2 * pair : 2 * pair + 2], atol=1e-5)


def test_the_embedding_adds_a_position_vector_to_every_token():
    emb = A.TokenAndPositionEmbedding(10, 8, 6)
    ids = torch.tensor([[3, 3, 3]])
    out = emb(ids)
    assert out.shape == (1, 3, 8) and not torch.allclose(out[0, 0], out[0, 1]), (
        "the same token at different positions must differ"
    )
    assert torch.allclose(out[0, 1] - out[0, 0], emb.pos.weight[1] - emb.pos.weight[0], atol=1e-6)


# ----------------------------------------------------------------------------- the experiments


def test_scaled_attention_keeps_its_entropy_as_d_grows_and_unscaled_attention_collapses():
    scaled = [d3.attention_entropy(d, scaled=True)[0] for d in (4, 64, 1024)]
    unscaled = [d3.attention_entropy(d, scaled=False)[0] for d in (4, 64, 1024)]
    assert max(scaled) - min(scaled) < 0.1 and min(scaled) > 2.9 and max(scaled) < math.log(32)
    assert unscaled[0] > unscaled[1] > unscaled[2] and unscaled[2] < 0.3
    assert d3.attention_entropy(1024, scaled=False)[1] > 0.9


def test_the_copy_task_targets_are_the_input_shifted_by_the_offset():
    x, y = d3.copy_task(4, 10, 7, 3, torch.Generator().manual_seed(0))
    assert (y[:, :3] == -100).all() and torch.equal(y[:, 3:], x[:, :-3]) and x.max() < 7


def test_a_one_layer_model_learns_to_copy_from_three_back_and_the_attention_shows_how():
    r = d3.train_copy(positions=True, steps=400)
    assert r["accuracy"] > 0.99 and sum(r["losses"][-20:]) / 20 < 0.05
    assert set(d3.attended_offsets(r["weights"], 3)) == {3}
    assert r["losses"][0] == pytest.approx(r["chance_loss"], abs=0.3), "it starts at ln(vocab)"


def test_without_positional_embeddings_the_same_model_cannot_solve_it():
    r = d3.train_copy(positions=False, steps=400)
    assert r["accuracy"] < 0.6, (
        "the causal mask leaks a little position information at the start of the sequence, nothing more"
    )


def test_heatmap_has_one_row_per_query_and_marks_the_strongest_cell():
    w = torch.eye(4)
    text = d3.heatmap(w).splitlines()
    assert len(text) == 5 and text[1].endswith("█   ") and text[4].endswith("   █")
    assert d3.attended_offsets(torch.eye(4), 0) == [0, 0, 0, 0]


def test_the_score_matrix_quadruples_when_the_sequence_doubles():
    rows = d3.attention_cost([64, 128, 256], repeats=1)
    assert [r["score_mb"] for r in rows] == pytest.approx(
        [8 * 64 * 64 * 4 / 1e6, 8 * 128 * 128 * 4 / 1e6, 8 * 256 * 256 * 4 / 1e6]
    )
    assert rows[1]["score_mb"] / rows[0]["score_mb"] == pytest.approx(4.0)
