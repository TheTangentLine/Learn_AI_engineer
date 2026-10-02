"""Tests for Week 10 Day 3: LoRA (against PEFT and by hand), merging, the SFT loop's pieces and behaviour, and the quantisation formats."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "weeks/week09_transformers-from-scratch/solutions"))

import blocks as B  # noqa: E402
import chatfmt as C  # noqa: E402
import lora as L  # noqa: E402
import quant as Q  # noqa: E402
import train_sft as T  # noqa: E402


def tiny(vocab=64, d=32, layers=2, seed=0) -> B.Decoder:
    torch.manual_seed(seed)
    m = B.Decoder(
        B.Config(
            vocab_size=vocab,
            d_model=d,
            n_layers=layers,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=64,
            tie_embeddings=True,
        )
    ).eval()
    for p in m.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.2)
    return m


# ----------------------------------------------------------------------------- the LoRA layer


def test_a_fresh_adapter_changes_nothing_and_the_forward_is_the_documented_formula():
    torch.manual_seed(0)
    base = nn.Linear(12, 20, bias=True)
    lin = L.LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 5, 12)
    assert torch.equal(lin(x), base(x)), "B starts at zero: the adapted layer equals the base layer"
    with torch.no_grad():
        lin.lora_B.normal_()
    expect = base(x) + 2.0 * (x @ lin.lora_A.T @ lin.lora_B.T)
    assert lin.scale == 2.0 and torch.allclose(lin(x), expect, atol=1e-6)
    assert lin.lora_A.shape == (4, 12) and lin.lora_B.shape == (20, 4)


def test_the_weight_update_is_low_rank_and_the_merged_layer_equals_the_adapted_one():
    torch.manual_seed(1)
    lin = L.LoRALinear(nn.Linear(16, 24, bias=True), r=3, alpha=6)
    with torch.no_grad():
        lin.lora_B.normal_()
    delta = lin.delta_weight()
    assert delta.shape == (24, 16) and torch.linalg.matrix_rank(delta).item() == 3
    merged = lin.merged_linear()
    x = torch.randn(7, 16)
    assert torch.allclose(merged(x), lin(x), atol=1e-5) and torch.allclose(
        merged.bias, lin.base.bias
    )


def test_at_initialisation_the_gradient_reaches_B_but_not_A():
    """dL/dA is proportional to B, which is zero: the classic first-step behaviour of LoRA."""
    lin = L.LoRALinear(nn.Linear(8, 8), r=2, alpha=4)
    lin(torch.randn(5, 8)).pow(2).sum().backward()
    assert (
        lin.lora_B.grad.abs().sum() > 0
        and lin.lora_A.grad.abs().sum() == 0
        and lin.base.weight.grad is not None
    )
    assert not any(
        p.requires_grad for p in []
    )  # (the base is frozen by add_lora, not by the layer: tested below)


def test_rank_must_be_positive_and_dropout_only_acts_in_training():
    with pytest.raises(ValueError):
        L.LoRALinear(nn.Linear(4, 4), r=0, alpha=1)
    lin = L.LoRALinear(nn.Linear(10, 10), r=4, alpha=8, dropout=0.5)
    with torch.no_grad():
        lin.lora_B.normal_()
    x = torch.randn(4, 10)
    lin.eval()
    assert torch.equal(lin(x), lin(x))
    lin.train()
    assert not torch.equal(lin(x), lin(x))


def test_parameter_count_formula():
    assert (
        L.lora_parameter_count(576, 576, 16) == 18432
        and L.lora_parameter_count(576, 1536, 8) == 16896
    )


# ----------------------------------------------------------------------------- wrapping a model


def test_add_lora_freezes_the_base_wraps_only_the_targets_and_trains_only_the_adapters():
    m = tiny()
    names = L.add_lora(m, r=4, alpha=8, targets=("q_proj", "v_proj"))
    assert len(names) == 4 and all(n.endswith(("q_proj", "v_proj")) for n in names)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable) and len(trainable) == 8
    assert all(not p.requires_grad for n, p in m.named_parameters() if "lora_" not in n)
    assert isinstance(m.layers[0].self_attn.k_proj, nn.Linear) and isinstance(
        m.layers[0].self_attn.q_proj, L.LoRALinear
    )
    assert isinstance(m.lm_head, nn.Linear), "the output projection is never wrapped"


def test_a_second_call_does_not_wrap_the_wrapped_and_no_match_is_an_error():
    m = tiny()
    first = L.add_lora(m, r=2, alpha=4)
    assert len(first) == 14
    with pytest.raises(ValueError, match="no Linear layer matched"):
        L.add_lora(m, r=2, alpha=4)  # everything that matches is already adapted
    assert len(L.lora_modules(m)) == 14 and not any(
        isinstance(mod.base, L.LoRALinear) for mod in L.lora_modules(m).values()
    )
    with pytest.raises(ValueError, match="no Linear layer matched"):
        L.add_lora(tiny(), targets=("nope",))


def test_the_layer_filter_adapts_only_the_chosen_blocks():
    m = tiny(layers=4)
    names = L.add_lora(m, r=2, alpha=4, targets=("q_proj",), layers=range(2, 4))
    assert names == ["layers.2.self_attn.q_proj", "layers.3.self_attn.q_proj"]


def test_an_adapted_model_equals_the_base_until_B_changes_and_counts_are_right():
    m = tiny()
    ids = torch.randint(0, 64, (2, 11))
    with torch.no_grad():
        base = m(ids)
    L.add_lora(m, r=4, alpha=8)
    with torch.no_grad():
        assert torch.equal(m(ids), base)
    c = L.count_parameters(m)
    expected = sum(
        L.lora_parameter_count(mod.in_features, mod.out_features, 4)
        for mod in L.lora_modules(m).values()
    )
    assert c["trainable"] == expected and c["total"] == c["trainable"] + c["frozen"]


def test_the_adapter_state_dict_round_trips_and_rejects_a_mismatched_one():
    m, m2 = tiny(), tiny()
    L.add_lora(m, r=4, alpha=8)
    L.add_lora(m2, r=4, alpha=8)
    with torch.no_grad():
        for p in (p for n, p in m.named_parameters() if "lora_" in n):
            p.normal_()
    state = L.lora_state_dict(m)
    assert len(state) == 28 and all("lora_" in k for k in state)
    L.load_lora_state_dict(m2, state)
    ids = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        assert torch.equal(m(ids), m2(ids))
    with pytest.raises(ValueError, match="mismatch"):
        L.load_lora_state_dict(m2, {k: v for k, v in state.items() if "layers.0" not in k})


def test_merging_bakes_the_adapters_into_the_weights_without_changing_the_outputs():
    m = tiny()
    L.add_lora(m, r=4, alpha=8)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_(std=0.1)
    ids = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        before = m(ids)
    L.merge_lora(m)
    with torch.no_grad():
        assert torch.allclose(m(ids), before, atol=1e-4)
    assert not L.lora_modules(m) and all(p.requires_grad for p in m.parameters())
    assert isinstance(m.layers[0].mlp.down_proj, nn.Linear)


def test_lora_matches_pefts_lora_with_the_same_adapters_and_after_merging():
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(3)
    cfg = LlamaConfig(
        vocab_size=70,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    pm = get_peft_model(
        LlamaForCausalLM(cfg).eval(),
        LoraConfig(r=4, lora_alpha=8, target_modules=list(L.DEFAULT_TARGETS), lora_dropout=0.0),
    )
    for n, p in pm.named_parameters():
        if "lora_" in n:
            torch.nn.init.normal_(p, std=0.1)
    pm.eval()
    mine = B.Decoder(B.config_from_hf(cfg)).eval()
    B.load_hf_state_dict(
        mine,
        {
            k.replace("base_model.model.", "").replace(".base_layer", ""): v
            for k, v in pm.state_dict().items()
            if "lora_" not in k
        },
    )
    L.add_lora(mine, r=4, alpha=8)
    L.load_lora_state_dict(
        mine,
        {
            k.replace("base_model.model.model.", "").replace(".default.weight", ""): v
            for k, v in pm.state_dict().items()
            if "lora_A" in k or "lora_B" in k
        },
    )
    ids = torch.randint(0, 70, (2, 12))
    with torch.no_grad():
        assert (pm(ids).logits - mine(ids)).abs().max() < 1e-5
        merged = pm.merge_and_unload()
        L.merge_lora(mine)
        assert (merged(ids).logits - mine(ids)).abs().max() < 1e-5


# ----------------------------------------------------------------------------- the SFT loop


def examples(n=12, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        plen, alen = rng.randint(5, 14), rng.randint(3, 6)
        ids = [rng.randrange(3, 60) for _ in range(plen + alen)]
        out.append(C.Example(ids, [C.IGNORE] * plen + ids[plen:], alen))
    return out


def test_the_learning_rate_schedule_warms_up_then_decays_to_the_floor():
    cfg = T.SFTConfig(lr=1e-3, warmup=10, min_lr_fraction=0.1)
    assert T.lr_at(0, 100, cfg) == pytest.approx(1e-4) and T.lr_at(9, 100, cfg) == pytest.approx(
        1e-3
    )
    assert T.lr_at(99, 100, cfg) == pytest.approx(1e-4, rel=0.05) and T.lr_at(
        55, 100, cfg
    ) == pytest.approx(1e-3 * 0.55, rel=0.05)


def test_batches_cover_every_example_once_and_have_similar_lengths():
    exs = examples(50)
    batches = T.make_batches(exs, 8, random.Random(0), bucket=16)
    flat = [e for b in batches for e in b]
    assert (
        len(flat) == 50
        and {id(e) for e in flat} == {id(e) for e in exs}
        and all(len(b) <= 8 for b in batches)
    )
    spread = sum(max(len(e) for e in b) - min(len(e) for e in b) for b in batches) / len(batches)
    shuffled = [exs[i : i + 8] for i in range(0, 48, 8)]
    assert spread < sum(max(len(e) for e in b) - min(len(e) for e in b) for b in shuffled) / len(
        shuffled
    )
    assert T.make_batches(exs, 8, random.Random(1)) != T.make_batches(exs, 8, random.Random(2))


def test_with_one_bucket_batches_are_sorted_by_length_inside_and_their_order_is_shuffled():
    exs = examples(64)
    batches = T.make_batches(exs, 8, random.Random(0), bucket=1000)
    inside = [[len(e) for e in b] for b in batches]
    assert all(lens == sorted(lens) for lens in inside)
    firsts = [lens[0] for lens in inside]
    assert firsts != sorted(firsts), (
        "the batches themselves are shuffled (otherwise training would see short sequences first)"
    )
    spread = sum(max(lens) - min(lens) for lens in inside) / len(inside)
    naive = [exs[i : i + 8] for i in range(0, 64, 8)]
    assert spread < 0.4 * sum(max(len(e) for e in b) - min(len(e) for e in b) for b in naive) / len(
        naive
    )


def test_each_step_backpropagates_the_per_token_mean_clips_the_gradient_and_clears_it(monkeypatch):
    m = tiny(d=48)
    L.add_lora(m, r=4, alpha=8)
    exs = examples(8)
    seen = []
    real = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *a, **k):
        total = real(params, max_norm, *a, **k)
        seen.append((float(max_norm), float(total)))
        return total

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    cfg = T.SFTConfig(epochs=1, batch=8, lr=1e-3, warmup=1, clip=0.7, log_every=1)
    T.train_sft(m, exs, exs[:2], cfg, pad_id=0)
    assert seen and all(mx == 0.7 for mx, _ in seen)
    # the gradient norm of the MEAN token loss of that one batch, computed independently on a fresh copy
    ref = tiny(d=48)
    L.add_lora(ref, r=4, alpha=8)
    loss, n, _ = T.answer_loss(ref, C.collate(exs, 0))
    (loss / n).backward()
    assert seen[0][1] == pytest.approx(
        math.sqrt(sum(p.grad.pow(2).sum().item() for p in ref.parameters() if p.grad is not None)),
        rel=1e-3,
    )
    assert all(p.grad is None for p in m.parameters()), "gradients are cleared after the last step"


def test_the_answer_loss_equals_the_naive_loss_on_full_logits():
    m = tiny()
    batch = C.collate(examples(5), pad_id=0)
    loss, n, correct = T.answer_loss(m, batch)
    logits = m(batch["input_ids"])[:, :-1]
    target = batch["labels"][:, 1:]
    naive = F.cross_entropy(
        logits.reshape(-1, 64), target.reshape(-1), ignore_index=C.IGNORE, reduction="sum"
    )
    assert loss.item() == pytest.approx(naive.item(), rel=1e-5) and n == int(
        (target != C.IGNORE).sum()
    )
    assert correct == int(((logits.argmax(-1) == target) & (target != C.IGNORE)).sum())


def test_padding_does_not_change_the_loss_of_the_real_tokens():
    m = tiny()
    exs = examples(3)
    alone = sum(T.answer_loss(m, C.collate([e], 0))[0].item() for e in exs)
    together = T.answer_loss(m, C.collate(exs, 0))[0].item()
    assert together == pytest.approx(alone, rel=1e-4)


def test_training_lowers_the_loss_moves_only_the_adapters_and_calls_back_each_epoch():
    m = tiny(d=48)
    L.add_lora(m, r=8, alpha=16)
    before = {n: p.detach().clone() for n, p in m.named_parameters() if "lora_" not in n}
    exs = examples(24)
    seen = []
    run = T.train_sft(
        m,
        exs,
        exs[:6],
        T.SFTConfig(epochs=4, batch=6, lr=2e-2, warmup=2, log_every=2),
        pad_id=0,
        on_epoch=lambda e, r: seen.append(e),
    )
    assert (
        seen == [1, 2, 3, 4]
        and len(run.epoch_dev) == 5
        and run.epoch_dev[-1]["loss"] < run.epoch_dev[0]["loss"] * 0.9
    )
    assert all(torch.equal(before[n], p) for n, p in m.named_parameters() if "lora_" not in n), (
        "the frozen weights did not move"
    )
    assert any(p.abs().sum() > 0 for n, p in m.named_parameters() if "lora_B" in n)
    assert run.steps[-1] == 16 and run.tokens_seen == 4 * sum(len(e) for e in exs)


def test_training_a_model_with_nothing_trainable_is_an_error():
    m = tiny()
    for p in m.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="nothing to train"):
        T.train_sft(m, examples(4), examples(2), T.SFTConfig(epochs=1), 0)


def test_evaluate_loss_is_deterministic_and_restores_training_mode():
    m = tiny()
    m.train()
    exs = examples(7)
    a = T.evaluate_loss(m, exs, 0)
    assert (
        m.training
        and a == T.evaluate_loss(m, exs, 0)
        and a["tokens"] == sum(e.n_answer for e in exs)
    )
    assert a["loss"] == pytest.approx(math.log(64), abs=1.5)


# ----------------------------------------------------------------------------- quantisation


def test_the_nf4_table_rebuilt_from_its_definition_equals_the_published_values():
    pytest.importorskip("scipy")
    cb = Q.nf4_codebook()
    assert (
        len(cb) == 16
        and (cb - torch.tensor(Q.NF4_PUBLISHED)).abs().max() < 1e-6
        and cb[7] == 0
        and cb[0] == -1
        and cb[-1] == 1
    )


def test_blockwise_round_trip_is_exact_at_codebook_points_and_pads_to_whole_blocks():
    cb = Q.uniform_codebook(4)
    w = torch.cat(
        [cb * 2.0, cb * 0.5]
    )  # two blocks of 16 whose values ARE codebook points times the block max
    idx, scales = Q.quantize_blockwise(w, cb, block=16)
    assert idx.shape == (2, 16) and scales.tolist() == pytest.approx([2.0, 0.5])
    assert torch.allclose(Q.dequantize_blockwise(idx, scales, cb, w.shape), w, atol=1e-6)
    odd = torch.randn(5, 7)
    idx, scales = Q.quantize_blockwise(odd, cb, block=16)
    assert (
        idx.shape == (3, 16)
        and Q.dequantize_blockwise(idx, scales, cb, odd.shape).shape == odd.shape
    )


def test_error_ordering_on_normal_weights_and_the_formats_bit_costs():
    torch.manual_seed(0)
    w = torch.randn(256, 256) * 0.02
    err = {f: Q.relative_error(w, Q.fake_quantize(w, f)) for f in ("nf4", "int4", "q4_0", "q8_0")}
    assert err["q8_0"] < 0.01 < err["nf4"] < err["int4"] and err["q4_0"] < err["int4"]
    assert (
        Q.BITS_PER_WEIGHT["q8_0"] == 8.5
        and Q.BITS_PER_WEIGHT["q4_0"] == 4.5
        and Q.BITS_PER_WEIGHT["nf4"] == 4.5
    )


def test_q4_0_and_q8_0_by_hand():
    block = torch.tensor([0.0] * 31 + [-2.0])
    rec = Q.q4_0(block)
    assert rec[-1] == pytest.approx(-2.0, abs=1e-3) and rec[0] == 0, (
        "the largest-magnitude value maps to the end of the 4-bit range"
    )
    pos = torch.tensor([0.0] * 31 + [3.5])
    assert Q.q4_0(pos)[-1] == pytest.approx(3.5, abs=0.05)
    assert Q.q8_0(torch.tensor([1.0, -1.0] + [0.0] * 30))[0] == pytest.approx(1.0, abs=1e-3)
    assert (
        torch.equal(Q.q8_0(torch.zeros(40)), torch.zeros(40))
        and Q.q4_0(torch.zeros(40)).abs().sum() == 0
    )


def test_quantising_a_model_touches_the_linear_weights_only_and_changes_its_outputs_a_little():
    m = tiny()
    ids = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        ref = m(ids)
    emb = m.embed_tokens.weight.clone()
    n = Q.quantize_model_(m, "q8_0")
    assert n > 0 and torch.equal(m.embed_tokens.weight, emb)
    with torch.no_grad():
        q8 = m(ids)
    m4 = tiny()
    Q.quantize_model_(m4, "int4")
    with torch.no_grad():
        q4 = m4(ids)
    assert (q8 - ref).abs().max() < (q4 - ref).abs().max() / 2
    with pytest.raises(ValueError):
        Q.fake_quantize(torch.zeros(4), "q5_k")
