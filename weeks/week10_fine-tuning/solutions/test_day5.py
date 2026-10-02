"""Tests for Week 10 Day 5: the DPO loss (by hand and by its gradients), sequence log-probabilities, ORPO and GRPO formulas, the training loop."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "weeks/week09_transformers-from-scratch/solutions"))

import blocks as B  # noqa: E402
import chatfmt as C  # noqa: E402
import dpo as D  # noqa: E402
import infer as I  # noqa: E402
import lora as L  # noqa: E402


def tiny(seed=0, vocab=50, d=32):
    torch.manual_seed(seed)
    m = B.Decoder(
        B.Config(vocab_size=vocab, d_model=d, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)
    ).eval()
    for p in m.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.2)
    return m


def example(prompt, answer):
    return C.Example(prompt + answer, [C.IGNORE] * len(prompt) + answer, len(answer))


def pairs(n=8, seed=0):
    import random

    rng = random.Random(seed)
    out = []
    for _ in range(n):
        prompt = [rng.randrange(3, 40) for _ in range(rng.randint(4, 8))]
        good = [rng.randrange(3, 40) for _ in range(4)]
        bad = [rng.randrange(3, 40) for _ in range(4)]
        if good == bad:
            bad[0] = (bad[0] + 1) % 40 + 3
        out.append(D.Pair(example(prompt, good), example(prompt, bad)))
    return out


# ----------------------------------------------------------------------------- log-probabilities


def test_sequence_logprobs_equal_the_sum_of_the_answer_tokens_log_probabilities():
    m = tiny()
    batch = C.collate(
        [example([5, 6, 7, 8], [9, 10, 11]), example([5, 6], [12, 13, 14, 15, 16])], pad_id=0
    )
    got = D.sequence_logprobs(m, batch)
    logp = F.log_softmax(m(batch["input_ids"]).float(), dim=-1)
    for i, (prompt, ans) in enumerate(
        [([5, 6, 7, 8], [9, 10, 11]), ([5, 6], [12, 13, 14, 15, 16])]
    ):
        want = sum(logp[i, len(prompt) + j - 1, t].item() for j, t in enumerate(ans))
        assert got[i].item() == pytest.approx(want, abs=1e-4)
    assert got.shape == (2,) and (got < 0).all()


def test_padding_does_not_change_a_sequences_log_probability():
    m = tiny()
    a, b = example([5, 6, 7], [8, 9]), example([5, 6, 7, 8, 9, 10], [11, 12, 13])
    alone = D.sequence_logprobs(m, C.collate([a], 0))[0]
    padded = D.sequence_logprobs(m, C.collate([a, b], 0))[0]
    assert alone.item() == pytest.approx(padded.item(), abs=1e-5)


# ----------------------------------------------------------------------------- the DPO loss


def test_at_the_reference_policy_the_loss_is_ln_two_and_the_margin_is_zero():
    x = torch.tensor([-10.0, -20.0, -5.0])
    loss, m = D.dpo_loss(x, x - 3, x, x - 3, beta=0.1)
    assert (
        loss.item() == pytest.approx(math.log(2)) and m["margin"] == 0 and m["reward_accuracy"] == 0
    )


def test_the_loss_by_hand():
    pc, pr, rc, rr = (
        torch.tensor([-8.0]),
        torch.tensor([-14.0]),
        torch.tensor([-10.0]),
        torch.tensor([-12.0]),
    )
    margin = 0.1 * ((-8 + 10) - (-14 + 12))  # chosen +2, rejected -2 -> 0.1 * 4
    loss, m = D.dpo_loss(pc, pr, rc, rr, beta=0.1)
    assert (
        loss.item() == pytest.approx(-math.log(1 / (1 + math.exp(-margin))), abs=1e-6)
        and m["margin"] == pytest.approx(0.4)
        and m["reward_accuracy"] == 1.0
    )
    assert m["chosen_reward"] == pytest.approx(0.2) and m["rejected_reward"] == pytest.approx(-0.2)


def test_the_gradient_raises_the_chosen_answer_and_lowers_the_rejected_one():
    pc, pr = torch.tensor([-10.0], requires_grad=True), torch.tensor([-10.0], requires_grad=True)
    loss, _ = D.dpo_loss(pc, pr, torch.tensor([-10.0]), torch.tensor([-10.0]), beta=0.5)
    loss.backward()
    assert pc.grad.item() < 0 < pr.grad.item(), (
        "descending the loss increases log pi(chosen) and decreases log pi(rejected)"
    )
    assert pc.grad.item() == pytest.approx(-pr.grad.item()) and pc.grad.item() == pytest.approx(
        -0.5 * 0.5
    )


def test_the_loss_falls_as_the_margin_grows_and_a_larger_beta_saturates_sooner():
    ref = torch.zeros(1)
    losses = [
        D.dpo_loss(torch.tensor([float(g)]), torch.tensor([0.0]), ref, ref, 0.1)[0].item()
        for g in (0, 5, 10, 30)
    ]
    assert losses == sorted(losses, reverse=True)
    assert (
        D.dpo_loss(torch.tensor([10.0]), torch.tensor([0.0]), ref, ref, 1.0)[0].item()
        < D.dpo_loss(torch.tensor([10.0]), torch.tensor([0.0]), ref, ref, 0.1)[0].item()
    )


def test_a_pair_the_policy_ranks_the_wrong_way_has_loss_above_ln_two_and_reward_accuracy_zero():
    z = torch.zeros(1)
    loss, m = D.dpo_loss(torch.tensor([-5.0]), torch.tensor([5.0]), z, z, 0.1)
    assert loss.item() > math.log(2) and m["reward_accuracy"] == 0.0


# ----------------------------------------------------------------------------- ORPO and GRPO


def test_orpo_is_the_sft_loss_plus_an_odds_ratio_term_and_needs_no_reference():
    lc, lr = torch.tensor([4 * math.log(0.5)]), torch.tensor([4 * math.log(0.25)])
    n = torch.tensor([4.0])
    loss, parts = D.orpo_loss(lc, lr, n, n, lam=0.5)
    assert parts["sft"] == pytest.approx(-math.log(0.5))
    # log odds: chosen p = 0.5 -> log(1) = 0 ; rejected p = 0.25 -> log(1/3)
    assert parts["odds_ratio_term"] == pytest.approx(
        -math.log(1 / (1 + math.exp(-(0 - math.log(1 / 3))))), abs=1e-5
    )
    assert loss.item() == pytest.approx(parts["sft"] + 0.5 * parts["odds_ratio_term"])
    worse, _ = D.orpo_loss(lr, lc, n, n)
    better, _ = D.orpo_loss(lc, lr, n, n)
    assert better < worse


def test_group_advantages_are_standardised_within_the_group():
    a = D.group_advantages(torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert (
        a.mean().abs() < 1e-6
        and a.std(unbiased=False) == pytest.approx(1.0, abs=1e-4)
        and a[0] > 0 > a[1]
    )
    assert torch.equal(D.group_advantages(torch.tensor([0.5, 0.5, 0.5])), torch.zeros(3)), (
        "an all-equal group has no signal"
    )
    r = torch.tensor([3.0, 1.0, 2.0])
    assert torch.allclose(D.group_advantages(r)[[2, 0, 1]], D.group_advantages(r[[2, 0, 1]]))


def test_the_clipped_objective_stops_pushing_once_the_ratio_leaves_the_trust_region():
    adv = torch.tensor([1.0])
    inside = torch.tensor([0.05], requires_grad=True)
    D.grpo_loss(inside, torch.zeros(1), adv).backward()
    assert inside.grad.abs().item() > 0
    outside = torch.tensor(
        [0.5], requires_grad=True
    )  # ratio e^0.5 = 1.65 > 1.2 with a positive advantage: clipped, no gradient
    D.grpo_loss(outside, torch.zeros(1), adv).backward()
    assert outside.grad.item() == 0
    neg = torch.tensor(
        [-0.5], requires_grad=True
    )  # ratio 0.61 < 0.8 with a NEGATIVE advantage: clipped
    D.grpo_loss(neg, torch.zeros(1), -adv).backward()
    assert neg.grad.item() == 0
    assert D.grpo_loss(
        torch.zeros(3), torch.zeros(3), torch.tensor([1.0, -1.0, 0.0])
    ).item() == pytest.approx(0.0)


# ----------------------------------------------------------------------------- the training loop


def test_reference_log_probabilities_match_the_per_sequence_values_and_start_at_ln_two():
    m = tiny()
    ps = pairs(6)
    ref = D.reference_logprobs(m, ps, 0, batch=4)
    for (c, r), p in zip(ref, ps, strict=True):
        assert c == pytest.approx(
            D.sequence_logprobs(m, C.collate([p.chosen], 0))[0].item(), abs=1e-4
        )
        assert r == pytest.approx(
            D.sequence_logprobs(m, C.collate([p.rejected], 0))[0].item(), abs=1e-4
        )
    lp = D.sequence_logprobs(m, D._stack(ps[:4], 0))
    loss, _ = D.dpo_loss(
        lp[:4], lp[4:], torch.tensor([r[0] for r in ref[:4]]), torch.tensor([r[1] for r in ref[:4]])
    )
    assert loss.item() == pytest.approx(math.log(2), abs=1e-4)


def test_dpo_training_raises_the_chosen_answers_relative_to_the_rejected_and_leaves_the_base_alone():
    m = tiny(d=48)
    L.add_lora(m, r=8, alpha=16)
    before = {n: p.detach().clone() for n, p in m.named_parameters() if "lora_" not in n}
    ps = pairs(16)
    ref = D.reference_logprobs(m, ps, 0)
    run = D.train_dpo(
        m, ps, ref, D.DPOConfig(epochs=6, batch=4, lr=3e-3, beta=0.2, warmup=2, log_every=4), 0
    )
    assert (
        run.loss[-1] < math.log(2) * 0.6 and run.reward_accuracy[-1] >= 0.9 and run.margin[-1] > 0.3
    )
    assert all(torch.equal(before[n], p) for n, p in m.named_parameters() if "lora_" not in n)
    lp = D.sequence_logprobs(m, D._stack(ps, 0))
    new_margin = (lp[: len(ps)] - torch.tensor([r[0] for r in ref])) - (
        lp[len(ps) :] - torch.tensor([r[1] for r in ref])
    )
    assert (new_margin > 0).float().mean() > 0.85


def test_the_first_training_step_starts_at_ln_two_and_clips_the_gradient(monkeypatch):
    m = tiny(d=48)
    L.add_lora(m, r=4, alpha=8)
    ps = pairs(8, seed=3)
    ref = D.reference_logprobs(m, ps, 0)
    clips = []
    real = torch.nn.utils.clip_grad_norm_
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda params, max_norm, *a, **k: clips.append(max_norm) or real(params, max_norm, *a, **k),
    )
    run = D.train_dpo(
        m,
        ps,
        ref,
        D.DPOConfig(epochs=1, batch=4, lr=1e-3, beta=0.1, warmup=1, clip=0.9, log_every=1),
        0,
    )
    assert run.loss[0] == pytest.approx(math.log(2), abs=1e-3), (
        "policy == reference at step 1, so the loss is ln 2 (with the reference columns swapped it would not be)"
    )
    assert clips == [0.9, 0.9]


def test_training_needs_something_trainable():
    m = tiny()
    for p in m.parameters():
        p.requires_grad_(False)
    ps = pairs(2)
    with pytest.raises(ValueError, match="nothing to train"):
        D.train_dpo(m, ps, [(0.0, 0.0)] * 2, D.DPOConfig(), 0)


def test_encode_pair_shares_the_prompt_and_rejects_identical_answers():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(I.BASE)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(str(exc))
    prompt = I.tuned_messages("Order A-1 please: 2 lamps. Sam")
    p = D.encode_pair(prompt, '{"a":1}', '{"a":2}', tok)
    n = p.chosen.labels.index(next(t for t in p.chosen.labels if t != C.IGNORE))
    assert (
        p.chosen.input_ids[:n] == p.rejected.input_ids[:n]
        and p.chosen.input_ids != p.rejected.input_ids
    )
    assert tok.decode([t for t in p.rejected.labels if t != C.IGNORE]) == '{"a":2}<|im_end|>'
    with pytest.raises(ValueError, match="identical"):
        D.encode_pair(prompt, "x", "x", tok)


def test_the_nll_term_keeps_the_chosen_answers_likelihood_from_falling():
    def run(nll):
        m = tiny(d=48)
        L.add_lora(m, r=8, alpha=16)
        ps = pairs(16, seed=5)
        ref = D.reference_logprobs(m, ps, 0)
        D.train_dpo(
            m,
            ps,
            ref,
            D.DPOConfig(
                epochs=8, batch=4, lr=1e-2, beta=0.1, nll_weight=nll, warmup=2, log_every=100
            ),
            0,
        )
        lp = D.sequence_logprobs(m.eval(), D._stack(ps, 0))
        return (lp[:16] - torch.tensor([r[0] for r in ref])).mean().item()

    assert run(1.0) > run(0.0), (
        "with the NLL term the chosen answers end up more likely than without it"
    )
