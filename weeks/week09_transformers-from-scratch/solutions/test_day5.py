"""Tests for Week 9 Day 5: the data pipeline (no leakage between training and validation), the training machinery (schedule, optimiser groups, evaluation),
the baselines, sampling, and that the whole thing can actually fit a small corpus."""

from __future__ import annotations

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
import day5_solution as d5  # noqa: E402
from bpe import BPE, END_OF_TEXT  # noqa: E402

_RNG = random.Random(11)
DOCS = [
    (
        f"doc{i}.md",
        f"Document number {i}. "
        + ("the cat sat on the mat and the dog sat on the log. " * (3 + i % 4))
        + (("".join(_RNG.choice("abcdefghij") for _ in range(6)) + " ") * 25)
        + f" unique-{i}-end",
    )
    for i in range(20)
]

# ----------------------------------------------------------------------------- the data


def test_the_split_is_by_document_disjoint_deterministic_and_never_empty():
    train, val = d5.split_documents(DOCS, 0.1)
    assert len(val) == 2 and len(train) == 18 and not {p for p, _ in train} & {p for p, _ in val}
    assert (train, val) == d5.split_documents(DOCS, 0.1)
    assert d5.split_documents(DOCS, 0.1, seed=1)[1] != val
    assert len(d5.split_documents(DOCS[:3], 0.01)[1]) == 1, (
        "even a tiny corpus gets a validation document"
    )


def test_the_repository_corpus_covers_the_lessons_and_source_but_not_tests_or_later_weeks():
    docs = d5.repo_documents()
    paths = [p for p, _ in docs]
    assert paths == sorted(paths, key=paths.index) and len(set(paths)) == len(paths)
    assert any(p.endswith(".md") and "week01" in p for p in paths) and any(
        p.startswith("common/") for p in paths
    )
    assert not any("test_" in Path(p).name for p in paths) and not any(
        "week09" in p or "week1" in p.split("/")[1] for p in paths if p.startswith("weeks")
    )
    assert sum(len(t) for _, t in docs) > 1_500_000


def test_validation_text_is_not_in_the_training_stream_and_round_trips():
    data = d5.build_data(300, docs=DOCS)
    assert data.n_docs == (18, 2) and data.tok.vocab_size <= 300
    val_text = data.tok.decode(data.val_ids.tolist())
    train_text = data.tok.decode(data.train_ids.tolist())
    val_markers = {w for w in val_text.split() if w.startswith("unique-")}
    train_markers = {w for w in train_text.split() if w.startswith("unique-")}
    assert len(val_markers) == 2 and len(train_markers) == 18 and not val_markers & train_markers
    assert data.train_ids.dtype == torch.long and END_OF_TEXT in val_text, (
        "documents are separated by the end-of-text token"
    )


def test_the_tokenizer_is_trained_on_training_documents_only():
    data = d5.build_data(300, docs=DOCS)
    train, _ = d5.split_documents(DOCS)

    def learn(docs):
        return BPE.train(
            END_OF_TEXT.join(t for _, t in docs).replace(END_OF_TEXT, "\n\n"),
            300,
            specials=[END_OF_TEXT],
        )

    assert data.tok.ranks == learn(train).ranks, (
        "exactly what training on the training documents alone gives"
    )
    assert data.tok.ranks != learn(DOCS).ranks, "and not what training on everything would give"


def test_batches_are_next_token_windows():
    ids = torch.arange(1000)
    x, y = d5.get_batch(ids, 6, 16, torch.Generator().manual_seed(0))
    assert (
        x.shape == y.shape == (6, 16)
        and torch.equal(y[:, :-1], x[:, 1:])
        and torch.equal(y[:, -1], x[:, -1] + 1)
    )
    x2, _ = d5.get_batch(ids, 6, 16, torch.Generator().manual_seed(0))
    assert torch.equal(x, x2) and x.max() < 1000 and y.max() <= 999


# ----------------------------------------------------------------------------- the schedule and the optimiser


def test_the_learning_rate_warms_up_then_decays_to_the_floor():
    c = d5.TrainConfig(steps=1000, lr=1e-2, warmup=100, min_lr_fraction=0.1)
    assert d5.lr_at(0, c) == pytest.approx(1e-4) and d5.lr_at(99, c) == pytest.approx(1e-2)
    assert d5.lr_at(100, c) == pytest.approx(1e-2, rel=1e-3)
    mid = d5.lr_at(550, c)
    assert mid == pytest.approx(1e-2 * (0.1 + 0.9 * 0.5), rel=1e-2)
    assert d5.lr_at(999, c) == pytest.approx(1e-3, rel=1e-2)
    series = [d5.lr_at(s, c) for s in range(100, 1000)]
    assert all(a >= b for a, b in zip(series, series[1:], strict=False))


def test_weight_decay_applies_to_weight_matrices_only():
    m = B.Decoder(B.Config(vocab_size=50, d_model=16, n_layers=2, n_heads=2))
    opt = d5.make_optimizer(m, 1e-3, 0.1)
    decayed = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    named = dict(m.named_parameters())
    assert (
        id(named["embed_tokens.weight"]) not in decayed and id(named["norm.weight"]) not in decayed
    )
    assert id(named["layers.0.input_layernorm.weight"]) not in decayed
    assert (
        id(named["layers.0.mlp.down_proj.weight"]) in decayed
        and id(named["layers.1.self_attn.q_proj.weight"]) in decayed
    )
    assert sum(len(g["params"]) for g in opt.param_groups) == len(named), (
        "every parameter is in exactly one group"
    )


def test_the_optimiser_uses_adamw_with_the_documented_betas():
    opt = d5.make_optimizer(tiny_model(), 1e-3, 0.1)
    assert isinstance(opt, torch.optim.AdamW) and opt.defaults["betas"] == (0.9, 0.95)


def test_every_training_step_clips_the_gradient_norm(monkeypatch):
    calls = []
    real = torch.nn.utils.clip_grad_norm_
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda params, max_norm, *a, **k: calls.append(max_norm) or real(params, max_norm, *a, **k),
    )
    data = d5.build_data(300, docs=DOCS)
    d5.train(
        tiny_model(data.tok.vocab_size),
        data,
        d5.TrainConfig(
            steps=7, batch=4, seq_len=16, warmup=2, eval_every=7, eval_tokens=500, clip=0.5
        ),
    )
    assert calls == [0.5] * 7


# ----------------------------------------------------------------------------- evaluation


def tiny_model(vocab: int = 40, seed: int = 0, **kw) -> B.Decoder:
    torch.manual_seed(seed)
    return B.Decoder(
        B.Config(
            vocab_size=vocab,
            d_model=kw.get("d", 32),
            n_layers=kw.get("layers", 2),
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=64,
        )
    )


def test_evaluate_is_the_mean_cross_entropy_over_non_overlapping_windows():
    m = tiny_model()
    ids = torch.randint(0, 40, (200,), generator=torch.Generator().manual_seed(1))
    got = d5.evaluate(m, ids, 16)
    x = ids[:192].view(-1, 16)
    y = ids[1:193].view(-1, 16)
    with torch.no_grad():
        m.eval()
        want = F.cross_entropy(m(x).reshape(-1, 40), y.reshape(-1)).item()
    assert got == pytest.approx(want, abs=1e-5) and d5.evaluate(m, ids, 16) == got
    assert d5.evaluate(m, ids, 16, max_tokens=64) != got, "max_tokens limits how much is scored"
    assert m.training, "evaluate puts the model back in training mode"


def test_a_fresh_model_scores_ln_vocabulary():
    m = tiny_model(vocab=256)
    ids = torch.randint(0, 256, (2000,), generator=torch.Generator().manual_seed(2))
    assert d5.evaluate(m, ids, 32) == pytest.approx(math.log(256), abs=0.1)


def test_bits_per_char_by_hand():
    assert d5.bits_per_char(math.log(2), tokens=100, chars=100) == pytest.approx(1.0)
    assert d5.bits_per_char(math.log(2), tokens=100, chars=200) == pytest.approx(0.5)


# ----------------------------------------------------------------------------- training


def test_one_fixed_batch_can_be_memorised():
    """The Day 1 sanity check, for the transformer: a model that cannot memorise one batch has a bug."""
    m = tiny_model(vocab=30, d=64, layers=2)
    x = torch.randint(0, 30, (4, 24), generator=torch.Generator().manual_seed(3))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(250):
        loss = F.cross_entropy(m(x)[:, :-1].reshape(-1, 30), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_training_lowers_validation_loss_records_the_best_checkpoint_and_is_reproducible():
    data = d5.build_data(300, docs=DOCS)
    c = d5.TrainConfig(
        steps=60, batch=8, seq_len=32, lr=5e-3, warmup=10, eval_every=20, eval_tokens=2000
    )
    run1 = d5.train(tiny_model(data.tok.vocab_size, d=48), data, c)
    run2 = d5.train(tiny_model(data.tok.vocab_size, d=48), data, c)
    assert run1.steps == [20, 40, 60] and len(run1.train_loss) == len(run1.val_loss) == 3
    assert run1.start_loss == pytest.approx(math.log(data.tok.vocab_size), abs=0.3)
    assert run1.val_loss[-1] < run1.start_loss - 1.0 and run1.train_loss == run2.train_loss
    assert (
        run1.best_val == min(run1.val_loss)
        and run1.best_step == run1.steps[run1.val_loss.index(run1.best_val)]
        and run1.best_state is not None
    )


def test_the_best_state_is_a_snapshot_not_a_reference_to_the_live_weights():
    data = d5.build_data(300, docs=DOCS)
    m = tiny_model(data.tok.vocab_size, d=48)
    run = d5.train(
        m,
        data,
        d5.TrainConfig(
            steps=40, batch=8, seq_len=32, lr=5e-3, warmup=5, eval_every=10, eval_tokens=2000
        ),
    )
    snap = run.best_state["layers.0.mlp.down_proj.weight"].clone()
    with torch.no_grad():
        m.layers[0].mlp.down_proj.weight.add_(1.0)
    assert torch.equal(run.best_state["layers.0.mlp.down_proj.weight"], snap)


def test_gradient_clipping_bounds_the_update_norm():
    m = tiny_model()
    x = torch.randint(0, 40, (4, 16))
    F.cross_entropy(m(x).reshape(-1, 40), x.reshape(-1)).backward()
    norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e-3)
    assert norm > 1e-3
    assert math.sqrt(
        sum(p.grad.pow(2).sum().item() for p in m.parameters() if p.grad is not None)
    ) == pytest.approx(1e-3, rel=1e-2)


# ----------------------------------------------------------------------------- baselines


def test_count_baselines_on_a_stream_with_known_entropy():
    tok = BPE.train("ab" * 50, 258)
    uniform = torch.tensor([i % 4 for i in range(4000)])
    cyc = d5.Data(tok, uniform, uniform, 1, 1, (1, 1))
    b = d5.count_baselines(cyc, k=0.01)
    assert b["unigram"] == pytest.approx(math.log(4), abs=0.05), "four equally likely tokens"
    assert b["bigram"] < 0.1, (
        "a deterministic cycle is perfectly predictable from the previous token"
    )


def test_smoothing_keeps_the_bigram_loss_finite_on_unseen_pairs():
    tok = BPE.train("ab" * 50, 258)
    train = torch.tensor([0, 1] * 200)
    val = torch.tensor([5, 6, 7, 5, 6, 7] * 10)
    b = d5.count_baselines(d5.Data(tok, train, val, 1, 1, (1, 1)))
    assert math.isfinite(b["bigram"]) and math.isfinite(b["unigram"]) and b["bigram"] > 2


# ----------------------------------------------------------------------------- sampling


class Fixed(nn.Module):
    """A 'model' whose logits are the same at every position: lets sampling behaviour be tested exactly."""

    def __init__(self, logits: list[float]):
        super().__init__()
        self.cfg = B.Config(
            vocab_size=len(logits), d_model=8, n_layers=1, n_heads=2, max_seq_len=64
        )
        self.logits = torch.tensor(logits)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(*ids.shape, -1)


def counts(model, n=3000, **kw):
    tok = BPE.train("abcd" * 20, 256)
    out = []
    for seed in range(n):
        text = d5.generate(model, tok, "a", 1, seed=seed, **kw)
        out.append(tok.encode(text)[1])
    return torch.bincount(torch.tensor(out), minlength=4)[:4].float()


def test_temperature_zero_is_greedy_and_top_k_one_is_greedy_too():
    m = Fixed([0.0, 1.0, 3.0, 2.0])
    tok = BPE.train("abcd" * 20, 256)
    assert tok.encode(d5.generate(m, tok, "a", 5, temperature=0))[1:] == [2] * 5
    assert tok.encode(d5.generate(m, tok, "a", 5, temperature=1.0, top_k=1, seed=9))[1:] == [2] * 5


def test_sampling_follows_the_softmax_of_the_logits_and_top_k_removes_the_tail():
    m = Fixed([0.0, 1.0, 2.0, 3.0])
    c = counts(m, temperature=1.0)
    p = F.softmax(torch.tensor([0.0, 1.0, 2.0, 3.0]), -1)
    assert torch.allclose(c / c.sum(), p, atol=0.03)
    k2 = counts(m, temperature=1.0, top_k=2)
    assert (
        k2[0] == 0
        and k2[1] == 0
        and k2[3] / k2.sum() == pytest.approx(math.e / (1 + math.e), abs=0.03)
    )


def test_lower_temperature_sharpens_and_higher_flattens():
    m = Fixed([0.0, 1.0, 2.0, 3.0])
    cold, hot = counts(m, temperature=0.3), counts(m, temperature=3.0)
    assert cold[3] / cold.sum() > 0.95 and hot[3] / hot.sum() < 0.45


def test_generation_is_reproducible_for_a_seed_and_starts_with_the_prompt():
    tok = BPE.train("the cat sat on the mat " * 30, 300)
    m = tiny_model(tok.vocab_size)
    a = d5.generate(m, tok, "the cat", 12, temperature=0.9, seed=4)
    assert a == d5.generate(m, tok, "the cat", 12, temperature=0.9, seed=4) and a.startswith(
        "the cat"
    )
    assert a != d5.generate(m, tok, "the cat", 12, temperature=0.9, seed=5)
    assert len(tok.encode(a)) >= len(tok.encode("the cat")) + 12 - 1


def test_an_empty_prompt_starts_from_the_end_of_text_token_or_is_refused():
    with_special = BPE.train("the cat sat " * 30, 300, specials=[END_OF_TEXT])
    m = tiny_model(with_special.vocab_size)
    assert isinstance(d5.generate(m, with_special, "", 5, temperature=0), str)
    without = BPE.train("the cat sat " * 30, 300)
    with pytest.raises(ValueError, match="end-of-text"):
        d5.generate(tiny_model(without.vocab_size), without, "", 5)


# ----------------------------------------------------------------------------- the analysis helpers


def test_loss_by_kind_scores_prose_and_code_separately_on_validation_documents_only():
    docs = [
        (f"d{i}.md" if i % 2 else f"d{i}.py", f"line {i} of text {'abc ' * 40}") for i in range(30)
    ]
    data = d5.build_data(300, docs=docs)
    m = tiny_model(data.tok.vocab_size)
    out = d5.loss_by_kind(m, data, docs, seq_len=16)
    assert set(out) == {"lessons (.md)", "source code (.py)"}
    _, val = d5.split_documents(docs)
    for kind, suffix in (("lessons (.md)", ".md"), ("source code (.py)", ".py")):
        n_expected = sum(1 for p, _ in val if p.endswith(suffix))
        assert (out[kind][1] > 0) == (n_expected > 0)
        if n_expected:
            ln_v = math.log(data.tok.vocab_size)
            assert ln_v - 1.0 < out[kind][0] < ln_v + 0.5, (
                "an untrained model scores about ln(vocabulary), a little less on repetitive text (tied embeddings favour repeating the input token)"
            )


def test_loss_by_position_has_one_entry_per_bucket_and_the_mean_matches_the_overall_loss():
    m = tiny_model(vocab=40)
    ids = torch.randint(0, 40, (800,), generator=torch.Generator().manual_seed(4))
    rows = d5.loss_by_position(m, ids, seq_len=16, buckets=(1, 4, 16))
    assert [name for name, _ in rows] == ["positions 0..0", "positions 1..3", "positions 4..15"]
    weights = [1, 3, 12]
    mean = sum(w * v for w, (_, v) in zip(weights, rows, strict=True)) / 16
    assert mean == pytest.approx(d5.evaluate(m, ids, 16), abs=1e-4)
