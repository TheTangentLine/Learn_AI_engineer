"""Tests for Week 9 Day 1: the data, the hand-written backward pass (and that the checkers can fail), the training loop, the sanity checks, the curve reader."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))

import day1_solution as d1  # noqa: E402

# ----------------------------------------------------------------------------- data


def test_spirals_are_deterministic_balanced_and_not_linearly_separable():
    x, y = d1.make_spirals(50, seed=3)
    x2, _ = d1.make_spirals(50, seed=3)
    assert torch.equal(x, x2) and not torch.equal(x, d1.make_spirals(50, seed=4)[0])
    assert x.shape == (150, 2) and x.dtype == torch.float32 and y.dtype == torch.int64
    assert torch.bincount(y).tolist() == [50, 50, 50]
    # a linear classifier (one Linear layer, trained to convergence) cannot solve it: that is why the task needs a hidden layer
    lin = nn.Linear(2, 3)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    for _ in range(500):
        loss = nn.functional.cross_entropy(lin(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert (lin(x).argmax(1) == y).float().mean() < 0.7


def test_split_is_shuffled_disjoint_and_the_right_size():
    x, y = d1.make_spirals(40)
    xt, yt, xv, yv = d1.split(x, y, 0.25)
    assert len(xt) == 90 and len(xv) == 30
    assert set(torch.bincount(yv).nonzero().flatten().tolist()) == {0, 1, 2}, (
        "the validation set must contain every class"
    )
    rows = {tuple(r) for r in xt.tolist()}
    assert not rows & {tuple(r) for r in xv.tolist()}


# ----------------------------------------------------------------------------- the hand-written network


@pytest.fixture
def batch():
    x, y = d1.make_spirals(20, seed=1)
    return x.numpy().astype(np.float64), y.numpy()


def test_backward_matches_autograd_to_machine_precision(batch):
    x, y = batch
    net = d1.NumpyMLP(2, 8, 3, seed=2)
    z, cache = net.forward(x)
    _, dz = net.loss_and_dlogits(z, y)
    mine, ref = net.backward(cache, dz), d1.autograd_gradients(net, x, y)
    for k in mine:
        assert np.abs(mine[k] - ref[k]).max() < 1e-12, k


def test_backward_matches_finite_differences(batch):
    x, y = batch
    assert d1.finite_difference_check(d1.NumpyMLP(2, 8, 3, seed=2), x, y) < 1e-5


def test_the_gradient_checkers_can_fail(batch, monkeypatch):
    """A check that cannot fail proves nothing: break the backward pass (forget the tanh derivative) and both checkers must notice."""
    x, y = batch
    net = d1.NumpyMLP(2, 8, 3, seed=2)

    def broken(self, cache, dz2):
        h, xx = cache["h"], cache["x"]
        dz1 = dz2 @ self.W2.T  # missing: * (1 - h**2)
        return {"W2": h.T @ dz2, "b2": dz2.sum(0), "W1": xx.T @ dz1, "b1": dz1.sum(0)}

    monkeypatch.setattr(d1.NumpyMLP, "backward", broken)
    z, cache = net.forward(x)
    _, dz = net.loss_and_dlogits(z, y)
    ref = d1.autograd_gradients(net, x, y)
    assert np.abs(net.backward(cache, dz)["W1"] - ref["W1"]).max() > 1e-3
    assert d1.finite_difference_check(net, x, y) > 1e-3


def test_softmax_cross_entropy_does_not_overflow_on_huge_logits():
    loss, dz = d1.NumpyMLP.loss_and_dlogits(
        np.array([[1000.0, 0.0, -1000.0], [0.0, 1000.0, 0.0]]), np.array([0, 1])
    )
    assert math.isfinite(loss) and np.isfinite(dz).all() and loss < 1e-6
    with np.errstate(over="ignore"):
        naive = np.exp(np.array([[1000.0, 0.0]]))
    assert np.isinf(naive).any(), "this is what the log-sum-exp trick avoids"


def test_the_loss_gradient_rows_sum_to_zero_because_probabilities_sum_to_one(batch):
    x, y = batch
    net = d1.NumpyMLP(2, 8, 3, seed=2)
    _, dz = net.loss_and_dlogits(net.forward(x)[0], y)
    assert np.allclose(dz.sum(axis=1), 0, atol=1e-12)


def test_numpy_and_pytorch_follow_the_same_loss_curve_from_the_same_weights(batch):
    x, y = batch
    a = d1.numpy_fit(d1.NumpyMLP(2, 8, 3, seed=5), x, y, lr=0.3, steps=60)
    b = d1.torch_fit(d1.torch_twin(d1.NumpyMLP(2, 8, 3, seed=5)), x, y, lr=0.3, steps=60)
    assert max(abs(p - q) for p, q in zip(a, b, strict=True)) < 1e-10
    assert a[-1] < a[0] * 0.8


# ----------------------------------------------------------------------------- the training loop and the sanity checks


def test_the_starting_loss_is_ln_of_the_number_of_classes():
    torch.manual_seed(0)
    x, y = d1.make_spirals(100)
    assert d1.initial_loss(d1.TorchMLP(2, 16, 3), x, y) == pytest.approx(math.log(3), abs=0.1)


def test_one_small_batch_can_be_memorised():
    assert d1.overfit_one_batch() < 0.01


def test_gradients_accumulate_unless_zeroed():
    lin = nn.Linear(1, 1, bias=False)
    x = torch.ones(1, 1)
    for _ in range(2):
        lin(x).sum().backward()
    assert lin.weight.grad.item() == pytest.approx(2.0), (
        "two backward calls without zero_grad add up"
    )
    lin.zero_grad()
    lin(x).sum().backward()
    assert lin.weight.grad.item() == pytest.approx(1.0)


def test_training_reaches_high_accuracy_and_the_history_has_one_entry_per_epoch():
    torch.manual_seed(0)
    x, y = d1.make_spirals(100)
    xt, yt, xv, yv = d1.split(x, y)
    h = d1.train(
        d1.TorchMLP(2, 32, 3, depth=2), xt, yt, xv, yv, lr=0.01, epochs=150, optimizer="adam"
    )
    assert len(h.train_loss) == len(h.val_loss) == len(h.val_acc) == 150
    assert h.val_acc[-1] >= 0.95 and h.train_loss[-1] < h.train_loss[0] / 10 and not h.diverged


def test_a_far_too_large_learning_rate_diverges_and_a_tiny_one_barely_moves():
    x, y = d1.make_spirals(100)
    xt, yt, xv, yv = d1.split(x, y)
    torch.manual_seed(0)
    big = d1.train(d1.TorchMLP(2, 16, 3), xt, yt, xv, yv, lr=10.0, epochs=100)
    torch.manual_seed(0)
    tiny = d1.train(d1.TorchMLP(2, 16, 3), xt, yt, xv, yv, lr=0.001, epochs=100)
    assert big.train_loss[-1] > 3 * math.log(3)
    assert tiny.train_loss[0] - tiny.train_loss[-1] < 0.2


def test_training_is_reproducible_for_a_seed():
    x, y = d1.make_spirals(60)
    xt, yt, xv, yv = d1.split(x, y)
    runs = []
    for _ in range(2):
        torch.manual_seed(7)
        runs.append(d1.train(d1.TorchMLP(2, 8, 3), xt, yt, xv, yv, lr=0.1, epochs=5).train_loss)
    assert runs[0] == runs[1]


# ----------------------------------------------------------------------------- reading a curve


@pytest.mark.parametrize(
    ("train", "val", "kwargs", "expected"),
    [
        ([1.0 * 0.95**i for i in range(60)], None, {}, "improving"),
        ([0.5] * 60, None, {}, "stalled"),
        ([0.5 + 0.0001 * i for i in range(60)], None, {}, "stalled"),
        ([1.0 * 0.997**i for i in range(60)], None, {}, "slowly improving"),
        ([1.0 + 0.05 * i for i in range(60)], None, {}, "getting worse"),
        (
            [1.0 * 0.97**i for i in range(60)],
            [1.0 * 0.97**i if i < 20 else 0.35 + 0.01 * i for i in range(60)],
            {},
            "overfitting",
        ),
        ([1.0] + [30.0] * 59, None, {"chance": 1.1}, "diverging"),
        ([1.0, 2.0, float("nan")] + [float("nan")] * 30, None, {}, "diverging"),
        ([0.5 * 0.9**i for i in range(80)], None, {}, "converged"),
        ([1.0, 0.9, 0.8], None, {}, "too short to say"),
    ],
)
def test_describe_curve_names_each_regime(train, val, kwargs, expected):
    assert d1.describe_curve(train, val, **kwargs) == expected


def test_the_sparkline_has_the_requested_width_and_follows_the_curve():
    line = d1.sparkline([float(i) for i in range(200)], width=20)
    assert len(line) == 20 and line[0] == "▁" and line[-1] == "█"
    assert d1.sparkline([1.0] * 10, width=5) == "▁" * 5
