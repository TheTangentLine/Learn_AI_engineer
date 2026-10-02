"""Week 9 Day 1 - Solution: PyTorch essentials by training an MLP from scratch, then reading its loss curve.

1. DATA      three interleaved spirals: not linearly separable, so a hidden layer is needed; generated from a seed (no downloads)
2. BY HAND   a NumPy MLP with a hand-written backward pass; its gradients are checked against PyTorch autograd AND finite differences
3. PYTORCH   the same network as an nn.Module; trained with the same data order, the two should follow the same loss curve
4. SANITY    the checks that catch most bugs before any tuning: the starting loss is ln(classes), one tiny batch can be memorised
5. READ      learning-rate sweep, and a classifier that names what a loss curve is doing (falling / plateau / diverging / overfitting)

  uv run python weeks/week09_transformers-from-scratch/solutions/day1_solution.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

# ----------------------------------------------------------------------------- 1. data


def make_spirals(
    n_per_class: int = 100, classes: int = 3, noise: float = 0.2, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """``classes`` interleaved spiral arms in 2-D. Deterministic for a seed."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(classes):
        t = np.linspace(0.1, 1.0, n_per_class)
        angle = t * 4.0 + c * (2 * math.pi / classes) + rng.normal(0, noise, n_per_class) * 0.5
        r = t
        xs.append(np.stack([r * np.sin(angle), r * np.cos(angle)], axis=1))
        ys.append(np.full(n_per_class, c))
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def split(x: torch.Tensor, y: torch.Tensor, val_fraction: float = 0.25, seed: int = 0):
    """A shuffled train / validation split (the data come ordered by class, so never split without shuffling)."""
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(x), generator=g)
    n_val = int(len(x) * val_fraction)
    return x[order[n_val:]], y[order[n_val:]], x[order[:n_val]], y[order[:n_val]]


# ----------------------------------------------------------------------------- 2. an MLP written by hand (NumPy, manual backward)


class NumpyMLP:
    """x -> Linear -> tanh -> Linear -> softmax cross-entropy. Everything PyTorch would do for you, written out:

        z1 = x W1 + b1          h = tanh(z1)         z2 = h W2 + b2          loss = mean_i( -log softmax(z2)_i[y_i] )

    The backward pass is the chain rule applied once per line, from the loss back to the weights:

        dz2 = (softmax(z2) - onehot(y)) / N          dW2 = h^T dz2        db2 = sum dz2
        dh  = dz2 W2^T                               dz1 = dh * (1 - h^2)       dW1 = x^T dz1       db1 = sum dz1
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / math.sqrt(n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 1 / math.sqrt(n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def params(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        z1 = x @ self.W1 + self.b1
        h = np.tanh(z1)
        z2 = h @ self.W2 + self.b2
        return z2, {"x": x, "h": h}

    @staticmethod
    def loss_and_dlogits(z2: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        shifted = z2 - z2.max(
            axis=1, keepdims=True
        )  # the log-sum-exp trick: softmax is unchanged, exp() cannot overflow
        logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        n = len(y)
        loss = float(-logp[np.arange(n), y].mean())
        dz2 = np.exp(logp)
        dz2[np.arange(n), y] -= 1.0
        return loss, dz2 / n

    def backward(self, cache: dict, dz2: np.ndarray) -> dict[str, np.ndarray]:
        h, x = cache["h"], cache["x"]
        dh = dz2 @ self.W2.T
        dz1 = dh * (1.0 - h**2)
        return {"W2": h.T @ dz2, "b2": dz2.sum(axis=0), "W1": x.T @ dz1, "b1": dz1.sum(axis=0)}

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.loss_and_dlogits(self.forward(x)[0], y)[0]


def finite_difference_check(
    net: NumpyMLP,
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-6,
    per_param: int = 5,
    seed: int = 0,
) -> float:
    """Largest relative error between the analytic gradient and (loss(w+eps) - loss(w-eps)) / 2eps on a few random weights of every parameter."""
    z2, cache = net.forward(x)
    _, dz2 = net.loss_and_dlogits(z2, y)
    grads = net.backward(cache, dz2)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for name, p in net.params().items():
        flat = p.reshape(-1)
        for idx in rng.choice(flat.size, size=min(per_param, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + eps
            up = net.loss(x, y)
            flat[idx] = old - eps
            down = net.loss(x, y)
            flat[idx] = old
            numeric = (up - down) / (2 * eps)
            analytic = grads[name].reshape(-1)[idx]
            worst = max(worst, abs(numeric - analytic) / max(1e-8, abs(numeric) + abs(analytic)))
    return worst


def autograd_gradients(net: NumpyMLP, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """The same network in PyTorch with the same weights, in float64, to compare against the hand-written backward pass."""
    w = {
        k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in net.params().items()
    }
    h = torch.tanh(torch.tensor(x, dtype=torch.float64) @ w["W1"] + w["b1"])
    loss = nn.functional.cross_entropy(h @ w["W2"] + w["b2"], torch.tensor(y))
    loss.backward()
    return {k: v.grad.numpy() for k, v in w.items()}


def numpy_fit(net: NumpyMLP, x: np.ndarray, y: np.ndarray, *, lr: float, steps: int) -> list[float]:
    """Full-batch gradient descent with the hand-written backward pass; returns the loss before every step."""
    losses = []
    for _ in range(steps):
        z2, cache = net.forward(x)
        loss, dz2 = net.loss_and_dlogits(z2, y)
        losses.append(loss)
        for name, g in net.backward(cache, dz2).items():
            net.params()[name] -= lr * g  # in place: params() returns the arrays themselves
    return losses


# ----------------------------------------------------------------------------- 3. the same thing in PyTorch


class TorchMLP(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int, depth: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_in
        for _ in range(depth):
            layers += [nn.Linear(width, n_hidden), nn.Tanh()]
            width = n_hidden
        layers.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def torch_twin(net: NumpyMLP) -> TorchMLP:
    """A float64 nn.Module holding exactly the NumPy network's weights (nn.Linear stores W transposed)."""
    model = TorchMLP(net.W1.shape[0], net.W1.shape[1], net.W2.shape[1]).double()
    with torch.no_grad():
        for layer, (W, b) in zip(
            (model.net[0], model.net[2]), ((net.W1, net.b1), (net.W2, net.b2)), strict=True
        ):
            layer.weight.copy_(torch.tensor(W.T))
            layer.bias.copy_(torch.tensor(b))
    return model


def torch_fit(
    model: nn.Module, x: np.ndarray, y: np.ndarray, *, lr: float, steps: int
) -> list[float]:
    xt, yt = torch.tensor(x, dtype=torch.float64), torch.tensor(y)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        loss = nn.functional.cross_entropy(model(xt), yt)
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return losses


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)

    @property
    def diverged(self) -> bool:
        return any(not math.isfinite(v) for v in self.train_loss)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    logits = model(x)
    return nn.functional.cross_entropy(logits, y).item(), (
        logits.argmax(1) == y
    ).float().mean().item()


def train(
    model: nn.Module,
    x,
    y,
    xv,
    yv,
    *,
    lr: float = 0.1,
    epochs: int = 200,
    batch: int = 32,
    seed: int = 0,
    optimizer: str = "sgd",
) -> History:
    """The loop every PyTorch training script is a variation of: batch -> forward -> loss -> zero_grad -> backward -> step."""
    opt = (
        torch.optim.SGD(model.parameters(), lr=lr)
        if optimizer == "sgd"
        else torch.optim.Adam(model.parameters(), lr=lr)
    )
    g = torch.Generator().manual_seed(seed)
    hist = History()
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x), generator=g)
        total, n = 0.0, 0
        for i in range(0, len(x), batch):
            idx = order[i : i + batch]
            loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad()  # gradients ACCUMULATE: forgetting this is the classic silent bug
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            n += len(idx)
        hist.train_loss.append(total / n)
        vl, va = evaluate(model, xv, yv)
        hist.val_loss.append(vl)
        hist.val_acc.append(va)
        if not math.isfinite(total):
            break
    return hist


# ----------------------------------------------------------------------------- 4. sanity checks


def initial_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    return evaluate(model, x, y)[0]


def overfit_one_batch(seed: int = 0, n: int = 12, steps: int = 400, lr: float = 0.05) -> float:
    """A network that cannot memorise 12 points has a bug (or a learning rate that is far too small), whatever the data."""
    torch.manual_seed(seed)
    x, y = make_spirals(n // 3, seed=seed)
    model = TorchMLP(2, 32, 3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


# ----------------------------------------------------------------------------- 5. reading a loss curve


def describe_curve(
    train: list[float],
    val: list[float] | None = None,
    *,
    window: int = 10,
    chance: float | None = None,
    converged: float = 0.02,
) -> str:
    """Name what a loss curve is doing, from its last ``window`` epochs.

    diverging         non-finite, or ended far above where it started, or above twice the chance-level loss (``chance`` = ln(classes))
    converged         the loss is essentially zero
    overfitting       training loss still falls while validation loss rises
    improving         still falling at more than 5% per window
    stalled           flat: either stuck OR crawling because the learning rate is tiny. The curve cannot tell you which: raise the rate 10x
    slowly improving  falling, but under 5% per window
    """
    if (
        any(not math.isfinite(v) for v in train)
        or (len(train) > 5 and train[-1] > 5 * train[0])
        or (chance is not None and train[-1] > 2 * chance)
    ):
        return "diverging"
    if len(train) < 2 * window:
        return "too short to say"
    if float(np.mean(train[-window:])) < converged:
        return "converged"
    first, last = float(np.mean(train[-2 * window : -window])), float(np.mean(train[-window:]))
    gain = (first - last) / max(first, 1e-9)
    if val is not None and len(val) >= 2 * window:
        v_first, v_last = float(np.mean(val[-2 * window : -window])), float(np.mean(val[-window:]))
        if gain > 0.01 and v_last > v_first * 1.02:
            return "overfitting"
    if gain > 0.05:
        return "improving"
    if abs(gain) <= 0.01:
        return "stalled"
    if gain < -0.01:
        return "getting worse"
    return "slowly improving"


def sparkline(values: list[float], width: int = 40) -> str:
    """A loss curve in one line (min to max scaled), for reading curves in a terminal."""
    bars = "▁▂▃▄▅▆▇█"
    step = max(1, len(values) // width)
    pts = [float(np.mean(values[i : i + step])) for i in range(0, len(values), step)]
    lo, hi = min(pts), max(pts)
    return "".join(
        bars[min(len(bars) - 1, int((v - lo) / (hi - lo + 1e-12) * len(bars)))] for v in pts
    )


def main(argv: list[str]) -> None:
    x, y = make_spirals(100, seed=0)
    xt, yt, xv, yv = split(x, y)
    print(
        f"data: {len(xt)} train / {len(xv)} validation points, 3 classes; majority-class baseline = {1 / 3:.0%}\n"
    )

    net = NumpyMLP(2, 16, 3, seed=1)
    z, cache = net.forward(xt.numpy())
    _, dz = net.loss_and_dlogits(z, yt.numpy())
    mine, ref = net.backward(cache, dz), autograd_gradients(net, xt.numpy(), yt.numpy())
    print(
        "hand-written backward vs autograd (max abs difference per tensor): "
        + ", ".join(f"{k}={np.abs(mine[k] - ref[k]).max():.1e}" for k in mine)
    )
    print(
        f"hand-written backward vs finite differences (worst relative error): {finite_difference_check(net, xt.numpy(), yt.numpy()):.1e}\n"
    )

    x64 = xt.numpy().astype(np.float64)
    a = numpy_fit(NumpyMLP(2, 16, 3, seed=1), x64, yt.numpy(), lr=0.5, steps=100)
    b = torch_fit(torch_twin(NumpyMLP(2, 16, 3, seed=1)), x64, yt.numpy(), lr=0.5, steps=100)
    gap = max(abs(p - q) for p, q in zip(a, b, strict=True))
    print(
        f"NumPy by hand vs PyTorch, same weights, 100 full-batch SGD steps: loss {a[0]:.4f} -> {a[-1]:.4f} vs {b[0]:.4f} -> {b[-1]:.4f}; largest difference {gap:.1e}\n"
    )

    torch.manual_seed(0)
    model = TorchMLP(2, 16, 3)
    print(
        f"initial loss {initial_loss(model, xt, yt):.3f}   (ln 3 = {math.log(3):.3f}: a network that knows nothing predicts uniformly)"
    )
    print(f"loss after 400 steps on 12 points: {overfit_one_batch():.4f}   (must be near 0)\n")

    print(f"{'lr':>7}{'final train':>13}{'final val':>11}{'val acc':>9}   what the curve does")
    for lr in (0.001, 0.01, 0.1, 1.0, 10.0):
        torch.manual_seed(0)
        h = train(TorchMLP(2, 16, 3), xt, yt, xv, yv, lr=lr, epochs=200)
        print(
            f"{lr:>7}{h.train_loss[-1]:>13.3f}{h.val_loss[-1]:>11.3f}{h.val_acc[-1]:>9.0%}   {describe_curve(h.train_loss, h.val_loss, chance=math.log(3)):<17}{sparkline(h.train_loss)}"
        )

    torch.manual_seed(0)
    h = train(TorchMLP(2, 32, 3, depth=2), xt, yt, xv, yv, lr=0.01, epochs=300, optimizer="adam")
    print("\nAdam, two hidden layers of 32, lr 0.01:")
    for e in (0, 5, 20, 50, 100, 200, 299):
        print(
            f"  epoch {e:>3}: train {h.train_loss[e]:.3f}   val {h.val_loss[e]:.3f}   val acc {h.val_acc[e]:.0%}"
        )
    print(
        f"  -> {describe_curve(h.train_loss, h.val_loss, chance=math.log(3))}   train {sparkline(h.train_loss)}"
    )


if __name__ == "__main__":
    main(sys.argv)
