"""Week 9 Day 5 - Solution: train a mini-GPT on your own corpus, and say honestly what it learned.

DATA     this repository: the lessons of weeks 1-8 and the Python source of weeks 1-8 and ``common/`` (about 1.9 million characters), split BY DOCUMENT
         (every file is entirely in training or entirely in validation, so a memorised paragraph cannot flatter the validation loss)
TOKENS   the Day 2 byte-level BPE trained on the training documents only
MODEL    the Day 4 decoder (RMSNorm, RoPE, SwiGLU, grouped-query attention), a few hundred thousand to a million parameters, on the CPU
TRAIN    AdamW, linear warm-up then cosine decay, gradient clipping, weight decay on matrices only; validation loss every few steps
COMPARE  a unigram and a bigram model of the same tokens: a neural network that cannot beat counting has not learned anything

  uv run python weeks/week09_transformers-from-scratch/solutions/day5_solution.py [--steps N] [--save PATH]
"""

from __future__ import annotations

import glob
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import blocks as B  # noqa: E402
from bpe import BPE, END_OF_TEXT  # noqa: E402

# ----------------------------------------------------------------------------- data


def repo_documents(root: Path = ROOT, last_week: int = 8) -> list[tuple[str, str]]:
    """(path, text) for the lessons and Python source of weeks 1..last_week and common/: deterministic order, no tests, no generated files."""
    paths: list[str] = []
    for w in range(1, last_week + 1):
        paths += sorted(glob.glob(str(root / f"weeks/week{w:02d}_*/day*.md")))
        for p in sorted(
            glob.glob(str(root / f"weeks/week{w:02d}_*/solutions/**/*.py"), recursive=True)
        ):
            if "test_" not in os.path.basename(p) and "__pycache__" not in p:
                paths.append(p)
    paths += sorted(glob.glob(str(root / "common/*.py")))
    return [(os.path.relpath(p, root), Path(p).read_text(errors="replace")) for p in paths]


def split_documents(docs: list[tuple[str, str]], val_fraction: float = 0.1, seed: int = 0):
    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(len(docs) * val_fraction))
    val = {order[i] for i in range(n_val)}
    return [d for i, d in enumerate(docs) if i not in val], [
        d for i, d in enumerate(docs) if i in val
    ]


@dataclass
class Data:
    tok: BPE
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    train_chars: int
    val_chars: int
    n_docs: tuple[int, int]


def build_data(
    vocab_size: int = 1024, seed: int = 0, docs: list[tuple[str, str]] | None = None
) -> Data:
    docs = docs if docs is not None else repo_documents()
    train_docs, val_docs = split_documents(docs, seed=seed)
    joined = lambda ds: (END_OF_TEXT).join(t for _, t in ds)  # noqa: E731 - one stream, documents separated by an end-of-text token
    train_text, val_text = joined(train_docs), joined(val_docs)
    tok = BPE.train(train_text.replace(END_OF_TEXT, "\n\n"), vocab_size, specials=[END_OF_TEXT])
    enc = lambda t: torch.tensor(tok.encode(t, allowed_special={END_OF_TEXT}), dtype=torch.long)  # noqa: E731
    return Data(
        tok,
        enc(train_text),
        enc(val_text),
        len(train_text),
        len(val_text),
        (len(train_docs), len(val_docs)),
    )


def get_batch(
    ids: torch.Tensor, batch: int, seq_len: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """``batch`` random windows of seq_len+1 tokens: inputs are the first seq_len, targets the same window shifted by one."""
    starts = torch.randint(0, len(ids) - seq_len - 1, (batch,), generator=gen)
    window = torch.stack([ids[s : s + seq_len + 1] for s in starts])
    return window[:, :-1], window[:, 1:]


# ----------------------------------------------------------------------------- training


@dataclass
class TrainConfig:
    steps: int = 1500
    batch: int = 32
    seq_len: int = 128
    lr: float = 3e-3
    min_lr_fraction: float = 0.1
    warmup: int = 100
    weight_decay: float = 0.1
    clip: float = 1.0
    eval_every: int = 100
    eval_tokens: int = 40000
    seed: int = 0


def lr_at(step: int, c: TrainConfig) -> float:
    """Linear warm-up to ``lr``, then cosine decay to ``min_lr_fraction * lr``."""
    if step < c.warmup:
        return c.lr * (step + 1) / c.warmup
    progress = (step - c.warmup) / max(1, c.steps - c.warmup)
    return c.lr * (
        c.min_lr_fraction + (1 - c.min_lr_fraction) * 0.5 * (1 + math.cos(math.pi * progress))
    )


def make_optimizer(model: B.Decoder, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """Weight decay on the 2-D weight matrices only: not on norm scales, and not on the embedding table (which is also the output matrix)."""
    decay = [p for n, p in model.named_parameters() if p.dim() == 2 and "embed_tokens" not in n]
    other = [
        p for n, p in model.named_parameters() if not (p.dim() == 2 and "embed_tokens" not in n)
    ]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": other, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
    )


@torch.no_grad()
def evaluate(
    model: B.Decoder, ids: torch.Tensor, seq_len: int, max_tokens: int | None = None
) -> float:
    """Mean next-token cross-entropy (nats per token) over consecutive, non-overlapping windows of ``ids``: the same windows every time."""
    model.eval()
    n = (len(ids) - 1) // seq_len * seq_len
    if max_tokens:
        n = min(n, max_tokens // seq_len * seq_len)
    x = ids[:n].view(-1, seq_len)
    y = ids[1 : n + 1].view(-1, seq_len)
    total, count = 0.0, 0
    for i in range(0, len(x), 64):
        logits = model(x[i : i + 64])
        total += F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y[i : i + 64].reshape(-1), reduction="sum"
        ).item()
        count += y[i : i + 64].numel()
    model.train()
    return total / count


@dataclass
class Run:
    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(
        default_factory=list
    )  # mean over the steps since the last evaluation
    val_loss: list[float] = field(default_factory=list)
    best_val: float = float("inf")
    best_step: int = 0
    best_state: dict | None = None
    seconds: float = 0.0
    start_loss: float = 0.0


def train(model: B.Decoder, data: Data, c: TrainConfig, *, log=None) -> Run:
    torch.manual_seed(c.seed)
    gen = torch.Generator().manual_seed(c.seed)
    opt = make_optimizer(model, c.lr, c.weight_decay)
    run = Run(start_loss=evaluate(model, data.val_ids, c.seq_len, c.eval_tokens))
    model.train()
    t0, window = time.time(), []
    for step in range(c.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, c)
        x, y = get_batch(data.train_ids, c.batch, c.seq_len, gen)
        loss = F.cross_entropy(model(x).reshape(-1, model.cfg.vocab_size), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip)
        opt.step()
        window.append(loss.item())
        if (step + 1) % c.eval_every == 0 or step + 1 == c.steps:
            v = evaluate(model, data.val_ids, c.seq_len, c.eval_tokens)
            run.steps.append(step + 1)
            run.train_loss.append(sum(window) / len(window))
            run.val_loss.append(v)
            window = []
            if v < run.best_val:
                run.best_val, run.best_step = v, step + 1
                run.best_state = {k: t.clone() for k, t in model.state_dict().items()}
            if log:
                log(
                    f"step {step + 1:>5}  lr {lr_at(step, c):.1e}  train {run.train_loss[-1]:.3f}  val {v:.3f}  ({time.time() - t0:.0f}s)"
                )
    run.seconds = time.time() - t0
    return run


# ----------------------------------------------------------------------------- baselines and metrics


def count_baselines(data: Data, k: float = 0.1) -> dict[str, float]:
    """Cross-entropy (nats/token) on the validation tokens of two models that only COUNT: unigram frequencies, and bigrams with add-k smoothing."""
    v = data.tok.vocab_size
    train, val = data.train_ids.tolist(), data.val_ids.tolist()
    uni = Counter(train)
    n = len(train)
    unigram = -sum(math.log((uni[t] + k) / (n + k * v)) for t in val[1:]) / (len(val) - 1)
    bi: dict[int, Counter] = {}
    for a, b in zip(train, train[1:], strict=False):
        bi.setdefault(a, Counter())[b] += 1
    total = 0.0
    for a, b in zip(val, val[1:], strict=False):
        row = bi.get(a)
        c_ab, c_a = (row[b], sum(row.values())) if row else (0, 0)
        total -= math.log((c_ab + k) / (c_a + k * v))
    return {"unigram": unigram, "bigram": total / (len(val) - 1)}


def bits_per_char(loss_nats_per_token: float, tokens: int, chars: int) -> float:
    """The same quality in a unit that does not depend on the tokenizer: bits needed per character of text."""
    return loss_nats_per_token * tokens / chars / math.log(2)


# ----------------------------------------------------------------------------- sampling


@torch.no_grad()
def generate(
    model: B.Decoder,
    tok: BPE,
    prompt: str,
    max_new_tokens: int = 80,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 0,
) -> str:
    """Autoregressive sampling, recomputing the whole prefix at each step (the KV cache that removes this cost is Day 7). temperature 0 = greedy."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    start = tok.encode(prompt)
    if not start:  # nothing to condition on: begin a new document, as training did
        if END_OF_TEXT not in tok.specials:
            raise ValueError(
                "an empty prompt needs a tokenizer with an end-of-text token to start from"
            )
        start = [tok.specials[END_OF_TEXT]]
    ids = torch.tensor([start], dtype=torch.long)
    for _ in range(max_new_tokens):
        logits = model(ids[:, -model.cfg.max_seq_len :])[0, -1]
        if temperature == 0:
            nxt = logits.argmax()
        else:
            logits = logits / temperature
            if top_k:
                logits = logits.masked_fill(
                    logits < torch.topk(logits, top_k).values[-1], float("-inf")
                )
            nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=g)[0]
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return tok.decode(ids[0].tolist())


def loss_by_kind(
    model: B.Decoder, data: Data, docs: list[tuple[str, str]], seq_len: int = 128
) -> dict[str, tuple[float, int]]:
    """Validation loss per kind of document (prose lessons against Python source): (nats per token, tokens). A single average hides which the model is better at."""
    _, val_docs = split_documents(docs)
    out = {}
    for kind, suffix in (("lessons (.md)", ".md"), ("source code (.py)", ".py")):
        text = END_OF_TEXT.join(t for p, t in val_docs if p.endswith(suffix))
        ids = torch.tensor(data.tok.encode(text, allowed_special={END_OF_TEXT}), dtype=torch.long)
        out[kind] = (evaluate(model, ids, seq_len), len(ids))
    return out


def loss_by_position(
    model: B.Decoder, ids: torch.Tensor, seq_len: int = 128, buckets=(1, 8, 32, 128)
) -> list[tuple[str, float]]:
    """Loss at each position inside the window: the first tokens have almost no context to predict from, so they are the hardest."""
    model.eval()
    n = (len(ids) - 1) // seq_len * seq_len
    x, y = ids[:n].view(-1, seq_len), ids[1 : n + 1].view(-1, seq_len)
    with torch.no_grad():
        per = torch.cat(
            [
                F.cross_entropy(
                    model(x[i : i + 64]).transpose(1, 2), y[i : i + 64], reduction="none"
                )
                for i in range(0, len(x), 64)
            ]
        ).mean(0)
    model.train()
    edges = (0, *buckets)
    return [
        (f"positions {a}..{b - 1}", per[a:b].mean().item())
        for a, b in zip(edges, edges[1:], strict=False)
    ]


def analyse(path: Path) -> None:
    ckpt = torch.load(path, weights_only=False)
    model = B.Decoder(ckpt["cfg"])
    model.load_state_dict(ckpt["state"])
    tok = BPE.load(path.with_suffix(".bpe.json"))
    docs = repo_documents()
    data = build_data(tok.vocab_size, docs=docs)
    for kind, (loss, n) in loss_by_kind(model, data, docs).items():
        print(
            f"validation loss on {kind}: {loss:.3f} nats/token ({math.exp(loss):.1f} perplexity) over {n:,} tokens"
        )
    print(
        "loss by position inside the 128-token window: "
        + "; ".join(f"{name} {v:.2f}" for name, v in loss_by_position(model, data.val_ids))
    )
    for t in (0.0, 0.5, 1.0, 1.5):
        print(
            f"\n--- temperature {t}\n"
            + generate(model, tok, "The retry limit is", 40, temperature=t, seed=3)
        )


def make_model(
    vocab_size: int,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    max_seq_len: int = 128,
) -> B.Decoder:
    d_ff = 8 * round(8 * d_model / 3 / 8)
    return B.Decoder(
        B.Config(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            rope_theta=10000.0,
        )
    )


def main(argv: list[str]) -> None:
    if "analyse" in argv:
        return analyse(Path(argv[argv.index("analyse") + 1]))
    steps = int(argv[argv.index("--steps") + 1]) if "--steps" in argv else 1500
    save = Path(argv[argv.index("--save") + 1]) if "--save" in argv else None
    torch.manual_seed(0)
    data = build_data(1024)
    print(
        f"corpus: {sum(data.n_docs)} documents ({data.n_docs[0]} train / {data.n_docs[1]} validation), {data.train_chars:,} + {data.val_chars:,} characters"
    )
    print(
        f"tokenizer: {data.tok.vocab_size} tokens -> {len(data.train_ids):,} training tokens ({data.train_chars / len(data.train_ids):.2f} chars/token), {len(data.val_ids):,} validation tokens\n"
    )
    base = count_baselines(data)
    ln_v = math.log(data.tok.vocab_size)
    print(
        f"reference points (nats per validation token): uniform ln V = {ln_v:.3f}   unigram counts {base['unigram']:.3f}   bigram counts {base['bigram']:.3f}"
    )
    model = make_model(data.tok.vocab_size)
    c = TrainConfig(steps=steps)
    print(
        f"model: {model.num_parameters():,} parameters ({model.cfg.n_layers} layers, width {model.cfg.d_model}, {model.cfg.n_heads} heads / {model.cfg.n_kv_heads} kv); "
        f"{steps} steps x {c.batch} x {c.seq_len} = {steps * c.batch * c.seq_len / 1e6:.1f}M tokens = {steps * c.batch * c.seq_len / len(data.train_ids):.1f} epochs\n"
    )
    run = train(model, data, c, log=print)
    model.load_state_dict(run.best_state)
    final_val = evaluate(model, data.val_ids, c.seq_len)
    vtoks = len(data.val_ids)
    print(
        f"\nstart {run.start_loss:.3f} (ln V = {ln_v:.3f}); best validation {run.best_val:.3f} at step {run.best_step}; "
        f"full validation set with that checkpoint {final_val:.3f} = {math.exp(final_val):.1f} perplexity = {bits_per_char(final_val, vtoks, data.val_chars):.2f} bits/char; {run.seconds:.0f}s"
    )
    for prompt in ("## Learning objectives\n", "def ", "The attention weights"):
        print(
            f"\n--- prompt {prompt!r}, temperature 0.8, top-k 40\n"
            + generate(model, data.tok, prompt, 70, temperature=0.8, top_k=40, seed=1)
        )
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": run.best_state, "cfg": model.cfg, "tokenizer": None}, save)
        data.tok.save(save.with_suffix(".bpe.json"))
        print(f"\nsaved {save} and {save.with_suffix('.bpe.json')}")


if __name__ == "__main__":
    main(sys.argv)
