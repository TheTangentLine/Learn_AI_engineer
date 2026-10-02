"""Supervised fine-tuning with LoRA: batches of chat examples, loss on the answer tokens only, AdamW with warm-up and cosine decay.

    cfg = SFTConfig(epochs=3, batch=8, lr=2e-4)
    run = train_sft(model, examples, dev_examples, cfg, on_epoch=save_adapters)

The loss is the mean over ALL answer tokens in the batch (not the mean of per-example means), so a long answer counts for more tokens, as in standard SFT.
The vocabulary-sized output projection is applied only at positions that have a label.
"""

from __future__ import annotations

import math
import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "weeks/week09_transformers-from-scratch/solutions"))

import blocks as B  # noqa: E402
import chatfmt as C  # noqa: E402


@dataclass
class SFTConfig:
    epochs: int = 3
    batch: int = 8
    lr: float = 2e-4
    min_lr_fraction: float = 0.1
    warmup: int = 20
    weight_decay: float = 0.0
    clip: float = 1.0
    seed: int = 0
    log_every: int = 10
    bucket: int = 64  # sort within windows of this many examples so a batch has similar lengths (less padding)


def lr_at(step: int, total: int, cfg: SFTConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, total - cfg.warmup)
    return cfg.lr * (
        cfg.min_lr_fraction + (1 - cfg.min_lr_fraction) * 0.5 * (1 + math.cos(math.pi * progress))
    )


def make_batches(
    examples: list[C.Example], batch: int, rng: random.Random, bucket: int = 64
) -> list[list[C.Example]]:
    """Shuffle, then sort by length inside buckets and cut into batches, then shuffle the batches: random composition, little padding."""
    order = list(range(len(examples)))
    rng.shuffle(order)
    batches = []
    for i in range(0, len(order), bucket):
        chunk = sorted(order[i : i + bucket], key=lambda k: len(examples[k]))
        batches += [
            [examples[k] for k in chunk[j : j + batch]] for j in range(0, len(chunk), batch)
        ]
    rng.shuffle(batches)
    return batches


def answer_loss(model: B.Decoder, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int, int]:
    """(summed cross-entropy over answer tokens, number of answer tokens, number correct by argmax). Position t predicts token t+1, so the label of
    position t is labels[t+1]; the output projection is applied only where that label is not -100."""
    ids, labels = batch["input_ids"], batch["labels"]
    hidden = model.hidden_states(ids)[:, :-1]
    target = labels[:, 1:]
    mask = target != C.IGNORE
    logits = model.lm_head(hidden[mask])
    tgt = target[mask]
    loss = F.cross_entropy(logits, tgt, reduction="sum")
    return loss, int(mask.sum()), int((logits.argmax(-1) == tgt).sum())


@torch.no_grad()
def evaluate_loss(
    model: B.Decoder, examples: list[C.Example], pad_id: int, batch: int = 16
) -> dict[str, float]:
    model.eval()
    total, n, correct = 0.0, 0, 0
    for i in range(0, len(examples), batch):
        loss, k, c = answer_loss(model, C.collate(examples[i : i + batch], pad_id))
        total, n, correct = total + loss.item(), n + k, correct + c
    model.train()
    return {"loss": total / n, "token_accuracy": correct / n, "tokens": n}


@dataclass
class SFTRun:
    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    epoch_dev: list[dict[str, float]] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)
    seconds: float = 0.0
    tokens_seen: int = 0


def train_sft(
    model: B.Decoder,
    train: list[C.Example],
    dev: list[C.Example],
    cfg: SFTConfig,
    pad_id: int,
    *,
    log: Callable[[str], None] | None = None,
    on_epoch: Callable[[int, SFTRun], None] | None = None,
) -> SFTRun:
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("nothing to train: add_lora() first (or unfreeze something)")
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(train) / cfg.batch)
    total = cfg.epochs * steps_per_epoch
    run = SFTRun()
    run.epoch_dev.append(
        evaluate_loss(model, dev, pad_id)
    )  # before any training: the base model's loss on the task
    if log:
        log(
            f"start: dev loss {run.epoch_dev[0]['loss']:.3f}, token accuracy {run.epoch_dev[0]['token_accuracy']:.1%}"
        )
    model.train()
    t0, step, window, win_tokens = time.time(), 0, 0.0, 0
    for epoch in range(cfg.epochs):
        for batch in make_batches(train, cfg.batch, rng, cfg.bucket):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, cfg)
            loss, n, _ = answer_loss(model, C.collate(batch, pad_id))
            (loss / n).backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            window, win_tokens = window + loss.item(), win_tokens + n
            run.tokens_seen += sum(len(e) for e in batch)
            if step % cfg.log_every == 0 or step == total:
                run.steps.append(step)
                run.train_loss.append(window / win_tokens)
                run.lrs.append(lr_at(step - 1, total, cfg))
                window, win_tokens = 0.0, 0
                if log:
                    log(
                        f"epoch {epoch + 1} step {step:>4}/{total}  lr {run.lrs[-1]:.1e}  train loss {run.train_loss[-1]:.4f}  ({time.time() - t0:.0f}s)"
                    )
        run.epoch_dev.append(evaluate_loss(model, dev, pad_id))
        run.seconds = time.time() - t0
        if log:
            d = run.epoch_dev[-1]
            log(
                f"== epoch {epoch + 1}: dev loss {d['loss']:.4f}, token accuracy {d['token_accuracy']:.2%}"
            )
        if on_epoch:
            on_epoch(epoch + 1, run)
    run.seconds = time.time() - t0
    return run
