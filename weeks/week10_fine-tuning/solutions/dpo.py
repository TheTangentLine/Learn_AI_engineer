"""Preference tuning from scratch: DPO, plus the loss formulas of ORPO and GRPO.

SFT teaches a model to imitate one good answer. Preference tuning teaches it to PREFER a good answer over a bad one, which is how you push down specific
mistakes the model actually makes. The data are triples (prompt, chosen, rejected).

DPO (Rafailov et al., 2023) needs no reward model and no sampling loop. With pi the model being trained and ref a frozen copy (the SFT model):

    loss = - log sigmoid( beta * [ (log pi(chosen) - log ref(chosen))  -  (log pi(rejected) - log ref(rejected)) ] )

where log pi(y) is the total log-probability of the answer tokens given the prompt. beta * (log pi - log ref) is the "implicit reward" of an answer. The loss pushes
the chosen answer's implicit reward above the rejected one's; beta controls how far the model may drift from ref. At pi = ref the loss is exactly ln 2.
"""

from __future__ import annotations

import math
import random
import sys
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
class Pair:
    chosen: C.Example
    rejected: C.Example


def encode_pair(
    prompt_messages: list[dict], chosen: str, rejected: str, tok, max_len: int = 512
) -> Pair:
    """Two training examples that share a prompt and differ in the assistant's answer."""
    if chosen == rejected:
        raise ValueError("chosen and rejected are identical")
    return Pair(
        C.encode_example(
            [*prompt_messages, {"role": "assistant", "content": chosen}], tok, max_len
        ),
        C.encode_example(
            [*prompt_messages, {"role": "assistant", "content": rejected}], tok, max_len
        ),
    )


def sequence_logprobs(model: B.Decoder, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Total log-probability of the labelled (answer) tokens of each sequence in the batch: shape (batch,). The output projection is applied only where labelled."""
    ids, labels = batch["input_ids"], batch["labels"]
    hidden = model.hidden_states(ids)[:, :-1]
    target = labels[:, 1:]
    mask = target != C.IGNORE
    logp = F.log_softmax(model.lm_head(hidden[mask]).float(), dim=-1).gather(
        -1, target[mask][:, None]
    )[:, 0]
    out = torch.zeros(ids.shape[0], dtype=logp.dtype)
    rows = torch.arange(ids.shape[0])[:, None].expand_as(target)[mask]
    return out.index_add(0, rows, logp)


def dpo_loss(
    pol_chosen: torch.Tensor,
    pol_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The DPO loss and diagnostics: the reward accuracy (how often the chosen answer's implicit reward is higher) and the mean margin."""
    chosen_reward = beta * (pol_chosen - ref_chosen)
    rejected_reward = beta * (pol_rejected - ref_rejected)
    margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(margin).mean()
    return loss, {
        "reward_accuracy": (margin > 0).float().mean().item(),
        "margin": margin.mean().item(),
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
    }


def orpo_loss(
    pol_chosen_logp: torch.Tensor,
    pol_rejected_logp: torch.Tensor,
    n_chosen: torch.Tensor,
    n_rejected: torch.Tensor,
    lam: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """ORPO (Hong et al., 2024): SFT loss on the chosen answer plus an odds-ratio term; NO reference model. With p = the per-token geometric-mean
    probability of an answer (exp of the mean log-prob), odds(p) = p / (1 - p), the extra term is -log sigmoid(log odds(chosen) - log odds(rejected))."""
    mc, mr = pol_chosen_logp / n_chosen, pol_rejected_logp / n_rejected
    log_odds = lambda m: m - torch.log1p(-torch.exp(m).clamp(max=1 - 1e-6))  # noqa: E731 - log(p / (1 - p)) from log p
    sft = -mc.mean()
    ratio = -F.logsigmoid(log_odds(mc) - log_odds(mr)).mean()
    return sft + lam * ratio, {"sft": sft.item(), "odds_ratio_term": ratio.item()}


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """GRPO's baseline: for G sampled answers to ONE prompt, the advantage of each is its reward standardised within the group: (r - mean) / (std + eps).
    No value network: the group is its own baseline. A group whose answers all got the same reward carries no signal (advantage 0)."""
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)


def grpo_loss(
    logp_new: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor, clip: float = 0.2
) -> torch.Tensor:
    """The clipped policy-gradient objective of PPO/GRPO for per-answer log-probabilities: -mean(min(ratio * A, clip(ratio, 1-c, 1+c) * A))."""
    ratio = torch.exp(logp_new - logp_old)
    return -torch.minimum(
        ratio * advantages, torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
    ).mean()


# ----------------------------------------------------------------------------- training


@dataclass
class DPOConfig:
    epochs: int = 1
    batch: int = 4
    lr: float = 5e-5
    beta: float = 0.1
    nll_weight: float = 0.0  # >0 adds the SFT loss on the chosen answer (per token): a guard against the chosen answer's likelihood falling
    warmup: int = 5
    clip: float = 1.0
    seed: int = 0
    log_every: int = 10


@dataclass
class DPORun:
    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    reward_accuracy: list[float] = field(default_factory=list)
    margin: list[float] = field(default_factory=list)
    seconds: float = 0.0


def _stack(batch: list[Pair], pad_id: int) -> dict[str, torch.Tensor]:
    """Chosen and rejected sequences of every pair in ONE padded batch (chosen first, then rejected)."""
    return C.collate([p.chosen for p in batch] + [p.rejected for p in batch], pad_id)


@torch.no_grad()
def reference_logprobs(
    model: B.Decoder, pairs: list[Pair], pad_id: int, batch: int = 8
) -> list[tuple[float, float]]:
    """Log-probabilities of every pair's answers under the frozen reference: computed ONCE before training, when the policy still equals the reference."""
    model.eval()
    out = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i : i + batch]
        lp = sequence_logprobs(model, _stack(chunk, pad_id))
        out += list(zip(lp[: len(chunk)].tolist(), lp[len(chunk) :].tolist(), strict=True))
    model.train()
    return out


def train_dpo(
    model: B.Decoder,
    pairs: list[Pair],
    ref: list[tuple[float, float]],
    cfg: DPOConfig,
    pad_id: int,
    *,
    log: Callable[[str], None] | None = None,
) -> DPORun:
    import time

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("nothing to train: add_lora() first")
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    idx = list(range(len(pairs)))
    total = cfg.epochs * math.ceil(len(pairs) / cfg.batch)
    run, step, t0 = DPORun(), 0, time.time()
    model.train()
    acc: list[dict[str, float]] = []
    for _ in range(cfg.epochs):
        rng.shuffle(idx)
        for i in range(0, len(idx), cfg.batch):
            chunk = [pairs[k] for k in idx[i : i + cfg.batch]]
            r = torch.tensor([ref[k] for k in idx[i : i + cfg.batch]])
            for g in opt.param_groups:
                g["lr"] = (
                    cfg.lr
                    * min(1.0, (step + 1) / cfg.warmup)
                    * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(1, total))))
                )
            lp = sequence_logprobs(model, _stack(chunk, pad_id))
            loss, m = dpo_loss(lp[: len(chunk)], lp[len(chunk) :], r[:, 0], r[:, 1], cfg.beta)
            if cfg.nll_weight:
                n_tok = torch.tensor([float(p.chosen.n_answer) for p in chunk])
                loss = loss + cfg.nll_weight * (-lp[: len(chunk)] / n_tok).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            acc.append({"loss": loss.item(), **m})
            if step % cfg.log_every == 0 or step == total:
                run.steps.append(step)
                run.loss.append(sum(a["loss"] for a in acc) / len(acc))
                run.reward_accuracy.append(sum(a["reward_accuracy"] for a in acc) / len(acc))
                run.margin.append(sum(a["margin"] for a in acc) / len(acc))
                acc = []
                if log:
                    log(
                        f"step {step:>4}/{total}  loss {run.loss[-1]:.4f}  reward accuracy {run.reward_accuracy[-1]:.0%}  margin {run.margin[-1]:+.3f}  ({time.time() - t0:.0f}s)"
                    )
    run.seconds = time.time() - t0
    return run
