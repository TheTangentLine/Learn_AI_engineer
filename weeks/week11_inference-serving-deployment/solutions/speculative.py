"""Speculative decoding from scratch, on the Week 9 decoder: a small DRAFT model proposes ``k`` tokens, the large TARGET model checks all of them in ONE forward pass.

Why it works: generating a token with a big model is limited by reading its weights from memory, not by arithmetic, so checking k tokens in parallel costs about as much as
generating one. If the draft guesses right most of the time the target emits several tokens per weight-read. The output is EXACTLY what the target would have produced
(greedy: the same tokens; sampling: the same distribution), so there is no quality trade-off, only a speed one that depends on how often the draft is right.

Rejection sampling (Leviathan et al., Chen et al., 2023): for each drafted token x with draft probability q(x) and target probability p(x), accept with probability
min(1, p(x)/q(x)); on the first rejection, sample the replacement from the residual distribution max(0, p - q) normalised; if every token is accepted, sample one BONUS token
from the target's distribution after the last one. ``temperature == 0`` is the greedy special case: accept while the draft equals the target's argmax.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

W9 = Path(__file__).resolve().parents[3] / "weeks/week09_transformers-from-scratch/solutions"
sys.path.append(str(W9))
sys.path.append(str(W9 / "weekly/fastgen"))

import blocks as B  # noqa: E402
from kvcache import KVCache, forward_cached  # noqa: E402


def truncate(cache: KVCache, length: int) -> None:
    """Forget everything after position ``length`` (the rejected drafts). The preallocated buffers are simply overwritten by later appends."""
    if length > cache.length:
        raise ValueError("cannot truncate to a longer length")
    for layer in cache.layers:
        layer.length = length


@dataclass
class SpecStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    tokens: list[int] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_target_forward(self) -> float:
        return len(self.tokens) / self.target_forwards if self.target_forwards else 0.0


def probs_of(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return torch.zeros_like(logits).scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
    return torch.softmax(logits.float() / temperature, dim=-1)


@torch.no_grad()
def speculative_generate(
    target: B.Decoder,
    draft: B.Decoder,
    prompt: list[int],
    max_new_tokens: int,
    *,
    k: int = 4,
    temperature: float = 0.0,
    stop_ids: set[int] | frozenset[int] = frozenset(),
    seed: int = 0,
) -> SpecStats:
    """Generate up to ``max_new_tokens`` tokens. Both models must share a vocabulary."""
    if target.cfg.vocab_size != draft.cfg.vocab_size:
        raise ValueError("draft and target must share a vocabulary")
    gen = torch.Generator().manual_seed(seed)
    total = len(prompt) + max_new_tokens + k + 1
    tc, dc = KVCache.for_model(target, 1, total), KVCache.for_model(draft, 1, total)
    st = SpecStats()
    t0 = time.perf_counter()
    ids = torch.tensor([prompt])
    t_logits = forward_cached(target, ids, tc)[:, -1]
    forward_cached(draft, ids, dc)
    st.target_forwards += 1
    st.draft_forwards += 1
    # the first token comes from the target's own prefill distribution
    p0 = probs_of(t_logits, temperature)
    first = int(p0.argmax(-1)) if temperature == 0 else int(torch.multinomial(p0, 1, generator=gen))
    st.tokens.append(first)
    last = first
    while len(st.tokens) < max_new_tokens and last not in stop_ids:
        n_draft = min(k, max_new_tokens - len(st.tokens))
        # 1. the draft proposes n_draft tokens autoregressively (remembering its distributions)
        cur, q_list, drafted = last, [], []
        for _ in range(n_draft):
            d_logits = forward_cached(draft, torch.tensor([[cur]]), dc)[:, -1]
            st.draft_forwards += 1
            q = probs_of(d_logits, temperature)
            x = (
                int(q.argmax(-1))
                if temperature == 0
                else int(torch.multinomial(q, 1, generator=gen))
            )
            q_list.append(q[0])
            drafted.append(x)
            cur = x
        # 2. the target scores [last, drafted...] in ONE pass: row i is its distribution for the token after position i
        verify = torch.tensor([[last, *drafted]])
        p_all = probs_of(forward_cached(target, verify, tc)[0], temperature)
        st.target_forwards += 1
        st.rounds += 1
        st.drafted += n_draft
        # 3. accept a prefix of the draft
        n_ok, replacement = 0, None
        for i, x in enumerate(drafted):
            p, q = p_all[i], q_list[i]
            if temperature == 0:
                ok = int(p.argmax()) == x
            else:
                ok = float(torch.rand(1, generator=gen)) < min(
                    1.0, float(p[x] / q[x].clamp_min(1e-12))
                )
            if ok:
                n_ok += 1
                continue
            if temperature == 0:
                replacement = int(p.argmax())
            else:
                resid = (p - q).clamp_min(0)
                resid = resid / resid.sum() if resid.sum() > 0 else p
                replacement = int(torch.multinomial(resid[None], 1, generator=gen))
            break
        if replacement is None:  # every draft accepted: one bonus token from the target
            pb = p_all[n_draft]
            replacement = (
                int(pb.argmax())
                if temperature == 0
                else int(torch.multinomial(pb[None], 1, generator=gen))
            )
        st.accepted += n_ok
        new = [*drafted[:n_ok], replacement]
        # 4. roll both caches back to the accepted prefix; the replacement token is not in either cache yet (it is next round's `last`)
        keep = tc.length - (n_draft - n_ok)
        truncate(tc, keep)
        truncate(dc, min(dc.length, keep))
        for tok in new:
            st.tokens.append(tok)
            if tok in stop_ids or len(st.tokens) >= max_new_tokens:
                break
        last = st.tokens[-1]
        if (
            dc.length < tc.length
        ):  # the draft never saw the last accepted draft token when all were accepted
            forward_cached(draft, torch.tensor([[drafted[-1]]]), dc)
            st.draft_forwards += 1
    st.tokens = st.tokens[:max_new_tokens]
    st.seconds = time.perf_counter() - t0
    return st


def expected_tokens_per_round(alpha: float, k: int) -> float:
    """Expected tokens produced per target forward pass when each draft token is accepted independently with probability ``alpha``: (1 - alpha^(k+1)) / (1 - alpha)."""
    return float(k + 1) if alpha >= 1.0 else (1 - alpha ** (k + 1)) / (1 - alpha)


def expected_speedup(alpha: float, k: int, cost_ratio: float) -> float:
    """Speed-up over plain decoding of the target: tokens per round divided by the round's cost in units of one target step (k draft steps at ``cost_ratio`` each plus one
    verification step that costs about the same as one target step)."""
    return expected_tokens_per_round(alpha, k) / (k * cost_ratio + 1.0)
