"""Text generation three ways, so they can be compared: without a cache, with a growing (concatenated) cache, and with a preallocated cache."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import blocks as B  # noqa: E402
import torch
from kvcache import KVCache, forward_cached
from sampling import sample


@dataclass
class Generation:
    tokens: list[int]  # the NEW tokens
    prefill_s: float = 0.0
    step_s: list[float] = field(default_factory=list)  # seconds for each decode step
    stopped_on: int | None = None

    @property
    def total_s(self) -> float:
        return self.prefill_s + sum(self.step_s)

    @property
    def tokens_per_second(self) -> float:
        """Decode speed: new tokens divided by the time spent after the prefill."""
        return len(self.step_s) / sum(self.step_s) if self.step_s else 0.0


@torch.no_grad()
def generate(
    model: B.Decoder,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    mode: str = "cache",
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_ids: set[int] | frozenset[int] = frozenset(),
    seed: int = 0,
    grouped: bool = True,
) -> Generation:
    """mode: 'nocache' (recompute the whole prefix every step), 'concat' (cache grown with torch.cat) or 'cache' (preallocated). With the same
    seed and settings all three produce the same tokens (greedy: exactly; sampled: the same random draws from the same distributions)."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    if ids.shape[1] + max_new_tokens > model.cfg.max_seq_len:
        raise ValueError("prompt plus new tokens exceeds the model's maximum length")
    out = Generation([])

    def pick(logits: torch.Tensor) -> torch.Tensor:
        return sample(logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=gen)

    cache = None
    if mode != "nocache":
        cache = KVCache.for_model(
            model, 1, ids.shape[1] + max_new_tokens, preallocate=(mode == "cache")
        )
    t0 = time.perf_counter()
    logits = forward_cached(model, ids, cache, grouped=grouped) if cache else model(ids)
    nxt = pick(logits[:, -1])
    out.prefill_s = time.perf_counter() - t0
    for _ in range(max_new_tokens):
        out.tokens.append(int(nxt))
        if int(nxt) in stop_ids:
            out.stopped_on = int(nxt)
            break
        if len(out.tokens) == max_new_tokens:
            break
        t0 = time.perf_counter()
        if cache is not None:
            logits = forward_cached(model, nxt[:, None], cache, grouped=grouped)
        else:
            ids = torch.cat([ids, nxt[:, None]], dim=1)
            logits = model(ids)
        nxt = pick(logits[:, -1])
        out.step_s.append(time.perf_counter() - t0)
    return out
