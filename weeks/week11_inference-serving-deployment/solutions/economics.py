"""GPU memory and cost arithmetic: will it fit, how many users, and when does self-hosting beat an API?

EVERY price and every GPU spec here is an INPUT (an assumption you supply and a lesson reports): none is built in and none was looked up. What the module provides is
the arithmetic that turns those inputs plus MEASURED token counts and throughputs into dollars, and the sensitivity of the answer to each assumption.

    fit = fit_on_gpu(spec, gpu_gb=24, context=4096, weight_bytes=2, kv_bytes=2)      # how many concurrent sequences fit
    cmp = compare(volume, api, selfhost)                                                # $ per month for each, and who is cheaper
    be = break_even_tokens_per_month(api, selfhost)                                      # the volume at which they cost the same
"""

from __future__ import annotations

import math
from dataclasses import dataclass

HOURS_PER_MONTH = 730.0


# ----------------------------------------------------------------------------- memory


@dataclass(frozen=True)
class Spec:
    """The size-defining fields of a decoder-only model."""

    params: int
    layers: int
    kv_heads: int
    head_dim: int
    active_params: int | None = None  # for a mixture of experts; defaults to all

    def kv_bytes_per_token(self, dtype_bytes: float = 2) -> float:
        return 2 * self.layers * self.kv_heads * self.head_dim * dtype_bytes

    def weight_bytes(self, bytes_per_param: float = 2) -> float:
        return self.params * bytes_per_param


@dataclass(frozen=True)
class Fit:
    weights_gb: float
    cache_budget_gb: float
    kv_per_sequence_gb: float
    max_concurrent: int
    fits: bool


def fit_on_gpu(
    spec: Spec,
    *,
    gpu_gb: float,
    context: int,
    weight_bytes: float = 2,
    kv_bytes: float = 2,
    overhead_gb: float = 1.5,
    gpus: int = 1,
) -> Fit:
    """Weights plus ``overhead_gb`` (activations, CUDA context, fragmentation: an assumption) leave a budget for KV caches; each concurrent sequence needs
    ``context`` tokens of cache. Weights are split across ``gpus`` (tensor parallelism); the cache budget is the sum of what is left on each."""
    total = gpu_gb * gpus
    weights = spec.weight_bytes(weight_bytes) / 1e9
    budget = total - weights - overhead_gb * gpus
    per_seq = spec.kv_bytes_per_token(kv_bytes) * context / 1e9
    n = max(0, math.floor(budget / per_seq)) if per_seq > 0 else 0
    return Fit(weights, max(0.0, budget), per_seq, n, budget > 0 and n >= 1)


def min_gpus(
    spec: Spec,
    *,
    gpu_gb: float,
    context: int,
    concurrent: int,
    weight_bytes: float = 2,
    kv_bytes: float = 2,
    overhead_gb: float = 1.5,
    limit: int = 64,
) -> int | None:
    """Smallest number of GPUs that holds the weights and ``concurrent`` sequences of ``context`` tokens, or None if ``limit`` is not enough."""
    for g in range(1, limit + 1):
        if (
            fit_on_gpu(
                spec,
                gpu_gb=gpu_gb,
                context=context,
                weight_bytes=weight_bytes,
                kv_bytes=kv_bytes,
                overhead_gb=overhead_gb,
                gpus=g,
            ).max_concurrent
            >= concurrent
        ):
            return g
    return None


def decode_tokens_per_second_ceiling(
    spec: Spec,
    *,
    bandwidth_gbs: float,
    weight_bytes: float = 2,
    context: int = 1024,
    kv_bytes: float = 2,
    batch: int = 1,
) -> float:
    """Memory-bound ceiling on the TOTAL tokens/second of a decode step: each step reads the weights once (shared by the batch) and every sequence's cache.
    tokens/s = batch * bandwidth / (weight bytes + batch * cache bytes per sequence). An upper bound, not a prediction."""
    read = spec.weight_bytes(weight_bytes) + batch * spec.kv_bytes_per_token(kv_bytes) * context
    return batch * bandwidth_gbs * 1e9 / read


# ----------------------------------------------------------------------------- money


@dataclass(frozen=True)
class ApiPrice:
    """Dollars per million tokens, as quoted by whoever you are comparing with (an input)."""

    input_per_m: float
    output_per_m: float

    def cost(self, input_tokens: float, output_tokens: float) -> float:
        return (input_tokens * self.input_per_m + output_tokens * self.output_per_m) / 1e6


@dataclass(frozen=True)
class SelfHost:
    gpu_per_hour: float  # an input
    tokens_per_second: (
        float  # sustained OUTPUT tokens/s of one GPU at your batch size (measure it; Day 2)
    )
    prefill_tokens_per_second: float  # prompt tokens/s of one GPU
    utilisation: float = (
        0.5  # share of the time the GPU is doing useful work: real traffic is bursty
    )
    min_gpus: int = 1  # you pay for these whether or not anyone calls
    ops_per_month: float = 0.0  # engineering time, monitoring, on-call, spread over the month (an input; usually the largest hidden cost)

    def gpus_needed(self, input_tokens: float, output_tokens: float) -> int:
        """GPUs to serve a month's tokens at the stated utilisation: busy seconds divided by the seconds a GPU has to offer."""
        seconds = (
            output_tokens / self.tokens_per_second + input_tokens / self.prefill_tokens_per_second
        )
        return max(self.min_gpus, math.ceil(seconds / (HOURS_PER_MONTH * 3600 * self.utilisation)))

    def cost(self, input_tokens: float, output_tokens: float) -> float:
        return (
            self.gpus_needed(input_tokens, output_tokens) * self.gpu_per_hour * HOURS_PER_MONTH
            + self.ops_per_month
        )


@dataclass(frozen=True)
class Comparison:
    api: float
    selfhost: float
    gpus: int
    cheaper: str
    ratio: float  # selfhost / api


def compare(input_tokens: float, output_tokens: float, api: ApiPrice, sh: SelfHost) -> Comparison:
    a, s = api.cost(input_tokens, output_tokens), sh.cost(input_tokens, output_tokens)
    return Comparison(
        a,
        s,
        sh.gpus_needed(input_tokens, output_tokens),
        "api" if a < s else "self-hosted" if s < a else "equal",
        s / a if a else math.inf,
    )


def break_even_tokens_per_month(
    api: ApiPrice, sh: SelfHost, *, input_ratio: float = 3.0, search_to: float = 1e14
) -> float | None:
    """Monthly OUTPUT tokens (with ``input_ratio`` input tokens per output token) at which self-hosting becomes cheaper than the API, found by bisection on the step
    function of GPU counts. None if the API stays cheaper over the whole range (a GPU can never pay for itself at these prices)."""

    def diff(out_tokens: float) -> float:
        return (
            compare(out_tokens * input_ratio, out_tokens, api, sh).selfhost
            - compare(out_tokens * input_ratio, out_tokens, api, sh).api
        )

    lo, hi = 1.0, search_to
    if diff(hi) >= 0:
        return None
    if diff(lo) <= 0:
        return lo
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if diff(mid) > 0:
            lo = mid
        else:
            hi = mid
    return hi


def sensitivity(
    base: SelfHost, api: ApiPrice, factors: dict[str, list[float]], *, input_ratio: float = 3.0
) -> dict[str, list[tuple[float, float | None]]]:
    """Break-even volume as ONE assumption at a time is multiplied by each factor (``gpu_per_hour``, ``tokens_per_second``, ``utilisation``, ``ops_per_month``)."""
    out: dict[str, list[tuple[float, float | None]]] = {}
    for field_name, fs in factors.items():
        rows = []
        for f in fs:
            kw = {**base.__dict__, field_name: getattr(base, field_name) * f}
            if field_name == "utilisation":
                kw["utilisation"] = min(1.0, kw["utilisation"])
            rows.append(
                (f, break_even_tokens_per_month(api, SelfHost(**kw), input_ratio=input_ratio))
            )
        out[field_name] = rows
    return out
