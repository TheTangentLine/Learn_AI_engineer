"""An asynchronous load generator for OpenAI-compatible chat endpoints: closed-loop (N concurrent users) and open-loop (Poisson arrivals), with
streaming so the time to the FIRST token is measured separately from the total latency.

    results = asyncio.run(closed_loop(url, make_request, concurrency=8, n_requests=64))
    summary = summarize(results, wall_seconds)         # p50/p95/p99 latency, TTFT, tokens/s, error count

Definitions used everywhere in Week 11:
  latency      from sending the request to receiving the last byte
  TTFT         from sending the request to receiving the first generated token (the user's "is it responding?")
  throughput   generated tokens per second over the whole run (all requests together)
  per-user     generated tokens per second for one request, between its first and last token
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx


@dataclass
class Result:
    ok: bool
    latency: float = 0.0
    ttft: float = 0.0
    tokens: int = 0
    status: int = 0
    error: str = ""
    started: float = 0.0  # seconds since the run began
    text: str = ""

    @property
    def per_user_tps(self) -> float:
        span = self.latency - self.ttft
        return (self.tokens - 1) / span if self.tokens > 1 and span > 0 else 0.0


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (the 'inclusive' definition: p = 50 is the median, p = 100 the maximum)."""
    if not values:
        return float("nan")
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


async def one_request(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    t0: float,
    *,
    path: str = "/v1/chat/completions",
    headers: dict | None = None,
) -> Result:
    """POST a streaming chat completion; count content deltas as tokens (an OpenAI-style server sends one token per event)."""
    sent = time.perf_counter()
    r = Result(ok=False, started=sent - t0)
    try:
        async with client.stream(
            "POST", url + path, json={**body, "stream": True}, headers=headers
        ) as resp:
            r.status = resp.status_code
            if resp.status_code != 200:
                await resp.aread()
                r.error = f"HTTP {resp.status_code}"
                r.latency = time.perf_counter() - sent
                return r
            parts: list[str] = []
            stream_error = ""
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                if (
                    isinstance(obj, dict) and "error" in obj
                ):  # an in-band failure: the 200 status line was sent long ago
                    stream_error = f"stream_error: {obj['error'].get('code', 'unknown')}"
                    continue
                try:
                    delta = obj["choices"][0].get("delta", {})
                except (
                    KeyError,
                    IndexError,
                    TypeError,
                ):  # TypeError: a non-chunk event such as the `sources` list of /v1/ask
                    continue
                piece = delta.get("content")
                if piece:
                    if r.tokens == 0:
                        r.ttft = time.perf_counter() - sent
                    r.tokens += 1
                    parts.append(piece)
            r.text = "".join(parts)
            r.ok = r.tokens > 0 and not stream_error
            if not r.ok:
                r.error = stream_error or "no tokens"
    except (httpx.HTTPError, OSError) as exc:
        r.error = f"{type(exc).__name__}: {exc}"
    r.latency = time.perf_counter() - sent
    return r


async def closed_loop(
    url: str,
    make_body: Callable[[int], dict],
    *,
    concurrency: int,
    n_requests: int,
    headers: dict | None = None,
    timeout: float = 120.0,
    path: str = "/v1/chat/completions",
) -> tuple[list[Result], float]:
    """``concurrency`` virtual users, each sending its next request as soon as the previous one finishes, until ``n_requests`` have been sent in all."""
    results: list[Result] = []
    counter = iter(range(n_requests))
    t0 = time.perf_counter()
    async with httpx.AsyncClient(
        timeout=timeout, limits=httpx.Limits(max_connections=concurrency + 4)
    ) as client:

        async def user() -> None:
            for i in counter:
                results.append(
                    await one_request(client, url, make_body(i), t0, headers=headers, path=path)
                )

        await asyncio.gather(*(user() for _ in range(concurrency)))
    return results, time.perf_counter() - t0


async def open_loop(
    url: str,
    make_body: Callable[[int], dict],
    *,
    rate: float,
    duration: float,
    seed: int = 0,
    headers: dict | None = None,
    timeout: float = 120.0,
    max_requests: int = 10_000,
    path: str = "/v1/chat/completions",
) -> tuple[list[Result], float]:
    """Requests arrive as a Poisson process of ``rate`` per second for ``duration`` seconds whether or not earlier ones have finished: the load a public
    service sees. Unlike the closed loop it can OVERLOAD the server (the queue grows), which is the point."""
    rng = random.Random(seed)
    t0 = time.perf_counter()
    tasks: list[asyncio.Task] = []
    async with httpx.AsyncClient(
        timeout=timeout, limits=httpx.Limits(max_connections=1000)
    ) as client:
        t, i = 0.0, 0
        while i < max_requests:
            t += rng.expovariate(rate)
            if t > duration:
                break
            delay = t - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(
                asyncio.create_task(
                    one_request(client, url, make_body(i), t0, headers=headers, path=path)
                )
            )
            i += 1
        results = list(await asyncio.gather(*tasks))
    return results, time.perf_counter() - t0


@dataclass
class Summary:
    n: int
    ok: int
    errors: dict[str, int] = field(default_factory=dict)
    wall: float = 0.0
    tokens: int = 0
    throughput: float = 0.0  # generated tokens per second, whole run
    requests_per_second: float = 0.0
    latency: dict[str, float] = field(default_factory=dict)
    ttft: dict[str, float] = field(default_factory=dict)
    per_user_tps: float = 0.0


def summarize(results: list[Result], wall: float) -> Summary:
    good = [r for r in results if r.ok]
    errors: dict[str, int] = {}
    for r in results:
        if not r.ok:
            errors[r.error.split(":")[0] or "unknown"] = (
                errors.get(r.error.split(":")[0] or "unknown", 0) + 1
            )
    tokens = sum(r.tokens for r in good)
    pu = [r.per_user_tps for r in good if r.per_user_tps > 0]
    return Summary(
        n=len(results),
        ok=len(good),
        errors=errors,
        wall=wall,
        tokens=tokens,
        throughput=tokens / wall if wall > 0 else 0.0,
        requests_per_second=len(good) / wall if wall > 0 else 0.0,
        latency={f"p{p}": percentile([r.latency for r in good], p) for p in (50, 95, 99)},
        ttft={f"p{p}": percentile([r.ttft for r in good], p) for p in (50, 95, 99)},
        per_user_tps=sum(pu) / len(pu) if pu else 0.0,
    )
