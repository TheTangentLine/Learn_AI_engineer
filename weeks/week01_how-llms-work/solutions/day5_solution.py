"""Week 1 Day 5 - Solution: concurrent batch processor with retries, backoff and timeouts.

``run_batch`` is provider-agnostic: give it a list of items and an ``async worker(item)``.
It adds, in this order of importance:

  1. bounded concurrency      (asyncio.Semaphore)        - don't stampede the rate limit
  2. retries with backoff     (exponential + full jitter) - only for *retryable* errors
  3. Retry-After honouring    (server tells us how long)
  4. per-attempt timeout      (asyncio.timeout)           - never hang forever
  5. order-preserving results + per-item error capture   - one bad item can't kill the batch
  6. stats                    (attempts, p50/p95 latency, wall time)

Part 1 runs offline against a deliberately flaky FakeClient so the behaviour is deterministic
and free. Part 2 runs the same runner against a real provider if a key is configured.

Run:  uv run python weeks/week01_how-llms-work/solutions/day5_solution.py
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # make `common` importable


# ----------------------------------------------------------------- error taxonomy


class RetryableError(Exception):
    """429 / 5xx / timeouts / connection resets: trying again can succeed."""

    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class FatalError(Exception):
    """400 / 401 / 403 / 404: retrying cannot help (bad request, bad key, wrong model)."""


def classify(exc: Exception) -> Exception:
    """Map SDK exceptions to Retryable/Fatal without importing a specific SDK at module level."""
    if isinstance(exc, RetryableError | FatalError):
        return exc
    if isinstance(exc, asyncio.TimeoutError | TimeoutError | ConnectionError):
        return RetryableError(f"timeout/connection: {exc!r}")
    status = getattr(exc, "status_code", None)  # anthropic.APIStatusError / openai.APIStatusError
    if status is not None:
        if status == 429 or status >= 500 or status in (408, 409):
            headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
            ra = headers.get("retry-after")
            return RetryableError(f"HTTP {status}", float(ra) if ra else None)
        return FatalError(f"HTTP {status}: {exc}")
    if type(exc).__name__ in ("APIConnectionError", "APITimeoutError"):
        return RetryableError(type(exc).__name__)
    return FatalError(repr(exc))


# ----------------------------------------------------------------- the runner


@dataclass
class ItemResult:
    index: int
    ok: bool
    value: Any = None
    error: str | None = None
    attempts: int = 0
    latency_s: float = 0.0  # duration of the successful request only


@dataclass
class BatchStats:
    results: list[ItemResult] = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def ok(self) -> int:
        return sum(r.ok for r in self.results)

    @property
    def total_attempts(self) -> int:
        return sum(r.attempts for r in self.results)

    def percentile(self, q: float) -> float:
        lat = sorted(r.latency_s for r in self.results if r.ok)
        return lat[min(len(lat) - 1, int(q * len(lat)))] if lat else 0.0


def backoff_delay(attempt: int, base: float, cap: float, rng: random.Random) -> float:
    """Exponential backoff with *full jitter*: uniform(0, min(cap, base * 2**attempt)).

    Jitter matters: without it, 100 clients that failed together retry together, forever.
    """
    return rng.uniform(0, min(cap, base * 2**attempt))


async def run_batch(
    items: list[Any],
    worker: Callable[[Any], Awaitable[Any]],
    *,
    concurrency: int = 8,
    max_retries: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    attempt_timeout: float = 60.0,
    seed: int | None = None,
) -> BatchStats:
    sem = asyncio.Semaphore(concurrency)
    rng = random.Random(seed)
    stats = BatchStats(results=[ItemResult(i, False) for i in range(len(items))])

    async def one(i: int) -> None:
        res = stats.results[i]
        for attempt in range(max_retries + 1):
            res.attempts = attempt + 1
            try:
                # The semaphore is held only while a request is *in flight*, never while
                # sleeping in backoff, so waiting retries don't block fresh work.
                async with sem, asyncio.timeout(attempt_timeout):
                    t_req = time.perf_counter()
                    res.value = await worker(items[i])
                    res.latency_s = time.perf_counter() - t_req  # the request, not the queueing
                res.ok = True
                break
            except Exception as exc:
                err = classify(exc)
                if isinstance(err, FatalError) or attempt == max_retries:
                    res.error = str(err)
                    break
                delay = backoff_delay(attempt, base_delay, max_delay, rng)
                if isinstance(err, RetryableError) and err.retry_after:
                    delay = max(delay, err.retry_after)  # the server's hint is a floor
                await asyncio.sleep(delay)

    t0 = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(len(items))))
    stats.wall_s = time.perf_counter() - t0
    return stats


# ----------------------------------------------------------------- offline fake client


class FakeClient:
    """Pretends to be an LLM API: variable latency, random 429/500s, one permanently bad item."""

    def __init__(self, seed: int = 0, p_429: float = 0.2, p_500: float = 0.1, bad_item: int = 7):
        self.rng = random.Random(seed)
        self.p_429, self.p_500, self.bad_item = p_429, p_500, bad_item
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(self, item: int) -> str:
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.rng.uniform(0.02, 0.08))  # network + generation time
            if item == self.bad_item:
                raise FatalError("HTTP 400: invalid request")
            r = self.rng.random()
            if r < self.p_429:
                raise RetryableError("HTTP 429", retry_after=0.05)
            if r < self.p_429 + self.p_500:
                raise RetryableError("HTTP 500")
            return f"summary of document {item}"
        finally:
            self.in_flight -= 1


def report(title: str, stats: BatchStats) -> None:
    failed = [r for r in stats.results if not r.ok]
    print(f"\n{title}")
    print(
        f"  items={len(stats.results)} ok={stats.ok} failed={len(failed)} "
        f"attempts={stats.total_attempts} (retries={stats.total_attempts - len(stats.results)})"
    )
    print(
        f"  wall={stats.wall_s:.2f}s  latency p50={stats.percentile(0.5):.2f}s "
        f"p95={stats.percentile(0.95):.2f}s"
    )
    for r in failed:
        print(f"  FAILED item {r.index}: {r.error} after {r.attempts} attempt(s)")


async def part1_offline() -> None:
    print(
        "=" * 80
        + "\nPART 1 - offline, deterministic (FakeClient: 20% 429s, 10% 500s, 1 bad item)\n"
        + "=" * 80
    )
    items = list(range(40))

    async def sequential():
        client = FakeClient(seed=1)
        return await run_batch(items, client.complete, concurrency=1, base_delay=0.02, seed=1)

    async def concurrent():
        client = FakeClient(seed=1)
        stats = await run_batch(items, client.complete, concurrency=8, base_delay=0.02, seed=1)
        return stats, client

    seq = await sequential()
    par, client = await concurrent()
    report("concurrency=1 (sequential)", seq)
    report("concurrency=8", par)
    print(
        f"\n  speed-up x{seq.wall_s / par.wall_s:.1f}; peak in-flight requests = {client.max_in_flight}"
    )

    # Acceptance checks
    assert client.max_in_flight <= 8, "semaphore must cap concurrency"
    assert par.ok == len(items) - 1, "only the permanently-bad item should fail"
    assert par.results[7].attempts == 1, "fatal errors must NOT be retried"
    assert [r.index for r in par.results] == list(range(len(items))), "order preserved"
    assert par.wall_s < seq.wall_s, "concurrency should be faster"

    # Retries exhausted: a client that always returns 500
    always_500 = FakeClient(p_429=0, p_500=1.0, bad_item=-1)
    stats = await run_batch([0], always_500.complete, max_retries=3, base_delay=0.01, seed=0)
    assert stats.results[0].attempts == 4 and not stats.results[0].ok
    print("  all assertions passed (cap, fatal-not-retried, order, retries exhausted)")


# ----------------------------------------------------------------- real provider


async def part2_real() -> None:
    from common.llm import acomplete, available_providers

    providers = [p for p in available_providers() if p != "ollama"] or available_providers()
    print("\n" + "=" * 80 + "\nPART 2 - real provider\n" + "=" * 80)
    if not providers:
        print("Skipped: no API key or running Ollama found.")
        return
    provider = providers[0]
    docs = [f"Document {i}: " + "The quarterly results beat expectations. " * 5 for i in range(12)]

    async def summarize(doc: str) -> str:
        r = await acomplete(f"Summarize in 8 words: {doc}", provider=provider, max_tokens=60)
        return r.text.strip()

    stats = await run_batch(docs, summarize, concurrency=4, attempt_timeout=60)
    report(f"{provider}: 12 summaries, concurrency=4", stats)
    for r in stats.results[:3]:
        print(f"  [{r.index}] {r.value!r}")


if __name__ == "__main__":
    asyncio.run(part1_offline())
    if os.getenv("RUN_REAL") == "1":  # opt in: real calls cost money
        asyncio.run(part2_real())
    else:
        print("\n(Set RUN_REAL=1 to also run the batch against a real provider.)")
