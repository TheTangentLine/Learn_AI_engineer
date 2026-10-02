"""Load-test the deployed API: a closed-loop sweep over concurrency (what a fixed pool of users sees) and an open-loop run at fixed arrival rates (what a public service sees),
on ``/v1/ask`` (retrieval + generation) with the labelled questions as the request mix. Latency, time to first token and throughput come from ``loadgen``."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

SOLUTIONS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOLUTIONS))

import loadgen as G  # noqa: E402


def ask_bodies(rows: list[dict], *, k: int = 4, max_tokens: int = 100):
    """Request ``i`` asks question ``i`` (cycling): greedy, capped, the mix the evaluation used."""

    def make(i: int) -> dict:
        r = rows[i % len(rows)]
        return {"question": r["question"], "k": k, "max_tokens": max_tokens, "temperature": 0}

    return make


def row_of(s: G.Summary) -> dict:
    return {
        "n": s.n,
        "ok": s.ok,
        "errors": dict(s.errors),
        "wall": s.wall,
        "throughput": s.throughput,
        "rps": s.requests_per_second,
        "lat50": s.latency.get("p50"),
        "lat95": s.latency.get("p95"),
        "ttft50": s.ttft.get("p50"),
        "ttft95": s.ttft.get("p95"),
        "per_user": s.per_user_tps,
    }


def closed_sweep(
    url: str, key: str, rows: list[dict], levels=(1, 2, 4, 8), per_level: int = 32
) -> list[dict]:
    out = []
    make = ask_bodies(rows)
    for users in levels:
        results, wall = asyncio.run(
            G.closed_loop(
                url,
                make,
                concurrency=users,
                n_requests=per_level,
                headers={"authorization": f"Bearer {key}"},
                path="/v1/ask",
            )
        )
        out.append({"users": users, **row_of(G.summarize(results, wall))})
    return out


def open_runs(
    url: str, key: str, rows: list[dict], rates=(0.5, 1.0, 2.0), duration: float = 30.0
) -> list[dict]:
    out = []
    make = ask_bodies(rows)
    for rate in rates:
        results, wall = asyncio.run(
            G.open_loop(
                url,
                make,
                rate=rate,
                duration=duration,
                seed=0,
                headers={"authorization": f"Bearer {key}"},
                path="/v1/ask",
            )
        )
        out.append({"rate": rate, **row_of(G.summarize(results, wall))})
    return out


def find_knee(rows: list[dict], *, factor: float = 3.0) -> float | None:
    """The first offered rate whose p95 latency exceeds ``factor`` times the p95 of the lowest rate (or that had errors): where the service stops keeping up."""
    if not rows:
        return None
    base = rows[0]["lat95"]
    for r in rows:
        if r["errors"] or (
            base is not None and r["lat95"] is not None and r["lat95"] > factor * base
        ):
            return r["rate"]
    return None
