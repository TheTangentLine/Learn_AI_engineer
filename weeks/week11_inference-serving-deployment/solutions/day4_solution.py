"""Week 11 Day 4 - Solution: the API in front of the model, exercised for real.

llama-server (the fine-tuned order extractor, Q8_0, 4 slots) + `python -m llmapi` (FastAPI, SSE, auth, per-user limits, backpressure), two real processes.

1. CONTRACT    the status codes and bodies a client sees: no key, a bad key, a bad request, a good request (JSON and streamed)
2. LIMITS      a user who bursts past the bucket gets 429 + Retry-After; a user over their daily quota is refused; an overloaded server sheds load with 503
3. STREAMING   time to first token against total time, and what a client that hangs up costs the server
4. OVERHEAD    the same load straight at llama-server and through the API: what the gateway adds
5. METRICS     the Prometheus text the API exports

  uv run python weeks/week11_inference-serving-deployment/solutions/day4_solution.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.append(str(ROOT / "weeks/week10_fine-tuning/solutions"))

import chatfmt as C  # noqa: E402
import llamacpp as L  # noqa: E402
import loadgen as G  # noqa: E402
import stack as K  # noqa: E402

ALICE, BOB = {"authorization": "Bearer sk-demo-alice"}, {"authorization": "Bearer sk-demo-bob"}


def bodies(n: int, **extra) -> list[dict]:
    recs = C.read_jsonl(ROOT / "outputs/w10_data/test.jsonl")
    return [
        {"messages": r["messages"][:-1], "max_tokens": 200, "temperature": 0, **extra}
        for r in recs[:n]
    ]


async def stream_timing(
    url: str, headers: dict, body: dict, *, hang_up_after: int | None = None
) -> dict:
    t0 = time.perf_counter()
    first, n, status = None, 0, None
    async with (
        httpx.AsyncClient(timeout=60) as c,
        c.stream(
            "POST", f"{url}/v1/chat/completions", json={**body, "stream": True}, headers=headers
        ) as r,
    ):
        status = r.status_code
        async for line in r.aiter_lines():
            if line.startswith("data:") and '"content"' in line:
                n += 1
                first = first or time.perf_counter() - t0
                if hang_up_after and n >= hang_up_after:
                    break
    return {"status": status, "ttft": first, "tokens": n, "seconds": time.perf_counter() - t0}


def main(argv: list[str]) -> None:
    model = L.GGUF_DIR / f"{L.STEM}-Q8_0.gguf"
    bs = bodies(40)
    with K.Stack(model, parallel=4, max_inflight=4, max_queue=4) as s:
        api = s.api.url
        print("1. CONTRACT (what a client sees)")
        c = httpx.Client(timeout=60)
        ok = c.post(f"{api}/v1/chat/completions", json=bs[0], headers=ALICE)
        print(
            f"   no key                 -> {c.post(f'{api}/v1/chat/completions', json=bs[0]).status_code} {c.post(f'{api}/v1/chat/completions', json=bs[0]).json()['error']['code']}"
        )
        print(
            f"   wrong key              -> {c.post(f'{api}/v1/chat/completions', json=bs[0], headers={'authorization': 'Bearer nope'}).status_code}"
        )
        bad = c.post(f"{api}/v1/chat/completions", json={"messages": []}, headers=ALICE)
        print(f"   empty messages         -> {bad.status_code} {bad.json()['error']['code']}")
        big = c.post(f"{api}/v1/chat/completions", json={**bs[0], "max_tokens": 5000}, headers=BOB)
        print(f"   max_tokens above cap   -> {big.status_code} {big.json()['error']['message']}")
        print(
            f"   a good request         -> {ok.status_code}; reply {ok.json()['choices'][0]['message']['content'][:70]!r}...; usage {ok.json()['usage']}; request id {ok.headers['x-request-id']}"
        )
        print(
            f"   /healthz {c.get(f'{api}/healthz').status_code}, /readyz {c.get(f'{api}/readyz').status_code}"
        )

        print("\n2. LIMITS")
        bob_rs = [
            c.post(f"{api}/v1/chat/completions", json={**bs[i], "max_tokens": 40}, headers=BOB)
            for i in range(6)
        ]
        print(
            f"   bob (burst 3, 30 requests/minute) sends 6 requests back to back: {[r.status_code for r in bob_rs]}; Retry-After on the first refusal: {next((r.headers.get('retry-after') for r in bob_rs if r.status_code == 429), None)} s"
        )
        print(f"   bob's usage so far: {c.get(f'{api}/v1/usage', headers=BOB).json()}")
        quota = None
        for _ in range(30):
            time.sleep(
                2.1
            )  # refill the bucket (30 per minute = one per 2 s) and spend the daily quota (5,000 tokens)
            r = c.post(f"{api}/v1/chat/completions", json={**bs[1], "max_tokens": 200}, headers=BOB)
            if r.status_code == 429 and r.json()["error"]["code"] == "quota_exceeded":
                quota = r
                break
        print(
            f"   bob keeps going until his daily token quota is spent -> {quota.status_code if quota else 'not reached'} {quota.json()['error'] if quota else ''}"
        )

    # overload: a deliberately small API (2 slots, queue of 2) in front of a 2-slot model server
    with K.Stack(model, parallel=2, max_inflight=2, max_queue=2) as s:
        results, wall = asyncio.run(
            G.closed_loop(
                s.api.url, lambda i: bs[i % len(bs)], concurrency=12, n_requests=36, headers=ALICE
            )
        )
        codes: dict[int, int] = {}
        for r in results:
            codes[r.status] = codes.get(r.status, 0) + 1
        print(
            f"\n   OVERLOAD: 12 concurrent users against 2 model slots and a queue of 2, 36 requests: status counts {dict(sorted(codes.items()))}"
        )
        ok_lat = sorted(r.latency for r in results if r.ok)
        shed = sorted(r.latency for r in results if r.status == 503)
        print(
            f"   served requests: latency p50 {G.percentile(ok_lat, 50):.2f}s, p95 {G.percentile(ok_lat, 95):.2f}s; refused requests were refused in {G.percentile(shed, 50) * 1000 if shed else float('nan'):.0f} ms (median) instead of waiting"
        )

    with K.Stack(model, parallel=4, max_inflight=4, max_queue=16) as s:
        api = s.api.url
        print("\n3. STREAMING")
        t = asyncio.run(stream_timing(api, ALICE, bs[2]))
        print(
            f"   one streamed request: first token after {t['ttft'] * 1000:.0f} ms, {t['tokens']} chunks in {t['seconds']:.2f} s total (the user sees output {t['seconds'] / t['ttft']:.0f}x sooner than a non-streaming client would)"
        )
        before = httpx.get(f"{s.llama.url}/metrics").text
        hang = asyncio.run(stream_timing(api, ALICE, {**bs[3], "max_tokens": 400}, hang_up_after=5))
        time.sleep(1.0)
        usage = httpx.get(f"{api}/v1/usage", headers=ALICE).json()
        slots = (
            httpx.get(f"{s.llama.url}/slots", timeout=5).json()
            if httpx.get(f"{s.llama.url}/slots", timeout=5).status_code == 200
            else []
        )
        busy = sum(1 for sl in slots if sl.get("is_processing"))
        print(
            f"   a client that hangs up after 5 chunks: API in-flight {usage['in_flight']}, model slots still processing {busy} (the upstream call was cancelled: {hang['tokens']} chunks read, {before.count('llamacpp')} metric lines scraped)"
        )

        print(
            "\n4. OVERHEAD OF THE GATEWAY: the same 40 requests at concurrency 1 and 4, straight to llama-server and through the API"
        )
        print(
            f"   {'path':<10}{'users':>6}{'tokens/s':>10}{'latency p50':>13}{'latency p95':>13}{'TTFT p50':>10}"
        )
        for users in (1, 4):
            for name, url, headers in (("direct", s.llama.url, None), ("via API", api, ALICE)):
                res, wall = asyncio.run(
                    G.closed_loop(
                        url,
                        lambda i: bs[i % len(bs)],
                        concurrency=users,
                        n_requests=40,
                        headers=headers,
                    )
                )
                sm = G.summarize(res, wall)
                print(
                    f"   {name:<10}{users:>6}{sm.throughput:>10.0f}{sm.latency['p50']:>12.2f}s{sm.latency['p95']:>12.2f}s{sm.ttft['p50'] * 1000:>9.0f}ms"
                )

        print("\n5. /metrics (excerpt)")
        text = httpx.get(f"{api}/metrics", headers={"authorization": "Bearer adm-demo"}).text
        for line in text.splitlines():
            if line.startswith(
                (
                    "llmapi_requests_total",
                    "llmapi_rejected_total",
                    "llmapi_tokens_total",
                    "llmapi_request_seconds_count",
                    "llmapi_request_seconds_sum",
                    "llmapi_ttft_seconds_count",
                    "llmapi_in_flight",
                    "llmapi_queue_depth",
                )
            ):
                print("   " + line)
        _ = json


if __name__ == "__main__":
    main(sys.argv)
