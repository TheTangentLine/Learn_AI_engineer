"""Week 11 Day 6 - Solution: containerise the API, run it with environment secrets, prove its health checks, and rehearse the failures.

1. CONTEXT   assemble the build context (the API package, a baked-in markdown corpus, requirements)
2. BUILD     `docker build`: size, time, the cached rebuild after a code change, the user it runs as, what the image's environment and history reveal
3. RUN       the container in front of a model server on the host; /healthz (liveness) against /readyz (readiness); an answer with citations from inside the container
4. FAILURES  kill the model server: readiness fails, liveness does not, and the container is NOT restarted; bring it back; `docker stop` with a request in flight
5. MODEL     the real llama.cpp server container serving the fine-tuned GGUF, for comparison with the host's Metal build

  uv run python weeks/week11_inference-serving-deployment/solutions/day6_solution.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "deploy"))

import build_context as BC  # noqa: E402
import dockerlab as D  # noqa: E402
import llamacpp as L  # noqa: E402

TAG = "order-api:w11"
KEYS = json.dumps(
    [
        {
            "user": "alice",
            "key": "sk-container-alice",
            "rpm": 600,
            "burst": 100,
            "daily_tokens": 5_000_000,
            "max_concurrent": 16,
            "max_tokens_cap": 400,
        }
    ]
)
AUTH = {"authorization": "Bearer sk-container-alice"}
SECRET = "super-secret-admin-token-9f3"


def main(argv: list[str]) -> None:
    if not D.available():
        print("docker is not available: nothing to run")
        return
    ctx = Path(tempfile.mkdtemp(prefix="w11-ctx-"))
    info = BC.build(ctx, limit_weeks=11)
    print(
        f"1. BUILD CONTEXT: {info['docs']} markdown files baked in, {info['bytes'] / 1e6:.1f} MB (no model weights, no secrets)"
    )
    secs, out = D.build(str(ctx), TAG)
    print(
        f"2. BUILD: {secs:.0f} s cold; image {D.image_size_mb(TAG):.0f} MB; runs as user {D.image_user(TAG) or 'root'!r}"
    )
    (ctx / "llmapi" / "app.py").write_text(
        (ctx / "llmapi" / "app.py").read_text() + "\n# touched\n"
    )
    secs2, out2 = D.build(str(ctx), TAG)
    cached = out2.count("CACHED")
    print(
        f"   rebuild after touching one source file: {secs2:.1f} s ({cached} cached layers: the dependency layer is reused)"
    )
    env = D.env_in_image(TAG)
    # names only, never values. GPG_KEY is inherited from the python base image: the public fingerprint Python's release signatures are checked against (not a secret)
    names = [e.split("=", 1)[0] for e in env]
    suspicious = [
        n
        for n in names
        if any(w in n.lower() for w in ("key", "token", "secret", "password")) and n != "GPG_KEY"
    ]
    print(
        f"   secret-looking variables baked into the image environment: {suspicious or 'none'} (the base image's public GPG_KEY fingerprint is excluded); "
        f"the demo secret appears in the layer history: {SECRET in D.image_history(TAG)}"
    )

    model = L.GGUF_DIR / f"{L.STEM}-Q8_0.gguf"
    with L.LlamaServer(model, parallel=4, ctx=4096) as llama:
        # Docker Desktop reaches the host through host.docker.internal
        backend = f"http://host.docker.internal:{llama.port}"
        port = L.free_port()
        c = D.Container(
            "w11-api",
            TAG,
            port,
            env={
                "LLMAPI_BACKEND_URL": backend,
                "LLMAPI_KEYS": KEYS,
                "LLMAPI_ADMIN_TOKEN": SECRET,
                "LLMAPI_MAX_INFLIGHT": "4",
            },
        ).run()
        try:
            wait = c.wait_healthy()
            print(
                f"\n3. RUN: container healthy after {wait:.1f} s (HEALTHCHECK = the process answers /healthz)"
            )
            print(
                f"   /healthz {httpx.get(c.url + '/healthz').status_code}, /readyz {httpx.get(c.url + '/readyz').status_code} {httpx.get(c.url + '/readyz').json()}"
            )
            r = httpx.post(
                c.url + "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "Extract the order from the email as JSON."},
                        {
                            "role": "user",
                            "content": "Hi, order A-5 for Dana Lee: 3 desk lamps. ASAP",
                        },
                    ],
                    "max_tokens": 150,
                },
                headers=AUTH,
                timeout=60,
            )
            print(
                f"   chat through the container: {r.status_code}, {r.json()['choices'][0]['message']['content'][:90]!r}"
            )
            r = httpx.post(
                c.url + "/v1/ask",
                json={
                    "question": "why do we divide attention scores by the square root of d",
                    "k": 3,
                    "max_tokens": 120,
                },
                headers=AUTH,
                timeout=60,
            )
            j = r.json()
            print(
                f"   /v1/ask (retrieval inside the container, answered by the fine-tuned extractor, so the TEXT is not meaningful): {r.status_code}, sources {[s['doc'].split('/')[-1] for s in j['sources']]}"
            )
            print(
                f"   container logs carry no prompts: {'desk lamps' not in c.logs() and 'attention' not in c.logs()}; secret in logs: {SECRET in c.logs()}"
            )

            print("\n4. FAILURES")
            llama.__exit__(None, None, None)  # kill the model server
            time.sleep(2)
            rz, hz = httpx.get(c.url + "/readyz"), httpx.get(c.url + "/healthz")
            print(
                f"   model server killed: /readyz {rz.status_code} {rz.json()['status']!r}, /healthz {hz.status_code}; container health {c.health()!r}, state {c.state()!r} (liveness is unaffected: restarting the API would not fix the model)"
            )
            t = time.perf_counter()
            bad = httpx.post(
                c.url + "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers=AUTH,
                timeout=30,
            )
            print(
                f"   a request meanwhile: {bad.status_code} {bad.json()['error']['code']} in {(time.perf_counter() - t) * 1000:.0f} ms (a clear 502, not a hang)"
            )
            llama2 = L.LlamaServer(model, parallel=4, ctx=4096)
            llama2.port = llama.port  # same port: the container's configured backend comes back
            llama2.__enter__()
            ok = D.wait_http(c.url + "/readyz", 30)
            print(f"   model server restarted on the same port: /readyz healthy again: {ok}")
            res: dict = {}

            def long_request() -> None:
                t0 = time.perf_counter()
                with httpx.stream(
                    "POST",
                    c.url + "/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "user", "content": "Write a long story about a lamp."}
                        ],
                        "max_tokens": 300,
                        "stream": True,
                    },
                    headers=AUTH,
                    timeout=60,
                ) as rr:
                    n = sum(1 for ln in rr.iter_lines() if '"content"' in ln)
                res.update(chunks=n, seconds=time.perf_counter() - t0)

            th = threading.Thread(target=long_request)
            th.start()
            time.sleep(0.7)
            stop = c.stop(timeout=15)
            th.join()
            print(
                f"   `docker stop` with a streaming request in flight: stopped in {stop:.1f} s, exit code {c.exit_code()}; the request received {res.get('chunks')} chunks before the connection ended"
            )
            llama2.__exit__(None, None, None)
        finally:
            c.remove()

    print(
        "\n5. THE MODEL SERVER IN A CONTAINER (ghcr.io/ggml-org/llama.cpp:server, CPU inside Docker's Linux VM) against the host's llama-server (Metal)"
    )
    gguf = L.GGUF_DIR
    port = L.free_port()
    t0 = time.time()
    p = D.docker(
        "run",
        "-d",
        "--name",
        "w11-llama",
        "-p",
        f"127.0.0.1:{port}:8080",
        "-v",
        f"{gguf}:/models:ro",
        "ghcr.io/ggml-org/llama.cpp:server",
        "-m",
        f"/models/{L.STEM}-Q8_0.gguf",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "-c",
        "4096",
        "-np",
        "4",
        check=False,
    )
    try:
        if p.returncode == 0 and D.wait_http(f"http://127.0.0.1:{port}/health", 60):
            print(f"   ready after {time.time() - t0:.1f} s")
            from loadgen import closed_loop, summarize

            bodies = [
                {
                    "messages": [
                        {"role": "system", "content": "Extract the order from the email as JSON."},
                        {
                            "role": "user",
                            "content": "Hi, order A-5 for Dana Lee: 3 desk lamps and 2 notebooks. ASAP",
                        },
                    ],
                    "max_tokens": 150,
                    "temperature": 0,
                }
            ] * 24
            import asyncio

            res_c, wall = asyncio.run(
                closed_loop(
                    f"http://127.0.0.1:{port}", lambda i: bodies[i], concurrency=4, n_requests=24
                )
            )
            sc = summarize(res_c, wall)
            with L.LlamaServer(L.GGUF_DIR / f"{L.STEM}-Q8_0.gguf", parallel=4, ctx=4096) as host:
                res_h, wall_h = asyncio.run(
                    closed_loop(host.url, lambda i: bodies[i], concurrency=4, n_requests=24)
                )
            sh = summarize(res_h, wall_h)
            print(
                f"   4 users, 24 requests: container (CPU) {sc.throughput:.0f} tokens/s, latency p50 {sc.latency['p50']:.2f}s; host (Metal) {sh.throughput:.0f} tokens/s, latency p50 {sh.latency['p50']:.2f}s"
            )
        else:
            print(
                "   the llama.cpp container did not start: "
                + (p.stderr or D.docker("logs", "w11-llama", check=False).stdout)[-300:]
            )
    finally:
        D.docker("rm", "-f", "w11-llama", check=False)


if __name__ == "__main__":
    main(sys.argv)
