"""Tests for Week 11 Day 4: keys, the limiter (with a fake clock), metrics, and the API end to end against a test-double backend: auth, validation, streaming, errors, backpressure, cleanup."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from llmapi import (  # noqa: E402
    ApiKeyStore,
    EchoBackend,
    Limiter,
    TokenBucket,
    User,
    create_app,
    make_key,
)
from llmapi.app import Settings  # noqa: E402
from llmapi.auth import bearer_token, hash_key  # noqa: E402
from llmapi.backends import BackendError, LlamaServerBackend  # noqa: E402
from llmapi.metrics import Metrics  # noqa: E402


class Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


# ----------------------------------------------------------------------------- keys


def test_keys_are_random_hashed_and_authenticated_by_hash():
    a, b = make_key(), make_key()
    assert a != b and a.startswith("sk-local-") and len(a) > 30
    store = ApiKeyStore()
    store.add(a, User("alice"))
    assert (
        store.authenticate(a).id == "alice"
        and store.authenticate(b) is None
        and store.authenticate("") is None
        and store.authenticate(None) is None
    )
    assert a not in str(store._by_hash) and hash_key(a) in store._by_hash and len(store) == 1


def test_bearer_parsing():
    assert bearer_token("Bearer abc") == "abc" and bearer_token("bearer  abc ") == "abc"
    assert (
        bearer_token("Basic abc") is None
        and bearer_token("Bearer") is None
        and bearer_token("") is None
        and bearer_token(None) is None
    )


# ----------------------------------------------------------------------------- limiter


def test_token_bucket_allows_a_burst_then_refills_at_the_rate():
    clock = Clock()
    b = TokenBucket(rate=1.0, burst=3, clock=clock)
    assert [b.acquire()[0] for _ in range(3)] == [True] * 3
    ok, wait = b.acquire()
    assert not ok and wait == pytest.approx(1.0)
    clock.t += 0.5
    assert b.acquire() == (False, pytest.approx(0.5))
    clock.t += 0.5
    assert b.acquire()[0]
    clock.t += 1000
    assert [b.acquire()[0] for _ in range(4)] == [True, True, True, False], (
        "the bucket never holds more than its capacity"
    )
    with pytest.raises(ValueError):
        TokenBucket(0, 1)


def test_the_wait_is_the_time_to_refill_the_missing_fraction_at_the_rate():
    clock = Clock()
    slow = TokenBucket(rate=0.5, burst=1, clock=clock)  # one token every 2 seconds
    assert slow.acquire()[0]
    assert slow.acquire() == (False, pytest.approx(2.0))
    clock.t += 0.5  # a quarter of a token back: 1.5 s still to wait
    assert slow.acquire() == (False, pytest.approx(1.5))
    fast = TokenBucket(rate=4.0, burst=1, clock=clock)  # one token every 0.25 seconds
    assert fast.acquire()[0] and fast.acquire() == (False, pytest.approx(0.25))
    assert fast.acquire(cost=1.0)[1] > 0


def test_the_limiter_enforces_rate_concurrency_and_a_daily_quota_per_user():
    clock, wall = Clock(), Clock(86400 * 100 + 10)
    lim = Limiter(clock, wall)
    u = User("u", rpm=60, burst=2, daily_tokens=100, max_concurrent=1)
    assert lim.admit(u).allowed
    d = lim.admit(u)
    assert not d.allowed and d.code == "too_many_concurrent"
    lim.release(u, 60)
    assert lim.admit(u).allowed
    lim.release(u, 60)  # 120 tokens used today
    d = lim.admit(u)
    assert (
        not d.allowed and d.code == "quota_exceeded" and d.retry_after == pytest.approx(86400 - 10)
    )
    wall.t += 86400  # the next UTC day: the quota resets
    clock.t += 86400
    assert lim.admit(u).allowed
    lim.release(u, 0)
    assert lim.usage(u)["rejected"] == 2


def test_a_burst_beyond_the_bucket_is_rate_limited_with_a_retry_hint_and_users_are_independent():
    lim = Limiter(Clock(), Clock())
    a, b = (
        User("a", rpm=60, burst=2, max_concurrent=10),
        User("b", rpm=60, burst=2, max_concurrent=10),
    )
    assert [lim.admit(a).allowed for _ in range(3)] == [True, True, False]
    d = lim.admit(a)
    assert d.code == "rate_limited" and d.retry_after == pytest.approx(1.0)
    assert lim.admit(b).allowed, "another user is unaffected"
    assert lim.usage(a)["in_flight"] == 2 and lim.usage(a)["requests"] == 2


def test_release_never_goes_negative_and_ignores_negative_usage():
    lim = Limiter(Clock(), Clock())
    u = User("u")
    lim.release(u, -5)
    assert lim.usage(u)["in_flight"] == 0 and lim.usage(u)["tokens_today"] == 0


# ----------------------------------------------------------------------------- metrics


def test_metrics_render_counters_gauges_and_cumulative_histograms():
    m = Metrics()
    m.describe("x_total", "counter", "things")
    m.inc("x_total", code="a")
    m.inc("x_total", 2, code="a")
    m.inc("x_total", code='b"q')
    m.set_gauge("depth", 7)
    for v in (0.04, 0.3, 0.3, 50.0):
        m.observe("lat", v)
    text = m.render()
    assert (
        'x_total{code="a"} 3' in text
        and 'x_total{code="b\\"q"} 1' in text
        and "# HELP x_total things" in text
        and "depth 7" in text
    )
    assert (
        'lat_bucket{le="0.05"} 1' in text
        and 'lat_bucket{le="0.5"} 3' in text
        and 'lat_bucket{le="+Inf"} 4' in text
    )
    assert "lat_count 4" in text and "lat_sum 50.64" in text and m.value("x_total", code="a") == 3


# ----------------------------------------------------------------------------- the API


KEY = make_key()
USER = User("alice", rpm=600, burst=50, daily_tokens=10_000, max_concurrent=8, max_tokens_cap=64)
HEADERS = {"authorization": f"Bearer {KEY}"}
BODY = {"messages": [{"role": "user", "content": "extract the order please now"}]}


def make(backend=None, **kw):
    keys = ApiKeyStore()
    keys.add(KEY, kw.pop("user", USER))
    backend = backend or EchoBackend()
    app = create_app(backend, keys, **kw)
    return app, backend


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def run(coro):
    return asyncio.run(coro)


def test_a_valid_request_returns_an_openai_shaped_completion_with_usage_and_a_request_id():
    app, backend = make()

    async def go():
        async with client(app) as c:
            return await c.post(
                "/v1/chat/completions", json=BODY, headers={**HEADERS, "x-request-id": "abc123"}
            )

    r = run(go())
    j = r.json()
    assert (
        r.status_code == 200
        and r.headers["x-request-id"] == "abc123"
        and j["object"] == "chat.completion"
        and j["id"] == "chatcmpl-abc123"
    )
    assert (
        j["choices"][0]["message"]
        == {"role": "assistant", "content": "extract the order please now "}
        and j["choices"][0]["finish_reason"] == "stop"
    )
    assert j["usage"]["total_tokens"] == 10 and backend.calls[0]["max_tokens"] == 64, (
        "the default max_tokens is capped by the user's own limit"
    )


def test_authentication_failures_are_401_with_a_bearer_challenge_and_an_openai_style_error():
    app, _ = make()

    async def go(headers):
        async with client(app) as c:
            return await c.post("/v1/chat/completions", json=BODY, headers=headers)

    for h in ({}, {"authorization": "Bearer wrong"}, {"authorization": "Basic abc"}):
        r = run(go(h))
        assert (
            r.status_code == 401
            and r.headers["www-authenticate"] == "Bearer"
            and r.json()["error"]["code"] == "invalid_api_key"
        )


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({"messages": []}, 400, "invalid_request"),
        ({"messages": [{"role": "robot", "content": "x"}]}, 400, "invalid_request"),
        ({"messages": [{"role": "user", "content": ""}]}, 400, "invalid_request"),
        ({"messages": BODY["messages"], "temperature": 5}, 400, "invalid_request"),
        ({"messages": BODY["messages"], "max_tokens": 0}, 400, "invalid_request"),
        ({"messages": BODY["messages"], "max_tokens": 65}, 400, "max_tokens_too_large"),
        ({"messages": [{"role": "user", "content": "x" * 17_000}]}, 400, "prompt_too_long"),
    ],
)
def test_bad_requests_are_400_with_a_machine_readable_code(body, status, code):
    app, backend = make()

    async def go():
        async with client(app) as c:
            return await c.post("/v1/chat/completions", json=body, headers=HEADERS)

    r = run(go())
    assert r.status_code == status and r.json()["error"]["code"] == code and backend.calls == [], (
        "the model is never called for a bad request"
    )


def parse_sse(text: str):
    events = [e[6:] for e in text.split("\n\n") if e.startswith("data: ")]
    return [json.loads(e) if e != "[DONE]" else "[DONE]" for e in events]


def test_streaming_sends_a_role_chunk_content_deltas_a_finish_chunk_and_done():
    app, _ = make()

    async def go():
        async with client(app) as c:
            return await c.post(
                "/v1/chat/completions", json={**BODY, "stream": True}, headers=HEADERS
            )

    r = run(go())
    ev = parse_sse(r.text)
    assert (
        r.headers["content-type"].startswith("text/event-stream")
        and r.headers["cache-control"] == "no-cache"
        and r.headers["x-accel-buffering"] == "no"
    )
    assert (
        ev[0]["choices"][0]["delta"] == {"role": "assistant"}
        and ev[-1] == "[DONE]"
        and ev[-3]["choices"][0]["finish_reason"] == "stop"
        and ev[-2]["choices"] == []
        and ev[-2]["usage"] == {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    )
    assert (
        "".join(e["choices"][0]["delta"].get("content", "") for e in ev[1:-3])
        == "extract the order please now "
    )


def test_the_gateway_is_invisible_to_the_openai_client_shape_a_stream_reassembles_to_the_non_streaming_answer():
    app, _ = make()

    async def go():
        async with client(app) as c:
            a = (await c.post("/v1/chat/completions", json=BODY, headers=HEADERS)).json()[
                "choices"
            ][0]["message"]["content"]
            s = parse_sse(
                (
                    await c.post(
                        "/v1/chat/completions", json={**BODY, "stream": True}, headers=HEADERS
                    )
                ).text
            )
            return a, "".join(e["choices"][0]["delta"].get("content", "") for e in s[1:-3])

    a, b = run(go())
    assert a == b


def test_a_backend_failure_before_the_stream_is_a_502_and_in_the_middle_of_it_an_error_event():
    app, _ = make(EchoBackend(fail_after=0))
    app2, _ = make(EchoBackend(fail_after=2))

    async def go():
        async with client(app) as c:
            plain = await c.post("/v1/chat/completions", json=BODY, headers=HEADERS)
        async with client(app2) as c:
            streamed = await c.post(
                "/v1/chat/completions", json={**BODY, "stream": True}, headers=HEADERS
            )
        return plain, streamed

    plain, streamed = run(go())
    assert plain.status_code == 502 and plain.json()["error"]["code"] == "backend_error"
    ev = parse_sse(streamed.text)
    assert (
        streamed.status_code == 200
        and ev[-1] == "[DONE]"
        and ev[-2]["error"]["code"] == "backend_error"
        and any(e["choices"][0]["delta"].get("content") for e in ev[1:-2])
    )
    assert (
        app2.state.metrics.value("llmapi_requests_total", outcome="backend_error", stream="true")
        == 1
    )


def test_a_failed_request_still_releases_the_users_slot_and_charges_what_was_used():
    app, _ = make(EchoBackend(fail_after=1))

    async def go():
        async with client(app) as c:
            await c.post("/v1/chat/completions", json=BODY, headers=HEADERS)
            return (await c.get("/v1/usage", headers=HEADERS)).json()

    u = run(go())
    assert u["in_flight"] == 0 and u["requests"] == 1


def test_per_user_limits_come_back_as_429_with_retry_after_and_the_model_is_not_called():
    user = User("alice", rpm=60, burst=2, max_concurrent=8, max_tokens_cap=64)
    app, backend = make(user=user, limiter=Limiter(Clock(), Clock()))

    async def go():
        async with client(app) as c:
            return [
                await c.post("/v1/chat/completions", json=BODY, headers=HEADERS) for _ in range(4)
            ]

    rs = run(go())
    assert [r.status_code for r in rs] == [200, 200, 429, 429] and len(backend.calls) == 2
    assert rs[2].headers["retry-after"] == "1" and rs[2].json()["error"] == {
        "message": "too many requests",
        "type": "rate_limit_error",
        "code": "rate_limited",
    }
    assert app.state.metrics.value("llmapi_rejected_total", code="rate_limited") == 2


def test_the_daily_token_quota_is_enforced_from_the_tokens_actually_used():
    user = User("alice", rpm=6000, burst=100, daily_tokens=20, max_concurrent=8, max_tokens_cap=64)
    app, _ = make(user=user, limiter=Limiter(Clock(), Clock()))

    async def go():
        async with client(app) as c:
            return [
                (await c.post("/v1/chat/completions", json=BODY, headers=HEADERS)).status_code
                for _ in range(4)
            ]

    assert run(go()) == [200, 200, 429, 429], "each request uses 10 tokens: after two, 20 >= 20"


def test_per_user_concurrency_is_capped_and_others_are_not_blocked():
    user = User("alice", rpm=6000, burst=100, max_concurrent=2, max_tokens_cap=64)
    backend = EchoBackend(delay=0.05)
    app, _ = make(backend, user=user, settings=Settings(max_inflight=8))

    async def go():
        async with client(app) as c:
            return await asyncio.gather(
                *(c.post("/v1/chat/completions", json=BODY, headers=HEADERS) for _ in range(4))
            )

    rs = run(go())
    assert sorted(r.status_code for r in rs) == [200, 200, 429, 429] and backend.peak == 2
    assert {r.json()["error"]["code"] for r in rs if r.status_code == 429} == {
        "too_many_concurrent"
    }


def test_backpressure_a_full_queue_is_503_instead_of_unbounded_waiting_and_the_slots_are_respected():
    backend = EchoBackend(delay=0.05)
    app, _ = make(backend, settings=Settings(max_inflight=1, max_queue=1))

    async def go():
        async with client(app) as c:
            return await asyncio.gather(
                *(c.post("/v1/chat/completions", json=BODY, headers=HEADERS) for _ in range(4))
            )

    rs = run(go())
    codes = sorted(r.status_code for r in rs)
    assert codes.count(200) == 2 and codes.count(503) == 2 and backend.peak == 1
    shed = next(r for r in rs if r.status_code == 503)
    assert shed.json()["error"]["code"] == "overloaded" and shed.headers["retry-after"] == "2"


def test_requests_that_wait_in_the_queue_are_served_in_order_after_the_slot_frees():
    backend = EchoBackend(delay=0.02)
    app, _ = make(backend, settings=Settings(max_inflight=1, max_queue=8))

    async def go():
        async with client(app) as c:
            return await asyncio.gather(
                *(
                    c.post(
                        "/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": f"req{i}"}]},
                        headers=HEADERS,
                    )
                    for i in range(4)
                )
            )

    rs = run(go())
    assert (
        [r.status_code for r in rs] == [200] * 4
        and [c["messages"][0]["content"] for c in backend.calls] == ["req0", "req1", "req2", "req3"]
        and backend.peak == 1
    )


async def asgi_stream(app, body: dict, headers: dict, *, disconnect_after: int | None):
    """Drive the ASGI app by hand so a client can hang up in the middle of a stream (httpx's test transport cannot)."""
    payload = json.dumps(body).encode()
    sent, chunks, hung_up = [False], [], asyncio.Event()

    async def receive():
        if not sent[0]:
            sent[0] = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await hung_up.wait()
        return {"type": "http.disconnect"}

    async def send(msg):
        if msg["type"] == "http.response.body" and msg.get("body"):
            chunks.append(msg["body"])
            if disconnect_after is not None and len(chunks) >= disconnect_after:
                hung_up.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [
            (k.encode(), v.encode())
            for k, v in {**headers, "content-type": "application/json"}.items()
        ],
        "client": ("1.2.3.4", 1),
        "server": ("t", 80),
        "scheme": "http",
        "root_path": "",
    }
    await asyncio.wait_for(app(scope, receive, send), 5)
    return chunks


def test_a_client_that_hangs_up_mid_stream_cancels_the_model_call_and_frees_everything():
    backend = EchoBackend(delay=0.01)
    app, _ = make(backend, settings=Settings(max_inflight=1))
    long_body = {
        "messages": [{"role": "user", "content": " ".join(f"w{i}" for i in range(60))}],
        "stream": True,
        "max_tokens": 60,
    }

    async def go():
        chunks = await asgi_stream(app, long_body, HEADERS, disconnect_after=3)
        async with client(app) as c:
            usage = (await c.get("/v1/usage", headers=HEADERS)).json()
            follow_up = await c.post("/v1/chat/completions", json=BODY, headers=HEADERS)
        return chunks, usage, follow_up

    chunks, usage, follow_up = run(go())
    assert len(chunks) < 20, "the stream stopped long before its 60 words"
    assert backend.cancelled == 1 and backend.in_flight == 0 and usage["in_flight"] == 0
    assert follow_up.status_code == 200, "the single model slot was released"
    assert (
        app.state.metrics.value(
            "llmapi_requests_total", outcome="client_disconnected", stream="true"
        )
        == 1
    )


def test_a_zero_length_queue_still_serves_a_request_when_a_slot_is_free():
    app, backend = make(settings=Settings(max_inflight=2, max_queue=0))

    async def go():
        async with client(app) as c:
            return (await c.post("/v1/chat/completions", json=BODY, headers=HEADERS)).status_code

    assert run(go()) == 200 and len(backend.calls) == 1


def test_the_in_flight_and_queue_gauges_show_what_is_running_and_waiting_right_now():
    backend = EchoBackend(delay=0.05)
    app, _ = make(backend, settings=Settings(max_inflight=1, max_queue=4, admin_token="adm"))

    async def go():
        async with client(app) as c:
            tasks = [
                asyncio.create_task(c.post("/v1/chat/completions", json=BODY, headers=HEADERS))
                for _ in range(3)
            ]
            await asyncio.sleep(0.06)  # one is generating, two are waiting for the slot
            text = (await c.get("/metrics", headers={"authorization": "Bearer adm"})).text
            await asyncio.gather(*tasks)
            after = (await c.get("/metrics", headers={"authorization": "Bearer adm"})).text
            return text, after

    during, after = run(go())
    assert "llmapi_in_flight 1" in during and "llmapi_queue_depth 2" in during
    assert "llmapi_in_flight 0" in after and "llmapi_queue_depth 0" in after


def test_health_ready_models_usage_and_metrics_endpoints():
    app, backend = make(settings=Settings(admin_token="adm"))

    async def go():
        async with client(app) as c:
            out = {
                "health": await c.get("/healthz"),
                "ready": await c.get("/readyz"),
                "models": await c.get("/v1/models", headers=HEADERS),
                "models_anon": await c.get("/v1/models"),
            }
            await c.post("/v1/chat/completions", json=BODY, headers=HEADERS)
            out["metrics_anon"] = await c.get("/metrics")
            out["metrics"] = await c.get("/metrics", headers={"authorization": "Bearer adm"})
            backend.is_ready = False
            out["not_ready"] = await c.get("/readyz")
            out["still_alive"] = await c.get("/healthz")
            return out

    o = run(go())
    assert (
        o["health"].json() == {"status": "ok"}
        and o["ready"].status_code == 200
        and o["models"].json()["data"][0]["id"] == "order-extractor"
        and o["models_anon"].status_code == 401
    )
    assert o["metrics_anon"].status_code == 401 and o["metrics"].status_code == 200
    assert (
        'llmapi_requests_total{outcome="ok",stream="false"} 1' in o["metrics"].text
        and 'llmapi_tokens_total{kind="completion"} 5' in o["metrics"].text
        and "llmapi_request_seconds_count 1" in o["metrics"].text
    )
    assert o["not_ready"].status_code == 503 and o["still_alive"].status_code == 200, (
        "liveness must not depend on the model server"
    )


def test_logs_identify_the_request_and_user_but_never_contain_the_prompt_or_the_key(caplog):
    app, _ = make()

    async def go():
        async with client(app) as c:
            await c.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "my secret card 4111 1111 1111 1111"}]
                },
                headers={**HEADERS, "x-request-id": "r1"},
            )

    with caplog.at_level(logging.INFO, logger="llmapi"):
        run(go())
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert (
        '"rid": "r1"' in text
        and '"user": "alice"' in text
        and "4111" not in text
        and KEY not in text
    )


def test_llama_server_backend_parses_the_openai_stream_including_usage_and_reports_readiness():
    sse = (
        "".join(
            [
                "data: " + json.dumps(e) + "\n\n"
                for e in [
                    {"choices": [{"delta": {"content": "Hel"}}]},
                    {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}},
                ]
            ]
        )
        + "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(request.content)
        assert (
            body["stream"] is True
            and body["max_tokens"] == 5
            and body["stream_options"] == {"include_usage": True}
        )
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    backend = LlamaServerBackend(
        "http://llama", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    async def go():
        deltas = [d async for d in backend.stream([{"role": "user", "content": "x"}], 5, 0.0)]
        return await backend.ready(), deltas

    ready, deltas = run(go())
    assert (
        ready
        and "".join(d.text for d in deltas) == "Hello"
        and deltas[-1].finish_reason == "stop"
        and (deltas[-1].prompt_tokens, deltas[-1].completion_tokens) == (7, 2)
    )


def test_llama_server_backend_turns_http_errors_and_an_unreachable_server_into_backend_errors():
    bad = LlamaServerBackend(
        "http://llama",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(500, text="oops"))
        ),
    )

    async def boom(request):
        raise httpx.ConnectError("refused")

    down = LlamaServerBackend(
        "http://llama", client=httpx.AsyncClient(transport=httpx.MockTransport(boom))
    )

    async def go(b):
        return [d async for d in b.stream([{"role": "user", "content": "x"}], 5, 0.0)]

    for b in (bad, down):
        with pytest.raises(BackendError):
            run(go(b))
    assert run(down.ready()) is False


def test_keys_load_from_a_config_list_with_hashes_or_plaintext_and_reject_mistakes():
    from llmapi.auth import load_keys

    store = load_keys(
        [
            {"user": "a", "key_sha256": hash_key("k1"), "rpm": 5, "max_tokens_cap": 32},
            {"user": "b", "key": "k2"},
        ]
    )
    assert (
        store.authenticate("k1").rpm == 5
        and store.authenticate("k1").max_tokens_cap == 32
        and store.authenticate("k2").id == "b"
        and store.authenticate("k3") is None
    )
    for bad in (
        [{"user": "a"}],
        [{"user": "a", "key": "x", "key_sha256": "y"}],
        [{"user": "a", "key": "x", "rmp": 5}],
    ):
        with pytest.raises(ValueError):
            load_keys(bad)


def test_the_server_refuses_to_start_without_keys(monkeypatch):
    from llmapi import __main__ as m

    monkeypatch.delenv("LLMAPI_KEYS", raising=False)
    monkeypatch.delenv("LLMAPI_KEYS_FILE", raising=False)
    with pytest.raises(SystemExit):
        m.build_app()
    monkeypatch.setenv("LLMAPI_KEYS", json.dumps([{"user": "a", "key": "k"}]))
    monkeypatch.setenv("LLMAPI_MAX_INFLIGHT", "3")
    app = m.build_app()
    assert app.state.settings.max_inflight == 3 and app.state.settings.admin_token is None


def test_one_gateway_serves_several_models_and_routes_by_the_model_field():
    a, b = EchoBackend(), EchoBackend()
    keys = ApiKeyStore()
    keys.add(KEY, USER)
    app = create_app({"extractor": a, "chat": b}, keys)

    async def go():
        async with client(app) as c:
            r1 = await c.post(
                "/v1/chat/completions", json={**BODY, "model": "chat"}, headers=HEADERS
            )
            r2 = await c.post(
                "/v1/chat/completions", json={**BODY, "model": "extractor"}, headers=HEADERS
            )
            r3 = await c.post(
                "/v1/chat/completions", json={**BODY, "model": "nope"}, headers=HEADERS
            )
            return r1, r2, r3, await c.get("/v1/models", headers=HEADERS), await c.get("/readyz")

    r1, r2, r3, models, ready = run(go())
    assert (len(a.calls), len(b.calls)) == (1, 1) and r1.status_code == r2.status_code == 200
    assert (
        r3.status_code == 404
        and r3.json()["error"]["code"] == "model_not_found"
        and "extractor, chat" in r3.json()["error"]["message"]
    )
    assert [m["id"] for m in models.json()["data"]] == ["extractor", "chat"] and ready.json()[
        "models"
    ] == {"extractor": True, "chat": True}


def test_the_gateway_is_not_ready_when_any_model_server_is_down():
    keys = ApiKeyStore()
    keys.add(KEY, USER)
    app = create_app({"up": EchoBackend(), "down": EchoBackend(is_ready=False)}, keys)

    async def go():
        async with client(app) as c:
            return await c.get("/readyz")

    r = run(go())
    assert r.status_code == 503 and r.json()["models"] == {"up": True, "down": False}
