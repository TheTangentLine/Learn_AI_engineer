"""The FastAPI application: OpenAI-compatible ``/v1/chat/completions`` (JSON or Server-Sent Events), API-key auth, per-user limits, backpressure, health checks, metrics.

Request path:  request id -> authenticate (401) -> validate (400/422) -> per-user admission: quota, concurrency, rate (429) -> a model slot, with a bounded queue (503) ->
               stream tokens -> charge the tokens actually used -> release everything, whatever happened (success, error, client disconnect).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from .auth import ApiKeyStore, User, bearer_token
from .backends import ABSTAIN, Backend, BackendError, Delta, FixedBackend
from .metrics import Metrics
from .ratelimit import Limiter

log = logging.getLogger("llmapi")


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=64)
    model: str = "default"  # selects the backend when the gateway serves several
    max_tokens: int | None = Field(
        default=None, ge=1
    )  # omitted: the server default, capped by the user's limit
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    rating: int  # +1 or -1
    reason: str = Field(
        default="other", pattern="^(wrong|unhelpful|bad_citation|rude|slow|unsafe|other)$"
    )
    comment: str = Field(default="", max_length=500)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=10)
    model: str = "default"
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False


@dataclass
class Settings:
    max_inflight: int = 4  # model slots (match the model server's parallelism)
    max_queue: int = 8  # requests allowed to WAIT for a slot; beyond this the API says 503 instead of letting latency grow without bound
    max_prompt_chars: int = 16_000
    request_timeout: float = 120.0
    model_name: str = "order-extractor"
    admin_token: str | None = None  # protects /metrics when set
    queue_retry_after: float = 2.0
    default_max_tokens: int = 256
    feedback_path: str | None = None  # append-only JSONL; None keeps feedback in memory only
    abstain_without_sources: bool = False  # /v1/ask with nothing retrieved: answer with the refusal sentence without calling the model
    ask_model: str | None = (
        None  # which backend answers /v1/ask when the request does not name one (needed with several models)
    )


def error_body(message: str, type_: str, code: str) -> dict:
    return {"error": {"message": message, "type": type_, "code": code}}


def create_app(
    backend: Backend | dict[str, Backend],
    keys: ApiKeyStore,
    *,
    settings: Settings | None = None,
    limiter: Limiter | None = None,
    metrics: Metrics | None = None,
    retriever=None,
) -> FastAPI:
    """``backend`` is one Backend (every request uses it, whatever ``model`` says) or a dict {model name: Backend}: the request's ``model`` field then selects the model,
    and an unknown name is a 404 ``model_not_found``. Model servers are separate processes (one per model), so one gateway can serve several."""
    cfg = settings or Settings()
    backends: dict[str, Backend] = backend if isinstance(backend, dict) else {}
    default_backend: Backend | None = None if isinstance(backend, dict) else backend
    lim = limiter or Limiter()
    met = metrics or Metrics()
    for name, kind, text in [
        ("llmapi_requests_total", "counter", "Chat requests by outcome."),
        (
            "llmapi_rejected_total",
            "counter",
            "Requests refused before reaching the model, by reason.",
        ),
        ("llmapi_tokens_total", "counter", "Tokens processed, by kind."),
        ("llmapi_in_flight", "gauge", "Requests currently holding a model slot."),
        ("llmapi_queue_depth", "gauge", "Requests waiting for a model slot."),
        ("llmapi_request_seconds", "histogram", "End-to-end request latency."),
        ("llmapi_ttft_seconds", "histogram", "Time to the first generated token."),
        ("llmapi_retrieval_seconds", "histogram", "Time spent retrieving passages for /v1/ask."),
        ("llmapi_feedback_total", "counter", "Thumbs up and down, by reason."),
        (
            "llmapi_abstained_total",
            "counter",
            "/v1/ask requests answered with the refusal sentence because nothing relevant was retrieved.",
        ),
    ]:
        met.describe(name, kind, text)
    slots = asyncio.Semaphore(cfg.max_inflight)
    state = {"waiting": 0, "running": 0}
    app = FastAPI(title="LLM API", docs_url=None, redoc_url=None)
    app.state.metrics, app.state.limiter, app.state.settings, app.state.backend = (
        met,
        lim,
        cfg,
        backend,
    )
    app.state.backends = backends or {cfg.model_name: default_backend}
    app.state.feedback = []

    # ------------------------------------------------------------------ plumbing
    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.rid = rid
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return JSONResponse(
            error_body(
                f"invalid request: {where}: {first.get('msg', 'bad value')}",
                "invalid_request_error",
                "invalid_request",
            ),
            status_code=400,
        )

    def reject(
        status: int,
        code: str,
        message: str,
        kind: str = "invalid_request_error",
        retry_after: float | None = None,
    ) -> JSONResponse:
        met.inc("llmapi_rejected_total", code=code)
        headers = {"retry-after": str(max(1, math.ceil(retry_after)))} if retry_after else {}
        if status == 401:
            headers["www-authenticate"] = "Bearer"
        return JSONResponse(error_body(message, kind, code), status_code=status, headers=headers)

    def authenticate(request: Request) -> User | None:
        return keys.authenticate(bearer_token(request.headers.get("authorization")))

    # ------------------------------------------------------------------ probes and metadata
    @app.get("/healthz")
    async def healthz():
        """Liveness: the PROCESS is up. It must not depend on the model server (a restart will not fix the model server)."""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        """Readiness: this instance can serve traffic RIGHT NOW (its model server answers). A load balancer stops routing here when it fails."""
        status = {name: await b.ready() for name, b in app.state.backends.items()}
        ok = all(status.values())
        return JSONResponse(
            {"status": "ready" if ok else "model server unavailable", "models": status},
            status_code=200 if ok else 503,
        )

    @app.get("/metrics")
    async def metrics_endpoint(request: Request):
        if cfg.admin_token and request.headers.get("authorization") != f"Bearer {cfg.admin_token}":
            return reject(401, "unauthorized", "metrics need the admin token")
        met.set_gauge("llmapi_in_flight", state["running"])
        met.set_gauge("llmapi_queue_depth", state["waiting"])
        return PlainTextResponse(met.render(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models(request: Request):
        if authenticate(request) is None:
            return reject(401, "invalid_api_key", "a valid API key is required")
        return {
            "object": "list",
            "data": [{"id": name, "object": "model"} for name in app.state.backends],
        }

    @app.post("/v1/feedback")
    async def feedback(body: FeedbackRequest, request: Request):
        """A thumbs up or down on one answer, tied to its request id (never its text: the API does not keep prompts). The comment is user text: it is capped, and a
        deployment must treat it as personal data (Week 8)."""
        user = authenticate(request)
        if user is None:
            return reject(401, "invalid_api_key", "a valid API key is required")
        if body.rating not in (-1, 1):
            return reject(400, "invalid_rating", "rating must be +1 or -1")
        record = {
            "at": time.time(),
            "user": user.id,
            "request_id": body.request_id,
            "rating": body.rating,
            "reason": body.reason,
            "comment": body.comment,
        }
        app.state.feedback.append(record)
        if cfg.feedback_path:
            with open(cfg.feedback_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        met.inc(
            "llmapi_feedback_total", rating="up" if body.rating > 0 else "down", reason=body.reason
        )
        return {"status": "recorded"}

    @app.get("/v1/usage")
    async def usage(request: Request):
        user = authenticate(request)
        if user is None:
            return reject(401, "invalid_api_key", "a valid API key is required")
        return {"user": user.id, **lim.usage(user)}

    # ------------------------------------------------------------------ chat
    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest, request: Request):
        user = authenticate(request)
        if user is None:
            return reject(
                401, "invalid_api_key", "a valid API key is required (Authorization: Bearer ...)"
            )
        return await handle(request, body, user)

    async def handle(
        request: Request,
        body: ChatRequest,
        user: User,
        *,
        sources: list[dict] | None = None,
        override: Backend | None = None,
    ):
        """Everything after authentication: validation, admission, a model slot, generation, accounting. Shared by chat and ask (``sources`` are the retrieved
        passages of an ask: they travel with the answer, as the first SSE event or a field of the JSON)."""
        rid = request.state.rid
        chosen = override or default_backend or backends.get(body.model)
        if chosen is None:
            return reject(
                404,
                "model_not_found",
                f"unknown model {body.model!r}; available: {', '.join(backends)}",
            )
        max_tokens = (
            body.max_tokens
            if body.max_tokens is not None
            else min(cfg.default_max_tokens, user.max_tokens_cap)
        )
        if max_tokens > user.max_tokens_cap:
            return reject(
                400, "max_tokens_too_large", f"max_tokens may be at most {user.max_tokens_cap}"
            )
        if sum(len(m.content) for m in body.messages) > cfg.max_prompt_chars:
            return reject(
                400,
                "prompt_too_long",
                f"the conversation may be at most {cfg.max_prompt_chars} characters",
            )
        decision = lim.admit(user)
        if not decision.allowed:
            return reject(
                429,
                decision.code,
                {
                    "rate_limited": "too many requests",
                    "quota_exceeded": "the daily token quota is used up",
                    "too_many_concurrent": "too many requests in flight",
                }[decision.code],
                "rate_limit_error",
                decision.retry_after,
            )
        # from here on the user's concurrency slot is held: every path below must release it
        used = {"tokens": 0, "prompt": 0, "completion": 0}
        released = {"done": False}

        def finish(outcome: str) -> None:
            if not released["done"]:
                released["done"] = True
                lim.release(user, used["tokens"])
                met.inc("llmapi_requests_total", outcome=outcome, stream=str(body.stream).lower())
                log.info(
                    json.dumps(
                        {
                            "event": "request",
                            "rid": rid,
                            "user": user.id,
                            "outcome": outcome,
                            "tokens": used["tokens"],
                            "stream": body.stream,
                        }
                    )
                )

        if state["waiting"] >= cfg.max_queue and slots.locked():
            finish("overloaded")
            return reject(
                503,
                "overloaded",
                "the server is busy; retry shortly",
                "server_error",
                cfg.queue_retry_after,
            )
        state["waiting"] += 1
        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(slots.acquire(), timeout=cfg.request_timeout)
        except TimeoutError:
            state["waiting"] -= 1
            finish("queue_timeout")
            return reject(
                503,
                "queue_timeout",
                "timed out waiting for a model slot",
                "server_error",
                cfg.queue_retry_after,
            )
        except asyncio.CancelledError:
            state["waiting"] -= 1
            finish("cancelled")
            raise
        state["waiting"] -= 1
        state["running"] += 1
        messages = [m.model_dump() for m in body.messages]

        async def generate() -> AsyncIterator[Delta]:
            async with contextlib.aclosing(
                chosen.stream(messages, max_tokens, body.temperature)
            ) as it:
                async for d in it:
                    yield d

        def note(d: Delta, first: list[float]) -> None:
            if d.text and not first:
                first.append(time.perf_counter() - t0)
                met.observe("llmapi_ttft_seconds", first[0])
            if d.prompt_tokens is not None:
                used["prompt"] += d.prompt_tokens
                used["tokens"] += d.prompt_tokens
                met.inc("llmapi_tokens_total", d.prompt_tokens, kind="prompt")
            if d.completion_tokens is not None:
                used["completion"] += d.completion_tokens
                used["tokens"] += d.completion_tokens
                met.inc("llmapi_tokens_total", d.completion_tokens, kind="completion")

        def release_slot(outcome: str) -> None:
            state["running"] -= 1
            slots.release()
            met.observe("llmapi_request_seconds", time.perf_counter() - t0)
            finish(outcome)

        created, cid = int(time.time()), f"chatcmpl-{rid}"

        if not body.stream:
            parts: list[str] = []
            finish_reason, first = "stop", []
            try:
                async for d in generate():
                    note(d, first)
                    parts.append(d.text)
                    finish_reason = d.finish_reason or finish_reason
            except BackendError as exc:
                release_slot("backend_error")
                return reject(
                    502, "backend_error", f"the model server failed: {exc}", "server_error"
                )
            except BaseException:
                release_slot("cancelled")
                raise
            release_slot("ok")
            return {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": cfg.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "".join(parts)},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": used["prompt"],
                    "completion_tokens": used["completion"],
                    "total_tokens": used["tokens"],
                },
                **({"sources": sources} if sources is not None else {}),
            }

        async def sse() -> AsyncIterator[str]:
            outcome, first = "ok", []

            def chunk(delta: dict, reason: str | None = None) -> str:
                payload = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": cfg.model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
                }
                return "data: " + json.dumps(payload) + "\n\n"

            try:
                if sources is not None:
                    yield "event: sources\ndata: " + json.dumps(sources) + "\n\n"
                yield chunk({"role": "assistant"})
                async for d in generate():
                    note(d, first)
                    if d.text:
                        yield chunk({"content": d.text})
                    if d.finish_reason:
                        yield chunk({}, d.finish_reason)
                usage = {
                    "prompt_tokens": used["prompt"],
                    "completion_tokens": used["completion"],
                    "total_tokens": used["tokens"],
                }
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": cfg.model_name,
                            "choices": [],
                            "usage": usage,
                        }
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"
            except BackendError as exc:
                # the status line has long been sent: the only honest thing left is an error event, then the end of the stream
                outcome = "backend_error"
                yield (
                    "data: "
                    + json.dumps(
                        error_body(
                            f"the model server failed: {exc}", "server_error", "backend_error"
                        )
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"
            except (asyncio.CancelledError, GeneratorExit):
                outcome = "client_disconnected"
                raise
            finally:
                release_slot(outcome)

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    if retriever is not None:
        preparing = {"n": 0}
        prepare_slots = asyncio.Semaphore(cfg.max_inflight)

        @app.post("/v1/ask")
        async def ask(body: AskRequest, request: Request):
            """Retrieval-augmented answer: find passages, put them in the prompt, answer with citations. Passes through the SAME admission, queue, slots and metrics as chat."""
            user = authenticate(request)
            if user is None:
                return reject(
                    401,
                    "invalid_api_key",
                    "a valid API key is required (Authorization: Bearer ...)",
                )
            # The retriever may be slow (a cross-encoder, a model call): it runs in a worker thread so the event loop stays free for health checks and other users, and it is
            # bounded the same way model slots are: at most ``max_inflight`` run at once, and when that many plus ``max_queue`` more are already preparing the answer is a fast 503
            # (without this, a slow retriever makes every request wait in the operating system's queue BEFORE admission control can say no).
            if preparing["n"] >= cfg.max_inflight + cfg.max_queue:
                return reject(503, "overloaded", "the server is busy; retry shortly", "server_error", cfg.queue_retry_after)
            preparing["n"] += 1
            try:
                async with prepare_slots:
                    t = time.perf_counter()
                    messages, sources = await asyncio.to_thread(retriever, body.question, body.k)
                    met.observe("llmapi_retrieval_seconds", time.perf_counter() - t)
            finally:
                preparing["n"] -= 1
            chat_body = ChatRequest(
                messages=[Message(**m) for m in messages],
                model=body.model if body.model != "default" else (cfg.ask_model or body.model),
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                stream=body.stream,
            )
            override = None
            if not sources and cfg.abstain_without_sources:
                met.inc("llmapi_abstained_total")
                override = FixedBackend(ABSTAIN)
            return await handle(request, chat_body, user, sources=sources, override=override)

    return app
