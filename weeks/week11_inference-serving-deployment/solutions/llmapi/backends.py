"""Model backends: anything that turns a chat request into a stream of text deltas. The API is written against ``Backend`` so the same app serves a real llama.cpp server,
a test double, or (by writing another class) vLLM, an Ollama endpoint or a hosted API."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

import httpx

ABSTAIN = "I don't know based on the provided sources."  # the refusal sentence /v1/ask asks the model for, and sends itself when retrieval found nothing


@dataclass
class Delta:
    text: str = ""
    finish_reason: str | None = None
    prompt_tokens: int | None = None  # filled on the last delta when the backend knows them
    completion_tokens: int | None = None


class BackendError(RuntimeError):
    """The model server failed or is unreachable (the API turns this into a 502/503 or, mid-stream, an error event)."""


class Backend(Protocol):
    async def ready(self) -> bool: ...

    def stream(
        self, messages: list[dict], max_tokens: int, temperature: float
    ) -> AsyncIterator[Delta]: ...


class EchoBackend:
    """A deterministic test double: streams the last user message back word by word (``delay`` seconds between words), optionally failing part-way."""

    def __init__(self, delay: float = 0.0, fail_after: int | None = None, is_ready: bool = True):
        self.delay, self.fail_after, self.is_ready = delay, fail_after, is_ready
        self.calls: list[dict] = []
        self.cancelled = 0
        self.in_flight = 0
        self.peak = 0

    async def ready(self) -> bool:
        return self.is_ready

    async def stream(
        self, messages: list[dict], max_tokens: int, temperature: float
    ) -> AsyncIterator[Delta]:
        self.calls.append(
            {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        )
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        words = [
            w + " "
            for w in next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
            ).split()
        ][:max_tokens] or ["(empty) "]
        try:
            for i, w in enumerate(words):
                if self.fail_after is not None and i >= self.fail_after:
                    raise BackendError("the model server went away")
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield Delta(w)
            n_prompt = sum(len(m["content"].split()) for m in messages)
            yield Delta("", "length" if len(words) == max_tokens else "stop", n_prompt, len(words))
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.in_flight -= 1


class FixedBackend:
    """Answers every request with one fixed text and never touches a model: what the gateway uses to abstain when retrieval found nothing relevant (instant, free, and exactly the
    refusal sentence: a small model cannot be trusted to produce it)."""

    def __init__(self, text: str):
        self.text = text

    async def ready(self) -> bool:
        return True

    async def stream(
        self, messages: list[dict], max_tokens: int, temperature: float
    ) -> AsyncIterator[Delta]:
        yield Delta(self.text)
        yield Delta("", "stop", 0, 0)


class LlamaServerBackend:
    """Streams from a llama-server's OpenAI-compatible endpoint (``/v1/chat/completions`` with ``stream: true``)."""

    def __init__(
        self, base_url: str, *, timeout: float = 120.0, client: httpx.AsyncClient | None = None
    ):
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=3.0))

    async def ready(self) -> bool:
        try:
            r = await self.client.get(f"{self.base_url}/health", timeout=2)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except (httpx.HTTPError, ValueError):
            return False

    async def stream(
        self, messages: list[dict], max_tokens: int, temperature: float
    ) -> AsyncIterator[Delta]:
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        try:
            async with self.client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=body
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise BackendError(f"the model server answered HTTP {resp.status_code}")
                finish, usage = None, {}
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
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for ch in obj.get("choices", []):
                        text = (ch.get("delta") or {}).get("content")
                        if text:
                            yield Delta(text)
                        finish = ch.get("finish_reason") or finish
                yield Delta(
                    "", finish or "stop", usage.get("prompt_tokens"), usage.get("completion_tokens")
                )
        except httpx.HTTPError as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc
