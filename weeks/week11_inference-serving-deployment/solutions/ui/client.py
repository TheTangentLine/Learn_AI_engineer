"""The client side of the chat UI: talks to the API, turns the SSE stream into typed events, and does the arithmetic the UI shows (cost, speed, citation checks).

No Streamlit in this module: everything here is plain Python so it can be tested without a browser. ``ChatApp`` (``chat_app.py``) only draws what this module returns.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx

CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Event:
    kind: str  # sources | token | done | error
    data: object = None


@dataclass
class AnswerStats:
    request_id: str = ""
    status: int = 0
    ttft: float | None = None
    seconds: float = 0.0
    chunks: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    error: dict | None = None
    sources: list[dict] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        span = self.seconds - (self.ttft or 0.0)
        return (self.chunks - 1) / span if self.chunks > 1 and span > 0 else 0.0


@dataclass(frozen=True)
class PriceCard:
    """Dollars per million tokens: an input the user types into the sidebar (an assumption, never a built-in price)."""

    input_per_m: float = 0.0
    output_per_m: float = 0.0

    def cost(self, prompt_tokens: float | None, completion_tokens: float | None) -> float:
        return (
            (prompt_tokens or 0) * self.input_per_m + (completion_tokens or 0) * self.output_per_m
        ) / 1e6


def parse_sse(lines: Iterator[str]) -> Iterator[Event]:
    """SSE lines -> events. ``event: sources`` carries the retrieved passages; ``data:`` lines carry OpenAI chunks, an error object, or [DONE]."""
    pending = None
    for line in lines:
        if line.startswith("event:"):
            pending = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            yield Event("done")
            return
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        if pending == "sources":
            yield Event("sources", obj)
        elif "error" in obj:
            yield Event("error", obj["error"])
        else:
            if obj.get("usage"):
                yield Event("usage", obj["usage"])
            for ch in obj.get("choices", []):
                text = (ch.get("delta") or {}).get("content")
                if text:
                    yield Event("token", text)
        pending = None


class ApiClient:
    """``ask`` (retrieval + answer) and ``chat`` (plain) as generators of Events, filling an AnswerStats as they go. ``transport`` lets tests inject a fake server."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = {"authorization": f"Bearer {api_key}"}
        self.client = httpx.Client(timeout=timeout, transport=transport)

    def _stream(self, path: str, body: dict, stats: AnswerStats) -> Iterator[Event]:
        t0 = time.perf_counter()
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}{path}",
                json={**body, "stream": True},
                headers=self.headers,
            ) as r:
                stats.status = r.status_code
                stats.request_id = r.headers.get("x-request-id", "")
                if r.status_code != 200:
                    r.read()
                    try:
                        err = r.json()["error"]
                    except (ValueError, KeyError):
                        err = {"message": f"HTTP {r.status_code}", "code": "http_error"}
                    err["retry_after"] = r.headers.get("retry-after")
                    stats.error = err
                    yield Event("error", err)
                    return
                for ev in parse_sse(r.iter_lines()):
                    if ev.kind == "token":
                        stats.chunks += 1
                        if stats.ttft is None:
                            stats.ttft = time.perf_counter() - t0
                    elif ev.kind == "sources":
                        stats.sources = ev.data
                    elif ev.kind == "usage":
                        stats.prompt_tokens = ev.data.get("prompt_tokens")
                        stats.completion_tokens = ev.data.get("completion_tokens")
                        stats.total_tokens = ev.data.get("total_tokens")
                    elif ev.kind == "error":
                        stats.error = ev.data
                    yield ev
        except httpx.HTTPError as exc:
            stats.error = {"message": f"could not reach the API: {exc}", "code": "unreachable"}
            yield Event("error", stats.error)
        finally:
            stats.seconds = time.perf_counter() - t0

    def ask(
        self,
        question: str,
        stats: AnswerStats,
        *,
        k: int = 4,
        model: str = "default",
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Iterator[Event]:
        yield from self._stream(
            "/v1/ask",
            {
                "question": question,
                "k": k,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            stats,
        )

    def chat(
        self,
        messages: list[dict],
        stats: AnswerStats,
        *,
        model: str = "default",
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Iterator[Event]:
        yield from self._stream(
            "/v1/chat/completions",
            {
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            stats,
        )

    def feedback(
        self, request_id: str, rating: int, reason: str = "other", comment: str = ""
    ) -> bool:
        try:
            r = self.client.post(
                f"{self.base_url}/v1/feedback",
                json={
                    "request_id": request_id,
                    "rating": rating,
                    "reason": reason,
                    "comment": comment,
                },
                headers=self.headers,
            )
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def usage(self) -> dict | None:
        try:
            r = self.client.get(f"{self.base_url}/v1/usage", headers=self.headers)
            return r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None


def estimate_tokens(text: str) -> int:
    """About one token per 4 characters: only used when the API did not report usage on a stream (the UI labels it an estimate)."""
    return max(1, round(len(text) / 4))


def used_citations(answer: str) -> list[int]:
    return sorted({int(n) for n in CITATION.findall(answer)})


def citation_report(answer: str, n_sources: int) -> dict:
    """Which sources the answer cites, which of those do not exist (a fabricated reference), which sources were retrieved but never cited, and whether it abstained."""
    used = used_citations(answer)
    return {
        "cited": [n for n in used if 1 <= n <= n_sources],
        "invalid": [n for n in used if not 1 <= n <= n_sources],
        "unused": [n for n in range(1, n_sources + 1) if n not in used],
        "abstained": "i don't know" in answer.lower(),
        "uncited_claims": bool(answer.strip())
        and not used
        and "i don't know" not in answer.lower(),
    }


def render_with_links(answer: str, n_sources: int) -> str:
    """Markdown: a valid [n] becomes bold ([**n**]); an invalid one is struck through so the reader sees the model cited something that does not exist."""
    return CITATION.sub(
        lambda m: (
            f"[**{m.group(1)}**]" if 1 <= int(m.group(1)) <= n_sources else f"~~[{m.group(1)}]~~"
        ),
        answer,
    )


def format_cost(price: PriceCard, stats: AnswerStats, answer: str) -> dict:
    """What to show under an answer. Token counts come from the API's usage when it reported them and are ESTIMATES otherwise (flagged)."""
    exact = stats.prompt_tokens is not None and stats.completion_tokens is not None
    prompt = stats.prompt_tokens if stats.prompt_tokens is not None else 0
    completion = (
        stats.completion_tokens if stats.completion_tokens is not None else estimate_tokens(answer)
    )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "estimated": not exact,
        "dollars": price.cost(prompt, completion),
        "ttft": stats.ttft,
        "seconds": stats.seconds,
        "tokens_per_second": stats.tokens_per_second,
    }
