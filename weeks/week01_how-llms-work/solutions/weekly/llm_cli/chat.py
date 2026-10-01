"""Chat session logic: history, trimming, streaming with retries, cost meter, persistence.

No terminal I/O here - everything is injectable (stream function, sleep, token counter) so the
whole thing is unit-testable offline. ``__main__.py`` is the thin terminal layer on top.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.llm import DEFAULT_MODELS, LLMResponse, Usage, resolve
from common.llm import stream as default_stream

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def is_retryable(exc: BaseException) -> bool:
    """429 / 5xx / timeouts / connection errors can succeed on retry; 4xx cannot."""
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in RETRYABLE_STATUS
    return isinstance(exc, TimeoutError | ConnectionError) or type(exc).__name__ in (
        "APIConnectionError",
        "APITimeoutError",
    )


def default_token_counter(text: str) -> int:
    """Cheap estimate used only for *trimming decisions* (billing uses the API's usage block)."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:  # offline first run, etc.
        return len(text) // 4 + 1


@dataclass
class SendResult:
    text: str = ""
    response: LLMResponse | None = None  # final usage/cost; None if interrupted or failed
    ttft_s: float | None = None
    retries: int = 0
    dropped_messages: int = 0  # old messages trimmed to fit the budget
    interrupted: bool = False
    error: Exception | None = None
    over_budget: bool = False  # the newest message alone exceeds the context budget


@dataclass
class Meter:
    calls: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    by_model: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def record(self, resp: LLMResponse) -> None:
        self.calls += 1
        self.usage = self.usage + resp.usage
        self.cost_usd += resp.cost_usd
        self.by_model[f"{resp.provider}/{resp.model}"] += resp.cost_usd


class ChatSession:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        system: str | None = None,
        max_context_tokens: int = 8000,
        max_retries: int = 3,
        base_delay: float = 1.0,
        stream_fn: Callable[..., Iterator[str]] = default_stream,
        sleep: Callable[[float], None] = time.sleep,
        token_counter: Callable[[str], int] = default_token_counter,
    ):
        self.provider, self.model = resolve(provider, model)
        self.system = system
        self.max_context_tokens = max_context_tokens
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._stream_fn, self._sleep, self._count = stream_fn, sleep, token_counter
        self.messages: list[dict[str, Any]] = []  # {"role","content"[, "partial": True]}
        self.meter = Meter()

    # ---------------------------------------------------------------- history

    def api_messages(self) -> list[dict[str, str]]:
        """What we actually send: role/content only (strip our own bookkeeping keys)."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def context_tokens(self) -> int:
        total = self._count(self.system) if self.system else 0
        return total + sum(self._count(m["content"]) + 4 for m in self.messages)

    def trim(self) -> int:
        """Sliding window: drop the oldest messages until we fit the budget.

        Always keeps the newest message, and never leaves an assistant message first
        (conversations must start with a user turn). Returns how many were dropped.
        """
        dropped = 0
        while len(self.messages) > 1 and self.context_tokens() > self.max_context_tokens:
            self.messages.pop(0)
            dropped += 1
            while len(self.messages) > 1 and self.messages[0]["role"] != "user":
                self.messages.pop(0)
                dropped += 1
        return dropped

    def clear(self) -> None:
        self.messages.clear()

    def pop_last_exchange(self) -> str | None:
        """Remove the last assistant reply and user message; return the user text (for /retry)."""
        while self.messages and self.messages[-1]["role"] == "assistant":
            self.messages.pop()
        return self.messages.pop()["content"] if self.messages else None

    # ---------------------------------------------------------------- provider switching

    def set_provider(self, provider: str, model: str | None = None) -> None:
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"unknown provider {provider!r}; choose from {list(DEFAULT_MODELS)}")
        self.provider, self.model = provider, model or DEFAULT_MODELS[provider]
        # History is stored provider-neutral, so it carries over unchanged.

    # ---------------------------------------------------------------- the main call

    def send(
        self,
        user_text: str,
        on_chunk: Callable[[str], None] | None = None,
        on_retry: Callable[[int, float, Exception], None] | None = None,
    ) -> SendResult:
        result = SendResult()
        self.messages.append({"role": "user", "content": user_text})
        result.dropped_messages = self.trim()
        result.over_budget = self.context_tokens() > self.max_context_tokens

        parts: list[str] = []
        captured: list[LLMResponse] = []
        t0 = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            try:
                for chunk in self._stream_fn(
                    self.api_messages(),
                    system=self.system,
                    provider=self.provider,
                    model=self.model,
                    on_done=captured.append,
                ):
                    if result.ttft_s is None:
                        result.ttft_s = time.perf_counter() - t0
                    parts.append(chunk)
                    if on_chunk:
                        on_chunk(chunk)
                result.error = None
                break
            except KeyboardInterrupt:
                result.interrupted = True
                break
            except Exception as exc:
                # Retry only if nothing was shown yet (can't un-print half an answer) and the
                # error is transient. Otherwise surface it and keep whatever text we have.
                if parts or not is_retryable(exc) or attempt == self.max_retries:
                    result.error = exc
                    break
                result.retries += 1
                delay = random.uniform(0, self.base_delay * 2**attempt)  # full jitter
                if on_retry:
                    on_retry(attempt + 1, delay, exc)
                self._sleep(delay)

        result.text = "".join(parts)
        if captured:
            result.response = captured[-1]
            self.meter.record(result.response)

        if result.text:
            entry: dict[str, Any] = {"role": "assistant", "content": result.text}
            if result.interrupted or result.error:
                entry["partial"] = True
            self.messages.append(entry)
        else:
            self.messages.pop()  # nothing came back: don't leave a dangling user turn
        return result

    # ---------------------------------------------------------------- persistence

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "provider": self.provider,
                    "model": self.model,
                    "system": self.system,
                    "messages": self.messages,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return path

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self.provider, self.model = data["provider"], data["model"]
        self.system, self.messages = data.get("system"), data["messages"]
