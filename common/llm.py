"""Provider-agnostic LLM wrapper used throughout the roadmap.

You build a minimal version of this file in Week 1 Day 1; this is the extended
reference version every later lesson imports:

    from common.llm import complete, stream, structured

    resp = complete("Explain KV cache in one sentence.")
    print(resp.text, resp.usage, f"${resp.cost_usd:.5f}")

Providers
---------
- ``anthropic`` - Claude via the official ``anthropic`` SDK (Messages API)
- ``openai``    - GPT via the official ``openai`` SDK (Responses API)
- ``ollama``    - local open-weights models through Ollama's OpenAI-compatible
                  Chat Completions endpoint (free, no key)

Pick one with ``LLM_PROVIDER`` in ``.env`` or ``provider=`` per call.
Provider-specific options (``thinking``, ``reasoning``, ``temperature``...) go
through ``**extra`` untouched, so the wrapper never hides the real SDK.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Literal, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

Provider = Literal["anthropic", "openai", "ollama"]
Messages = str | list[dict[str, Any]]
T = TypeVar("T", bound=BaseModel)

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-6.1-sol",
    "ollama": "llama3.2:3b",
    "local": "Qwen/Qwen2.5-0.5B-Instruct",  # in-process Hugging Face model: free, offline, weak
}

# A cheaper model per provider for bulk work, judges and sub-tasks.
CHEAP_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-6-luna",
    "ollama": "llama3.2:3b",
    "local": "Qwen/Qwen2.5-0.5B-Instruct",
}

# USD per 1M tokens: (input, cached-input read, output).
# Snapshot from the providers' pricing pages (Oct 2026) - re-check before relying on it:
#   https://www.anthropic.com/pricing   https://developers.openai.com/api/docs/pricing
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-fable-5-1": (10.00, 1.00, 50.00),
    "claude-opus-5-5": (4.00, 0.20, 20.00),
    "claude-opus-5": (5.00, 0.50, 25.00),
    "claude-sonnet-5": (2.00, 0.20, 10.00),
    "claude-haiku-4-5": (1.00, 0.10, 5.00),
    "gpt-6-astra": (10.00, 1.00, 50.00),
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
}
# Anthropic bills cache *writes* (5-minute TTL) at 1.25x the input price.
ANTHROPIC_CACHE_WRITE_MULTIPLIER = 1.25


# --------------------------------------------------------------------------- types


@dataclass
class Usage:
    """Normalised token usage. ``input_tokens`` excludes cached tokens on every provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        cached = self.cache_read_tokens + self.cache_write_tokens
        return self.input_tokens + self.output_tokens + cached

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: Usage
    cost_usd: float
    latency_s: float
    stop_reason: str | None = None
    raw: Any = field(default=None, repr=False)


def cost_usd(model: str, usage: Usage) -> float:
    """Price a call. Unknown models (e.g. local Ollama) cost 0."""
    price = PRICES.get(model)
    if price is None:
        return 0.0
    inp, cached, out = price
    write = inp * ANTHROPIC_CACHE_WRITE_MULTIPLIER if model.startswith("claude") else inp
    return (
        usage.input_tokens * inp
        + usage.cache_read_tokens * cached
        + usage.cache_write_tokens * write
        + usage.output_tokens * out
    ) / 1_000_000


@dataclass
class SessionTracker:
    """Running totals across every call in this process (handy for cost meters)."""

    calls: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0

    def record(self, resp: LLMResponse) -> None:
        self.calls += 1
        self.usage = self.usage + resp.usage
        self.cost_usd += resp.cost_usd


SESSION = SessionTracker()


# ------------------------------------------------------------------------- helpers


def resolve(provider: str | None = None, model: str | None = None) -> tuple[str, str]:
    """Return (provider, model) using arguments first, then .env, then defaults."""
    provider = provider or os.getenv("LLM_PROVIDER") or "anthropic"
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unknown provider {provider!r}; use one of {list(DEFAULT_MODELS)}")
    env_model = os.getenv("LLM_MODEL") if provider == (os.getenv("LLM_PROVIDER") or "") else None
    return provider, model or env_model or DEFAULT_MODELS[provider]


def _to_messages(messages: Messages) -> list[dict[str, Any]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return list(messages)


def _client_opts() -> dict[str, Any]:
    return {
        "max_retries": int(os.getenv("LLM_MAX_RETRIES", "3")),
        "timeout": float(os.getenv("LLM_TIMEOUT", "120")),
    }


@cache
def _anthropic(async_: bool = False):
    import anthropic

    return (anthropic.AsyncAnthropic if async_ else anthropic.Anthropic)(**_client_opts())


@cache
def _openai(async_: bool = False):
    import openai

    return (openai.AsyncOpenAI if async_ else openai.OpenAI)(**_client_opts())


@cache
def _ollama(async_: bool = False):
    import openai

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    cls = openai.AsyncOpenAI if async_ else openai.OpenAI
    return cls(base_url=base_url, api_key="ollama", **_client_opts())


# Normalised result of one provider call: (text, usage, stop_reason).
Parsed = tuple[str, Usage, str | None]


def _finish(provider: str, model: str, t0: float, raw: Any, parsed: Parsed) -> LLMResponse:
    text, usage, stop = parsed
    resp = LLMResponse(
        text=text,
        provider=provider,
        model=model,
        usage=usage,
        cost_usd=cost_usd(model, usage),
        latency_s=time.perf_counter() - t0,
        stop_reason=stop,
        raw=raw,
    )
    SESSION.record(resp)
    if os.getenv("LLM_LOG") == "1":
        u = resp.usage
        print(
            f"[llm] {provider}/{model} in={u.input_tokens} cached={u.cache_read_tokens} "
            f"out={u.output_tokens} ${resp.cost_usd:.5f} {resp.latency_s:.2f}s",
            file=sys.stderr,
        )
    return resp


# --- request builders (shared by sync + async) ---


def _anthropic_kwargs(model, msgs, system, max_tokens, cache_prompt, extra) -> dict[str, Any]:
    kw: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": msgs, **extra}
    if system:
        kw["system"] = system
    if cache_prompt:
        # Top-level auto-caching: caches the prefix up to the last cacheable block.
        kw["cache_control"] = {"type": "ephemeral"}
    return kw


def _openai_kwargs(model, msgs, system, max_tokens, extra) -> dict[str, Any]:
    kw: dict[str, Any] = {"model": model, "input": msgs, "max_output_tokens": max_tokens, **extra}
    if system:
        kw["instructions"] = system
    return kw


def _ollama_kwargs(model, msgs, system, max_tokens, extra) -> dict[str, Any]:
    if system:
        msgs = [{"role": "system", "content": system}, *msgs]
    return {"model": model, "messages": msgs, "max_tokens": max_tokens, **extra}


# --- response parsers ---


def _anthropic_usage(u) -> Usage:
    return Usage(
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_read_tokens=u.cache_read_input_tokens or 0,
        cache_write_tokens=u.cache_creation_input_tokens or 0,
    )


def _openai_usage(u) -> Usage:
    if u is None:
        return Usage()
    cached = u.input_tokens_details.cached_tokens if u.input_tokens_details else 0
    return Usage(u.input_tokens - cached, u.output_tokens, cache_read_tokens=cached)


def _ollama_usage(u) -> Usage:
    return Usage(u.prompt_tokens, u.completion_tokens) if u else Usage()


def _parse_anthropic(msg) -> Parsed:
    # Content is a list of blocks (thinking, text, tool_use...); keep only the text.
    text = "".join(b.text for b in msg.content if b.type == "text")
    return text, _anthropic_usage(msg.usage), msg.stop_reason


def _parse_openai(r) -> Parsed:
    if r is None:
        return "", Usage(), None
    stop = r.incomplete_details.reason if r.incomplete_details else r.status
    return r.output_text, _openai_usage(r.usage), stop


def _parse_ollama(r) -> Parsed:
    choice = r.choices[0]
    return choice.message.content or "", _ollama_usage(r.usage), choice.finish_reason


# ---------------------------------------------------------------------- public API


def _complete_local(
    msgs: list[dict[str, Any]], system: str | None, model: str, max_tokens: int, t0: float
):
    """The in-process local model: supports complete() and common.chat.turn() only (no streaming/structured)."""
    from .local_llm import LocalChat

    user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
    text = LocalChat(model)(
        system, user if isinstance(user, str) else str(user), min(max_tokens, 256)
    )
    return _finish("local", model, t0, None, (text, Usage(), "stop"))


def _no_local(provider: str, what: str) -> None:
    if provider == "local":
        raise NotImplementedError(
            f"the 'local' provider does not support {what}; use complete() or chat.turn()"
        )


def complete(
    messages: Messages,
    *,
    system: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 16000,
    cache_prompt: bool = False,
    **extra: Any,
) -> LLMResponse:
    """One blocking call. ``messages`` is a prompt string or a list of role/content dicts."""
    provider, model = resolve(provider, model)
    msgs = _to_messages(messages)
    t0 = time.perf_counter()

    if provider == "anthropic":
        kw = _anthropic_kwargs(model, msgs, system, max_tokens, cache_prompt, extra)
        r = _anthropic().messages.create(**kw)
        return _finish(provider, model, t0, r, _parse_anthropic(r))
    if provider == "openai":
        r = _openai().responses.create(**_openai_kwargs(model, msgs, system, max_tokens, extra))
        return _finish(provider, model, t0, r, _parse_openai(r))
    if provider == "local":
        return _complete_local(msgs, system, model, max_tokens, t0)
    r = _ollama().chat.completions.create(**_ollama_kwargs(model, msgs, system, max_tokens, extra))
    return _finish(provider, model, t0, r, _parse_ollama(r))


async def acomplete(
    messages: Messages,
    *,
    system: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 16000,
    cache_prompt: bool = False,
    **extra: Any,
) -> LLMResponse:
    """Async twin of :func:`complete` - use with ``asyncio.gather`` for concurrency."""
    provider, model = resolve(provider, model)
    msgs = _to_messages(messages)
    t0 = time.perf_counter()

    if provider == "anthropic":
        kw = _anthropic_kwargs(model, msgs, system, max_tokens, cache_prompt, extra)
        r = await _anthropic(True).messages.create(**kw)
        return _finish(provider, model, t0, r, _parse_anthropic(r))
    if provider == "openai":
        kw = _openai_kwargs(model, msgs, system, max_tokens, extra)
        r = await _openai(True).responses.create(**kw)
        return _finish(provider, model, t0, r, _parse_openai(r))
    if provider == "local":
        return _complete_local(msgs, system, model, max_tokens, t0)
    kw = _ollama_kwargs(model, msgs, system, max_tokens, extra)
    r = await _ollama(True).chat.completions.create(**kw)
    return _finish(provider, model, t0, r, _parse_ollama(r))


def stream(
    messages: Messages,
    *,
    system: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 16000,
    cache_prompt: bool = False,
    on_done: Callable[[LLMResponse], None] | None = None,
    **extra: Any,
) -> Iterator[str]:
    """Yield text chunks as they arrive. ``on_done`` receives the final LLMResponse."""
    provider, model = resolve(provider, model)
    _no_local(provider, "streaming")
    msgs = _to_messages(messages)
    t0 = time.perf_counter()
    parts: list[str] = []

    if provider == "anthropic":
        kw = _anthropic_kwargs(model, msgs, system, max_tokens, cache_prompt, extra)
        with _anthropic().messages.stream(**kw) as s:
            for chunk in s.text_stream:
                parts.append(chunk)
                yield chunk
            final = s.get_final_message()
        resp = _finish(provider, model, t0, final, _parse_anthropic(final))
    elif provider == "openai":
        kw = _openai_kwargs(model, msgs, system, max_tokens, extra)
        final = None
        for event in _openai().responses.create(**kw, stream=True):
            if event.type == "response.output_text.delta":
                parts.append(event.delta)
                yield event.delta
            elif event.type in ("response.completed", "response.incomplete"):
                final = event.response
        resp = _finish(provider, model, t0, final, _parse_openai(final))
    else:
        kw = _ollama_kwargs(model, msgs, system, max_tokens, extra)
        usage, stop = Usage(), None
        for chunk in _ollama().chat.completions.create(
            **kw, stream=True, stream_options={"include_usage": True}
        ):
            if chunk.choices:
                stop = chunk.choices[0].finish_reason or stop
                if delta := chunk.choices[0].delta.content:
                    parts.append(delta)
                    yield delta
            if chunk.usage:
                usage = _ollama_usage(chunk.usage)
        resp = _finish(provider, model, t0, None, ("".join(parts), usage, stop))

    if on_done:
        on_done(resp)


async def astream(
    messages: Messages,
    *,
    system: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 16000,
    cache_prompt: bool = False,
    on_done: Callable[[LLMResponse], None] | None = None,
    **extra: Any,
) -> AsyncIterator[str]:
    """Async twin of :func:`stream` (used by the FastAPI backend in Week 11)."""
    provider, model = resolve(provider, model)
    _no_local(provider, "async streaming")
    msgs = _to_messages(messages)
    t0 = time.perf_counter()
    parts: list[str] = []

    if provider == "anthropic":
        kw = _anthropic_kwargs(model, msgs, system, max_tokens, cache_prompt, extra)
        async with _anthropic(True).messages.stream(**kw) as s:
            async for chunk in s.text_stream:
                parts.append(chunk)
                yield chunk
            final = await s.get_final_message()
        resp = _finish(provider, model, t0, final, _parse_anthropic(final))
    elif provider == "openai":
        kw = _openai_kwargs(model, msgs, system, max_tokens, extra)
        final = None
        async for event in await _openai(True).responses.create(**kw, stream=True):
            if event.type == "response.output_text.delta":
                parts.append(event.delta)
                yield event.delta
            elif event.type in ("response.completed", "response.incomplete"):
                final = event.response
        resp = _finish(provider, model, t0, final, _parse_openai(final))
    else:
        kw = _ollama_kwargs(model, msgs, system, max_tokens, extra)
        usage, stop = Usage(), None
        async for chunk in await _ollama(True).chat.completions.create(
            **kw, stream=True, stream_options={"include_usage": True}
        ):
            if chunk.choices:
                stop = chunk.choices[0].finish_reason or stop
                if delta := chunk.choices[0].delta.content:
                    parts.append(delta)
                    yield delta
            if chunk.usage:
                usage = _ollama_usage(chunk.usage)
        resp = _finish(provider, model, t0, None, ("".join(parts), usage, stop))

    if on_done:
        on_done(resp)


def structured(
    messages: Messages,
    schema: type[T],
    *,
    system: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 16000,
    **extra: Any,
) -> tuple[T, LLMResponse]:
    """Return a validated Pydantic object using each provider's native structured output."""
    provider, model = resolve(provider, model)
    _no_local(provider, "structured output")
    msgs = _to_messages(messages)
    t0 = time.perf_counter()

    if provider == "anthropic":
        kw = _anthropic_kwargs(model, msgs, system, max_tokens, False, extra)
        r = _anthropic().messages.parse(**kw, output_format=schema)
        resp, parsed = _finish(provider, model, t0, r, _parse_anthropic(r)), r.parsed_output
    elif provider == "openai":
        kw = _openai_kwargs(model, msgs, system, max_tokens, extra)
        r = _openai().responses.parse(**kw, text_format=schema)
        resp, parsed = _finish(provider, model, t0, r, _parse_openai(r)), r.output_parsed
    else:
        kw = _ollama_kwargs(model, msgs, system, max_tokens, extra)
        r = _ollama().chat.completions.parse(**kw, response_format=schema)
        parsed = r.choices[0].message.parsed
        resp = _finish(provider, model, t0, r, _parse_ollama(r))

    if parsed is None:
        raise ValueError(f"{provider}/{model} returned no parseable output ({resp.stop_reason=})")
    return parsed, resp


def available_providers() -> list[str]:
    """Providers you can call right now: API key present, or a local Ollama server answering."""
    found = []
    if os.getenv("ANTHROPIC_API_KEY"):
        found.append("anthropic")
    if os.getenv("OPENAI_API_KEY"):
        found.append("openai")
    if _ollama_running():
        found.append("ollama")
    import importlib.util

    if importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"):
        found.append("local")
    return found


def _ollama_running() -> bool:
    """Is an OpenAI-compatible local server (Ollama, vLLM...) answering GET {base}/models?"""
    import urllib.request

    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=0.5):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Say hello in five words."
    for chunk in stream(prompt, on_done=lambda r: print(f"\n\n{r.usage} ${r.cost_usd:.5f}")):
        print(chunk, end="", flush=True)
