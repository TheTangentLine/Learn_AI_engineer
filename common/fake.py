"""A scripted fake LLM for offline tests and ``--offline`` demos.

It patches the public functions of :mod:`common.llm` (``complete``, ``acomplete``, ``stream``,
``astream``, ``structured``) so any code that calls ``llm.complete(...)`` runs unchanged:

    from common import llm
    from common.fake import fake_llm

    with fake_llm([(r"capital of France", "Paris"), (r"(?i)summar", lambda p: p[:20])]) as fake:
        print(llm.complete("What is the capital of France?").text)   # -> Paris
        print(fake.calls[-1].prompt)

Rules are ``(regex, response)`` pairs checked in order against the *whole prompt text* (system
prompt + all messages). ``response`` is a string or a callable ``(prompt_text) -> str``; a
callable may also take ``(prompt_text, call)`` to see the full ``Call`` record.
Use a list as ``response`` to cycle through scripted answers (handy for "fail once, then succeed").

This is for testing *plumbing*, never for judging model quality: results printed in an
``--offline`` run are labelled as such by the lessons that use it.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from . import chat, llm

Responder = str | dict | list | Callable[..., Any] | Exception


@dataclass
class Call:
    kind: str  # complete | stream | structured
    prompt: str  # system + all messages, joined
    system: str | None
    messages: list[dict[str, Any]]
    provider: str
    model: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    response: str = ""


def _flatten(messages: Any) -> list[dict[str, Any]]:
    return [{"role": "user", "content": messages}] if isinstance(messages, str) else list(messages)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    # list of content blocks (Anthropic / OpenAI style)
    return " ".join(
        b.get("text", "") if isinstance(b, dict) else str(b)
        for b in content  # type: ignore[union-attr]
    )


class FakeLLM:
    def __init__(self, rules: list[tuple[str, Responder]] | None = None, default: str = "OK"):
        self.rules = [(re.compile(p, re.S), r) for p, r in (rules or [])]
        self.default = default
        self.calls: list[Call] = []
        self._cursor: dict[int, int] = {}

    # ------------------------------------------------------------- core
    def _respond(self, call: Call) -> str:
        for i, (pattern, responder) in enumerate(self.rules):
            if not pattern.search(call.prompt):
                continue
            if isinstance(responder, Exception):
                raise responder
            if isinstance(responder, list):
                k = self._cursor.get(i, 0)
                self._cursor[i] = k + 1
                item = responder[min(k, len(responder) - 1)]
                if isinstance(item, Exception):
                    raise item
                return item
            if callable(responder):
                nparams = len(inspect.signature(responder).parameters)
                return responder(call.prompt, call) if nparams >= 2 else responder(call.prompt)
            return responder
        return self.default

    def _make_call(self, kind, messages, system, provider, model, kwargs) -> Call:
        msgs = _flatten(messages)
        prompt = "\n".join(([system] if system else []) + [_text_of(m["content"]) for m in msgs])
        provider, model = llm.resolve(provider, model)
        return Call(kind, prompt, system, msgs, provider, model, kwargs)

    def _response(self, call: Call) -> llm.LLMResponse:
        call.response = self._respond(call)
        self.calls.append(call)
        usage = llm.Usage(max(1, len(call.prompt) // 4), max(1, len(call.response) // 4))
        resp = llm.LLMResponse(
            text=call.response,
            provider=call.provider,
            model=call.model,
            usage=usage,
            cost_usd=llm.cost_usd(call.model, usage),
            latency_s=0.0,
            stop_reason="end_turn",
        )
        llm.SESSION.record(resp)
        return resp

    # ------------------------------------------------------------- patched API
    def complete(self, messages, *, system=None, provider=None, model=None, **kw):
        return self._response(self._make_call("complete", messages, system, provider, model, kw))

    async def acomplete(self, messages, *, system=None, provider=None, model=None, **kw):
        return self.complete(messages, system=system, provider=provider, model=model, **kw)

    def stream(self, messages, *, system=None, provider=None, model=None, on_done=None, **kw):
        call = self._make_call("stream", messages, system, provider, model, kw)
        resp = self._response(call)
        words = re.findall(r"\S+\s*", resp.text)
        yield from words
        if on_done:
            on_done(resp)

    async def astream(
        self, messages, *, system=None, provider=None, model=None, on_done=None, **kw
    ):
        for chunk in self.stream(
            messages, system=system, provider=provider, model=model, on_done=on_done, **kw
        ):
            yield chunk

    def structured(
        self, messages, schema: type[BaseModel], *, system=None, provider=None, model=None, **kw
    ):
        call = self._make_call("structured", messages, system, provider, model, kw)
        resp = self._response(call)
        return schema.model_validate_json(resp.text), resp

    def turn(
        self,
        messages,
        tools=None,
        *,
        system=None,
        provider=None,
        model=None,
        tool_choice=None,
        **kw,
    ):
        """Scripted tool-calling turn. A rule's response may be a str (text only) or a dict
        ``{"text": str, "tool_calls": [{"name": str, "args": dict}, ...]}`` (see ``tool_calls()``)."""
        parts = [system] if system else []
        for m in messages:
            parts.append(_text_of(m.get("content") or ""))
            parts += [f"{c['name']}({c['args']})" for c in m.get("tool_calls", [])]
        provider, model = llm.resolve(provider, model)
        call = Call(
            "turn",
            "\n".join(parts),
            system,
            list(messages),
            provider,
            model,
            {"tools": tools, "tool_choice": tool_choice, **kw},
        )
        raw = self._respond(call)
        self.calls.append(call)
        spec = raw if isinstance(raw, dict) else {"text": raw}
        calls = [
            chat.ToolCall(f"call_{len(self.calls)}_{i}", c["name"], c["args"])
            for i, c in enumerate(spec.get("tool_calls", []))
        ]
        call.response = str(spec)
        usage = llm.Usage(max(1, len(call.prompt) // 4), max(1, len(call.response) // 4))
        resp = llm.LLMResponse(
            spec.get("text", ""),
            provider,
            model,
            usage,
            llm.cost_usd(model, usage),
            0.0,
            "end_turn",
        )
        llm.SESSION.record(resp)
        return chat.Turn(
            spec.get("text", ""),
            calls,
            "tool_use" if calls else "end_turn",
            usage,
            resp.cost_usd,
            0.0,
            provider,
            model,
        )

    # ------------------------------------------------------------- helpers
    @property
    def prompts(self) -> list[str]:
        return [c.prompt for c in self.calls]

    def calls_matching(self, pattern: str) -> list[Call]:
        return [c for c in self.calls if re.search(pattern, c.prompt, re.S)]


_PATCHED = ("complete", "acomplete", "stream", "astream", "structured")


@contextmanager
def fake_llm(
    rules: list[tuple[str, Responder]] | None = None, default: str = "OK"
) -> Iterator[FakeLLM]:
    """Temporarily replace ``common.llm``'s call functions with a scripted fake."""
    fake = FakeLLM(rules, default)
    originals = {name: getattr(llm, name) for name in _PATCHED}
    original_turn = chat.turn
    for name in _PATCHED:
        setattr(llm, name, getattr(fake, name))
    chat.turn = fake.turn
    try:
        yield fake
    finally:
        for name, fn in originals.items():
            setattr(llm, name, fn)
        chat.turn = original_turn


def tool_calls(*calls: tuple[str, dict], text: str = "") -> dict:
    """Build a scripted turn that requests tools: ``tool_calls(("multiply", {"a": 2, "b": 3}), text="Computing")``."""
    return {"text": text, "tool_calls": [{"name": n, "args": a} for n, a in calls]}
