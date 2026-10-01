"""Provider-agnostic *tool-calling* turns.

    from common import chat
    t = chat.turn(messages, tools=[spec], system="...", provider="anthropic")
    t.text, t.tool_calls            # [ToolCall(id, name, args), ...]
    messages += chat.assistant_message(t)                    # keep the model's turn in the history
    messages.append(chat.tool_message(call, "result text"))   # answer EVERY call, then call turn() again

One conversation format for every provider (plain dicts, JSON-serialisable):

    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [{"id": "c1", "name": "get_weather", "args": {...}}]}
    {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "22C", "is_error": False}

and one tool spec: ``{"name": ..., "description": ..., "parameters": <JSON Schema object>}``.

Adapters translate to each wire format (they differ more than you would expect):
  Anthropic   assistant ``tool_use`` blocks; ALL results for a turn go in ONE user message of ``tool_result`` blocks
  OpenAI      Responses API ``function_call`` items answered by ``function_call_output`` items (matched by call_id)
  Ollama/vLLM Chat Completions ``tool_calls`` answered by ``role: "tool"`` messages
  local       Qwen chat template: ``<tool_call>{...}</tool_call>`` text, results as ``role: "tool"``
"""

from __future__ import annotations

import functools
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from . import llm
from .llm import LLMResponse, Usage


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: (
        str | None
    )  # normalised: "tool_use" | "end_turn" | "max_tokens" | other provider values
    usage: Usage
    cost_usd: float
    latency_s: float
    provider: str
    model: str
    raw: Any = field(default=None, repr=False)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


def assistant_message(t: Turn) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": t.text}
    if t.tool_calls:
        msg["tool_calls"] = [c.as_dict() for c in t.tool_calls]
    return msg


def tool_message(call: ToolCall | dict, content: str, is_error: bool = False) -> dict[str, Any]:
    c = call if isinstance(call, dict) else call.as_dict()
    return {
        "role": "tool",
        "tool_call_id": c["id"],
        "name": c["name"],
        "content": content,
        "is_error": is_error,
    }


# ----------------------------------------------------------------------------- request adapters


def to_anthropic(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            blocks: list[dict] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            blocks += [
                {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                for c in m.get("tool_calls", [])
            ]
            out.append({"role": "assistant", "content": blocks or ""})
        elif m["role"] == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m["content"],
            }
            if m.get("is_error"):
                block["is_error"] = True
            prev = out[-1] if out else None
            if (
                prev
                and prev["role"] == "user"
                and isinstance(prev["content"], list)
                and prev.get("_results")
            ):
                prev["content"].append(
                    block
                )  # all results of one assistant turn share ONE user message
            else:
                out.append({"role": "user", "content": [block], "_results": True})
    for m in out:
        m.pop("_results", None)
    return out


def to_openai_input(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            if m.get("content"):
                out.append({"role": "assistant", "content": m["content"]})
            out += [
                {
                    "type": "function_call",
                    "call_id": c["id"],
                    "name": c["name"],
                    "arguments": json.dumps(c["args"]),
                }
                for c in m.get("tool_calls", [])
            ]
        elif m["role"] == "tool":
            out.append(
                {
                    "type": "function_call_output",
                    "call_id": m["tool_call_id"],
                    "output": m["content"],
                }
            )
    return out


def to_chat_completions(messages: list[dict], system: str | None) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": m.get("content") or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
                        }
                        for c in m["tool_calls"]
                    ],
                }
            )
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def to_local(messages: list[dict]) -> list[dict]:
    """Qwen's template wants arguments as dicts and results as role 'tool' (it wraps them in <tool_response>)."""
    out: list[dict] = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": m.get("content") or "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["args"]},
                        }
                        for c in m["tool_calls"]
                    ],
                }
            )
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def _anthropic_choice(choice: str | None) -> dict | None:
    if choice in (None, "auto"):
        return None
    if choice in ("any", "required"):
        return {"type": "any"}
    if choice == "none":
        return {"type": "none"}
    return {"type": "tool", "name": choice}


def _openai_choice(choice: str | None) -> str | dict | None:
    if choice in (None, "auto"):
        return None
    if choice in ("any", "required"):
        return "required"
    if choice == "none":
        return "none"
    return {"type": "function", "name": choice}


# ----------------------------------------------------------------------------- response parsers

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_local_output(text: str) -> tuple[str, list[ToolCall]]:
    calls: list[ToolCall] = []
    for i, m in enumerate(_TOOL_CALL.finditer(text)):
        try:
            obj = json.loads(m.group(1))
            calls.append(ToolCall(f"call_{i}", obj["name"], obj.get("arguments", {})))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue  # a malformed call is dropped (the model gets no result for it)
    plain = _TOOL_CALL.sub("", text).replace("<|im_end|>", "").strip()
    return plain, calls


def _safe_json(s: str | dict | None) -> dict:
    if isinstance(s, dict):
        return s
    try:
        out = json.loads(s or "{}")
        return out if isinstance(out, dict) else {"_value": out}
    except json.JSONDecodeError:
        return {"_unparseable": s}


# ----------------------------------------------------------------------------- the entry point


def turn(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    system: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 4096,
    tool_choice: str | None = None,
    **extra: Any,
) -> Turn:
    """One model turn, possibly requesting tool calls. See the module docstring for the formats."""
    provider, model = llm.resolve(provider, model)
    tools = tools or []
    t0 = time.perf_counter()

    if provider == "anthropic":
        kw: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": to_anthropic(messages),
            **extra,
        }
        if system:
            kw["system"] = system
        if tools:
            kw["tools"] = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
                for t in tools
            ]
            if tc := _anthropic_choice(tool_choice):
                kw["tool_choice"] = tc
        r = llm._anthropic().messages.create(**kw)
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
        usage, stop, raw = llm._anthropic_usage(r.usage), r.stop_reason, r
    elif provider == "openai":
        kw = {
            "model": model,
            "input": to_openai_input(messages),
            "max_output_tokens": max_tokens,
            **extra,
        }
        if system:
            kw["instructions"] = system
        if tools:
            kw["tools"] = [
                {
                    "type": "function",
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                    "strict": t.get("strict", False),
                }
                for t in tools
            ]
            if tc := _openai_choice(tool_choice):
                kw["tool_choice"] = tc
        r = llm._openai().responses.create(**kw)
        text = r.output_text
        calls = [
            ToolCall(i.call_id, i.name, _safe_json(i.arguments))
            for i in r.output
            if i.type == "function_call"
        ]
        usage = llm._openai_usage(r.usage)
        stop = (
            "tool_use"
            if calls
            else (r.incomplete_details.reason if r.incomplete_details else "end_turn")
        )
        raw = r
    elif provider == "ollama":
        kw = {
            "model": model,
            "messages": to_chat_completions(messages, system),
            "max_tokens": max_tokens,
            **extra,
        }
        if tools:
            kw["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    },
                }
                for t in tools
            ]
            if tool_choice in ("any", "required", "none"):
                kw["tool_choice"] = "required" if tool_choice in ("any", "required") else "none"
        r = llm._ollama().chat.completions.create(**kw)
        msg = r.choices[0].message
        text = msg.content or ""
        calls = [
            ToolCall(c.id, c.function.name, _safe_json(c.function.arguments))
            for c in (msg.tool_calls or [])
        ]
        usage = llm._ollama_usage(r.usage)
        stop = (
            "tool_use"
            if calls
            else {"stop": "end_turn", "length": "max_tokens"}.get(
                r.choices[0].finish_reason, r.choices[0].finish_reason
            )
        )
        raw = r
    else:  # local
        chat_model = _local_chat(model)
        # tool_choice="none": show the model no tools at all (the local template has no such switch)
        specs = [] if tool_choice == "none" else [_as_openai_tool(t) for t in tools]
        local_msgs = to_local(messages)
        raw_text = chat_model.chat(local_msgs, specs, system, max_new_tokens=min(max_tokens, 400))
        text, calls = parse_local_output(raw_text)
        usage = Usage(
            chat_model.count_tokens(local_msgs, specs, system), chat_model.text_tokens(raw_text)
        )
        stop, raw = ("tool_use" if calls else "end_turn"), raw_text

    resp = llm._finish(provider, model, t0, raw, (text, usage, stop))
    return _as_turn(resp, text, calls, stop, raw)


@functools.lru_cache(maxsize=4)
def _local_chat(model: str):
    """One LocalChat per model: loading the weights (and the cache file) on every turn is far too slow."""
    from .local_llm import LocalChat

    return LocalChat(model)


def _as_openai_tool(t: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"],
        },
    }


def _as_turn(
    resp: LLMResponse, text: str, calls: list[ToolCall], stop: str | None, raw: Any
) -> Turn:
    return Turn(
        text=text,
        tool_calls=calls,
        stop_reason="tool_use" if calls else stop,
        usage=resp.usage,
        cost_usd=resp.cost_usd,
        latency_s=resp.latency_s,
        provider=resp.provider,
        model=resp.model,
        raw=raw,
    )
