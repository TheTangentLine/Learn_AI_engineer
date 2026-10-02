"""An OpenAI-compatible Chat Completions server backed by the local Qwen model (``common/local_llm.py``).

Why: every agent framework (OpenAI Agents SDK, PydanticAI, LangChain ...) can talk to "an OpenAI-compatible base
URL". Pointing them at this server lets a REAL (small) model drive a REAL framework with no API key and no Ollama.

    with LocalOpenAIServer() as srv:
        client = OpenAI(base_url=srv.url, api_key="local")        # srv.url == "http://127.0.0.1:PORT/v1"
        client.chat.completions.create(model="local-qwen", messages=[...], tools=[...])

Supports: messages (system/user/assistant+tool_calls/tool), tools, parallel tool calls, usage, and ``tool_choice``
("required" or a named function: the reply is started with a forced ``<tool_call>`` prefix; "none" hides the tools). No streaming
(returns HTTP 400 so a framework falls back or you notice). Generations are serialised (one model, one thread) and
cached on disk by ``LocalChat``, so a repeated run is instant and reproducible.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .chat import forced_prefill, parse_local_output
from .local_llm import DEFAULT_MODEL, LocalChat

MODEL_NAME = "local-qwen"


def _flatten(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def json_instruction(response_format: dict | None) -> str:
    """Best-effort JSON mode: a small local model cannot be CONSTRAINED to a schema here, so we ask for it explicitly."""
    if not response_format or response_format.get("type") not in ("json_object", "json_schema"):
        return ""
    schema = (response_format.get("json_schema") or {}).get("schema")
    text = "Reply with ONLY a single valid JSON object, no prose and no code fences."
    if schema:
        text += " It must match this JSON Schema: " + json.dumps(schema)
    return text


def _choice_name(choice) -> str | None:
    """OpenAI ``tool_choice`` -> our vocabulary: 'required', a tool name, or None."""
    if isinstance(choice, dict):
        return (choice.get("function") or {}).get("name")
    return choice


def to_local_messages(messages: list[dict]) -> list[dict]:
    """OpenAI chat messages -> the Qwen template's shape (tool-call arguments as dicts, content as strings)."""
    out = []
    for m in messages:
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                args = c["function"].get("arguments") or "{}"
                calls.append(
                    {
                        "type": "function",
                        "function": {
                            "name": c["function"]["name"],
                            "arguments": json.loads(args) if isinstance(args, str) else args,
                        },
                    }
                )
            out.append(
                {"role": "assistant", "content": _flatten(m.get("content")), "tool_calls": calls}
            )
        elif role == "tool":
            out.append({"role": "tool", "content": _flatten(m.get("content"))})
        else:
            out.append(
                {
                    "role": "system" if role == "developer" else role,
                    "content": _flatten(m.get("content")),
                }
            )
    return out


class LocalOpenAIServer:
    def __init__(self, model: str = DEFAULT_MODEL, max_new_tokens: int = 400):
        self.chat = LocalChat(model)
        self.max_new_tokens = max_new_tokens
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self.url = ""

    def complete(self, body: dict) -> dict:
        messages = to_local_messages(body["messages"])
        if hint := json_instruction(body.get("response_format")):
            if messages and messages[0]["role"] == "system":
                messages[0] = {**messages[0], "content": messages[0]["content"] + "\n\n" + hint}
            else:
                messages.insert(0, {"role": "system", "content": hint})
        tools = body.get("tools") or []
        choice = body.get("tool_choice")
        if choice == "none":
            tools = []
        prefill = forced_prefill(_choice_name(choice), tools)
        max_new = min(
            int(body.get("max_tokens") or body.get("max_completion_tokens") or self.max_new_tokens),
            self.max_new_tokens,
        )
        with self._lock:  # one model, one generation at a time
            raw = self.chat.chat(messages, tools, None, max_new_tokens=max_new, prefill=prefill)
            prompt_tokens = self.chat.count_tokens(messages, tools, None)
            completion_tokens = self.chat.text_tokens(raw)
        text, calls = parse_local_output(raw)
        n = len(self.requests)
        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if calls:
            message["tool_calls"] = [
                {
                    "id": f"call_{n}_{i}",
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.args)},
                }
                for i, c in enumerate(calls)
            ]
        return {
            "id": f"chatcmpl-local-{n}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", MODEL_NAME),
            "choices": [
                {"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def __enter__(self) -> LocalOpenAIServer:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj, status=200):
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    return self._json(
                        {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}
                    )
                self._json({"error": {"message": "unknown route"}}, 404)

            def do_POST(self):
                body = json.loads(
                    self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}"
                )
                outer.requests.append({"path": self.path, "body": body})
                if not self.path.endswith("/chat/completions"):
                    return self._json(
                        {
                            "error": {
                                "message": f"{self.path} is not supported; use /v1/chat/completions"
                            }
                        },
                        404,
                    )
                if body.get("stream"):
                    return self._json(
                        {"error": {"message": "streaming is not supported by the local server"}},
                        400,
                    )
                try:
                    self._json(outer.complete(body))
                except Exception as exc:  # surface model/template errors to the client as HTTP 500
                    self._json({"error": {"message": f"{type(exc).__name__}: {exc}"}}, 500)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
