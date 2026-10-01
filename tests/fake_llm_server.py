"""A tiny local HTTP server that speaks the wire formats of the three APIs `common.llm` uses.

Lets us test the wrapper end to end through the *real* SDKs (request building, SSE streaming,
usage parsing, structured-output parsing) with no network and no API keys.

Routes: POST /v1/messages (Anthropic), /v1/responses (OpenAI Responses),
        /v1/chat/completions (OpenAI-compatible, e.g. Ollama).
Behaviour knobs (set on the server object): fail_next = [status, ...] to inject errors.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWER_TEXT = "Hello from the fake server"
CHUNKS = ["Hello ", "from the ", "fake server"]
STRUCT_JSON = '{"name": "Ada", "age": 36}'


def _sse(event: str | None, data: dict) -> bytes:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data)}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep test output quiet
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/models"):
            return self._json({"object": "list", "data": [{"id": "fake", "object": "model"}]})
        self._json({"error": "unknown route"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        srv = self.server
        srv.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
        if srv.fail_next:
            status = srv.fail_next.pop(0)
            return self._json(
                {"error": {"message": f"injected {status}", "type": "x"}, "type": "error"}, status
            )
        spec = srv.script.pop(0) if getattr(srv, "script", None) else None
        if spec is not None:
            return self._scripted(body, spec)
        wants_struct = "json_schema" in json.dumps(body) or "response_format" in body
        text = STRUCT_JSON if wants_struct else ANSWER_TEXT
        if self.path.endswith("/messages"):
            return self._anthropic(body, text)
        if self.path.endswith("/responses"):
            return self._responses(body, text)
        if self.path.endswith("/chat/completions"):
            return self._chat(body, text)
        self._json({"error": "unknown route"}, 404)

    # ---- scripted turns (tool calls) in each wire format
    def _scripted(self, body, spec):
        text, calls = spec.get("text", ""), spec.get("tool_calls", [])
        n = len(self.server.requests)
        if self.path.endswith("/messages"):
            content = ([{"type": "text", "text": text}] if text else []) + [
                {"type": "tool_use", "id": f"toolu_{n}_{i}", "name": c["name"], "input": c["args"]}
                for i, c in enumerate(calls)
            ]
            return self._json(
                {
                    "id": "msg_s",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": content or [{"type": "text", "text": ""}],
                    "stop_reason": "tool_use" if calls else "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            )
        if self.path.endswith("/responses"):
            out = (
                [
                    {
                        "type": "message",
                        "id": "m1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ]
                if text
                else []
            )
            out += [
                {
                    "type": "function_call",
                    "id": f"fc_{n}_{i}",
                    "call_id": f"call_{n}_{i}",
                    "name": c["name"],
                    "arguments": json.dumps(c["args"]),
                    "status": "completed",
                }
                for i, c in enumerate(calls)
            ]
            obj = self._response_obj(body, text)
            obj["output"] = out
            return self._json(obj)
        msg = {"role": "assistant", "content": text or None}
        if calls:
            msg["tool_calls"] = [
                {
                    "id": f"call_{n}_{i}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
                }
                for i, c in enumerate(calls)
            ]
        return self._json(
            {
                "id": "chat_s",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {"index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": msg}
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            }
        )

    # ---- Anthropic Messages
    def _anthropic(self, body, text):
        model = body["model"]
        msg = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
        }
        usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 7,
        }
        if not body.get("stream"):
            return self._json({**msg, "content": [{"type": "text", "text": text}], "usage": usage})
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        w = self.wfile.write
        w(
            _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        **msg,
                        "content": [],
                        "stop_reason": None,
                        "usage": {**usage, "output_tokens": 1},
                    },
                },
            )
        )
        w(
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        for c in CHUNKS:
            w(
                _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": c},
                    },
                )
            )
        w(_sse("content_block_stop", {"type": "content_block_stop", "index": 0}))
        w(
            _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 5},
                },
            )
        )
        w(_sse("message_stop", {"type": "message_stop"}))

    # ---- OpenAI Responses
    def _response_obj(self, body, text):
        return {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": body["model"],
            "incomplete_details": None,
            "error": None,
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 20,
                "input_tokens_details": {"cached_tokens": 8},
                "output_tokens": 6,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 26,
            },
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

    def _responses(self, body, text):
        obj = self._response_obj(body, text)
        if not body.get("stream"):
            return self._json(obj)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for i, c in enumerate(CHUNKS):
            self.wfile.write(
                _sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": "msg_1",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": c,
                        "sequence_number": i,
                        "logprobs": [],
                    },
                )
            )
        self.wfile.write(
            _sse(
                "response.completed",
                {"type": "response.completed", "response": obj, "sequence_number": 99},
            )
        )

    # ---- OpenAI-compatible Chat Completions (Ollama path)
    def _chat(self, body, text):
        base = {
            "id": "chat_1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": body["model"],
        }
        if not body.get("stream"):
            return self._json(
                {
                    **base,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": text},
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
                }
            )
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for c in CHUNKS:
            self.wfile.write(
                _sse(
                    None,
                    {
                        **base,
                        "choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}],
                    },
                )
            )
        self.wfile.write(
            _sse(None, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        )
        self.wfile.write(
            _sse(
                None,
                {
                    **base,
                    "choices": [],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
                },
            )
        )
        self.wfile.write(b"data: [DONE]\n\n")


class FakeLLMServer(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.requests: list[dict] = []
        self.fail_next: list[int] = []
        self.script: list[
            dict
        ] = []  # scripted turns: {"text": str, "tool_calls": [{"name", "args"}]}
        self._t = threading.Thread(
            target=lambda: self.serve_forever(poll_interval=0.01), daemon=True
        )

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self.shutdown()
        self.server_close()
