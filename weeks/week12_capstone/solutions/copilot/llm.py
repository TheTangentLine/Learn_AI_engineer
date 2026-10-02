"""A chat client for any OpenAI-compatible server (llama.cpp's ``llama-server`` here; vLLM, Ollama or a hosted API with the same shape), as the callable ``LlmAnswerer`` expects.

chat = OpenAIChat("http://127.0.0.1:8081", max_tokens=160)
text, prompt_tokens, completion_tokens = chat(messages)
"""

from __future__ import annotations

import httpx


class ChatError(RuntimeError):
    """The model server failed, timed out, or answered something that is not a chat completion."""


class OpenAIChat:
    def __init__(
        self,
        base_url: str,
        *,
        model: str = "",
        max_tokens: int = 160,
        temperature: float = 0.0,
        timeout: float = 60.0,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(timeout=timeout, headers=headers, transport=transport)
        self.calls = 0

    def __call__(self, messages: list[dict]) -> tuple[str, int, int]:
        body = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.model:
            body["model"] = self.model
        self.calls += 1
        try:
            r = self.client.post(f"{self.base_url}/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise ChatError(f"{type(exc).__name__}: {exc}") from exc
        if r.status_code != 200:
            raise ChatError(f"the model server answered HTTP {r.status_code}: {r.text[:200]}")
        try:
            j = r.json()
            text = j["choices"][0]["message"]["content"] or ""
            usage = j.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ChatError(f"unexpected response: {r.text[:200]}") from exc
        return text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
