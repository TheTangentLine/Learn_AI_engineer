"""Serve the Copilot with the Week 11 gateway.

The gateway (``llmapi``) already does identity, rate limits, quotas, a bounded queue, SSE streaming, health/readiness, metrics and feedback. The Copilot is not a model behind an HTTP endpoint;
it is a pipeline. The seam is the gateway's own: its ``/v1/ask`` route takes a *retriever* (question -> messages, sources) and a *backend* (messages -> a stream of text deltas). Here

    retriever = CopilotBackend.prepare   runs the whole pipeline once and remembers the result; returns the sources the UI shows before the first word
    backend   = CopilotBackend           streams the remembered answer

so authentication, limits, back-pressure, streaming, metrics and the Streamlit UI work unchanged. (The pipeline runs synchronously inside the request handler: fine for the load measured on Day 6,
and the first thing to move to a worker pool when it is not.)

    python -m copilot.service        # configuration from the environment (see ``from_env``)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator

from llmapi import Delta, create_app  # noqa: E402
from llmapi.app import Settings  # noqa: E402
from llmapi.auth import load_keys  # noqa: E402

from .core import Copilot, Result  # noqa: E402


def approx_tokens(text: str) -> int:
    """About four characters per token: usage for a pipeline with no tokenizer of its own (and no model, in extractive mode). Labelled an estimate wherever it is shown."""
    return max(1, round(len(text) / 4)) if text else 0


class CopilotBackend:
    def __init__(self, copilot: Copilot, remember: int = 64, words_per_chunk: int = 3):
        self.copilot = copilot
        self._results: OrderedDict[str, Result] = OrderedDict()
        self.remember, self.words_per_chunk = remember, words_per_chunk
        self.calls = 0
        self._lock = threading.Lock()  # the pipeline (torch, SQLite caches, the memo) is not thread-safe: one run at a time, in a worker thread so the event loop stays free

    async def ready(self) -> bool:
        return bool(self.copilot.retriever.index.chunks)

    def _ask_locked(self, question: str) -> Result:
        with self._lock:
            self.calls += 1
            return self.copilot.ask(question)

    def prepare(self, question: str, k: int = 5) -> tuple[list[dict], list[dict]]:
        """The gateway's ``retriever`` hook: answer now, return what the client should see first."""
        res = self._ask_locked(question)
        self._results[question] = res
        while len(self._results) > self.remember:
            self._results.popitem(last=False)
        return [{"role": "user", "content": question}], [s.as_dict() for s in res.sources[:k]]

    async def stream(
        self, messages: list[dict], max_tokens: int, temperature: float
    ) -> AsyncIterator[Delta]:
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        res = self._results.pop(question, None)
        if (
            res is None
        ):  # a plain /v1/chat/completions call: treat the last user message as the question
            res = await asyncio.to_thread(self._ask_locked, question)
        words = res.answer.split(" ")
        step = self.words_per_chunk
        for i in range(0, len(words), step):
            yield Delta(" ".join(words[i : i + step]) + (" " if i + step < len(words) else ""))
        yield Delta(
            "",
            "stop",
            res.prompt_tokens or approx_tokens(question),
            res.completion_tokens or approx_tokens(res.answer),
        )


def build_app(
    copilot: Copilot,
    keys_spec: list[dict],
    *,
    admin_token: str | None = None,
    max_inflight: int = 4,
    max_queue: int = 8,
    feedback_path: str | None = None,
):
    backend = CopilotBackend(copilot)
    app = create_app(
        backend,
        load_keys(keys_spec),
        settings=Settings(
            max_inflight=max_inflight,
            max_queue=max_queue,
            admin_token=admin_token,
            model_name="course-copilot",
            feedback_path=feedback_path,
        ),
        retriever=backend.prepare,
    )
    app.state.copilot = copilot
    return app


def gate_from_env(env, reranker_factory=None):
    """The relevance gate named by ``COPILOT_GATE``: ``rerank`` (default), ``cosine``, ``cascade`` (cosine first, the cross-encoder only for the uncertain middle: ``COPILOT_CASCADE_BAND``
    "low,high") or ``none``. ``reranker_factory`` is injectable so the choice can be tested without loading a model."""
    from .gate import CascadeGate, CosineGate, RerankGate

    kind = env.get("COPILOT_GATE", "rerank")
    if kind == "none":
        return None
    if kind == "cosine":
        return CosineGate(float(env.get("COPILOT_GATE_TAU", "0.66")))
    if kind in ("rerank", "cascade"):
        if reranker_factory is None:
            from common.rerank import get_reranker as reranker_factory
        rerank = RerankGate(reranker_factory(), float(env.get("COPILOT_GATE_TAU", "-1.94")))
        if kind == "rerank":
            return rerank
        low, high = (float(x) for x in env.get("COPILOT_CASCADE_BAND", "0.677,0.716").split(","))
        return CascadeGate(rerank, low, high)
    raise ValueError(f"COPILOT_GATE must be rerank, cascade, cosine or none, not {kind!r}")


def from_env():
    """Build the whole service from environment variables (the same way a container is configured).

    COPILOT_INDEX_DIR   a directory written by ``ingest.build_index`` (chunks.jsonl, vectors.npy, manifest.json)      [required]
    COPILOT_KEYS        JSON list of API-key entries (see ``llmapi.auth.load_keys``)                                   [required]
    COPILOT_ADMIN_TOKEN protects /metrics                                                                              [optional]
    COPILOT_GATE        ``rerank`` (default), ``cascade``, ``cosine`` or ``none``; COPILOT_GATE_TAU the re-ranker/cosine threshold; COPILOT_CASCADE_BAND "low,high" cosines
    COPILOT_LLM_URL     set to answer with a chat model behind an OpenAI-compatible server (verified, extractive fallback); unset = extractive
    COPILOT_FEEDBACK    path of an append-only JSONL file for /v1/feedback
    """
    from common.embed import get_embedder

    from .answer import ExtractiveAnswerer, LlmAnswerer
    from .core import idf_from_index
    from .guard import InputGuard, OutputGuard
    from .ingest import RagIndex
    from .llm import OpenAIChat
    from .retrieve import Retriever

    raw = os.environ.get("COPILOT_KEYS", "")
    index_dir = os.environ.get("COPILOT_INDEX_DIR", "")
    if not raw or not index_dir:
        sys.exit(
            "set COPILOT_KEYS and COPILOT_INDEX_DIR: the service refuses to start without keys or an index"
        )
    index = RagIndex(get_embedder(), index_dir)
    if not index.chunks:
        sys.exit(
            f"COPILOT_INDEX_DIR={index_dir} holds no index (or one built with a different embedder)"
        )
    gate = gate_from_env(os.environ)
    extractive = ExtractiveAnswerer(idf_from_index(index))
    answerer = (
        LlmAnswerer(OpenAIChat(os.environ["COPILOT_LLM_URL"]), extractive)
        if os.environ.get("COPILOT_LLM_URL")
        else extractive
    )
    copilot = Copilot(
        Retriever(index),
        answerer,
        gate=gate,
        input_guard=InputGuard(),
        output_guard=OutputGuard(
            secrets_=[os.environ["COPILOT_ADMIN_TOKEN"]]
            if os.environ.get("COPILOT_ADMIN_TOKEN")
            else []
        ),
    )
    return build_app(
        copilot,
        json.loads(raw),
        admin_token=os.environ.get("COPILOT_ADMIN_TOKEN") or None,
        feedback_path=os.environ.get("COPILOT_FEEDBACK") or None,
    )


def main() -> None:  # pragma: no cover - a process entry point
    import logging

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(
        from_env(),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        log_level="warning",
        timeout_graceful_shutdown=20,
    )  # noqa: S104


if __name__ == "__main__":
    main()
