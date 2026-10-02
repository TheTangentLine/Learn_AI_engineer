"""Run the API:  python -m llmapi   (configuration from environment variables, so the same command works on a laptop, in a container and on a platform).

LLMAPI_BACKEND_URL    where the model server is (default http://127.0.0.1:8080)
LLMAPI_BACKENDS       several models: a JSON object {"model name": "http://server:port"}; the request's ``model`` field picks one (overrides LLMAPI_BACKEND_URL)
LLMAPI_DOCS_DIR       a directory of Markdown files: turns on POST /v1/ask (BM25 retrieval + citations) over them
LLMAPI_MIN_SCORE     relevance floor for retrieval (BM25 score); passages at or below it are dropped, and with none left the model is told there are no sources
LLMAPI_ABSTAIN_WITHOUT_SOURCES  1: /v1/ask answers with the refusal sentence, without calling the model, when retrieval found nothing above the floor
LLMAPI_ASK_MODEL      which backend answers /v1/ask when the request names none (needed with LLMAPI_BACKENDS)
LLMAPI_KEYS_FILE      a JSON list of key entries (see auth.load_keys); or LLMAPI_KEYS with the JSON itself
LLMAPI_ADMIN_TOKEN    protects /metrics
LLMAPI_MAX_INFLIGHT   model slots (match llama-server's -np); LLMAPI_MAX_QUEUE
PORT / HOST           where to listen (default 0.0.0.0:8000)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import uvicorn

from .app import Settings, create_app
from .auth import load_keys
from .backends import LlamaServerBackend
from .rag import Bm25Index, load_corpus, make_retriever


def build_app():
    raw = os.environ.get("LLMAPI_KEYS") or (
        Path(os.environ["LLMAPI_KEYS_FILE"]).read_text()
        if os.environ.get("LLMAPI_KEYS_FILE")
        else ""
    )
    if not raw:
        sys.exit(
            "set LLMAPI_KEYS_FILE (or LLMAPI_KEYS): the API refuses to start without any API keys"
        )
    settings = Settings(
        max_inflight=int(os.environ.get("LLMAPI_MAX_INFLIGHT", "4")),
        max_queue=int(os.environ.get("LLMAPI_MAX_QUEUE", "8")),
        admin_token=os.environ.get("LLMAPI_ADMIN_TOKEN") or None,
        feedback_path=os.environ.get("LLMAPI_FEEDBACK_PATH") or None,
        ask_model=os.environ.get("LLMAPI_ASK_MODEL") or None,
        abstain_without_sources=os.environ.get("LLMAPI_ABSTAIN_WITHOUT_SOURCES") == "1",
    )
    if os.environ.get("LLMAPI_BACKENDS"):
        backend = {
            name: LlamaServerBackend(url)
            for name, url in json.loads(os.environ["LLMAPI_BACKENDS"]).items()
        }
    else:
        backend = LlamaServerBackend(os.environ.get("LLMAPI_BACKEND_URL", "http://127.0.0.1:8080"))
    retriever = None
    if os.environ.get("LLMAPI_DOCS_DIR"):
        chunks = load_corpus(Path(os.environ["LLMAPI_DOCS_DIR"]))
        if not chunks:
            sys.exit(f"LLMAPI_DOCS_DIR={os.environ['LLMAPI_DOCS_DIR']} holds no Markdown to index")
        retriever = make_retriever(
            Bm25Index(chunks), float(os.environ.get("LLMAPI_MIN_SCORE", "0"))
        )
    return create_app(backend, load_keys(json.loads(raw)), settings=settings, retriever=retriever)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(
        build_app(),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        log_level="warning",
        timeout_graceful_shutdown=20,
    )  # noqa: S104 - a container must listen on all interfaces


if __name__ == "__main__":
    main()
