"""Embeddings for the roadmap: a local model by default, OpenAI optionally, a fake for tests.

    from common.embed import get_embedder
    emb = get_embedder()                       # local BAAI/bge-small-en-v1.5 (384 dims, ~130 MB)
    D = emb.embed_documents(["KV cache stores keys and values", "Paris is in France"])
    q = emb.embed_query("how does the cache work?")
    scores = D @ q                              # vectors are L2-normalised: dot product == cosine

Choose with ``EMBED_PROVIDER=local|openai`` (default local). Vectors are float32 and L2-normalised,
so every similarity in the course is a plain dot product. Results are cached in SQLite
(``outputs/embed_cache.sqlite``) keyed by (model, kind, text), so re-running a lesson is instant.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np

LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
# bge models are trained so that *queries* carry this prefix and documents do not.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
OPENAI_MODEL = "text-embedding-3-small"
OPENAI_PRICE_PER_MTOK = 0.02
CACHE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "embed_cache.sqlite"


def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


class _Cache:
    """Tiny SQLite key/value store for embedding vectors."""

    def __init__(self, path: Path | None):
        self.db = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(path, check_same_thread=False)  # a server calls us from worker threads; callers must serialise access (the Week 12 service holds a lock)
            self.db.execute("CREATE TABLE IF NOT EXISTS v (k TEXT PRIMARY KEY, dim INT, b BLOB)")

    @staticmethod
    def key(model: str, kind: str, text: str) -> str:
        return hashlib.sha1(f"{model}\x00{kind}\x00{text}".encode()).hexdigest()

    def get(self, k: str) -> np.ndarray | None:
        if self.db is None:
            return None
        row = self.db.execute("SELECT b FROM v WHERE k=?", (k,)).fetchone()
        return np.frombuffer(row[0], dtype=np.float32) if row else None

    def put_many(self, items: list[tuple[str, np.ndarray]]) -> None:
        if self.db is None:
            return
        self.db.executemany(
            "INSERT OR REPLACE INTO v VALUES (?,?,?)",
            [(k, len(v), v.astype(np.float32).tobytes()) for k, v in items],
        )
        self.db.commit()


class Embedder:
    """Base class: subclasses implement ``_embed(texts, kind)`` returning raw vectors."""

    name = "base"
    dim = 0

    def __init__(self, cache_path: Path | None = CACHE_PATH):
        self._cache = _Cache(cache_path)
        self.cache_hits = 0
        self.cache_misses = 0

    def _embed(self, texts: list[str], kind: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _embed_cached(self, texts: Sequence[str], kind: str) -> np.ndarray:
        out: list[np.ndarray | None] = []
        missing: list[int] = []
        for i, t in enumerate(texts):
            hit = self._cache.get(_Cache.key(self.name, kind, t))
            out.append(hit)
            if hit is None:
                missing.append(i)
        self.cache_hits += len(texts) - len(missing)
        self.cache_misses += len(missing)
        if missing:
            fresh = normalize(self._embed([texts[i] for i in missing], kind))
            self._cache.put_many(
                [
                    (_Cache.key(self.name, kind, texts[i]), v)
                    for i, v in zip(missing, fresh, strict=True)
                ]
            )
            for i, v in zip(missing, fresh, strict=True):
                out[i] = v
        return np.stack(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed_cached(list(texts), "doc")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_cached([text], "query")[0]


class LocalEmbedder(Embedder):
    """Hugging Face encoder + CLS pooling (what bge models use) + L2 normalisation."""

    def __init__(
        self, model: str = LOCAL_MODEL, batch_size: int = 32, cache_path: Path | None = CACHE_PATH
    ):
        super().__init__(cache_path)
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name, self._torch = model, torch
        self._tok = AutoTokenizer.from_pretrained(model)
        self._model = AutoModel.from_pretrained(model).eval()
        self.dim = self._model.config.hidden_size
        self.batch_size = batch_size
        self.max_tokens = min(512, self._tok.model_max_length)

    def count_tokens(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))

    def _embed(self, texts: list[str], kind: str) -> np.ndarray:
        if kind == "query" and "bge" in self.name:
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = self._tok(
                texts[i : i + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )
            with self._torch.no_grad():
                out.append(self._model(**batch).last_hidden_state[:, 0].numpy())  # CLS token
        return np.concatenate(out)


class OpenAIEmbedder(Embedder):
    name = OPENAI_MODEL
    dim = 1536

    def __init__(self, model: str = OPENAI_MODEL, cache_path: Path | None = CACHE_PATH):
        super().__init__(cache_path)
        from openai import OpenAI

        self.name, self._client = model, OpenAI()
        self.tokens_billed = 0

    def _embed(self, texts: list[str], kind: str) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), 256):
            r = self._client.embeddings.create(model=self.name, input=texts[i : i + 256])
            self.tokens_billed += r.usage.total_tokens
            vecs += [d.embedding for d in r.data]
        return np.array(vecs, dtype=np.float32)

    @property
    def cost_usd(self) -> float:
        return self.tokens_billed * OPENAI_PRICE_PER_MTOK / 1e6


class HashEmbedder(Embedder):
    """Deterministic bag-of-hashed-words embedder: NOT semantic, but instant and dependency-free.

    Texts sharing words get high cosine similarity. Used by tests and ``--offline`` demos so
    they don't need a model download; lessons say clearly when they use it.
    """

    name = "hash-384"
    dim = 384

    def __init__(self, dim: int = 384, cache_path: Path | None = None):
        super().__init__(cache_path)
        self.dim = dim
        self.name = f"hash-{dim}"

    def _embed(self, texts: list[str], kind: str) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[r, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return out


_SINGLETON: dict[str, Embedder] = {}


def get_embedder(provider: str | None = None) -> Embedder:
    """The embedder selected by ``EMBED_PROVIDER`` (local | openai | hash). Cached per process."""
    provider = provider or os.getenv("EMBED_PROVIDER", "local")
    if provider not in _SINGLETON:
        _SINGLETON[provider] = {
            "local": LocalEmbedder,
            "openai": OpenAIEmbedder,
            "hash": HashEmbedder,
        }[provider]()
    return _SINGLETON[provider]


def cosine_top_k(
    doc_vecs: np.ndarray, query_vec: np.ndarray, k: int = 5
) -> list[tuple[int, float]]:
    """Top-k (index, score) by dot product. With normalised vectors this is cosine similarity."""
    scores = doc_vecs @ query_vec
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return [(int(i), float(scores[i])) for i in idx[np.argsort(-scores[idx])]]
