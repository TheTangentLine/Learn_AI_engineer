"""Cross-encoder re-ranking: score (query, passage) pairs jointly, more precise than a bi-encoder.

    from common.rerank import get_reranker
    rr = get_reranker()                       # local BAAI/bge-reranker-base (278M params)
    order = rr.rerank("how do I avoid retry storms?", passages, top_k=5)   # [(index, score), ...]

A bi-encoder embeds query and passage *separately* (fast, can be pre-computed). A cross-encoder
reads them *together* (slow, cannot be pre-computed, but sees word-level interactions). The
standard recipe: retrieve ~20-100 candidates cheaply, then re-rank only those. Scores are raw
logits: only their order within one query is meaningful. Results are cached in SQLite, keyed by
(model, query, passage).
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "rerank_cache.sqlite"
LOCAL_MODEL = "BAAI/bge-reranker-base"


class Reranker:
    name = "base"

    def __init__(self, cache_path: Path | None = CACHE_PATH):
        self.db = None
        self.calls = 0  # model forward passes (pairs), excluding cache hits
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(cache_path, check_same_thread=False)  # a server calls us from worker threads; callers must serialise access (the Week 12 service holds a lock)
            self.db.execute("CREATE TABLE IF NOT EXISTS s (k TEXT PRIMARY KEY, v REAL)")

    def _key(self, q: str, p: str) -> str:
        return hashlib.sha1(f"{self.name}\x00{q}\x00{p}".encode()).hexdigest()

    def _score(self, query: str, passages: list[str]) -> list[float]:  # pragma: no cover
        raise NotImplementedError

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        out: list[float | None] = [None] * len(passages)
        missing = []
        for i, p in enumerate(passages):
            row = (
                self.db.execute("SELECT v FROM s WHERE k=?", (self._key(query, p),)).fetchone()
                if self.db
                else None
            )
            if row:
                out[i] = row[0]
            else:
                missing.append(i)
        if missing:
            fresh = self._score(query, [passages[i] for i in missing])
            self.calls += len(missing)
            for i, v in zip(missing, fresh, strict=True):
                out[i] = v
            if self.db:
                self.db.executemany(
                    "INSERT OR REPLACE INTO s VALUES (?,?)",
                    [
                        (self._key(query, passages[i]), v)
                        for i, v in zip(missing, fresh, strict=True)
                    ],
                )
                self.db.commit()
        return [float(v) for v in out]  # type: ignore[arg-type]

    def rerank(
        self, query: str, passages: Sequence[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        sc = self.scores(query, passages)
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        return [(i, sc[i]) for i in order[: top_k or len(order)]]


class LocalReranker(Reranker):
    def __init__(
        self, model: str = LOCAL_MODEL, batch_size: int = 16, cache_path: Path | None = CACHE_PATH
    ):
        super().__init__(cache_path)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name, self._torch, self.batch_size = model, torch, batch_size
        self._tok = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model, dtype=torch.float32
        ).eval()

    def _score(self, query: str, passages: list[str]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(passages), self.batch_size):
            batch = self._tok(
                [query] * len(passages[i : i + self.batch_size]),
                passages[i : i + self.batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with self._torch.no_grad():
                out += self._model(**batch).logits.view(-1).tolist()
        return out


class OverlapReranker(Reranker):
    """Instant, non-neural stand-in for tests: scores by shared words. NOT a real reranker."""

    name = "overlap"

    def __init__(self):
        super().__init__(cache_path=None)

    def _score(self, query: str, passages: list[str]) -> list[float]:
        import re

        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        return [len(q & set(re.findall(r"[a-z0-9]+", p.lower()))) / (len(q) or 1) for p in passages]


_SINGLETON: dict[str, Reranker] = {}


def get_reranker(kind: str = "local") -> Reranker:
    if kind not in _SINGLETON:
        _SINGLETON[kind] = {"local": LocalReranker, "overlap": OverlapReranker}[kind]()
    return _SINGLETON[kind]
