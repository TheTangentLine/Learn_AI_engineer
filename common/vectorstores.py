"""One interface over five vector stores, so lessons can swap them with a single line.

    from common.vectorstores import make_store
    store = make_store("qdrant", dim=384)           # numpy | hnsw | chroma | qdrant | pgvector
    store.add(ids, vectors, metadatas, texts)
    hits = store.search(query_vec, k=5, where={"week": 2})     # equality filters, AND-ed
    hits[0].id, hits[0].score, hits[0].metadata, hits[0].text

Scores are always **cosine similarity** (higher = closer), whatever the store uses internally.
Vectors must be L2-normalised (``common.embed`` guarantees this).

``pgvector`` needs Postgres with the pgvector extension:
    docker compose -f weeks/week03_embeddings-and-rag/docker-compose.yml up -d
(override the connection with ``PGVECTOR_DSN``).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_DSN = "postgresql://roadmap:roadmap@localhost:55432/roadmap"


@dataclass
class Hit:
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str = ""


def _matches(meta: dict, where: dict | None) -> bool:
    return not where or all(meta.get(k) == v for k, v in where.items())


class VectorStore:
    name = "base"

    def __init__(self, dim: int):
        self.dim = dim

    def add(
        self, ids: list[str], vectors: np.ndarray, metadatas: list[dict], texts: list[str]
    ) -> None:
        raise NotImplementedError

    def search(self, query: np.ndarray, k: int = 5, where: dict | None = None) -> list[Hit]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ----------------------------------------------------------------- exact, in memory


class NumpyStore(VectorStore):
    """Exact brute-force search. The reference every approximate index is judged against."""

    name = "numpy"

    def __init__(self, dim: int):
        super().__init__(dim)
        self.ids: list[str] = []
        self.meta: list[dict] = []
        self.texts: list[str] = []
        self.mat = np.zeros((0, dim), dtype=np.float32)

    def add(self, ids, vectors, metadatas, texts):
        self.ids += list(ids)
        self.meta += list(metadatas)
        self.texts += list(texts)
        self.mat = np.vstack([self.mat, np.asarray(vectors, dtype=np.float32)])

    def search(self, query, k=5, where=None):
        scores = self.mat @ np.asarray(query, dtype=np.float32)
        if where:  # pre-filter: exact and never returns fewer than k if k matches exist
            mask = np.array([_matches(m, where) for m in self.meta])
            scores = np.where(mask, scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [Hit(self.ids[i], float(scores[i]), self.meta[i], self.texts[i]) for i in idx]

    def count(self):
        return len(self.ids)


# ----------------------------------------------------------------- HNSW library


class HnswStore(VectorStore):
    """hnswlib: the HNSW graph index that Chroma, Qdrant and pgvector are built around."""

    name = "hnsw"

    def __init__(
        self,
        dim: int,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        max_elements: int = 200_000,
    ):
        super().__init__(dim)
        import hnswlib

        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=m)
        self.index.set_ef(ef_search)
        self.ids: list[str] = []
        self.meta: list[dict] = []
        self.texts: list[str] = []

    def set_ef(self, ef: int) -> None:
        self.index.set_ef(ef)

    def add(self, ids, vectors, metadatas, texts):
        start = len(self.ids)
        self.ids += list(ids)
        self.meta += list(metadatas)
        self.texts += list(texts)
        self.index.add_items(np.asarray(vectors, dtype=np.float32), np.arange(start, len(self.ids)))

    def search(self, query, k=5, where=None):
        flt = (lambda i: _matches(self.meta[i], where)) if where else None
        k = min(k, len(self.ids))
        while k > 0:
            try:
                labels, dist = self.index.knn_query(
                    np.asarray(query, dtype=np.float32), k=k, filter=flt
                )
                break
            except RuntimeError:
                # hnswlib raises when a selective filter leaves fewer than k reachable candidates.
                # A known HNSW + filter limitation: we degrade to fewer results (see Day 2 lesson).
                k -= 1
        else:
            return []
        return [
            Hit(self.ids[i], 1.0 - float(d), self.meta[i], self.texts[i])
            for i, d in zip(labels[0], dist[0], strict=True)
        ]

    def count(self):
        return len(self.ids)


# ----------------------------------------------------------------- Chroma


class ChromaStore(VectorStore):
    name = "chroma"

    def __init__(self, dim: int):
        super().__init__(dim)
        import chromadb

        self.client = chromadb.EphemeralClient()
        self.col = self.client.create_collection(
            f"c{uuid.uuid4().hex[:12]}", configuration={"hnsw": {"space": "cosine"}}
        )

    def add(self, ids, vectors, metadatas, texts):
        for i in range(0, len(ids), 4000):  # Chroma limits the batch size
            s = slice(i, i + 4000)
            self.col.add(
                ids=list(ids[s]),
                embeddings=np.asarray(vectors[s]).tolist(),
                metadatas=list(metadatas[s]),
                documents=list(texts[s]),
            )

    def search(self, query, k=5, where=None):
        flt = None
        if where:
            conds = [{key: {"$eq": v}} for key, v in where.items()]
            flt = conds[0] if len(conds) == 1 else {"$and": conds}
        r = self.col.query(query_embeddings=[np.asarray(query).tolist()], n_results=k, where=flt)
        return [
            Hit(i, 1.0 - d, m, t)
            for i, d, m, t in zip(
                r["ids"][0], r["distances"][0], r["metadatas"][0], r["documents"][0], strict=True
            )
        ]

    def count(self):
        return self.col.count()


# ----------------------------------------------------------------- Qdrant


class QdrantStore(VectorStore):
    name = "qdrant"

    def __init__(self, dim: int):
        super().__init__(dim)
        from qdrant_client import QdrantClient, models

        self.models = models
        self.client = QdrantClient(":memory:")
        self.collection = f"c{uuid.uuid4().hex[:12]}"
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        self._ids: dict[int, str] = {}

    def add(self, ids, vectors, metadatas, texts):
        m = self.models
        base = len(self._ids)
        for i, sid in enumerate(ids):
            self._ids[base + i] = sid
        for s in range(0, len(ids), 1000):
            pts = [
                m.PointStruct(
                    id=base + i,
                    vector=np.asarray(vectors[i]).tolist(),
                    payload={**metadatas[i], "_text": texts[i]},
                )
                for i in range(s, min(s + 1000, len(ids)))
            ]
            self.client.upsert(self.collection, points=pts)

    def search(self, query, k=5, where=None):
        m = self.models
        flt = (
            m.Filter(
                must=[
                    m.FieldCondition(key=key, match=m.MatchValue(value=v))
                    for key, v in where.items()
                ]
            )
            if where
            else None
        )
        res = self.client.query_points(
            self.collection,
            query=np.asarray(query).tolist(),
            limit=k,
            query_filter=flt,
            with_payload=True,
        )
        out = []
        for p in res.points:
            payload = dict(p.payload or {})
            text = payload.pop("_text", "")
            out.append(Hit(self._ids[p.id], float(p.score), payload, text))
        return out

    def count(self):
        return self.client.count(self.collection).count


# ----------------------------------------------------------------- Postgres + pgvector


class PgvectorStore(VectorStore):
    name = "pgvector"

    def __init__(self, dim: int, dsn: str | None = None, hnsw: bool = True):
        super().__init__(dim)
        import psycopg
        from pgvector.psycopg import register_vector

        self.dsn = dsn or os.getenv("PGVECTOR_DSN", DEFAULT_DSN)
        self.conn = psycopg.connect(self.dsn, autocommit=True, connect_timeout=3)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.table = f"chunks_{uuid.uuid4().hex[:10]}"
        self.conn.execute(
            f"CREATE TABLE {self.table} (id text PRIMARY KEY, text text, metadata jsonb, "
            f"embedding vector({dim}))"
        )
        self._hnsw = hnsw

    @classmethod
    def available(cls, dsn: str | None = None) -> bool:
        try:
            import psycopg

            with psycopg.connect(dsn or os.getenv("PGVECTOR_DSN", DEFAULT_DSN), connect_timeout=2):
                return True
        except Exception:
            return False

    def add(self, ids, vectors, metadatas, texts):
        with self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} VALUES (%s, %s, %s::jsonb, %s)",
                [
                    (i, t, json.dumps(m), np.asarray(v, dtype=np.float32))
                    for i, t, m, v in zip(ids, texts, metadatas, vectors, strict=True)
                ],
            )
        if self._hnsw:  # build the index after the bulk load (much faster than incremental)
            self.conn.execute(
                f"CREATE INDEX ON {self.table} USING hnsw (embedding vector_cosine_ops)"
            )

    def search(self, query, k=5, where=None):
        clauses, params = [], []
        for key, v in (where or {}).items():
            clauses.append("metadata->>%s = %s")
            params += [key, str(v)]
        sql = (
            f"SELECT id, text, metadata, 1 - (embedding <=> %s) AS score FROM {self.table} "
            + (f"WHERE {' AND '.join(clauses)} " if clauses else "")
            + "ORDER BY embedding <=> %s LIMIT %s"
        )
        q = np.asarray(query, dtype=np.float32)
        rows = self.conn.execute(sql, [q, *params, q, k]).fetchall()
        return [Hit(r[0], float(r[3]), r[2], r[1]) for r in rows]

    def count(self):
        return self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]

    def close(self):
        try:
            self.conn.execute(f"DROP TABLE IF EXISTS {self.table}")
        finally:
            self.conn.close()


_BACKENDS = {
    "numpy": NumpyStore,
    "hnsw": HnswStore,
    "chroma": ChromaStore,
    "qdrant": QdrantStore,
    "pgvector": PgvectorStore,
}


def make_store(name: str, dim: int, **kw) -> VectorStore:
    return _BACKENDS[name](dim, **kw)
