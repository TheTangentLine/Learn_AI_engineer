"""The candidate upgrades, each a `Config` over the same chunks (so differences are the technique's)."""

from __future__ import annotations

from day3_solution import (  # noqa: E402 (path set in __init__)
    MiniIndex,
    hypothetical_questions,
    parent_child,
)

from common.crag import corrective_search
from common.rag import RagIndex

from .harness import Config

CRAG_TAU = -1.3  # median leave-one-out threshold measured in Week 4 Day 4 (a cross-encoder logit)


class CragRetrieve:
    """Corrective retrieval that counts its own work, so its cost is MEASURED, not guessed."""

    def __init__(self, index: RagIndex, reranker, rewriter, k: int):
        self.index, self.reranker, self.rewriter, self.k = index, reranker, rewriter, k
        self.queries = self.pairs = self.llm = 0

    def __call__(self, query: str) -> list[tuple[str, str]]:
        res = corrective_search(
            self.index,
            query,
            self.reranker,
            CRAG_TAU,
            rewriter=self.rewriter,
            k=20,
            depth=20,
            max_steps=4,
        )
        self.queries += 1
        self.pairs += res.rerank_pairs
        self.llm += int(res.corrected and self.rewriter is not None)
        return [(h.metadata["doc"], h.text) for h in res.hits]

    def per_query_cost(self) -> tuple[float, float]:
        n = max(self.queries, 1)
        return self.pairs / n, self.llm / n


def build_configs(index: RagIndex, emb, reranker, chat=None) -> list[Config]:
    chunks = index.chunks
    docs, texts = [c["doc"] for c in chunks], [c["text"] for c in chunks]
    n = len(chunks)

    def hybrid(alpha: float, rerank: bool):
        def run(q: str) -> list[tuple[str, str]]:
            hits = index.search(
                q, 20, alpha=alpha, reranker=reranker if rerank else None, rerank_depth=20
            )
            return [(h.metadata["doc"], h.text) for h in hits]

        return run

    pc = parent_child(emb, chunks, 350)
    configs = [
        Config("baseline (hybrid, k=4)", hybrid(0.5, False), 4, index_rows=n),
        Config("hybrid a=0.25, k=4", hybrid(0.25, False), 4, index_rows=n, note="more BM25 weight"),
        Config("hybrid, k=8", hybrid(0.5, False), 8, index_rows=n, note="just send more chunks"),
        Config("hybrid + rerank, k=4", hybrid(0.5, True), 4, rerank_pairs=20, index_rows=n),
        Config("hybrid + rerank, k=8", hybrid(0.5, True), 8, rerank_pairs=20, index_rows=n),
        Config(
            "parent-child (350), k=4", lambda q: pc.search(q, 20), 4, index_rows=len(pc.index_texts)
        ),
    ]
    if chat is not None:
        qs = [hypothetical_questions(chat, t) for t in texts]  # cached from Day 3
        hq = MiniIndex(emb, docs, texts, [f"{t}\n\n{x}" for t, x in zip(texts, qs, strict=True)])
        configs.append(
            Config(
                "hypothetical Qs, k=4",
                lambda q: hq.search(q, 20),
                4,
                index_rows=n,
                note="ingest-time LLM calls",
            )
        )
    crag = CragRetrieve(index, reranker, None, 4)
    configs.append(Config("CRAG (rerank + fallbacks), k=4", crag, 4, index_rows=n))
    return configs
