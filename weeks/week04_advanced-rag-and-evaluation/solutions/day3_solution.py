"""Week 4 Day 3 - Solution: advanced indexing, measured.

Variants (each is the SAME hybrid retriever over differently-prepared text):
  baseline              heading-aware chunks
  extractive context    prefix every chunk with its lesson title + first learning objective  (free)
  parent-child          embed small CHILD chunks, return the PARENT chunk                    (free)
  LLM context           a local LLM writes one sentence situating each chunk in its lesson   (--llm, slow)
  hypothetical Qs       a local LLM writes 2 questions each chunk answers; appended to index (--llm, slow)

Scored on the Day 1 golden set (50 questions) with paired bootstrap tests against the baseline.

  uv run python .../day3_solution.py              # baseline, extractive context, parent-child
  uv run python .../day3_solution.py --llm        # + the two LLM variants (generations are cached)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "weeks/week03_embeddings-and-rag/solutions/weekly"))

from docs_qa.app import course_sources  # noqa: E402

from common.chunking import recursive  # noqa: E402
from common.embed import get_embedder  # noqa: E402
from common.evalkit import (  # noqa: E402
    bootstrap_ci,
    evaluate_retriever,
    fmt_ci,
    load_golden,
    paired_bootstrap,
)
from common.local_llm import LocalChat  # noqa: E402
from common.rag import RagIndex, _tokenize  # noqa: E402

PINNED_WEEKS = (1, 2, 3)
GOLDEN = ROOT / "outputs" / "golden_week4.jsonl"


# ----------------------------------------------------------------- a tiny index over arbitrary texts


@dataclass
class MiniIndex:
    """Hybrid (BM25 + vector) search over `index_texts`, returning `(doc, display_text)`.

    `group` maps each indexed row to a parent id: results are collapsed to the best row per parent,
    which is how parent-child retrieval works (search children, return parents).
    """

    emb: object
    docs: list[str]
    display: list[str]
    index_texts: list[str]
    group: list[int] | None = None
    alpha: float = 0.5

    def __post_init__(self):
        self.vecs = self.emb.embed_documents(self.index_texts)
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.index_texts])
        self.parents: dict[int, tuple[str, str]] = {}

    def search(self, query: str, k: int = 20) -> list[tuple[str, str]]:
        n01 = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)  # noqa: E731
        dense = self.vecs @ self.emb.embed_query(query)
        score = self.alpha * n01(dense) + (1 - self.alpha) * n01(
            self.bm25.get_scores(_tokenize(query))
        )
        order = [int(i) for i in np.argsort(-score, kind="stable")]
        if self.group is None:
            return [(self.docs[i], self.display[i]) for i in order[:k]]
        seen, out = set(), []
        for i in order:
            if self.group[i] not in seen:
                seen.add(self.group[i])
                out.append((self.docs[i], self.display[i]))
                if len(out) == k:
                    break
        return out


# ----------------------------------------------------------------- the variants


def extractive_context(doc_text: str, title: str) -> str:
    """Title + the lesson's first learning objective: free, deterministic 'what is this document about?'."""
    m = re.search(r"##\s+Learning objectives\s*\n+\s*[-*]\s+(.+)", doc_text)
    return f"{title}. {m.group(1).strip()}" if m else title


def doc_title(text: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else ""


def parent_child(emb, chunks: list[dict], child_size: int = 350) -> MiniIndex:
    docs, display, index_texts, group = [], [], [], []
    for pid, c in enumerate(chunks):
        pieces = recursive(c["text"], child_size, 50) or [c["text"]]
        for piece in pieces:
            docs.append(c["doc"])
            display.append(c["text"])  # what we RETURN: the whole parent
            index_texts.append(piece)  # what we SEARCH: the small child
            group.append(pid)
    return MiniIndex(emb, docs, display, index_texts, group)


def llm_context(chat: LocalChat, doc_title_: str, doc_intro: str, chunk_text: str) -> str:
    """Anthropic-style contextual retrieval: one sentence situating the chunk within its document."""
    prompt = (
        f"<document_title>{doc_title_}</document_title>\n<document_start>\n{doc_intro[:600]}\n"
        f"</document_start>\n<chunk>\n{chunk_text[:700]}\n</chunk>\n\nWrite ONE short sentence that "
        "situates this chunk within the document so that it can be found by search. Output only the sentence."
    )
    out = chat("You write search context for documentation chunks.", prompt, max_new_tokens=45)
    return out.strip().splitlines()[0] if out.strip() else ""


def hypothetical_questions(chat: LocalChat, chunk_text: str) -> str:
    prompt = (
        f"<chunk>\n{chunk_text[:700]}\n</chunk>\n\nWrite 2 different questions that this chunk answers, "
        "one per line, no numbering."
    )
    out = chat("You write search questions for documentation chunks.", prompt, max_new_tokens=60)
    return " ".join(line.strip(" -*0123456789.)") for line in out.splitlines() if line.strip())[
        :300
    ]


# ----------------------------------------------------------------- evaluation


def main() -> None:
    use_llm = "--llm" in sys.argv
    emb = get_embedder()
    sources = [d for d in course_sources() if d.meta["week"] in PINNED_WEEKS]
    index = RagIndex(emb, ROOT / "outputs" / "week4_index")
    index.sync(sources)
    chunks = index.chunks
    golden = load_golden(GOLDEN)
    by_doc = {d.id: d for d in sources}
    print(
        f"{len(chunks)} chunks | golden: {len(golden)} questions "
        f"({sum(g.kind == 'manual' for g in golden)} manual, {sum(g.kind == 'synthetic' for g in golden)} synthetic)"
    )

    docs = [c["doc"] for c in chunks]
    texts = [c["text"] for c in chunks]
    variants: dict[str, MiniIndex] = {"baseline": MiniIndex(emb, docs, texts, texts)}
    ctx = {d: extractive_context(by_doc[d].text, doc_title(by_doc[d].text)) for d in by_doc}
    variants["extractive context"] = MiniIndex(
        emb, docs, texts, [f"{ctx[d]}\n\n{t}" for d, t in zip(docs, texts, strict=True)]
    )
    variants["parent-child (350)"] = parent_child(emb, chunks, 350)
    if use_llm:
        chat = LocalChat()
        gen = [
            llm_context(chat, doc_title(by_doc[d].text), by_doc[d].text[:600], t)
            for d, t in zip(docs, texts, strict=True)
        ]
        print(f"LLM context example: {gen[len(gen) // 2]!r}")
        variants["LLM context (0.5B)"] = MiniIndex(
            emb, docs, texts, [f"{g}\n\n{t}" for g, t in zip(gen, texts, strict=True)]
        )
        qs = [hypothetical_questions(chat, t) for t in texts]
        print(f"hypothetical-question example: {qs[len(qs) // 2]!r}")
        variants["hypothetical Qs (0.5B)"] = MiniIndex(
            emb, docs, texts, [f"{t}\n\n{q}" for t, q in zip(texts, qs, strict=True)]
        )

    results = {
        n: evaluate_retriever(n, lambda q, v=v: v.search(q, 20), golden)
        for n, v in variants.items()
    }
    kinds = [g.kind for g in golden]
    print(f"\n{'variant':<24}{'hit@1':>16}{'hit@5':>16}{'MRR':>20}   MRR manual / synthetic")
    for n, r in results.items():
        man = np.mean([m for m, k in zip(r.mrr, kinds, strict=True) if k == "manual"])
        syn = np.mean([m for m, k in zip(r.mrr, kinds, strict=True) if k == "synthetic"])
        print(
            f"{n:<24}{fmt_ci(bootstrap_ci(r.hit1)):>16}{fmt_ci(bootstrap_ci(r.hit5)):>16}"
            f"{fmt_ci(bootstrap_ci(r.mrr), pct=False):>20}   {man:.2f} / {syn:.2f}"
        )

    print("\nPaired comparison against the baseline (MRR; diff = variant - baseline)")
    for n, r in results.items():
        if n == "baseline":
            continue
        p = paired_bootstrap(r.mrr, results["baseline"].mrr)
        h = paired_bootstrap(r.hit5, results["baseline"].hit5)
        print(
            f"  {n:<24} MRR {p['diff']:+.3f} [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}] p={p['p']:.3f} "
            f"W/L/T {p['wins']}/{p['losses']}/{p['ties']} | hit@5 {h['diff']:+.2f} p={h['p']:.2f}"
        )
    sizes = {
        n: (len(v.index_texts), float(np.mean([len(t) for t in v.index_texts])))
        for n, v in variants.items()
    }
    print(
        "\nIndex size (rows, mean chars): "
        + "; ".join(f"{n}: {r} rows / {c:.0f}" for n, (r, c) in sizes.items())
    )
    ctx_cost = sum(len(ctx[d]) for d in docs) // 4
    print(
        f"Extractive context adds ~{ctx_cost:,} tokens of index text; LLM context adds one generation per chunk "
        f"({len(chunks)} calls)."
    )


if __name__ == "__main__":
    main()
