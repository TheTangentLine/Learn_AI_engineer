"""Week 4 Day 1 - Solution: a 50-question golden set and honest retrieval scores.

  1. Manual golden queries (20, from Week 3) + SYNTHETIC ones generated from chunks by a local LLM and
     passed through strict quality filters (30).
  2. Score four retrievers (BM25, vector, hybrid, hybrid + rerank) with 95% bootstrap intervals.
  3. Compare them with PAIRED bootstrap tests, split by manual vs. synthetic, and read the biases.

  uv run python weeks/week04_advanced-rag-and-evaluation/solutions/day1_solution.py
(first run generates ~90 questions with Qwen2.5-0.5B and scores ~1,000 re-ranker pairs; both are cached)
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "weeks/week03_embeddings-and-rag/solutions/weekly"))

from docs_qa.app import course_sources  # noqa: E402
from docs_qa.golden import ANSWERABLE  # noqa: E402

from common.embed import get_embedder  # noqa: E402
from common.evalkit import (  # noqa: E402
    GoldQuery,
    bootstrap_ci,
    evaluate_retriever,
    fmt_ci,
    norm,
    paired_bootstrap,
    save_golden,
)
from common.local_llm import LocalChat  # noqa: E402
from common.rag import RagIndex  # noqa: E402
from common.rerank import get_reranker  # noqa: E402

PINNED_WEEKS = (1, 2, 3)  # pin the corpus so numbers stay reproducible as later weeks are added
GOLDEN_PATH = (
    ROOT
    / "outputs"
    / ("golden_week4_loose.jsonl" if "--loose" in sys.argv else "golden_week4.jsonl")
)
N_SYNTHETIC = 30

STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "and",
    "in",
    "is",
    "it",
    "for",
    "on",
    "that",
    "this",
    "with",
    "how",
    "what",
    "why",
    "do",
    "does",
    "can",
    "you",
    "your",
    "are",
    "be",
    "or",
    "as",
    "by",
    "at",
    "from",
    "i",
    "my",
    "should",
    "when",
    "which",
    "not",
    "if",
    "so",
    "than",
    "then",
    "use",
    "using",
}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in STOP and len(w) > 2}


META = re.compile(
    r"\b(passage|the text|this text|document|the author|article|above|following|snippet)\b", re.I
)
MIN_SHARED = 2  # a good question shares at least this many content words with its source sentence


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


# ----------------------------------------------------------------- synthetic question generation

GEN_SYSTEM = "You write questions for a documentation search test set."


def target_sentence(chunk_text: str) -> str | None:
    """The most informative prose sentence of a chunk (no code, tables or headings)."""
    best = None
    in_code = False
    for line in chunk_text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        line = line.strip()
        if in_code or not line or line.startswith(("|", "#", "[", "<", ">", "-" * 3)):
            continue
        line = re.sub(r"^[-*]\s+|^\d+\.\s+", "", line)
        for sent in re.split(r"(?<=[.!?])\s+", line):
            if 70 <= len(sent) <= 220 and sent.count("`") <= 2 and sent[-1] in ".!?":
                if best is None or len(sent) > len(best):
                    best = sent
    return best


def clean_question(raw: str) -> str | None:
    line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    line = re.sub(r"^(question|q)\s*[:\-]\s*", "", line, flags=re.I).strip(" \"'*")
    return line if line.endswith("?") and 25 <= len(line) <= 160 else None


def generate_synthetic(
    index: RagIndex,
    chat: LocalChat,
    n: int,
    seed: int = 0,
    max_overlap: float = 0.40,
    strict: bool = True,
) -> tuple[list[GoldQuery], dict]:
    emb = index.emb
    rng = random.Random(seed)
    by_doc: dict[str, list[dict]] = {}
    for c in index.chunks:
        if len(c["text"]) >= 400 and c["text"].count("```") < 2:
            by_doc.setdefault(c["doc"], []).append(c)
    pool = [c for docs in by_doc.values() for c in rng.sample(docs, min(len(docs), 6))]
    rng.shuffle(pool)

    out: list[GoldQuery] = []
    stats = {
        "tried": 0,
        "no_sentence": 0,
        "bad_format": 0,
        "meta_reference": 0,
        "not_grounded": 0,
        "too_similar_to_passage": 0,
        "duplicate": 0,
    }
    kept_vecs: list[np.ndarray] = []
    for c in pool:
        if len(out) >= n:
            break
        sent = target_sentence(c["text"])
        if not sent:
            stats["no_sentence"] += 1
            continue
        stats["tried"] += 1
        raw = chat(
            GEN_SYSTEM,
            f"Passage:\n{sent}\n\nWrite ONE question that a developer could ask and that "
            "this passage answers. Use your own words: do not reuse the passage's distinctive "
            "words. Output only the question.",
            max_new_tokens=40,
        )
        q = clean_question(raw)
        if not q:
            stats["bad_format"] += 1
            continue
        if strict and META.search(q):  # "what does the passage suggest..." is about the test set
            stats["meta_reference"] += 1
            continue
        shared = content_words(q) & content_words(sent)
        if strict and len(shared) < MIN_SHARED:  # not grounded in THIS sentence: arbitrary "answer"
            stats["not_grounded"] += 1
            continue
        if jaccard(content_words(q), content_words(sent)) > max_overlap:
            stats["too_similar_to_passage"] += (
                1  # a question that copies the answer isn't a retrieval test
            )
            continue
        v = emb.embed_query(q)
        if any(float(v @ u) > 0.9 for u in kept_vecs):
            stats["duplicate"] += 1
            continue
        phrase = norm(sent)[:90]
        gq = GoldQuery(f"s{len(out) + 1:02d}", q, c["doc"], phrase, kind="synthetic")
        if not gq.is_relevant(c["doc"], c["text"]):  # the gold must be satisfiable by its own chunk
            stats["bad_format"] += 1
            continue
        kept_vecs.append(v)
        out.append(gq)
    return out, stats


# ----------------------------------------------------------------- evaluation


def main() -> None:
    emb = get_embedder()
    docs = [d for d in course_sources() if d.meta["week"] in PINNED_WEEKS]
    index = RagIndex(emb, ROOT / "outputs" / "week4_index")
    rep = index.sync(docs)
    print(f"index: {len(index.chunks)} chunks from {len(docs)} lessons ({rep})")

    manual = [
        GoldQuery(f"m{i + 1:02d}", q, lesson, phrase)
        for i, (q, lesson, phrase) in enumerate(ANSWERABLE)
    ]
    bad = [
        g.id for g in manual if not any(g.is_relevant(c["doc"], c["text"]) for c in index.chunks)
    ]
    assert not bad, f"manual gold not present in the corpus: {bad}"
    strict = "--loose" not in sys.argv
    print(
        f"synthetic filters: {'STRICT (meta words, grounding overlap)' if strict else 'LOOSE (format + copy filter only)'}"
    )
    synthetic, stats = generate_synthetic(index, LocalChat(), N_SYNTHETIC, strict=strict)
    golden = manual + synthetic
    save_golden(GOLDEN_PATH, golden)
    print(
        f"golden set: {len(manual)} manual + {len(synthetic)} synthetic = {len(golden)} -> {GOLDEN_PATH.name}"
    )
    print(f"synthetic funnel: {stats}")
    for g in synthetic[:4]:
        print(f"   e.g. {g.query!r}  <-  {g.phrase[:60]!r}")

    rr = get_reranker()
    retrievers = {
        "BM25": lambda q: [(h.metadata["doc"], h.text) for h in index.search(q, 20, alpha=0.0)],
        "vector": lambda q: [(h.metadata["doc"], h.text) for h in index.search(q, 20, alpha=1.0)],
        "hybrid (a=.5)": lambda q: [
            (h.metadata["doc"], h.text) for h in index.search(q, 20, alpha=0.5)
        ],
        "hybrid + rerank": lambda q: [
            (h.metadata["doc"], h.text)
            for h in index.search(q, 20, alpha=0.5, reranker=rr, rerank_depth=20)
        ],
    }
    results = {name: evaluate_retriever(name, fn, golden) for name, fn in retrievers.items()}
    kinds = [g.kind for g in golden]

    def subset(res, kind):
        idx = [i for i, k in enumerate(kinds) if kind in (None, k)]
        return {m: [getattr(res, m)[i] for i in idx] for m in ("hit1", "hit5", "mrr", "ndcg5")}

    for label, kind in (
        ("ALL 50", None),
        ("MANUAL 20", "manual"),
        (f"SYNTHETIC {len(synthetic)}", "synthetic"),
    ):
        n = sum(1 for k in kinds if kind in (None, k))
        print(f"\n== {label}  (95% bootstrap intervals; n={n})")
        print(f"{'retriever':<18}{'hit@1':>20}{'hit@5':>20}{'MRR':>22}")
        for name, res in results.items():
            s = subset(res, kind)
            print(
                f"{name:<18}{fmt_ci(bootstrap_ci(s['hit1'])):>20}{fmt_ci(bootstrap_ci(s['hit5'])):>20}"
                f"{fmt_ci(bootstrap_ci(s['mrr']), pct=False):>22}"
            )

    print(
        "\n== PAIRED comparisons on MRR, all queries (diff = first minus second; wins/losses/ties by query)"
    )
    for a, b in (
        ("hybrid (a=.5)", "BM25"),
        ("hybrid (a=.5)", "vector"),
        ("hybrid + rerank", "hybrid (a=.5)"),
        ("vector", "BM25"),
    ):
        r = paired_bootstrap(results[a].mrr, results[b].mrr)
        verdict = "significant" if r["p"] < 0.05 else "NOT significant"
        print(
            f"{a:>16} - {b:<14} diff {r['diff']:+.3f}  95% [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  "
            f"p={r['p']:.3f}  W/L/T {r['wins']}/{r['losses']}/{r['ties']}  -> {verdict}"
        )

    print("\n== Synthetic-question bias check: BM25 MRR on manual vs synthetic queries")
    for kind in ("manual", "synthetic"):
        print(
            f"   {kind:<10} BM25 {np.mean(subset(results['BM25'], kind)['mrr']):.2f}   "
            f"vector {np.mean(subset(results['vector'], kind)['mrr']):.2f}"
        )
    w20 = bootstrap_ci(results["hybrid (a=.5)"].hit5[:20])
    w50 = bootstrap_ci(results["hybrid (a=.5)"].hit5)
    print(
        f"\nInterval width for hybrid hit@5: n=20 -> {w20[2] - w20[1]:.0%} wide, n=50 -> {w50[2] - w50[1]:.0%} wide"
    )


if __name__ == "__main__":
    main()
