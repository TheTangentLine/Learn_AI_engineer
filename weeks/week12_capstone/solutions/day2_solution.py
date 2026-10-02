"""Week 12 Day 2 - Solution: the data and retrieval layer, measured.

1. INGEST     the corpus becomes chunks with metadata; indexing is incremental (a changed document re-embeds only itself)
2. RETRIEVE   BM25 against dense against hybrid, with and without week scoping and re-ranking, on the golden questions (hit@k, MRR, multi-lesson coverage, paired comparisons)
3. GATE       how well three signals (cosine, BM25 score, cross-encoder score) separate answerable from out-of-scope questions; the threshold is chosen on dev only
4. LATENCY    what each retrieval stage costs on this machine

  uv run python weeks/week12_capstone/solutions/day2_solution.py
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time

from copilot import evaluate as E
from copilot import gate as GT
from copilot import ingest as I
from lab import ROOT, WEEKS, Lab

from common.rag import _tokenize

VARIANTS = {
    "BM25 only": dict(alpha=0.0, scope_by_week=False),
    "dense only": dict(alpha=1.0, scope_by_week=False),
    "hybrid": dict(alpha=0.5, scope_by_week=False),
    "hybrid + week scope": dict(alpha=0.5, scope_by_week=True),
    "hybrid + scope + rerank": dict(alpha=0.5, scope_by_week=True, rerank=True),
}


def retrieval_metrics(lab: Lab, cfg: dict, items) -> dict:
    r = lab.retriever(**cfg)
    per = {}
    for it in items:
        docs = [s.doc for s in r.retrieve(it.question).sources]
        ranks = [docs.index(c) + 1 for c in it.must_cite if c in docs]
        first = min(ranks) if ranks else None
        per[it.id] = {
            "first": first,
            "all": all(c in docs for c in it.must_cite),
            "multi": it.kind == "multi",
        }
    return per


def main() -> None:
    lab = Lab()
    print(
        f"corpus: {len(lab.docs)} lessons -> {len(lab.index.chunks)} chunks ({lab.index.emb.name}, {lab.index.emb.dim} dims); setup {lab.setup_seconds:.1f} s (warm caches)"
    )
    print(
        "chunks per week:",
        ", ".join(f"{w}:{n}" for w, n in sorted(lab.ingest_report.per_week.items())),
    )

    print("\n1. INCREMENTAL INGESTION (a scratch copy of the index)")
    tmp = tempfile.mkdtemp(prefix="w12-ingest-")
    try:
        shutil.copytree(lab.index.dir, tmp, dirs_exist_ok=True)
        docs = I.load_corpus(WEEKS)
        _, same = I.build_index(lab.embedder, tmp, docs)
        print(f"   nothing changed: {same.sync} in {same.seconds:.2f} s")
        changed = [
            d
            if i
            else type(d)(
                d.id,
                d.text + "\n\nA new paragraph about speculative decoding that was added today.\n",
                d.meta,
            )
            for i, d in enumerate(docs)
        ]
        _, one = I.build_index(lab.embedder, tmp, changed)
        print(f"   one document edited: {one.sync} in {one.seconds:.2f} s")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    items = [i for i in lab.items if i.answerable]
    print(
        f"\n2. RETRIEVAL on {len(items)} answerable golden questions ({sum(i.kind == 'multi' for i in items)} need two lessons)"
    )
    print(
        f"   {'configuration':<26}{'hit@1':>7}{'hit@3':>7}{'hit@5':>16}{'MRR':>6}{'both lessons (multi)':>24}"
    )
    results = {}
    for name, cfg in VARIANTS.items():
        per = retrieval_metrics(lab, cfg, items)
        results[name] = per
        n = len(per)

        def h(k, per=per):
            return sum(1 for v in per.values() if v["first"] and v["first"] <= k)

        mrr = sum(1 / v["first"] for v in per.values() if v["first"]) / n
        multi = [v for v in per.values() if v["multi"]]
        print(
            f"   {name:<26}{h(1) / n:>7.2f}{h(3) / n:>7.2f}{E.fmt(E.rate(h(5), n)):>16}{mrr:>6.2f}{sum(v['all'] for v in multi):>20}/{len(multi)}"
        )
    base, best = results["hybrid"], results["hybrid + week scope"]
    for name in ("BM25 only", "dense only", "hybrid + scope + rerank"):
        other = results[name]
        wins = sum(bool(best[i]["first"]) and not other[i]["first"] for i in best)
        losses = sum(bool(other[i]["first"]) and not best[i]["first"] for i in best)
        print(
            f"   hybrid + scope against {name}: found a lesson in the top 5 on {wins} questions the other missed, missed {losses} that it found"
        )
    miss = [i for i, v in best.items() if not v["first"]]
    print(f"   misses with hybrid + week scope: {miss}")
    _ = base

    print("\n3. THE GATE: three signals for 'can this question be answered from the corpus?'")
    r = lab.retriever()
    cos, bm, ce = ([], []), ([], []), ([], [])
    for it in lab.items:
        if it.kind == "adversarial":
            continue
        ret = r.retrieve(it.question)
        side = 0 if it.answerable else 1
        cos[side].append(ret.top_cosine)
        bm[side].append(float(lab.index._bm25_index().get_scores(_tokenize(it.question)).max()))
        ce[side].append(GT.RerankGate(lab.reranker, 0).score(it.question, ret))
    for name, (pos, neg) in (
        ("cosine of the best source", cos),
        ("BM25 score of the best chunk", bm),
        ("cross-encoder score (top 3)", ce),
    ):
        print(
            f"   {name:<32} AUC {GT.auc(pos, neg):.3f}   answerable median {sorted(pos)[len(pos) // 2]:.2f}, out-of-scope median {sorted(neg)[len(neg) // 2]:.2f}"
        )
    gate, cal = lab.calibrated_gate()
    print(
        f"   threshold chosen on DEV: {cal['threshold']:.2f} (dev recall {cal['recall']:.0%}, dev refusal {cal['refusal']:.0%})"
    )
    pos_t, neg_t = lab.gate_scores(r, "test")
    print(
        f"   applied once to TEST: admits {sum(p >= gate.threshold for p in pos_t)}/{len(pos_t)} answerable, refuses {sum(n < gate.threshold for n in neg_t)}/{len(neg_t)} out-of-scope"
    )

    print(
        "\n4. WHAT EACH STAGE COSTS (this machine, CPU; every question is made unique so no cache can answer it)"
    )
    stamp = f"{time.time():.0f}"
    qs = [
        f"{i.question} (run {stamp}-{n})"
        for n, i in enumerate([i for i in lab.items if i.answerable][:20])
    ]
    fresh = lab.retriever()
    t0 = time.perf_counter()
    rets = [fresh.retrieve(q) for q in qs]
    t_search = (time.perf_counter() - t0) / len(qs)
    t0 = time.perf_counter()
    for q in qs:
        fresh.retrieve(q)
    t_memo = (time.perf_counter() - t0) / len(qs)
    t0 = time.perf_counter()
    for q, ret in zip(qs, rets, strict=True):
        GT.RerankGate(lab.reranker, 0).score(q, ret)
    t_ce = (time.perf_counter() - t0) / len(qs)
    print(
        f"   hybrid search (embed the question, dense + BM25) {t_search * 1000:.0f} ms; the same question again from the memo {t_memo * 1000:.3f} ms; cross-encoder gate over 3 sources {t_ce * 1000:.0f} ms"
    )
    _ = (ROOT, math)


if __name__ == "__main__":
    main()
