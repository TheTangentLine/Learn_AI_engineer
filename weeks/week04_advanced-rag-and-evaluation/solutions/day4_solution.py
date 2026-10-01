"""Week 4 Day 4 - Solution: corrective RAG, measured by how many baseline misses it rescues.

  1. Baseline: hybrid search -> cross-encoder rerank -> top-5.
  2. The reranker's top score is the *grader*: how well does it predict that the top-5 is a hit? (AUC)
  3. Calibrate the confidence threshold tau with LEAVE-ONE-OUT (each question's tau never saw that question).
  4. Corrective search (common/crag.py): when confidence is low, rewrite the query / switch strategy,
     pool candidates, re-rank against the ORIGINAL question.
  5. Report rescue rate, harm rate, paired tests, and the extra cost.

Also: `agentic_search`, an LLM-driven retrieval loop (the model decides to search again), tested with a
scripted model. Real measurement of agent loops comes in Week 5.

  uv run python weeks/week04_advanced-rag-and-evaluation/solutions/day4_solution.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "weeks/week03_embeddings-and-rag/solutions/weekly"))

from docs_qa.app import course_sources  # noqa: E402

from common import llm  # noqa: E402
from common.crag import corrective_search  # noqa: E402
from common.embed import get_embedder  # noqa: E402
from common.evalkit import bootstrap_ci, fmt_ci, load_golden, paired_bootstrap  # noqa: E402
from common.local_llm import LocalChat  # noqa: E402
from common.rag import RagIndex  # noqa: E402
from common.rerank import get_reranker  # noqa: E402

PINNED_WEEKS = (1, 2, 3)
GOLDEN = ROOT / "outputs" / "golden_week4.jsonl"
K = 5


# ----------------------------------------------------------------- calibration


def auc(pos: list[float], neg: list[float]) -> float:
    """P(random positive scores higher than random negative): 0.5 = useless, 1.0 = perfect grader."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_tau(scores: list[float], hit: list[bool]) -> float:
    """Cut-off on the top score maximising balanced accuracy at predicting 'the top-k contains a hit'."""
    cands = sorted(set(scores))
    cuts = [(a + b) / 2 for a, b in zip(cands, cands[1:], strict=False)] or [cands[0] - 1e-9]

    def balanced(t: float) -> float:
        tpr = np.mean([s >= t for s, h in zip(scores, hit, strict=True) if h]) if any(hit) else 0.0
        tnr = (
            np.mean([s < t for s, h in zip(scores, hit, strict=True) if not h])
            if not all(hit)
            else 0.0
        )
        return (tpr + tnr) / 2

    return max(cuts, key=balanced)


def loo_taus(scores: list[float], hit: list[bool]) -> list[float]:
    """For each question, the threshold learned from all the OTHER questions."""
    return [
        best_tau(scores[:i] + scores[i + 1 :], hit[:i] + hit[i + 1 :]) for i in range(len(scores))
    ]


# ----------------------------------------------------------------- rewriters and an LLM-driven loop


def clean_lines(text: str, limit: int) -> list[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:(?:[-*•]|\d+[.)])\s*)+", "", line).strip().strip('"')
        if len(line) > 8 and line.lower() not in {x.lower() for x in out}:
            out.append(line)
    return out[:limit]


def make_rewriter(chat: LocalChat, n: int = 2):
    def rewrite(q: str) -> list[str]:
        raw = chat(
            "You write search queries for a technical documentation search engine.",
            f"Write {n} different search queries that would find documentation answering the question "
            f"below. Keep the key technical terms. One per line, no numbering.\n\nQuestion: {q}",
            80,
        )
        return clean_lines(raw, n)

    return rewrite


AGENT_SYSTEM = (
    "You answer questions about documentation by searching it. Reply with ONE JSON object per turn: "
    '{"action": "search", "query": "..."} to search, or {"action": "answer", "text": "..."} when the '
    "evidence you have is enough (or after you have searched enough). Never invent facts."
)


def agentic_search(search_fn, question: str, max_steps: int = 3) -> dict:
    """An LLM-driven loop: the MODEL decides whether to search again. Returns the trace and final text.

    `search_fn(query) -> list[str]` returns result snippets. Guards: bounded steps, malformed JSON is
    surfaced as an error step (not silently ignored), and the model is forced to answer at the last step.
    """
    transcript = f"Question: {question}"
    trace = []
    for step in range(1, max_steps + 1):
        last = step == max_steps
        prompt = transcript + ("\n\nThis was your last search: answer now." if last else "")
        raw = llm.complete(prompt, system=AGENT_SYSTEM, max_tokens=300).text
        try:
            action = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        except (AttributeError, ValueError):
            trace.append({"step": step, "error": "unparseable action", "raw": raw[:80]})
            transcript += "\n\nYour last reply was not valid JSON. Reply with one JSON object."
            continue
        if (
            last and action.get("action") != "answer"
        ):  # the budget is spent: no more searching allowed
            trace.append({"step": step, "error": "did not answer on the final step"})
            break
        if action.get("action") == "answer":
            trace.append({"step": step, "action": "answer"})
            return {
                "answer": action.get("text", ""),
                "trace": trace,
                "searches": sum("query" in t for t in trace),
            }
        query = str(action.get("query", question))
        results = search_fn(query)
        trace.append({"step": step, "action": "search", "query": query, "n_results": len(results)})
        transcript += f"\n\nSearch: {query}\nResults:\n" + "\n".join(
            f"- {r[:200]}" for r in results
        )
    return {"answer": "", "trace": trace, "searches": sum("query" in t for t in trace)}


# ----------------------------------------------------------------- evaluation


def main() -> None:
    emb = get_embedder()
    index = RagIndex(emb, ROOT / "outputs" / "week4_index")
    index.sync([d for d in course_sources() if d.meta["week"] in PINNED_WEEKS])
    golden = load_golden(GOLDEN)
    rr = get_reranker()
    chat = LocalChat()
    rewriter = make_rewriter(chat)

    def relevant(g, hits) -> list[bool]:
        return [g.is_relevant(h.metadata["doc"], h.text) for h in hits]

    # baseline: hybrid -> rerank; record the top score (the grader's signal)
    base_hits, top_scores = [], []
    for g in golden:
        hits = index.search(g.query, 20, alpha=0.5)
        sc = rr.scores(g.query, [h.text for h in hits])
        order = sorted(range(len(hits)), key=lambda i: -sc[i])
        base_hits.append([hits[i] for i in order])
        top_scores.append(sc[order[0]])
    hit5 = [any(relevant(g, h[:K])) for g, h in zip(golden, base_hits, strict=True)]
    hit20 = [any(relevant(g, h)) for g, h in zip(golden, base_hits, strict=True)]
    rr_base = [
        next((1 / r for r, ok in enumerate(relevant(g, h), 1) if ok), 0.0)
        for g, h in zip(golden, base_hits, strict=True)
    ]
    print(
        f"{len(golden)} questions | baseline (hybrid + rerank): hit@5 {fmt_ci(bootstrap_ci([float(x) for x in hit5]))}, "
        f"MRR {np.mean(rr_base):.2f} | answer present in the 20 candidates for {sum(hit20)}/{len(golden)}"
    )
    first_rank = [
        next((r for r, ok in enumerate(relevant(g, h), 1) if ok), None)
        for g, h in zip(golden, base_hits, strict=True)
    ]
    curve = {k: sum(r is not None and r <= k for r in first_rank) for k in (1, 3, 5, 10, 20)}
    print(
        "where the answer sits after re-ranking (hit@k): "
        + ", ".join(f"@{k}: {v}/{len(golden)}" for k, v in curve.items())
    )
    print(
        "   -> misses at k=5 whose answer is at rank 6-20 are RANKING failures; re-retrieval cannot fix those"
    )
    pos = [s for s, h in zip(top_scores, hit5, strict=True) if h]
    neg = [s for s, h in zip(top_scores, hit5, strict=True) if not h]
    print(
        f"\nGrader: top rerank score when top-5 HAS the answer: mean {np.mean(pos):+.2f} (n={len(pos)}); "
        f"when it MISSES: mean {np.mean(neg):+.2f} (n={len(neg)}); AUC = {auc(pos, neg):.2f}"
    )

    taus = loo_taus(top_scores, hit5)
    print(
        f"leave-one-out tau: median {np.median(taus):+.2f} (range {min(taus):+.2f} .. {max(taus):+.2f})"
    )

    # corrective search with the LOO threshold
    c_hits, c_rr, steps_n, retrievals, pairs, llm_calls = [], [], [], [], [], 0
    traces = []
    for g, tau in zip(golden, taus, strict=True):
        calls_before = chat.generated
        res = corrective_search(
            index, g.query, rr, tau, rewriter=rewriter, k=K, depth=20, max_steps=4
        )
        llm_calls += chat.generated - calls_before
        rel = relevant(g, res.hits)
        c_hits.append(any(rel))
        c_rr.append(next((1 / r for r, ok in enumerate(rel, 1) if ok), 0.0))
        steps_n.append(len(res.steps))
        retrievals.append(res.retrievals)
        pairs.append(res.rerank_pairs)
        traces.append(res)
    corrected = [t.corrected for t in traces]
    rescued = [i for i in range(len(golden)) if not hit5[i] and c_hits[i]]
    harmed = [i for i in range(len(golden)) if hit5[i] and not c_hits[i]]
    misses = sum(not h for h in hit5)
    print(f"\nCRAG triggered a correction on {sum(corrected)}/{len(golden)} questions")
    print(f"{'':<22}{'hit@5':>20}{'MRR':>8}")
    print(
        f"{'baseline':<22}{fmt_ci(bootstrap_ci([float(x) for x in hit5])):>20}{np.mean(rr_base):>8.2f}"
    )
    print(
        f"{'corrective (CRAG)':<22}{fmt_ci(bootstrap_ci([float(x) for x in c_hits])):>20}{np.mean(c_rr):>8.2f}"
    )
    p = paired_bootstrap([float(x) for x in c_hits], [float(x) for x in hit5])
    pm = paired_bootstrap(c_rr, rr_base)
    print(
        f"paired hit@5: {p['diff']:+.2f} [{p['ci_low']:+.2f}, {p['ci_high']:+.2f}] p={p['p']:.2f} | "
        f"paired MRR: {pm['diff']:+.3f} [{pm['ci_low']:+.3f}, {pm['ci_high']:+.3f}] p={pm['p']:.2f}"
    )
    print(
        f"rescued {len(rescued)} of {misses} baseline misses; harmed {len(harmed)} of {sum(hit5)} baseline hits"
    )
    base_pairs = 20 * len(golden)
    print(
        f"cost per question: {np.mean(retrievals):.2f} searches (baseline 1), "
        f"{(base_pairs + sum(max(0, c - 20) for c in pairs)) / len(golden):.1f} rerank pairs (baseline 20), "
        f"{llm_calls / len(golden):.2f} LLM calls (baseline 0); p95 steps {int(np.percentile(steps_n, 95))}"
    )
    for i in rescued[:2]:
        print(f"\nrescued: {golden[i].query!r}")
        for s in traces[i].steps:
            print(
                f"   {s.action:<13} {s.query[:55]!r:<58} pool={s.candidates:<3} top={s.top_score:+.2f} -> {s.grade}"
            )
    out_of_reach = [i for i in range(len(golden)) if not hit20[i]]
    print(
        f"\nUnreachable by any re-ranking of these candidates: {len(out_of_reach)} questions (answer not in the top-20)."
    )


if __name__ == "__main__":
    main()
