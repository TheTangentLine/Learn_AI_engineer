"""Evaluate the deployed ``/v1/ask`` endpoint on a labelled question set: did retrieval find the right lesson, did the model cite what it was given, did it abstain when it should.

Everything that does not need a server is a pure function (tested); ``run_questions`` drives a live API through the same client the chat UI uses.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
UI = HERE.parents[1] / "ui"
sys.path.insert(0, str(UI))

import client as UC  # noqa: E402

from common.abtest import wilson as _wilson  # noqa: E402


def wilson(successes: int, n: int) -> tuple[float, float, float]:
    """(rate, low, high) with the repository's Wilson score interval (``common.abtest``); NaNs when there is nothing to measure."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    lo, hi = _wilson(successes, n)
    return successes / n, lo, hi


def fmt(ci: tuple[float, float, float]) -> str:
    p, lo, hi = ci
    return "n/a" if math.isnan(p) else f"{p:.0%} [{lo:.0%}, {hi:.0%}]"


def basename(doc: str) -> str:
    return doc.rsplit("/", 1)[-1]


def retrieval_hit(sources: list[dict], expected: list[str]) -> bool:
    """True if any retrieved source comes from an expected lesson."""
    want = set(expected)
    return any(basename(s.get("doc", "")) in want for s in sources)


def retrieval_rank(sources: list[dict], expected: list[str]) -> int | None:
    """1-based rank of the first expected source, or None."""
    want = set(expected)
    for i, s in enumerate(sources, 1):
        if basename(s.get("doc", "")) in want:
            return i
    return None


def judge(row: dict, sources: list[dict], answer: str) -> dict:
    """One question's outcome. ``cited_valid``: cites at least one source number that exists and none that do not. ``abstained``: the exact refusal sentence."""
    rep = UC.citation_report(answer, len(sources))
    return {
        "id": row["id"],
        "split": row["split"],
        "in_scope": row["in_scope"],
        "n_sources": len(sources),
        "hit": retrieval_hit(sources, row["expected"]) if row["in_scope"] else None,
        "rank": retrieval_rank(sources, row["expected"]) if row["in_scope"] else None,
        "abstained": rep["abstained"],
        "cited_valid": bool(rep["cited"]) and not rep["invalid"],
        "invalid_citation": bool(rep["invalid"]),
        "error": None,
        "answer": answer,
    }


def summarize(outcomes: list[dict], split: str | None = None) -> dict:
    """Rates with Wilson intervals. In scope: retrieval hit, abstained (a miss), cited validly. Out of scope: nothing retrieved, abstained (the right behaviour), answered anyway."""
    rows = [o for o in outcomes if split is None or o["split"] == split]
    ins, outs = [o for o in rows if o["in_scope"]], [o for o in rows if not o["in_scope"]]
    return {
        "errors": sum(bool(o.get("error")) for o in rows),
        "n_in": len(ins),
        "n_out": len(outs),
        "hit": wilson(sum(bool(o["hit"]) for o in ins), len(ins)),
        "abstained_in": wilson(sum(o["abstained"] for o in ins), len(ins)),
        "cited_in": wilson(sum(o["cited_valid"] for o in ins), len(ins)),
        "answered_cited_in": wilson(
            sum(o["cited_valid"] for o in ins if o["hit"]), sum(bool(o["hit"]) for o in ins)
        ),
        "no_sources_out": wilson(sum(o["n_sources"] == 0 for o in outs), len(outs)),
        "abstained_out": wilson(sum(o["abstained"] for o in outs), len(outs)),
        "answered_out": wilson(sum(not o["abstained"] for o in outs), len(outs)),
    }


def run_questions(
    url: str,
    key: str,
    rows: list[dict],
    *,
    k: int = 4,
    max_tokens: int = 160,
    model: str = "default",
) -> list[dict]:
    """Ask every question through the UI's client (streaming, sources first) and judge the answers."""
    c = UC.ApiClient(url, key)
    out = []
    for row in rows:
        stats, text, sources = UC.AnswerStats(), "", []
        for ev in c.ask(row["question"], stats, k=k, model=model, max_tokens=max_tokens):
            if ev.kind == "sources":
                sources = ev.data
            elif ev.kind == "token":
                text += ev.data
        o = judge(row, sources, text)
        o.update(
            status=stats.status,
            ttft=stats.ttft,
            seconds=stats.seconds,
            tokens=stats.completion_tokens,
            error=(stats.error or {}).get(
                "code"
            ),  # an in-band stream error arrives after a 200 status line
        )
        out.append(o)
    return out


def best_floor(top_scores: list[tuple[float, bool]], *, max_in_scope_loss: float = 0.1) -> float:
    """Choose a relevance floor from (top retrieval score, in_scope) pairs on the DEV split: the highest floor that still keeps at least ``1 - max_in_scope_loss`` of the in-scope
    questions retrieving something. (Out-of-scope questions are then dropped as far as the scores allow.) Returns 0.0 when no floor helps."""
    ins = sorted(s for s, ok in top_scores if ok)
    if not ins:
        return 0.0
    allowed_loss = int(len(ins) * max_in_scope_loss)
    return max(0.0, ins[allowed_loss] - 1e-6) if allowed_loss < len(ins) else 0.0


def retrieval_only(
    search: Callable[[str, int, float], list],
    rows: list[dict],
    *,
    k: int = 4,
    min_score: float = 0.0,
) -> list[dict]:
    """No model, no server: run ``search(question, k, min_score)`` (a ``Bm25Index.search``) and report hits and the top score per question."""
    out = []
    for row in rows:
        hits = search(row["question"], k, min_score)
        sources = [{"doc": c.doc} for c, _ in hits]
        out.append(
            {
                "id": row["id"],
                "split": row["split"],
                "in_scope": row["in_scope"],
                "top_score": hits[0][1] if hits else 0.0,
                "n_sources": len(hits),
                "hit": retrieval_hit(sources, row["expected"]) if row["in_scope"] else None,
                "rank": retrieval_rank(sources, row["expected"]) if row["in_scope"] else None,
            }
        )
    return out
