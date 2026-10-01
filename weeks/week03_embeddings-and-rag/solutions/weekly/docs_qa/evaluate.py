"""Evaluate the bot against the golden set and render a Markdown report.

Metrics (each isolates one stage, so you know what to fix):
  retrieval hit@k         is a chunk from the right lesson containing the answer phrase in the top-k?
  answered / over-refused answered the question / refused although the answer WAS retrieved
  cited the answer        the cited source contains the answer phrase (citation correctness)
  clean validation        no citation-validation issues after the repair attempt
  groundedness            share of the answer's words found in its cited sources
  abstention (unanswerable)  refused questions the corpus cannot answer
  follow-ups              hit@k when the follow-up is rewritten vs. used raw
"""

from __future__ import annotations

import re
import statistics

import numpy as np

from common.rag import RagBot


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", t)).lower()


def has_answer(hits, lesson: str, phrase: str) -> bool:
    return any(h.metadata["doc"] == lesson and phrase in norm(h.text) for h in hits)


def check_gold(bot: RagBot, golden) -> list[str]:
    """Gold that no chunk contains would silently cap every metric: report it loudly."""
    texts = {(c["doc"], norm(c["text"])) for c in bot.index.chunks}
    return [
        f"{lesson}: {phrase!r}"
        for _, lesson, phrase in golden
        if not any(d == lesson and phrase in t for d, t in texts)
    ]


def evaluate(bot: RagBot, answerable, unanswerable, followups=()) -> tuple[str, dict]:
    bad_gold = check_gold(bot, answerable)
    rows = []
    for q, lesson, phrase in answerable:
        a = bot.ask(q)
        rows.append(
            {
                "q": q,
                "retrieved": has_answer(a.retrieved, lesson, phrase),
                "abstained": a.abstained,
                "cited": has_answer(a.cited, lesson, phrase),
                "clean": not a.issues,
                "support": a.support,
                "seconds": sum(a.seconds.values()),
                "reason": a.reason,
            }
        )
    n = len(rows)
    answered = [r for r in rows if not r["abstained"]]
    over_refused = [r for r in rows if r["retrieved"] and r["abstained"]]
    unans = [bot.ask(q) for q in unanswerable]
    lat = sorted(r["seconds"] for r in rows)

    fu = {"raw follow-up": 0, "rewritten": 0}
    for hist, follow_up, lesson, phrase in followups:
        fu["raw follow-up"] += has_answer(
            bot.index.search(follow_up, bot.k, bot.alpha), lesson, phrase
        )
        standalone = bot._standalone(follow_up, [(hist, "")])
        fu["rewritten"] += has_answer(
            bot.index.search(standalone, bot.k, bot.alpha), lesson, phrase
        )

    m = {
        "n": n,
        "retrieved": sum(r["retrieved"] for r in rows),
        "answered": len(answered),
        "over_refused": len(over_refused),
        "cited_answer": sum(r["cited"] for r in rows),
        "clean": sum(r["clean"] for r in rows),
        "groundedness": float(np.mean([r["support"] for r in answered])) if answered else 0.0,
        "abstained_unanswerable": sum(a.abstained for a in unans),
        "n_unanswerable": len(unans),
        "followups": fu,
        "n_followups": len(followups),
        "p50_s": statistics.median(lat) if lat else 0.0,
        "p95_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0,
        "bad_gold": bad_gold,
        "tau": bot.tau,
        "chunks": len(bot.index.chunks),
        "docs": len(bot.index.manifest),
    }
    out = [
        "# docs_qa evaluation",
        "",
        f"- index: **{m['docs']} lessons, {m['chunks']} chunks** | gate tau = {bot.tau:.3f} | k = {bot.k}",
        f"- retrieval: gold chunk in top-{bot.k}: **{m['retrieved']}/{n}**",
        f"- answered: **{m['answered']}/{n}** | over-refused (answer retrieved but bot refused): "
        f"**{m['over_refused']}**",
        f"- cited the chunk that contains the answer: **{m['cited_answer']}/{n}**",
        f"- clean citation validation: **{m['clean']}/{n}** | groundedness of answers: "
        f"**{m['groundedness']:.2f}**",
        f"- unanswerable questions refused: **{m['abstained_unanswerable']}/{m['n_unanswerable']}**",
        f"- latency p50 {m['p50_s']:.2f}s, p95 {m['p95_s']:.2f}s per question",
    ]
    if followups:
        out.append(
            f"- follow-ups ({len(followups)}): hit@{bot.k} raw **{fu['raw follow-up']}**, "
            f"rewritten **{fu['rewritten']}**"
        )
    if bad_gold:
        out += ["", "## WARNING: gold phrases not found in any chunk", ""] + [
            f"- {g}" for g in bad_gold
        ]
    out += [
        "",
        "## Per-question",
        "",
        "| retrieved | answered | cited | question |",
        "|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {'yes' if r['retrieved'] else 'no'} | {'no (' + r['reason'][:22] + ')' if r['abstained'] else 'yes'} "
            f"| {'yes' if r['cited'] else 'no'} | {r['q'][:70]} |"
        )
    leaked = [a.question for a in unans if not a.abstained]
    if leaked:
        out += ["", "## Unanswerable questions the bot ANSWERED (hallucination risk)", ""]
        out += [f"- {q}" for q in leaked]
    return "\n".join(out), m
