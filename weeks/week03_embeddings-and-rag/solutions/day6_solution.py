"""Week 3 Day 6 - Solution: query transformation (multi-query, HyDE, decomposition, follow-up rewriting).

All four use an LLM to improve the QUERY before retrieval. Here the LLM is a small local model by
default (honest: weak models are weak rewriters), or your hosted model with --api.

  uv run python weeks/week03_embeddings-and-rag/solutions/day6_solution.py                 # local Qwen
  uv run python weeks/week03_embeddings-and-rag/solutions/day6_solution.py --model Qwen/Qwen2.5-1.5B-Instruct
  uv run python weeks/week03_embeddings-and-rag/solutions/day6_solution.py --api           # your provider

Retrieval baseline = the Day 5 weighted hybrid (BM25 + vectors, alpha 0.5). No re-ranker, to keep
the comparison about the QUERY, not the ranker.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from day3_solution import GOLD, norm  # noqa: E402
from day5_solution import (  # noqa: E402
    LEXICAL_GOLD,
    Corpus,
    build_corpus,
    gold_rank,
    rrf,
    summarise,
    tokenize,
)

from common import llm  # noqa: E402
from common.embed import get_embedder, normalize  # noqa: E402
from common.fake import fake_llm  # noqa: E402
from common.local_llm import LocalChat  # noqa: E402

# ----------------------------------------------------------------- the transformations (LLM calls)

MULTI_SYSTEM = "You write search queries for a technical documentation search engine."
HYDE_SYSTEM = "You are a technical writer who writes concise software documentation."


def clean_lines(text: str, limit: int) -> list[str]:
    """Models add numbering, bullets and quotes even when told not to; normalise defensively."""
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:(?:[-*•]|\d+[.)])\s*)+", "", line).strip().strip('"').strip()
        if len(line) > 3 and line.lower() not in {x.lower() for x in out}:
            out.append(line)
    return out[:limit]


def multi_query(q: str, n: int = 3) -> list[str]:
    prompt = (
        f"Write {n} different search queries that would find documentation answering the question "
        "below. Keep the key technical terms. One query per line, no numbering, no explanations.\n\n"
        f"Question: {q}"
    )
    variants = clean_lines(llm.complete(prompt, system=MULTI_SYSTEM, max_tokens=120).text, n)
    return [q] + [v for v in variants if v.lower() != q.lower()]


def hyde(q: str) -> str:
    """Hypothetical Document Embeddings: embed an invented ANSWER, which looks like a document."""
    prompt = (
        "Write a short passage (2-3 sentences) from software documentation that answers this question. "
        f"State the technique by name. Do not mention the question.\n\nQuestion: {q}"
    )
    return llm.complete(prompt, system=HYDE_SYSTEM, max_tokens=120).text.strip()


def decompose(q: str, limit: int = 3) -> list[str]:
    prompt = (
        "Split the question into independent sub-questions, one per line, no numbering. If it is already "
        "a single question, repeat it unchanged.\n\nExample\nQuestion: What is a KV cache and how do I "
        "retry failed calls?\nWhat is a KV cache?\nHow do I retry failed calls?\n\n"
        f"Question: {q}"
    )
    subs = clean_lines(
        llm.complete(prompt, system="You split questions.", max_tokens=120).text, limit
    )
    return subs if len(subs) > 1 else [q]


def rewrite_followup(history: str, follow_up: str) -> str:
    prompt = (
        f"Conversation so far:\nUser: {history}\n\nFollow-up: {follow_up}\n\n"
        "Rewrite the follow-up as a complete standalone question that includes the topic from the "
        "conversation. Output only the question."
    )
    out = llm.complete(
        prompt, system="You rewrite follow-up questions.", max_tokens=80
    ).text.strip()
    return out.splitlines()[0].strip('" ') if out else follow_up


# ----------------------------------------------------------------- retrieval variants


def hybrid_rank(
    c: Corpus, emb, q: str, dense: np.ndarray | None = None, alpha: float = 0.5
) -> list[int]:
    """Weighted BM25 + vector fusion; `dense` overrides the query vector (used by HyDE)."""
    n01 = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)  # noqa: E731
    dv = emb.embed_query(q) if dense is None else dense
    score = alpha * n01(c.vecs @ dv) + (1 - alpha) * n01(c.bm25.get_scores(tokenize(q)))
    return list(np.argsort(-score, kind="stable"))


def multi_query_rank(c, emb, q: str) -> list[int]:
    return rrf([hybrid_rank(c, emb, v) for v in multi_query(q)])


def hyde_rank(c, emb, q: str) -> list[int]:
    # average the query embedding with the embedding of the hypothetical passage (document side)
    dense = normalize((emb.embed_query(q) + emb.embed_documents([hyde(q)])[0])[None, :])[0]
    return hybrid_rank(c, emb, q, dense=dense)


def looks_like_identifier(q: str) -> bool:
    """Exact-term queries (identifiers, flags, short code-ish strings) should NOT be rewritten."""
    return bool(re.search(r"[_.\-\\]|[a-z][A-Z]|\d", q)) and len(q.split()) <= 4


def routed_multi_query_rank(c, emb, q: str) -> list[int]:
    return hybrid_rank(c, emb, q) if looks_like_identifier(q) else multi_query_rank(c, emb, q)


# ----------------------------------------------------------------- extra evaluation sets

COMPOUND = [  # one question that needs TWO lessons: [(lesson, phrase), (lesson, phrase)]
    (
        "How can I stop concurrent requests from stampeding the API, and also reuse the prompt prefix to "
        "save money?",
        [
            ("week01/day5", "unbounded gather = stampede"),
            ("week01/day4", "stable first, volatile last"),
        ],
    ),
    (
        "What makes models bad at counting letters, and how does temperature change randomness?",
        [
            ("week01/day2", "counting letters is guesswork"),
            ("week01/day3", 'temperature isn\'t "creativity", it\'s "risk"'),
        ],
    ),
    (
        "How do I get valid JSON from a model and how do I keep a chat within a token budget?",
        [
            ("week02/day3", "constrains decoding"),
            ("week02/day5", "enforce the budget at assembly time"),
        ],
    ),
    (
        "Why does tuning a prompt on a small dev set mislead me, and how do I pick the cheapest model that is "
        "good enough?",
        [
            ("week02/day6", "selection bias"),
            ("week01/day6", "pick the cheapest model that clears it"),
        ],
    ),
    (
        "How can voting across samples tell me when to trust an answer, and how do I route tickets to "
        "specialised prompts?",
        [
            ("week02/day2", "agreement is a confidence signal"),
            ("week02/day4", "specialised prompts are shorter and sharper"),
        ],
    ),
]
FOLLOWUPS = [  # (earlier user question, ambiguous follow-up, (lesson, phrase))
    (
        "How do I retry failed API calls?",
        "And how do I limit how many run at once?",
        ("week01/day5", "unbounded gather = stampede"),
    ),
    (
        "Why can't models count letters in a word?",
        "Does the same thing explain mistakes with decimal numbers?",
        ("week01/day2", "numbers are chopped arbitrarily"),
    ),
    (
        "How do I get structured output from a model?",
        "What if its JSON fails my schema?",
        ("week02/day3", "cap attempts"),
    ),
    (
        "What is prompt caching?",
        "What kinds of things break it?",
        ("week01/day4", "silent invalidator"),
    ),
    (
        "How does self-consistency work?",
        "When does it not help?",
        ("week02/day2", "when it does not help"),
    ),
]


def has(c: Corpus, idx: list[int], lesson: str, phrase: str) -> bool:
    return any(c.docs[i] == lesson and phrase in norm(c.texts[i]) for i in idx)


def eval_queries(c, emb, name: str, fn, sets: dict) -> None:
    row = f"{name:<26}"
    every = []
    for gold in sets.values():
        ranks = [gold_rank(c, fn(q), le, ph) for q, le, ph in gold]
        s = summarise(ranks)
        row += f"| {s['hit@1']:>5.0%}  {s['hit@5']:>5.0%}  {s['mrr']:>5.2f}   "
        every += ranks
    print(row + f"| {summarise(every)['mrr']:.2f}")


def main() -> None:
    api = "--api" in sys.argv
    model = (
        sys.argv[sys.argv.index("--model") + 1]
        if "--model" in sys.argv
        else "Qwen/Qwen2.5-0.5B-Instruct"
    )
    emb = get_embedder()
    c = build_corpus(emb)
    sets = {"paraphrased": GOLD, "exact-term": LEXICAL_GOLD}
    for pairs in [p for _, p in COMPOUND] + [[f[2]] for f in FOLLOWUPS]:
        for lesson, phrase in pairs:  # invalid gold would silently invalidate the experiment
            assert has(c, range(len(c.texts)), lesson, phrase), (lesson, phrase)

    ctx = None
    if not api:
        chat = LocalChat(model)
        ctx = fake_llm([(r"(?s).*", chat.as_responder(120))])
        ctx.__enter__()
        print(f"LLM for query rewriting: LOCAL {model} (weak: expect noisy or harmful rewrites)\n")
    else:
        print(f"LLM for query rewriting: {llm.resolve()}\n")
    try:
        q = GOLD[0][0]
        print(f"Example, query: {q!r}")
        print("  multi-query :", multi_query(q)[1:])
        print("  HyDE passage:", hyde(q)[:200].replace("\n", " "), "...\n")

        print(f"{'method':<26}" + "".join(f"| {s:^22}" for s in sets) + "| all 29")
        print(f"{'':<26}" + "| hit@1  hit@5   MRR     " * len(sets) + "| MRR")
        eval_queries(c, emb, "baseline (hybrid)", lambda q: hybrid_rank(c, emb, q), sets)
        eval_queries(c, emb, "multi-query (3) + RRF", lambda q: multi_query_rank(c, emb, q), sets)
        eval_queries(c, emb, "HyDE", lambda q: hyde_rank(c, emb, q), sets)
        eval_queries(
            c, emb, "multi-query, routed", lambda q: routed_multi_query_rank(c, emb, q), sets
        )
        routed = sum(looks_like_identifier(q) for g in sets.values() for q, _, _ in g)
        print(f"(router sent {routed} identifier-like queries straight to the baseline)\n")

        print("COMPOUND questions (need two lessons): coverage = both answers in the top-6")
        base_cov = dec_cov = 0
        for q, pairs in COMPOUND:
            direct = [int(i) for i in hybrid_rank(c, emb, q)[:6]]
            subs = decompose(q)
            lists = [[int(i) for i in hybrid_rank(c, emb, s)[:3]] for s in subs]
            merged = (
                list(dict.fromkeys(i for grp in zip(*lists, strict=False) for i in grp))[:6]
                if len(lists) > 1
                else direct
            )
            b = all(has(c, direct, *p) for p in pairs)
            d = all(has(c, merged, *p) for p in pairs)
            base_cov += b
            dec_cov += d
            print(
                f"  direct={'both' if b else 'miss'}  decomposed({len(subs)} sub-q)={'both' if d else 'miss'}  {q[:60]!r}"
            )
        print(
            f"  coverage: direct {base_cov}/{len(COMPOUND)} | decomposed {dec_cov}/{len(COMPOUND)}\n"
        )

        print("FOLLOW-UP questions (ambiguous without the conversation): hit@5")
        res = {"follow-up alone": 0, "history + follow-up (concat)": 0, "LLM rewrite": 0}
        for hist, fu, (lesson, phrase) in FOLLOWUPS:
            rewritten = rewrite_followup(hist, fu)
            for name, query in (
                ("follow-up alone", fu),
                ("history + follow-up (concat)", f"{hist} {fu}"),
                ("LLM rewrite", rewritten),
            ):
                res[name] += has(c, hybrid_rank(c, emb, query)[:5], lesson, phrase)
            print(f"  {fu!r:<62} -> rewritten: {rewritten[:70]!r}")
        for name, hits in res.items():
            print(f"  {name:<30} {hits}/{len(FOLLOWUPS)}")
    finally:
        if ctx:
            ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
