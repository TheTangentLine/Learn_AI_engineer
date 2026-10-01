"""Week 3 Day 4 - Solution: RAG with citations, a retrieval gate, and citation validation.

Pipeline:   question -> embed -> retrieve top-k chunks -> GATE (score < tau? abstain, no LLM call)
            -> prompt with numbered <source> blocks -> structured answer {answerable, answer, citations}
            -> VALIDATE citations -> typed Answer

  uv run python weeks/week03_embeddings-and-rag/solutions/day4_solution.py --offline   # extractive fake LLM
  uv run python weeks/week03_embeddings-and-rag/solutions/day4_solution.py             # your provider

Retrieval here is REAL (bge embeddings + a vector store over the Week 1-2 lessons). With --offline
the *generator* is a scripted extractive stand-in that reads only the sources in the prompt, so the
plumbing (prompt assembly, gate, citation checks, abstention) is genuinely exercised; answer quality
is yours to judge with a real model.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from day3_solution import GOLD, by_headings, norm  # noqa: E402

from common import llm  # noqa: E402
from common.corpus import load_course_docs  # noqa: E402
from common.embed import get_embedder  # noqa: E402
from common.fake import fake_llm  # noqa: E402
from common.vectorstores import Hit, make_store  # noqa: E402

PINNED = ("week01", "week02")
IDK = "I don't know based on the provided sources."

UNANSWERABLE = [
    "what is the capital of Australia",
    "how do I deploy a service to Kubernetes",
    "who won the 2018 football world cup",
    "what is the recipe for sourdough bread",
    "how many moons does Jupiter have",
    "how do I configure a router for port forwarding",
]

# ----------------------------------------------------------------- index + retrieval


@dataclass
class Index:
    emb: object
    store: object
    n_chunks: int


def build_index(store_name: str = "numpy") -> Index:
    emb = get_embedder()
    docs = [d for d in load_course_docs() if d.short.startswith(PINNED)]
    texts, metas = [], []
    for d in docs:
        for c in by_headings(d.text, 1200):
            texts.append(c.text)
            metas.append({"doc": d.short, "heading": c.heading or d.title})
    store = make_store(store_name, emb.dim)
    store.add([f"c{i}" for i in range(len(texts))], emb.embed_documents(texts), metas, texts)
    return Index(emb, store, len(texts))


def retrieve(index: Index, query: str, k: int = 4) -> list[Hit]:
    return index.store.search(index.emb.embed_query(query), k)


def calibrate_threshold(index: Index, answerable: list[str], unanswerable: list[str]) -> dict:
    """Pick the top-1 score cut-off that best separates answerable from unanswerable questions."""
    pos = np.array([retrieve(index, q, 1)[0].score for q in answerable])
    neg = np.array([retrieve(index, q, 1)[0].score for q in unanswerable])
    best = (-1.0, 0.0)
    for tau in np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 200):
        balanced = ((pos >= tau).mean() + (neg < tau).mean()) / 2  # TPR and TNR averaged
        if balanced > best[0]:
            best = (balanced, float(tau))
    return {"tau": best[1], "balanced_acc": best[0], "pos": pos, "neg": neg}


# ----------------------------------------------------------------- generation + validation


class Grounded(BaseModel):
    answerable: bool = Field(description="True only if the sources contain the answer")
    answer: str = Field(
        description="Answer using ONLY the sources, with a citation like [1] after each claim. "
        f"If the sources do not contain the answer, write exactly: {IDK}"
    )
    citations: list[int] = Field(description="Ids of the sources actually used")


SYSTEM = (
    "You answer questions using only the numbered sources provided. Cite the source id in square "
    "brackets after each claim, e.g. [2]. If the sources do not contain the answer, set "
    f'answerable to false and answer exactly "{IDK}". Never use outside knowledge.'
)


def _attr(text: str) -> str:
    """Make text safe inside a tag attribute (heading paths contain '>', titles contain quotes)."""
    return text.replace(">", "\u203a").replace("<", "\u2039").replace('"', "'")


def build_prompt(question: str, hits: list[Hit]) -> str:
    # NB: attribute values must not contain '>' (nested heading paths do!). A naive `<source ...>`
    # parser, or a confused model, would end the tag early and lose the source. Found while testing.
    blocks = []
    for i, h in enumerate(hits, 1):
        ref = _attr(f"{h.metadata['doc']} \u203a {h.metadata['heading']}")
        blocks.append(f'<source id="{i}" ref="{ref}">\n{h.text}\n</source>')
    body = "\n".join(blocks)
    return f"<sources>\n{body}\n</sources>\n\n<question>{question}</question>"


def validate_grounded(g: Grounded, n_sources: int) -> list[str]:
    """Mechanical checks a model cannot be trusted to do on itself."""
    issues = []
    markers = {int(m) for m in re.findall(r"\[(\d+)\]", g.answer)}
    if not g.answer.strip():
        issues.append("empty answer")
    if g.answerable:
        if not g.citations:
            issues.append("answerable but no citations")
        if not markers:
            issues.append("answer has no inline [n] markers")
        if IDK.lower() in g.answer.lower():
            issues.append("claims answerable but says it doesn't know")
    elif g.citations:
        issues.append("not answerable but lists citations")
    bad = (markers | set(g.citations)) - set(range(1, n_sources + 1))
    if bad:
        issues.append(f"cites sources that were not provided: {sorted(bad)}")
    if g.answerable and markers and set(g.citations) != markers:
        issues.append("citations list disagrees with inline markers")
    return issues


_STOP = {
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
    "you",
    "are",
    "be",
    "or",
    "as",
    "with",
    "by",
    "at",
    "from",
    "not",
    "can",
    "your",
    "they",
    "which",
    "will",
}


def support_score(answer: str, cited: list[Hit]) -> float:
    """Cheap groundedness proxy: share of the answer's content words that occur in the cited sources.
    ~1.0 for quotes, lower for paraphrase, near 0 for invention. (LLM-judge faithfulness: Week 4.)"""
    words = [
        w
        for w in re.findall(r"[a-z0-9]+", re.sub(r"\[\d+\]", " ", answer.lower()))
        if w not in _STOP
    ]
    if not words or not cited:
        return 0.0
    source = set(re.findall(r"[a-z0-9]+", " ".join(h.text for h in cited).lower()))
    return sum(w in source for w in words) / len(words)


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool
    reason: str = ""
    cited: list[Hit] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def support(self) -> float:
        return support_score(self.text, self.cited)


def ask(index: Index, question: str, k: int = 4, tau: float = 0.0) -> Answer:
    hits = retrieve(index, question, k)
    if (
        not hits or hits[0].score < tau
    ):  # GATE: don't pay for (or trust) an LLM call on thin evidence
        score = hits[0].score if hits else 0
        return Answer(question, IDK, True, f"gate: top score {score:.2f} < tau {tau:.2f}", [], hits)
    g, _ = llm.structured(build_prompt(question, hits), Grounded, system=SYSTEM, max_tokens=800)
    issues = validate_grounded(g, len(hits))
    if issues:  # one repair attempt: tell the model exactly what was wrong
        fix = (
            build_prompt(question, hits)
            + "\n\nYour previous answer was invalid: "
            + "; ".join(issues)
        )
        g, _ = llm.structured(fix, Grounded, system=SYSTEM, max_tokens=800)
        issues = validate_grounded(g, len(hits))
    cited = [hits[i - 1] for i in g.citations if 1 <= i <= len(hits)]
    abstained = not g.answerable
    return Answer(
        question,
        g.answer,
        abstained,
        "model said unanswerable" if abstained else "",
        cited,
        hits,
        issues,
    )


# ----------------------------------------------------------------- offline extractive generator


READER_THRESHOLD = (
    0.60  # min question-sentence cosine for the stand-in reader to commit to an answer
)


def extractive_model(prompt: str) -> str:
    """Stand-in 'LLM': a semantic extractive reader. It sees ONLY the question and the numbered
    sources in the prompt, embeds the sources' sentences, and answers with the best-matching one
    (or says it can't). A real model reads and synthesises; this only checks the plumbing."""
    emb = get_embedder()
    question = re.search(r"<question>(.*?)</question>", prompt, re.S).group(1)
    cands = []
    for sid, body in re.findall(r'<source id="(\d+)"[^>]*>\n(.*?)\n</source>', prompt, re.S):
        for sent in re.split(r"(?<=[.!?])\s+|\n", body):
            sent = re.sub(r"[*`#|]", "", sent).strip()
            # real sentences only: skip titles, bullets and table fragments
            if (
                50 <= len(sent) <= 400
                and sent[-1] in ".!?"
                and not sent.startswith(("<", "[", "-", "Week"))
            ):
                cands.append((int(sid), sent))
    if not cands:
        return json.dumps({"answerable": False, "answer": IDK, "citations": []})
    sims = emb.embed_documents([c[1] for c in cands]) @ emb.embed_query(question)
    best = int(np.argmax(sims))
    if sims[best] < READER_THRESHOLD:
        return json.dumps({"answerable": False, "answer": IDK, "citations": []})
    sid, sent = cands[best]
    return json.dumps({"answerable": True, "answer": f"{sent} [{sid}]", "citations": [sid]})


def sloppy_model(prompt: str) -> str:
    """A flawed model for testing the validator: cites a source that doesn't exist."""
    return json.dumps({"answerable": True, "answer": "It depends. [9]", "citations": [9]})


# ----------------------------------------------------------------- evaluation


def evaluate(index: Index, tau: float, k: int = 4) -> None:
    retr = cited_ok = valid = 0
    answers = []
    print(f"\n{'':2}{'question':<62} {'result':<10} gold chunk retrieved?")
    for query, lesson, phrase in GOLD:
        a = ask(index, query, k, tau)
        answers.append(a)
        has_gold = lambda hits, lesson=lesson, phrase=phrase: any(  # noqa: E731
            h.metadata["doc"] == lesson and phrase in norm(h.text) for h in hits
        )
        retr += has_gold(a.retrieved)
        cited_ok += has_gold(a.cited)
        valid += not a.issues
        status = "answered" if not a.abstained else "ABSTAINED"
        print(
            f"{'ok' if has_gold(a.cited) else '--':2}{query[:60]:<62} {status:<10} {has_gold(a.retrieved)}"
        )
    n = len(GOLD)
    answered = [a for a in answers if not a.abstained]
    support = float(np.mean([a.support for a in answered])) if answered else 0.0
    print(
        f"\nanswerable questions ({n}): gold chunk in top-{k}: {retr} | answered: {len(answered)} | "
        f"cited the chunk with the answer: {cited_ok} | clean validation: {valid} | "
        f"mean groundedness of answers: {support:.2f}"
    )
    print(
        "   (the answer rate is capped by retrieval: if the gold chunk isn't retrieved, a faithful model\n"
        "    should abstain. Offline, a toy reader picks ONE sentence by similarity, so 'cited the chunk\n"
        "    with the answer' reflects that toy, not RAG quality: judge it with a real model.)"
    )
    abst = [ask(index, q, k, tau) for q in UNANSWERABLE]
    print(f"unanswerable questions: abstained {sum(a.abstained for a in abst)}/{len(abst)}")
    for a in abst:
        if not a.abstained:
            print(f"   !! answered an unanswerable question: {a.question!r} -> {a.text[:70]!r}")


def main() -> None:
    offline = "--offline" in sys.argv
    index = build_index()
    print(f"index: {index.n_chunks} chunks from {PINNED}; embedder {index.emb.name}\n")
    answerable = [q for q, _, _ in GOLD]
    cal = calibrate_threshold(index, answerable, UNANSWERABLE)
    print(
        f"top-1 cosine, answerable   : min {cal['pos'].min():.2f}  mean {cal['pos'].mean():.2f}  max {cal['pos'].max():.2f}"
    )
    print(
        f"top-1 cosine, unanswerable : min {cal['neg'].min():.2f}  mean {cal['neg'].mean():.2f}  max {cal['neg'].max():.2f}"
    )
    print(
        f"best gate tau = {cal['tau']:.3f} (balanced accuracy {cal['balanced_acc']:.0%}); based on only "
        f"{len(cal['pos']) + len(cal['neg'])} questions, so the margin is thin: re-calibrate on real traffic\n"
    )

    ctx = fake_llm([(r"(?s).*", extractive_model)]) if offline else None
    if ctx:
        ctx.__enter__()
        print("*** OFFLINE: extractive stand-in generator (reads only the numbered sources) ***")
    try:
        for q in (GOLD[0][0], GOLD[4][0], UNANSWERABLE[0]):
            a = ask(index, q, 4, cal["tau"])
            print(
                f"\nQ: {q}\nA: {a.text}  [{'abstained: ' + a.reason if a.abstained else 'answered'}]"
            )
            for h in a.cited:
                print(
                    f"   cited: {h.metadata['doc']} > {h.metadata['heading']} (score {h.score:.2f})"
                )
        evaluate(index, cal["tau"])
        if offline:
            print("\n--- validator demo: a model that cites a source that was never provided ---")
            with fake_llm([(r"(?s).*", sloppy_model)]):
                a = ask(index, GOLD[0][0], 4, 0.0)
            print(f"issues after one repair attempt: {a.issues}")
            assert any("not provided" in i for i in a.issues)
            print("\nself-test passed")
    finally:
        if ctx:
            ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
