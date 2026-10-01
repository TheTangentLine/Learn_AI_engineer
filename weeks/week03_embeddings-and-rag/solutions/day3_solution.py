"""Week 3 Day 3 - Solution: chunking strategies, compared by what they do to RETRIEVAL.

Chunkers (all written from scratch):  fixed-size | recursive | heading-aware | semantic
Ground truth is chunker-independent: each query has a lesson and a KEY PHRASE that answers it.
A retrieval counts as a hit if a returned chunk comes from that lesson AND contains the phrase.
We also report 'complete' (the paragraph around the answer was not cut) and a hit rate under a fixed
context budget, because hit@5 alone rewards huge chunks.

  uv run python weeks/week03_embeddings-and-rag/solutions/day3_solution.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.corpus import Doc, load_course_docs  # noqa: E402
from common.embed import cosine_top_k, get_embedder  # noqa: E402

PINNED = ("week01", "week02")


@dataclass
class Chunk:
    text: str
    doc: str  # Doc.short, e.g. week01/day5
    heading: str = ""


# (query, lesson, key phrase). The phrase is the sentence that actually answers the query.
GOLD = [
    (
        "how do I stop my app from hammering an API that keeps failing",
        "week01/day5",
        "100 clients that failed at the same instant",
    ),
    (
        "why can't a language model count the letters in a word",
        "week01/day2",
        "counting letters is guesswork",
    ),
    (
        "how much GPU memory does remembering a long conversation take",
        "week01/day4",
        "kv bytes = 2 (k and v)",
    ),
    (
        "what setting makes the output more random or more deterministic",
        "week01/day3",
        'temperature isn\'t "creativity", it\'s "risk"',
    ),
    (
        "ways to get machine-readable JSON out of a model without parse errors",
        "week02/day3",
        "constrains decoding",
    ),
    (
        "keep a chat from growing too expensive while still remembering the user's name",
        "week02/day5",
        "enforce the budget at assembly time",
    ),
    (
        "asking the same question several times and voting to be more accurate",
        "week02/day2",
        "agreement is a confidence signal",
    ),
    (
        "send each support ticket to a specialised handler",
        "week02/day4",
        "specialised prompts are shorter and sharper",
    ),
    (
        "how to split data so I don't fool myself when tuning a prompt",
        "week02/day6",
        "selection bias",
    ),
    (
        "picking the cheapest model that is still good enough",
        "week01/day6",
        "pick the cheapest model that clears it",
    ),
    (
        "reuse the start of a prompt across requests to save money",
        "week01/day4",
        "stable first, volatile last",
    ),
    ("what should a well written prompt contain", "week02/day1", "examples beat adjectives"),
    (
        "can I trust the probabilities a model assigns to its words",
        "week01/day2",
        "a confidently wrong model scores high too",
    ),
    (
        "running many requests at once without hitting rate limits",
        "week01/day5",
        "unbounded gather = stampede",
    ),
    (
        "why do small changes in the prompt text change token counts and cost",
        "week01/day2",
        "minifying the json you send cut input by 45%",
    ),
]


def norm(t: str) -> str:
    """Match phrases without markdown noise: strip * and `, collapse whitespace, lowercase."""
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", t)).lower()


# ----------------------------------------------------------------- chunkers


def fixed(text: str, size: int = 800, overlap: int = 0) -> list[str]:
    """The naive baseline: cut every `size` characters, ignoring structure."""
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def _merge(parts: list[str], sep: str, size: int, overlap: int) -> list[str]:
    """Greedily pack small parts into chunks <= size, carrying `overlap` chars into the next chunk."""
    chunks, cur = [], []
    for p in parts:
        if cur and len(sep.join([*cur, p])) > size:
            chunks.append(sep.join(cur))
            while cur and len(sep.join(cur)) > overlap:
                cur.pop(0)
        cur.append(p)
    if cur:
        chunks.append(sep.join(cur))
    return chunks


def recursive(
    text: str, size: int = 800, overlap: int = 0, seps=("\n\n", "\n", ". ", " ")
) -> list[str]:
    """Split on the coarsest separator that works (paragraph > line > sentence > word), recursing
    into any piece that is still too big, then re-pack neighbours up to `size`."""
    if len(text) <= size:
        return [text] if text.strip() else []
    for i, sep in enumerate(seps):
        if sep not in text:
            continue
        parts = []
        for piece in text.split(sep):
            parts += recursive(piece, size, 0, seps[i + 1 :]) if len(piece) > size else [piece]
        return [c for c in _merge(parts, sep, size, overlap) if c.strip()]
    return fixed(text, size, overlap)  # one giant token: hard cut


_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.M)


def by_headings(
    text: str, size: int = 1200, min_chars: int = 200, prefix: bool = True
) -> list[Chunk]:
    """Structure-aware: one chunk per markdown section (sub-split if too big, merge if tiny).
    With prefix=True each chunk starts with its heading path, which gives the embedder context."""
    marks = [(m.start(), len(m.group(1)), m.group(2)) for m in _HEADING.finditer(text)]
    sections, path = [], []
    bounds = [*marks, (len(text), 0, "")]
    if not marks:  # a document with no headings is one section (never silently drop it)
        sections.append(("", text))
    elif marks[0][0] > 0:
        sections.append(("", text[: marks[0][0]]))
    for (start, level, title), nxt in zip(marks, bounds[1:], strict=True):
        path = [*path[: level - 1], title]
        sections.append((" > ".join(path), text[start : nxt[0]]))
    out: list[Chunk] = []
    buf_head, buf = "", ""
    for head, body in sections:
        if len(body) > size:
            if buf:
                out.append(Chunk(buf, "", buf_head))
                buf = ""
            out += [Chunk(c, "", head) for c in recursive(body, size, 100)]
            continue
        if buf and len(buf) + len(body) > size:
            out.append(Chunk(buf, "", buf_head))
            buf = ""
        if not buf:
            buf_head = head
        buf += body
        if len(buf) >= min_chars and len(buf) + 200 > size:
            out.append(Chunk(buf, "", buf_head))
            buf = ""
    if buf:
        out.append(Chunk(buf, "", buf_head))
    if prefix:
        for c in out:
            c.text = (f"[{c.heading}]\n" if c.heading else "") + c.text
    return [c for c in out if c.text.strip()]


_BLOCKS = re.compile(r"(```.*?```|\n\s*\n)", re.S)


def semantic(text: str, emb, max_chars: int = 1200, percentile: float = 30.0) -> list[str]:
    """Break where the topic shifts: embed paragraph/code blocks, and start a new chunk when the
    similarity between neighbouring blocks falls below a percentile of this document's own."""
    blocks = [b.strip() for b in _BLOCKS.split(text) if b and b.strip()]
    if len(blocks) < 2:
        return blocks
    vecs = emb.embed_documents(blocks)
    sims = np.array([float(vecs[i] @ vecs[i + 1]) for i in range(len(blocks) - 1)])
    thr = np.percentile(sims, percentile)
    chunks, cur = [], [blocks[0]]
    for i, b in enumerate(blocks[1:]):
        too_big = len("\n\n".join([*cur, b])) > max_chars
        if sims[i] < thr or too_big:
            chunks.append("\n\n".join(cur))
            cur = []
        cur.append(b)
    chunks.append("\n\n".join(cur))
    out = []
    for c in chunks:  # a single huge block must still respect the cap
        out += recursive(c, max_chars, 0) if len(c) > max_chars else [c]
    return out


# ----------------------------------------------------------------- evaluation


def build(docs: list[Doc], chunker) -> list[Chunk]:
    out = []
    for d in docs:
        for c in chunker(d):
            ch = c if isinstance(c, Chunk) else Chunk(c, "")
            out.append(Chunk(ch.text, d.short, ch.heading))
    return out


CONTEXT_BUDGET = 2000  # chars of retrieved text we can afford to show the LLM per question


def answer_paragraph(doc: Doc, phrase: str) -> str:
    """The whole paragraph/code block that contains the key phrase (what a reader needs intact)."""
    for block in _BLOCKS.split(doc.text):
        if phrase in norm(block):
            return norm(block)
    raise ValueError(f"phrase {phrase!r} not in {doc.short}")


def evaluate(chunks: list[Chunk], emb, docs: list[Doc]) -> dict:
    """hit@1 / hit@5 / MRR, plus two size-fair measures:
    complete : the full paragraph around the answer lives inside ONE chunk (not cut in two)
    hit@2k   : the answer chunk is among the top results that fit in a 2,000-char budget.
    (hit@5 alone rewards big chunks: a 3,000-char chunk 'contains' everything but floods the LLM.)"""
    by_short = {d.short: d for d in docs}
    vecs = emb.embed_documents([c.text for c in chunks])
    hit1 = hit5 = hitb = complete = rr = ctx = 0.0
    for query, lesson, phrase in GOLD:
        has = [(c.doc == lesson and phrase in norm(c.text)) for c in chunks]
        para = answer_paragraph(by_short[lesson], phrase)
        complete += any(c.doc == lesson and para in norm(c.text) for c in chunks)
        ranked = [i for i, _ in cosine_top_k(vecs, emb.embed_query(query), 30)]
        first = next((r for r, i in enumerate(ranked, 1) if has[i]), None)
        hit1 += first == 1
        hit5 += first is not None and first <= 5
        rr += 1 / first if first else 0
        used, within = 0, set()
        for i in ranked:  # fill the budget in rank order (the first chunk is always allowed)
            if used and used + len(chunks[i].text) > CONTEXT_BUDGET:
                break
            used += len(chunks[i].text)
            within.add(i)
        hitb += any(has[i] for i in within)
        ctx += sum(len(chunks[i].text) for i in ranked[:5])
    n = len(GOLD)
    sizes = [len(c.text) for c in chunks]
    return {
        "n": len(chunks),
        "mean": float(np.mean(sizes)),
        "complete": complete / n,
        "hit@1": hit1 / n,
        "hit@5": hit5 / n,
        "hit@2k": hitb / n,
        "mrr": rr / n,
        "ctx5": ctx / n,
    }


def row(name: str, r: dict) -> None:
    print(
        f"{name:<30} {r['n']:>5} {r['mean']:>5.0f} {r['complete']:>8.0%} {r['hit@1']:>6.0%} "
        f"{r['hit@5']:>6.0%} {r['hit@2k']:>7.0%} {r['mrr']:>5.2f} {r['ctx5']:>9.0f}"
    )


def parse_pdf_demo() -> None:
    """Parsing matters too: extract text PAGE BY PAGE so every chunk can carry a page number."""
    import tempfile

    from pypdf import PdfReader
    from reportlab.pdfgen import canvas

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo.pdf"
        c = canvas.Canvas(str(path))
        for n in (1, 2, 3):
            c.drawString(72, 750, f"Page {n}: the refund policy window is {n * 10} days.")
            c.showPage()
        c.save()
        pages = [
            (i, (p.extract_text() or "").strip())
            for i, p in enumerate(PdfReader(str(path)).pages, 1)
        ]
    print("PDF parse demo (page-level metadata):")
    for page_no, text in pages:
        print(f"   page {page_no}: {text!r}")


def main() -> None:
    docs = [d for d in load_course_docs() if d.short.startswith(PINNED)]
    emb = get_embedder()
    total = sum(len(d.text) for d in docs)
    print(
        f"corpus: {len(docs)} lessons, {total:,} chars | embedder {emb.name} (max ~512 tokens ~ 2000 chars)"
    )
    print(
        f"{len(GOLD)} queries. complete = answer paragraph not cut | hit@2k = answer found within a "
        f"{CONTEXT_BUDGET}-char context budget | ctx5 = avg chars in the top-5\n"
    )
    print(
        f"{'chunker':<30} {'n':>5} {'mean':>5} {'complete':>8} {'hit@1':>6} {'hit@5':>6} {'hit@2k':>7} "
        f"{'MRR':>5} {'ctx5':>9}"
    )

    print("-- strategy comparison (target ~800 chars) --")
    row("fixed 800, overlap 0", evaluate(build(docs, lambda d: fixed(d.text, 800, 0)), emb, docs))
    row(
        "fixed 800, overlap 150",
        evaluate(build(docs, lambda d: fixed(d.text, 800, 150)), emb, docs),
    )
    row(
        "recursive 800, overlap 100",
        evaluate(build(docs, lambda d: recursive(d.text, 800, 100)), emb, docs),
    )
    row(
        "headings (+ path prefix)",
        evaluate(build(docs, lambda d: by_headings(d.text, 1200)), emb, docs),
    )
    row(
        "headings (no prefix)",
        evaluate(build(docs, lambda d: by_headings(d.text, 1200, prefix=False)), emb, docs),
    )
    row(
        "semantic (p30, max 1200)",
        evaluate(build(docs, lambda d: semantic(d.text, emb, 1200)), emb, docs),
    )

    print("-- chunk size sweep (recursive, overlap = 12%) --")
    for size in (200, 400, 800, 1600, 3200):
        row(
            f"recursive {size}",
            evaluate(build(docs, lambda d, s=size: recursive(d.text, s, s // 8)), emb, docs),
        )
    print(
        "-> past ~2000 chars the embedder silently truncates (512-token limit): the tail of each chunk is invisible.\n"
    )
    parse_pdf_demo()


if __name__ == "__main__":
    main()
