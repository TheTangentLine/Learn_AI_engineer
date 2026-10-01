"""Chunkers used by the RAG engine (production versions of the Week 3 Day 3 lesson code).

    from common.chunking import by_headings, recursive
    chunks = by_headings(markdown_text, size=1200)         # -> list[Chunk(text, heading)]

Heading-aware chunking kept answer paragraphs whole 93% of the time in our Day 3 experiment
(naive fixed-size cuts: 40%), and the heading path in each chunk gives the embedder context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    doc: str = ""  # filled in by the caller (source document id)
    heading: str = ""


def fixed(text: str, size: int = 800, overlap: int = 0) -> list[str]:
    """The naive baseline: cut every `size` characters, ignoring structure."""
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def _merge(parts: list[str], sep: str, size: int, overlap: int) -> list[str]:
    """Greedily pack small parts into chunks <= size, carrying `overlap` chars forward."""
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
