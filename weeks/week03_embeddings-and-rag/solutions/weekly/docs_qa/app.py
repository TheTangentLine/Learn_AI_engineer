"""Wiring: load the course lessons as SourceDocs, build/sync the persistent index, build the bot."""

from __future__ import annotations

import re
from pathlib import Path

from common.corpus import ROOT, load_course_docs
from common.embed import get_embedder
from common.rag import RagBot, RagIndex, SourceDoc

INDEX_DIR = ROOT / "outputs" / "docs_qa_index"


def course_sources() -> list[SourceDoc]:
    out = []
    for d in load_course_docs():
        week = int(re.search(r"week(\d+)", d.path).group(1))
        out.append(SourceDoc(d.short, d.text, {"week": week, "title": d.title, "path": d.path}))
    return out


def build_index(
    embedder=None, directory: Path | None = INDEX_DIR, docs: list[SourceDoc] | None = None
):
    index = RagIndex(embedder or get_embedder(), directory)
    report = index.sync(docs if docs is not None else course_sources())
    return index, report


def build_bot(
    index: RagIndex,
    answerable: list[str] | None = None,
    unanswerable: list[str] | None = None,
    reranker=None,
    k: int = 4,
) -> RagBot:
    bot = RagBot(index, k=k, reranker=reranker)
    if answerable and unanswerable:
        bot.calibrate(answerable, unanswerable)
    return bot
