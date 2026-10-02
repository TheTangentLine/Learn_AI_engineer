"""Ingestion for the capstone product: the course's lessons become ``SourceDoc``s with metadata (week, day, title) and are indexed incrementally.

    docs = load_corpus(ROOT / "weeks", max_week=11)           # lessons only: no READMEs, nothing from Week 12 (the product must not index its own spec)
    index, report = build_index(embedder, "outputs/w12_index", docs)

Why the corpus is exactly this (decisions the Day 1 design document records):
* lessons only (``day*.md``): READMEs repeat the headline numbers and would make the golden set answerable from two places;
* Week 12 is excluded: its documents quote the golden questions, so indexing them would make retrieval look better than it is (circular evidence);
* each document keeps ``week`` and ``day`` so a question that names a week can be scoped to it (a metadata filter, Week 3 Day 2).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from common.rag import RagIndex, SourceDoc, SyncReport  # noqa: E402

LESSON = re.compile(r"^week(\d{2})_.+/day(\d)_.+\.md$")
TITLE = re.compile(r"^#\s+(.+?)\s*$", re.M)


def doc_id(path: Path, weeks_dir: Path) -> str:
    return path.relative_to(weeks_dir).as_posix()


def load_corpus(weeks_dir: Path, max_week: int = 11) -> list[SourceDoc]:
    """Every lesson ``weekNN_*/dayN_*.md`` up to ``max_week``, sorted by path, with its metadata. Files that are not lessons are ignored."""
    docs = []
    for p in sorted(Path(weeks_dir).glob("week*/day*.md")):
        rel = doc_id(p, Path(weeks_dir))
        m = LESSON.match(rel)
        if not m or int(m.group(1)) > max_week:
            continue
        text = p.read_text()
        title = TITLE.search(text)
        docs.append(
            SourceDoc(
                rel,
                text,
                {
                    "week": int(m.group(1)),
                    "day": int(m.group(2)),
                    "title": title.group(1) if title else rel,
                },
            )
        )
    return docs


@dataclass
class IngestReport:
    docs: int
    chunks: int
    sync: SyncReport
    seconds: float
    per_week: dict[int, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.docs} documents -> {self.chunks} chunks in {self.seconds:.1f} s ({self.sync}); "
            f"chunks per week: {', '.join(f'{w}:{n}' for w, n in sorted(self.per_week.items()))}"
        )


def build_index(embedder, directory: str | Path | None, docs: list[SourceDoc]):
    """Create (or load) the index at ``directory`` and bring it in sync with ``docs``: only new or changed documents are embedded. Returns (index, IngestReport)."""
    index = RagIndex(embedder, directory)
    t0 = time.perf_counter()
    sync = index.sync(docs)
    per_week: dict[int, int] = {}
    for c in index.chunks:
        w = c["meta"].get("week")
        if w is not None:
            per_week[w] = per_week.get(w, 0) + 1
    return index, IngestReport(
        len(docs), len(index.chunks), sync, time.perf_counter() - t0, per_week
    )
