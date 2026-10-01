"""The document set used for the Week 3-4 RAG exercises: this course's own lessons.

Why: it is real prose with headings, code and tables, it is already on disk (works offline), and
you can judge answers yourself because you have just read the material.

    from common.corpus import load_course_docs, split_sentences
    docs = load_course_docs()   # [Doc(path='weeks/week01_.../day2_....md', title=..., text=...)]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEEKS = ROOT / "weeks"


@dataclass(frozen=True)
class Doc:
    path: str  # relative to the repo root, e.g. weeks/week01_how-llms-work/day2_...md
    title: str
    text: str

    @property
    def short(self) -> str:
        """e.g. 'week01/day2' for compact citations."""
        m = re.search(r"week(\d+)[^/]*/day(\d+)", self.path)
        return f"week{m.group(1)}/day{m.group(2)}" if m else Path(self.path).stem


def load_course_docs(weeks_dir: Path = WEEKS) -> list[Doc]:
    """All lesson files (day*.md) under weeks/, sorted by path."""
    docs = []
    for p in sorted(weeks_dir.glob("week*/day*.md")):
        text = p.read_text(encoding="utf-8")
        title = next(
            (ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")), p.stem
        )
        docs.append(Doc(str(p.relative_to(ROOT)), title, text))
    return docs


_FENCE = re.compile(r"```.*?```", re.S)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`*(\[])")


def split_sentences(text: str, min_chars: int = 40) -> list[str]:
    """Prose sentences only: drops code fences, tables, headings and bare list markers."""
    text = _FENCE.sub(" ", text)
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("|", "#", "<", ">", "---")):
            continue
        line = re.sub(r"^[-*]\s+|^\d+\.\s+", "", line)
        line = re.sub(r"[*_`]", "", line)
        for s in _SENT.split(line):
            s = s.strip()
            if len(s) >= min_chars:
                out.append(s)
    return out
