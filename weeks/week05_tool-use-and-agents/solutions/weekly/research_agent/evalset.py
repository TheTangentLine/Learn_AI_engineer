"""A small research evaluation: questions with the facts a correct report must contain and the pages it must have read.

Scoring is deliberately strict and mostly mechanical:
  facts      every required regex appears in the report body
  sources    every required page was actually fetched
  verified   every claim is cited, cites a fetched page, and is supported by it (report.verify_report)
  abstains   (unanswerable question) the report says the pages do not answer it AND makes no cited claims
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .agent import ResearchResult


@dataclass
class ResearchQuestion:
    id: str
    question: str
    facts: list[str] = field(default_factory=list)  # regexes, all required
    sources: set[str] = field(default_factory=set)  # page slugs that must have been fetched
    answerable: bool = True


QUESTIONS = [
    ResearchQuestion(
        "tool-design",
        "In the Week 5 tool-design experiment, what success rate did the badly designed tools and the redesigned tools achieve?",
        [r"15\s?%", r"75\s?%"],
        {"week05-day3"},
    ),
    ResearchQuestion(
        "crag",
        "How many baseline retrieval misses did corrective RAG rescue in the Week 4 experiment?",
        [r"\b0\b.{0,12}\b9\b|rescued 0|no baseline misses|zero"],
        {"week04-day4"},
    ),
    ResearchQuestion(
        "context",
        "How many input tokens did the naive agent need in the Week 5 context-engineering experiment, and which strategy kept all five codes?",
        [r"437,072|437k|437 ?000", r"notes|scratchpad"],
        {"week05-day5"},
    ),
    ResearchQuestion(
        "sandbox",
        "According to Week 5 Day 6, what can a subprocess sandbox not stop?",
        [r"network", r"\bread"],
        {"week05-day6"},
    ),
    ResearchQuestion(
        "bm25",
        "In Week 3 Day 1, which retrieval method beat embeddings at hit@1?",
        [r"BM25"],
        {"week03-day1"},
    ),
    ResearchQuestion(
        "unanswerable", "Who won the 2026 football World Cup final?", [], set(), answerable=False
    ),
]

ABSTAIN = re.compile(
    r"(do(?:es)? not|don't|cannot|can't|could not|couldn't|no) .{0,40}(answer|contain|cover|mention|information|evidence|find)|not (?:found|covered|available|mentioned)|insufficient",
    re.I,
)


@dataclass
class Score:
    id: str
    facts: bool
    sources: bool
    verified: bool
    abstains: bool
    passed: bool
    notes: str = ""


def score(q: ResearchQuestion, r: ResearchResult) -> Score:
    body = r.body
    if not q.answerable:
        abstains = bool(ABSTAIN.search(body)) and not any(c.cites for c in r.verification.claims)
        return Score(
            q.id,
            True,
            True,
            True,
            abstains,
            r.run.ok and abstains,
            "abstained" if abstains else "answered an unanswerable question or cited something",
        )
    facts = all(re.search(rx, body, re.I) for rx in q.facts)
    sources = q.sources <= r.fetched_slugs
    verified = r.verification.ok
    notes = []
    if not facts:
        notes.append("missing facts")
    if not sources:
        notes.append(f"did not read {sorted(q.sources - r.fetched_slugs)}")
    if not verified:
        notes.append(r.verification.summary())
    return Score(
        q.id,
        facts,
        sources,
        verified,
        False,
        r.run.ok and facts and sources and verified,
        "; ".join(notes),
    )
