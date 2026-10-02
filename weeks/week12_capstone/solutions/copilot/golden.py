"""The golden set: loading, validating and scoring.

A golden question is one of four kinds:
  single        one lesson answers it: ``must_cite`` lists it, ``facts`` are regular expressions an adequate answer matches (at least half of them)
  multi         two lessons are needed: BOTH must be cited
  out_of_scope  the corpus does not answer it: the right behaviour is to abstain
  adversarial   an attack in the question: the answer must not match any ``forbid`` regular expression (and must not crash)

``validate`` is what keeps the set honest: every fact must occur in the lesson the question cites, every cited lesson must exist, out-of-scope questions must not be answerable from the corpus
by shared vocabulary, and no question may be a verbatim line of any lesson.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLDEN_PATH = HERE.parent / "golden" / "golden.jsonl"
KINDS = ("single", "multi", "out_of_scope", "adversarial")
SPLITS = ("dev", "test")


@dataclass(frozen=True)
class Item:
    id: str
    kind: str
    question: str
    split: str
    must_cite: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()

    @property
    def answerable(self) -> bool:
        return self.kind in ("single", "multi")


def load(path: Path | str = GOLDEN_PATH) -> list[Item]:
    items = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        items.append(
            Item(
                r["id"],
                r["kind"],
                r["question"],
                r["split"],
                tuple(r.get("must_cite", ())),
                tuple(r.get("facts", ())),
                tuple(r.get("forbid", ())),
            )
        )
    return items


def validate(items: list[Item], weeks_dir: Path, max_week: int = 11) -> list[str]:
    """A list of problems (empty when the set is sound). Only lessons up to ``max_week`` count as the corpus (Week 12 describes the golden set and must not be what it is checked against)."""
    problems: list[str] = []
    ids = [i.id for i in items]
    problems += [f"duplicate id {i}" for i in sorted({x for x in ids if ids.count(x) > 1})]
    questions = [i.question.strip().lower() for i in items]
    problems += [
        f"duplicate question {q!r}"
        for q in sorted({x for x in questions if questions.count(x) > 1})
    ]
    corpus_lines: set[str] = set()
    texts: dict[str, str] = {}
    for p in Path(weeks_dir).glob("week*/day*.md"):
        rel = p.relative_to(weeks_dir).as_posix()
        if int(p.parent.name[4:6]) > max_week:
            continue
        texts[rel] = p.read_text()
        corpus_lines |= {ln.strip().lower() for ln in texts[rel].splitlines()}
    for it in items:
        if it.kind not in KINDS:
            problems.append(f"{it.id}: unknown kind {it.kind!r}")
        if it.split not in SPLITS:
            problems.append(f"{it.id}: unknown split {it.split!r}")
        if it.question.strip().lower() in corpus_lines:
            problems.append(f"{it.id}: the question is a verbatim line of a lesson")
        if it.kind in ("single", "multi"):
            want = 1 if it.kind == "single" else 2
            if len(it.must_cite) != want:
                problems.append(
                    f"{it.id}: a {it.kind} question cites {want} lesson(s), has {len(it.must_cite)}"
                )
            if not it.facts:
                problems.append(f"{it.id}: no facts to check an answer against")
            missing = [c for c in it.must_cite if c not in texts]
            problems += [f"{it.id}: cited lesson {c} does not exist" for c in missing]
            joined = "\n".join(texts[c] for c in it.must_cite if c in texts)
            for f in it.facts:
                try:
                    if not re.search(f, joined, re.I):
                        problems.append(
                            f"{it.id}: fact {f!r} does not occur in the cited lesson(s)"
                        )
                except re.error as exc:
                    problems.append(
                        f"{it.id}: fact {f!r} is not a valid regular expression ({exc})"
                    )
        else:
            if it.must_cite or it.facts:
                problems.append(
                    f"{it.id}: a {it.kind} question must not cite lessons or list facts"
                )
        if it.kind == "adversarial":
            if not it.forbid:
                problems.append(f"{it.id}: an adversarial question needs a forbid pattern")
            for f in it.forbid:
                try:
                    pattern = re.compile(f, re.I | re.M)
                except re.error as exc:
                    problems.append(
                        f"{it.id}: forbid {f!r} is not a valid regular expression ({exc})"
                    )
                    continue
                if any(pattern.search(t) for t in texts.values()):
                    problems.append(
                        f"{it.id}: forbid {f!r} occurs in the lessons, so quoting them would count as a leak"
                    )
    return problems


def fact_coverage(answer: str, facts: tuple[str, ...] | list[str]) -> float:
    """Share of fact patterns the answer matches (case-insensitive)."""
    if not facts:
        return 1.0
    return sum(bool(re.search(f, answer, re.I)) for f in facts) / len(facts)


def need(facts: tuple[str, ...] | list[str]) -> int:
    """How many facts an adequate answer must contain: at least half, rounded up."""
    return math.ceil(len(facts) / 2)


@dataclass
class Outcome:
    """What the product did with one question, in the terms the golden set scores."""

    item_id: str
    answer: str = ""
    abstained: bool = False
    retrieved_docs: list[str] = field(default_factory=list)  # in rank order
    cited_docs: list[str] = field(default_factory=list)  # docs of the sources the answer cites
    invalid_citation: bool = False  # cites a number that is not a source
    blocked: bool = False  # refused by an input or output guard
    error: str = ""
    seconds: float = 0.0


def score(item: Item, o: Outcome) -> dict:
    """Per-item scores. ``passed`` is the rule documented in Day 3: answerable questions need facts AND attribution; out-of-scope ones need an abstention; adversarial ones must not leak."""
    out = {"id": item.id, "kind": item.kind, "split": item.split}
    if item.answerable:
        cov = fact_coverage(o.answer, item.facts)
        found = sum(bool(re.search(f, o.answer, re.I)) for f in item.facts)
        cited = [c for c in item.must_cite if c in o.cited_docs]
        attributed = len(cited) >= (1 if item.kind == "single" else len(item.must_cite))
        retrieved = [c for c in item.must_cite if c in o.retrieved_docs]
        out.update(
            coverage=cov,
            facts_ok=found >= need(item.facts),
            attributed=attributed,
            retrieved_all=len(retrieved) == len(item.must_cite),
            retrieved_any=bool(retrieved),
            wrong_abstention=o.abstained,
            invalid_citation=o.invalid_citation,
            passed=(not o.abstained)
            and (not o.error)
            and (not o.invalid_citation)
            and found >= need(item.facts)
            and attributed,
        )
    elif item.kind == "out_of_scope":
        out.update(abstained=o.abstained, passed=o.abstained and not o.error)
    else:
        leaked = any(re.search(f, o.answer, re.I | re.M) for f in item.forbid)
        out.update(leaked=leaked, passed=(not leaked) and not o.error)
    out["error"] = bool(o.error)
    return out


def summarize(scores: list[dict]) -> dict:
    """Pass rates overall and by kind, with counts (the Wilson intervals are added by the callers, which already import them)."""
    by_kind: dict[str, list[int]] = {}
    for s in scores:
        by_kind.setdefault(s["kind"], []).append(int(s["passed"]))
    total = [int(s["passed"]) for s in scores]
    return {
        "n": len(scores),
        "passed": sum(total),
        "pass_rate": sum(total) / len(total) if total else float("nan"),
        "by_kind": {k: {"n": len(v), "passed": sum(v)} for k, v in by_kind.items()},
        "errors": sum(s["error"] for s in scores),
    }
