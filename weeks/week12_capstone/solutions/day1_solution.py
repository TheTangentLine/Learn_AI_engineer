"""Week 12 Day 1 - Solution: the design document and the golden set, checked.

1. THE DESIGN DOC   the reference ``design/DESIGN.md`` against the structural checker
2. THE GOLDEN SET   validated against the corpus (every fact occurs in the lesson it cites, no leaked questions), then described: kinds, splits, weeks covered, difficulty
3. FLOORS           what trivial systems score, so a real score has something to be compared with: always abstain; dump the top BM25 chunk; and the CEILING of any answerer on these retrieved sources

  uv run python weeks/week12_capstone/solutions/day1_solution.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from lab import WEEKS, Lab

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from copilot import design as D  # noqa: E402
from copilot import evaluate as E  # noqa: E402
from copilot import golden as G  # noqa: E402
from copilot.retrieve import RetrievalConfig, Retriever  # noqa: E402


def main() -> None:
    print("1. THE DESIGN DOCUMENT")
    text = (HERE / "design" / "DESIGN.md").read_text()
    problems = D.check(text)
    secs = D.sections(text)
    print(
        f"   {len(secs)} of {len(D.SECTIONS)} sections; {len(D.table_rows(secs['Requirements']))} requirements; {len(D.table_rows(secs['Risks']))} risks; {len(D.table_rows(secs['Decisions and alternatives']))} decisions; checker problems: {problems or 'none'}"
    )

    print("\n2. THE GOLDEN SET")
    items = G.load()
    problems = G.validate(items, WEEKS)
    print(f"   {len(items)} questions; validator problems: {problems or 'none'}")
    print(f"   {'kind':<14}{'dev':>5}{'test':>6}{'total':>7}")
    for k in G.KINDS:
        d = sum(i.kind == k and i.split == "dev" for i in items)
        t = sum(i.kind == k and i.split == "test" for i in items)
        print(f"   {k:<14}{d:>5}{t:>6}{d + t:>7}")
    weeks = sorted({int(c.split("/")[0][4:6]) for i in items for c in i.must_cite})
    per_week = {
        w: sum(any(c.startswith(f"week{w:02d}") for c in i.must_cite) for i in items) for w in weeks
    }
    print("   questions touching each week:", ", ".join(f"{w}:{n}" for w, n in per_week.items()))
    facts = [len(i.facts) for i in items if i.answerable]
    print(
        f"   facts per answerable question: min {min(facts)}, median {sorted(facts)[len(facts) // 2]}, max {max(facts)}; an answer needs at least half of them"
    )
    named = sum(
        bool(re.search(r"\bweek\s+\d", i.question, re.I)) for i in items if i.kind == "multi"
    )
    print(
        f"   caution: {named} of the 8 two-lesson questions name both weeks in the question itself (I wrote them that way); real users often will not"
    )

    print("\n3. FLOORS AND CEILINGS (test split)")
    lab = Lab(load_reranker=False)
    test = [i for i in items if i.split == "test"]
    abstain = [
        G.score(
            i,
            G.Outcome(
                i.id,
                "I don't know based on the provided sources.",
                True,
            ),
        )
        for i in test
    ]
    print(
        f"   always abstain:          {sum(s['passed'] for s in abstain)}/{len(test)} pass ({sum(s['passed'] for s in abstain) / len(test):.0%}) (all of it from refusing and surviving attacks)"
    )
    r = Retriever(lab.index, RetrievalConfig(alpha=0.0, scope_by_week=False, k=1))
    dump = []
    for i in test:
        src = r.retrieve(i.question).sources
        o = G.Outcome(
            i.id, src[0].text if src else "", False, [s.doc for s in src], [s.doc for s in src]
        )
        dump.append(G.score(i, o))
    print(
        f"   dump the top BM25 chunk: {sum(s['passed'] for s in dump)}/{len(test)} pass ({sum(s['passed'] for s in dump) / len(test):.0%}), never abstains: it passes {sum(s['passed'] for s, i in zip(dump, test, strict=True) if i.answerable)} of {sum(i.answerable for i in test)} answerable questions"
    )
    r5 = Retriever(lab.index, RetrievalConfig(k=5))
    ceiling = 0
    answerable = [i for i in items if i.answerable]
    for i in answerable:
        joined = "\n".join(s.text for s in r5.retrieve(i.question).sources)
        ceiling += G.fact_coverage(joined, i.facts) * len(i.facts) >= G.need(i.facts)
    print(
        f"   ceiling: in {ceiling} of {len(answerable)} answerable questions ({ceiling / len(answerable):.0%}) at least half of the fact patterns occur somewhere in the top 5 sources. No answerer that quotes or paraphrases those sources can do better"
    )
    print(
        f"   the golden set's score on any system is therefore bounded: {E.fmt(E.rate(ceiling, len(answerable)))} of answerable questions"
    )


if __name__ == "__main__":
    main()
