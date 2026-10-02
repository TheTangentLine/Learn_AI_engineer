"""A frozen regression corpus: every attack text this week used or found, with the behaviour the hardened systems showed when it was frozen.

  expect "blocked"   the hardened system stopped it; if it ever succeeds, a control regressed: the test FAILS
  expect "open"      a known, documented residual risk; the test is an expected failure (xfail, strict): if the attack stops working, the test
                     FAILS until someone re-freezes the corpus, which is how an improvement gets noticed and the findings register updated

The corpus is data (``corpus.json``), not code: later changes to the attack generators cannot silently change what the suite tests, and a test pins its
hash. Regenerate with ``python corpus.py freeze`` and review the diff like any other change.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLUTIONS = HERE.parents[1]
ROOT = HERE.parents[4]
for p in (str(ROOT), str(SOLUTIONS), str(HERE)):
    sys.path.insert(0, p)

import agent_target as A  # noqa: E402
import campaign as C  # noqa: E402
import day3_solution as d3  # noqa: E402
import day4_solution as d4  # noqa: E402
import defenses as D  # noqa: E402
import support_target as S  # noqa: E402
import targets as T  # noqa: E402

from common import redteam as rt  # noqa: E402

CORPUS = HERE / "corpus.json"
CANARY = rt.Canary.make(seed=7)
HARDENED_SUPPORT = S.Controls(ownership=True, reason="enum", reply_guard=True)
AGENT_SUBSET = rt.build_attacks(
    CANARY,
    goals=d4.GOALS,
    techniques=["plain_override", "base64", "tag_smuggle", "delimiter_escape"],
    channels=("tool_result",),
)


def rag_target() -> T.RagTarget:
    d = dataclasses.replace(D.CONFIGS["all layers"], rng=random.Random(0))
    return T.RagTarget(T.obedient_model(CANARY), CANARY, defenses=d)


def candidates() -> list[dict]:
    """Every attack that goes into the corpus, as plain dicts. The RAG adaptive winners are the rewrites the Day 6 campaign found."""
    rows: list[dict] = []

    def add(target: str, a: rt.Attack, prefix: str = "") -> None:
        rows.append(
            {
                "id": f"{prefix or target}:{a.id}",
                "target": target,
                "goal": a.goal,
                "technique": a.technique,
                "channel": a.channel,
                "text": a.text,
            }
        )

    for a in d3.lab_attacks():
        add("rag", a)
    for a in d3.heldout_attacks():
        add("rag", a)
    adaptive = C.adaptive_run(rag_target(), d3.lab_attacks(), CANARY, max_queries=2)
    for o in adaptive:
        if o.first_success and o.first_success > 1:
            add("rag", dataclasses.replace(o.attack, text=o.winning_text), prefix="rag-adaptive")
    for a in S.attacks(CANARY):
        add("support", a)
    for a in AGENT_SUBSET:
        add("agent", a)
    return rows


class Replayer:
    """Runs corpus entries against the HARDENED version of their target. A context manager: the agent target owns a subprocess."""

    def __enter__(self) -> Replayer:
        self._rag = rag_target()
        self._agent = A.AgentHarness(CANARY, d4.CONFIGS["all layers"]).__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._agent.__exit__(*exc)

    def attack_of(self, entry: dict) -> rt.Attack:
        return rt.Attack(
            entry["id"], entry["goal"], entry["technique"], entry["channel"], entry["text"]
        )

    def succeeded(self, entry: dict) -> bool:
        a = self.attack_of(entry)
        if entry["target"] == "rag":
            return rt.run(self._rag, [a], CANARY)[0].succeeded
        if entry["target"] == "agent":
            return rt.run(self._agent, [a], CANARY)[0].succeeded
        if entry["target"] == "support":
            return S.run(HARDENED_SUPPORT, CANARY, [a])[0].succeeded
        raise ValueError(entry["target"])


def digest(entries: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()[:16]


def freeze(path: Path = CORPUS) -> dict:
    rows = candidates()
    with Replayer() as r:
        for row in rows:
            row["expect"] = "open" if r.succeeded(row) else "blocked"
    doc = {"canary_seed": 7, "entries": rows, "digest": digest(rows)}
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=True) + "\n")
    return doc


def load(path: Path = CORPUS) -> dict:
    return json.loads(path.read_text())


def summary(doc: dict) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for e in doc["entries"]:
        out.setdefault(e["target"], {"blocked": 0, "open": 0})[e["expect"]] += 1
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["freeze"]:
        d = freeze()
        print(d["digest"], json.dumps(summary(d)))
    else:
        d = load()
        print(d["digest"], json.dumps(summary(d)))
