"""Shared fakes for the capstone tests: a tiny index, a deterministic embedder, scripted chat and reranker, and a helper that builds a Copilot from them. No model or network is used."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from copilot.retrieve import Source  # noqa: E402

from common.vectorstores import Hit  # noqa: E402


def make_source(
    n,
    doc="week01_a/day1_x.md",
    text="The KV cache stores keys and values.",
    week=1,
    day=1,
    heading="Week 1, Day 1: X > 1. Cache",
    score=1.0,
    cosine=0.8,
):
    return Source(n, f"{doc}#{n}", doc, heading, text, score, cosine, week, day)


class FakeIndex:
    """An index whose ``search`` ranks chunks by shared words with the query; honours ``where`` on the week."""

    def __init__(self, chunks):
        # chunks: list of dicts {id, doc, week, day, heading, text}
        self.chunks = chunks
        self.calls = []

    def search(self, query, k=5, alpha=0.5, where=None, reranker=None, rerank_depth=20):
        self.calls.append((query, k, where))
        q = set(query.lower().replace("?", "").split())
        scored = []
        for c in self.chunks:
            if where and any(
                {"week": c["week"], "day": c["day"]}.get(a) != b for a, b in where.items()
            ):
                continue
            words = set(c["text"].lower().split())
            scored.append((len(q & words) + 0.001 * -self.chunks.index(c), c))
        scored.sort(key=lambda x: -x[0])
        return [
            Hit(
                c["id"],
                s,
                {
                    "doc": c["doc"],
                    "heading": c["heading"],
                    "cosine": min(1.0, 0.3 + 0.1 * s),
                    "week": c["week"],
                    "day": c["day"],
                },
                c["text"],
            )
            for s, c in scored[:k]
        ]


CHUNKS = [
    {
        "id": "a1",
        "doc": "week01_x/day4_cache.md",
        "week": 1,
        "day": 4,
        "heading": "Week 1, Day 4: Cache > 2. Why",
        "text": "[Week 1, Day 4: Cache > 2. Why]\nThe KV cache stores the key and value vectors of every earlier token so they are not recomputed. Prompt caching reuses the beginning of a prompt.",
    },
    {
        "id": "a2",
        "doc": "week01_x/day2_tokens.md",
        "week": 1,
        "day": 2,
        "heading": "Week 1, Day 2: Tokens > 1. Tokens",
        "text": "[Week 1, Day 2: Tokens > 1. Tokens]\nEverything is metered in tokens: price, context limit, rate limits, speed.",
    },
    {
        "id": "b1",
        "doc": "week11_y/day2_batching.md",
        "week": 11,
        "day": 2,
        "heading": "Week 11, Day 2: Batching > 5. Paged",
        "text": "[Week 11, Day 2: Batching > 5. Paged]\nPagedAttention splits the KV cache into blocks so memory is not wasted on empty slots. Blocks are allocated on demand.",
    },
    {
        "id": "c1",
        "doc": "week04_z/day1_eval.md",
        "week": 4,
        "day": 1,
        "heading": "Week 4, Day 1: Eval > 2. Golden",
        "text": "[Week 4, Day 1: Eval > 2. Golden]\nA golden set is a list of questions with known-correct answers and evidence.",
    },
]


@pytest.fixture
def index():
    return FakeIndex([dict(c) for c in CHUNKS])


class ScriptedChat:
    """A chat function that returns queued replies (text, prompt tokens, completion tokens) and records the messages it was given."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, messages):
        self.seen.append(messages)
        r = self.replies.pop(0) if self.replies else ("", 0, 0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeReranker:
    """Scores a passage by how many of the query's words it contains (so relevant passages score high, unrelated ones low)."""

    def __init__(self):
        self.calls = 0

    def scores(self, query, passages):
        self.calls += 1
        q = set(query.lower().replace("?", "").split())
        return [float(len(q & set(p.lower().split()))) - 2.0 for p in passages]

    def rerank(self, query, passages, top_k=None):
        sc = self.scores(query, passages)
        return sorted(enumerate(sc), key=lambda x: -x[1])[:top_k]
