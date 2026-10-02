"""A response cache that is hard to misuse: exact match on normalised text, an optional fuzzy tier, and the guards that
make a cache SAFE to put in front of a model.

    cache = ResponseCache(ttl_s=3600)
    scope = "tech|prompt-v3|model-x"                  # anything that changes the answer must be in the scope
    if (hit := cache.get("How do I reset my password?", scope=scope)):
        return hit.value
    answer = ask_the_model(...)
    cache.put("How do I reset my password?", answer, scope=scope)

Why the guards exist (each is tested):
  * SCOPE: the cache key includes the prompt version / model / agent, so a prompt change cannot serve yesterday's answer.
  * ENTITIES: ids, numbers and emails must match EXACTLY, even on a fuzzy hit. "refund INV-1001" and "refund INV-1002"
    are 90% similar words and different requests.
  * NEGATION: "my export works" and "my export does not work" share most of their words. A fuzzy hit across a negation
    flip is refused.
  * TTL and a size bound (LRU), so stale or unbounded entries cannot accumulate.
  * Only what the CALLER marks as safe goes in: the cache cannot know a reply contained personal data. Put nothing in
    that came from a customer-specific tool, and keep customer ids out of the scope-less key.
  * Thread-safe (agents run tools in threads).

The fuzzy tier here is token-overlap (Jaccard): a stand-in that needs no model. A semantic cache uses embedding
similarity instead (``similarity=``), with the same guards; embeddings catch paraphrases, and they also make
"similar but different" mistakes more likely, so the threshold must be validated on your own traffic.
"""

from __future__ import annotations

import re
import threading
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

NEGATIONS = {
    "not",
    "no",
    "never",
    "cannot",
    "cant",
    "can't",
    "won't",
    "wont",
    "doesn't",
    "doesnt",
    "don't",
    "dont",
    "isn't",
    "isnt",
    "without",
    "unable",
    "fail",
    "fails",
    "failed",
    "failing",
}
STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "to",
    "of",
    "i",
    "my",
    "me",
    "how",
    "do",
    "does",
    "can",
    "you",
    "please",
    "and",
    "or",
    "for",
    "in",
    "on",
    "it",
    "this",
    "that",
    "with",
}
_ENTITY = re.compile(r"[A-Za-z]+-\d+|\d+(?:[.,]\d+)*|[\w.+-]+@[\w-]+\.[\w.]+")


def normalize(text: str) -> str:
    """Lower-case, NFKC, punctuation to spaces (hyphens inside ids survive), whitespace collapsed."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[^\w\s@.+-]", " ", t)
    t = re.sub(
        r"(?<![\w])[-.]+|[-.]+(?![\w])", " ", t
    )  # a stray dash or full stop is punctuation, not part of a word
    return re.sub(r"\s+", " ", t).strip()


def entities(text: str) -> frozenset[str]:
    return frozenset(
        m.group().lower() for m in _ENTITY.finditer(unicodedata.normalize("NFKC", text))
    )


def words(text: str) -> frozenset[str]:
    return frozenset(w for w in normalize(text).split() if w not in STOPWORDS)


def has_negation(text: str) -> bool:
    toks = set(
        re.findall(r"[a-z']+", unicodedata.normalize("NFKC", text).lower().replace("’", "'"))
    )
    return bool(toks & NEGATIONS)


def jaccard(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


@dataclass
class CacheHit:
    value: str
    kind: str  # "exact" | "fuzzy"
    similarity: float
    age_s: float
    matched: str  # the stored question that matched


@dataclass
class _Entry:
    text: str
    value: str
    stored_at: float
    entities: frozenset[str]
    negated: bool


class ResponseCache:
    def __init__(
        self,
        *,
        max_entries: int = 1000,
        ttl_s: float = 3600.0,
        fuzzy: bool = False,
        threshold: float = 0.85,
        similarity: Callable[[str, str], float] = jaccard,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_entries, self.ttl_s, self.fuzzy, self.threshold = (
            max_entries,
            ttl_s,
            fuzzy,
            threshold,
        )
        self.similarity, self.clock = similarity, clock
        self._data: OrderedDict[tuple[str, str], _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = {
            "hits": 0,
            "exact_hits": 0,
            "fuzzy_hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0,
            "refused_entity": 0,
            "refused_negation": 0,
        }

    def __len__(self) -> int:
        return len(self._data)

    @property
    def hit_rate(self) -> float:
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total else 0.0

    def _expire(self, now: float) -> None:
        for key in [k for k, e in self._data.items() if now - e.stored_at >= self.ttl_s]:
            del self._data[key]
            self.stats["expired"] += 1

    def get(self, text: str, *, scope: str = "") -> CacheHit | None:
        now = self.clock()
        key = (scope, normalize(text))
        with self._lock:
            self._expire(now)
            if (e := self._data.get(key)) is not None:
                self._data.move_to_end(key)
                return self._hit(e, "exact", 1.0, now)
            if self.fuzzy:
                ents, neg = entities(text), has_negation(text)
                best: tuple[float, tuple[str, str]] | None = None
                for k, e in self._data.items():
                    if k[0] != scope:
                        continue
                    sim = self.similarity(text, e.text)
                    if sim < self.threshold or (best and sim <= best[0]):
                        continue
                    if e.entities != ents:
                        self.stats["refused_entity"] += 1
                        continue
                    if e.negated != neg:
                        self.stats["refused_negation"] += 1
                        continue
                    best = (sim, k)
                if best is not None:
                    self._data.move_to_end(best[1])
                    return self._hit(self._data[best[1]], "fuzzy", best[0], now)
            self.stats["misses"] += 1
            return None

    def _hit(self, e: _Entry, kind: str, sim: float, now: float) -> CacheHit:
        self.stats["hits"] += 1
        self.stats[f"{kind}_hits"] += 1
        return CacheHit(e.value, kind, sim, now - e.stored_at, e.text)

    def put(self, text: str, value: str, *, scope: str = "") -> None:
        key = (scope, normalize(text))
        with self._lock:
            self._data[key] = _Entry(text, value, self.clock(), entities(text), has_negation(text))
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self.stats["evictions"] += 1

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
