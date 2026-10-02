"""Labelled question pairs for a response cache: does a similarity score + threshold serve the SAME answer or a wrong one?

SAME pairs: two phrasings of one question; a cache hit is correct (both are answered by the same help-center article).
NEAR pairs: lexically close questions that need DIFFERENT answers (another article, another id, a flipped meaning);
            a cache hit is a wrong answer served confidently.

A cache's quality is the trade-off between the two: hit rate on SAME vs wrong-answer rate on NEAR, per threshold.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.cache import jaccard  # noqa: E402


@dataclass(frozen=True)
class Pair:
    a: str
    b: str
    same: bool  # True: a cached answer to `a` is also the right answer to `b`
    note: str = ""


SAME = [
    Pair("my export fails with error 0x5F", "getting error 0x5F when I export", True),
    Pair("what does error 0x5F mean", "error 0x5F on export, what is it", True),
    Pair("large export keeps failing with 0x5F", "0x5F error on a big export", True),
    Pair("how do I reset my password", "I forgot my password, how can I reset it", True),
    Pair("where is the forgot password link", "reset password link", True),
    Pair("how long does the password reset link last", "when does the reset link expire", True),
    Pair(
        "my account is locked after failed sign-ins",
        "locked out after too many wrong password attempts",
        True,
    ),
    Pair("how long is my account locked", "how many minutes does the account lock last", True),
    Pair("the mobile app crashes on start", "app crashes when I open it on my phone", True),
    Pair("how do I collect the diagnostic log", "where do I share logs with support", True),
    Pair("what is the API rate limit", "how many API requests per minute are allowed", True),
    Pair("I get HTTP 429 from the API", "API returns 429 too many requests", True),
    Pair("how do I change my billing email", "update the email address invoices are sent to", True),
    Pair("where do I set the billing contact", "change the billing contact address", True),
    Pair(
        "two-factor codes are not arriving",
        "I am not getting my 2FA code by SMS",
        True,
        "both negated: guard passes",
    ),
    Pair(
        "what if I lost my phone with the authenticator",
        "use backup codes if I lost my device",
        True,
    ),
    Pair("which export formats are available", "can I export as CSV or JSON", True),
    Pair("what separator do CSV exports use", "CSV export delimiter", True),
    Pair(
        "how do I reset my password",
        "I don't remember my password, how do I get back in",
        True,
        "negation guard will refuse: a true paraphrase lost to safety",
    ),
    Pair("my export does not work with error 0x5F", "export is not working, error 0x5F", True),
]

NEAR = [
    Pair(
        "how do I reset my password",
        "how do I reset my two-factor authentication",
        False,
        "another article",
    ),
    Pair(
        "my export fails with error 0x5F",
        "my export fails with error 0x7A",
        False,
        "another error code",
    ),
    Pair("what is the API rate limit", "what is the export rate limit", False, "another subject"),
    Pair("how do I change my billing email", "how do I change my account password", False),
    Pair(
        "my account is locked after failed sign-ins",
        "my account is locked after a failed payment",
        False,
        "billing, not security",
    ),
    Pair(
        "how do I export data as CSV",
        "how do I fix a failing CSV export",
        False,
        "formats vs the 0x5F failure",
    ),
    Pair(
        "the mobile app crashes on start", "the web app crashes on start", False, "another product"
    ),
    Pair("two-factor codes are not arriving", "invoice emails are not arriving", False),
    Pair(
        "I get HTTP 429 from the API", "I get HTTP 500 from the API", False, "another status code"
    ),
    Pair("my export works fine today", "my export does not work today", False, "negation flip"),
    Pair(
        "I can sign in again after the reset",
        "I cannot sign in even after the reset",
        False,
        "negation flip",
    ),
    Pair(
        "the reset link arrives within a minute",
        "the reset link never arrives",
        False,
        "negation flip",
    ),
    Pair(
        "how many requests per minute does key A get",
        "how many requests per minute does key B get",
        False,
        "another entity",
    ),
    Pair(
        "how do I reset my password on the mobile app",
        "how do I reset my password on the web app",
        False,
        "another platform",
    ),
    Pair("can I export as JSON", "can I import as JSON", False, "opposite action"),
    Pair(
        "how long is my account locked for 5 failed sign-ins",
        "how long is my account locked for 10 failed sign-ins",
        False,
        "another number",
    ),
    Pair(
        "where do I find the diagnostic log on iPhone",
        "where do I find the diagnostic log on Android",
        False,
        "another platform",
    ),
    Pair(
        "update my billing email to alice@example.com",
        "update my billing email to bob@example.com",
        False,
        "another email",
    ),
    Pair("why is my invoice higher this month", "why is my export slower this month", False),
    Pair(
        "how do I reset my password after being locked",
        "how do I reset my password after changing phones",
        False,
        "another situation",
    ),
]

PAIRS = SAME + NEAR


def curve(pairs, similarity, thresholds, *, guards) -> list[dict]:
    """For each threshold: share of SAME pairs that hit (useful) and of NEAR pairs that hit (a wrong answer served).
    ``guards`` is a ResponseCache-style check (entities + negation) applied before the threshold; ``None`` = similarity only."""
    rows = []
    sims = [(p, similarity(p.a, p.b)) for p in pairs]
    for t in thresholds:
        hits = [(p, s) for p, s in sims if s >= t and (guards is None or guards(p.a, p.b))]
        same = [p for p, _ in hits if p.same]
        near = [p for p, _ in hits if not p.same]
        n_same = sum(p.same for p in pairs)
        n_near = len(pairs) - n_same
        rows.append(
            {
                "threshold": t,
                "hit_rate_same": len(same) / n_same,
                "wrong_rate_near": len(near) / n_near,
                "precision": len(same) / len(hits) if hits else 1.0,
                "wrong_pairs": [p.a + " ~ " + p.b for p in near],
            }
        )
    return rows


def guards_ok(a: str, b: str) -> bool:
    from common.cache import entities, has_negation

    return entities(a) == entities(b) and has_negation(a) == has_negation(b)


def jaccard_similarity(a: str, b: str) -> float:
    return jaccard(a, b)
