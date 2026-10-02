"""A tiny knowledge base for the technical specialist (BM25 over short articles)."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

ARTICLES = {
    "KB-101": (
        "Export fails with error 0x5F",
        "Error 0x5F means the export queue is full. Wait a few minutes and retry. If the export status shows degraded, the backlog clears on its own; large exports (over 100 MB) should be split by date range.",
    ),
    "KB-102": (
        "Reset your password",
        "Use the Forgot password link on the sign-in page. We never ask for your password by chat or email. The reset link expires after 30 minutes.",
    ),
    "KB-103": (
        "Account locked after failed sign-ins",
        "After 5 failed sign-ins an account is locked for 15 minutes. Use the password reset link to unlock it immediately.",
    ),
    "KB-104": (
        "App crashes on start (mobile)",
        "Update to the latest version, then clear the app cache in Settings. If the crash persists, collect the diagnostic log (Settings, Support, Share logs) and send it to support.",
    ),
    "KB-105": (
        "API rate limits",
        "The API allows 100 requests per minute per key. Requests over the limit receive HTTP 429 with a Retry-After header; back off exponentially.",
    ),
    "KB-106": (
        "Change your billing email",
        "Billing emails are changed under Settings, Billing, Contact. Invoices are re-sent to the new address within an hour.",
    ),
    "KB-107": (
        "Two-factor authentication codes not arriving",
        "Check that the phone number is correct and that SMS is not blocked. Authenticator apps work without a network connection to us; use backup codes if you lost the device.",
    ),
    "KB-108": (
        "Data export formats",
        "Exports are available as CSV and JSON. CSV exports use UTF-8 and a comma separator.",
    ),
}


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class KnowledgeBase:
    def __init__(self, articles: dict[str, tuple[str, str]] | None = None):
        self.articles = articles or ARTICLES
        self.ids = list(self.articles)
        self.bm25 = BM25Okapi([_tok(t + " " + b) for t, b in self.articles.values()])

    def search(self, query: str, limit: int = 3) -> list[tuple[str, str, str, float]]:
        q = _tok(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        ranked = sorted(zip(self.ids, scores, strict=True), key=lambda p: (-p[1], p[0]))
        return [(i, *self.articles[i], float(s)) for i, s in ranked[:limit] if s > 0]
