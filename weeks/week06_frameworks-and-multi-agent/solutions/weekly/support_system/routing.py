"""A transparent keyword router: the cheap, auditable alternative (or first pass) to asking a model to classify.

Rules are ordered by cost of being wrong: an explicit request for a person wins; then money (billing); then product
trouble (technical). A message that matches nothing returns ``None`` so the caller can choose a fallback.
"""

from __future__ import annotations

import re

HUMAN = re.compile(
    r"\b(real person|human|speak to (?:someone|a person)|talk to (?:someone|a person)|manager|phone call|call me|not a bot|live agent)\b",
    re.I,
)
BILLING = re.compile(
    r"\b(invoices?|inv-\d+|refunds?|charged|charges?|billed|billing|payments?|money back|duplicate|double[- ]?charge|reimburse\w*)\b",
    re.I,
)
TECH = re.compile(
    r"\b(errors?|crash\w*|down|outage|exports?|passwords?|log ?in|sign(?:ed)? ?in|locked|api|rate limits?|two-factor|2fa|codes?|app|bugs?|broken|stuck|slow|fail\w*|reset|email|formats?|0x[0-9a-f]+)\b",
    re.I,
)


def rule_route(message: str) -> tuple[str | None, str]:
    """(category, reason). Category is billing | technical | human, or None when no rule fires."""
    if HUMAN.search(message):
        return "human", "asked for a person"
    if BILLING.search(message):
        return "billing", f"billing word: {BILLING.search(message).group().lower()}"
    if TECH.search(message):
        return "technical", f"technical word: {TECH.search(message).group().lower()}"
    return None, "no rule matched"
