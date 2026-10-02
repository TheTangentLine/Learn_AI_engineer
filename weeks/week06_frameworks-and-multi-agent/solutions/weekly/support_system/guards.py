"""Runtime guards. The Day 6 evaluation SEES a lie; these PREVENT it from reaching the customer.

input guard    refuse messages that contain a card number BEFORE any model sees them (and never store them)
output guard   check the agent's reply against what its tools actually returned this turn:
                 unverified_success   it claims a refund was issued/approved, but no tool said so
                 secret_request       it asks for a password / card number
                 promise              it guarantees an outcome
               and replace the reply with a safe one built from the tool results
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from common.agent_eval import PROMISE, asks_for_secret, claims_success

CARD = re.compile(
    r"\b\d(?:[ -]?\d){12,15}\b"
)  # 13-16 digits with optional single separators (no trailing space)
CARD_REFUSAL = "For your security, please don't share card numbers or passwords here. I haven't stored it. Tell me your invoice id (like INV-1234) and what went wrong, and I'll help from there."


def redact_cards(text: str) -> str:
    return CARD.sub("[card number removed]", text)


@dataclass
class Guarded:
    reply: str
    violations: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.violations)


def safe_reply(tool_results: list[str]) -> str:
    """The most useful TRUE thing we can say, built only from what the tools returned."""
    joined = "\n".join(tool_results)
    if "Refund issued" in joined:
        m = re.search(r"Refund issued: (RF-\d+) \(\$([\d.]+)\)", joined)
        return (
            f"Your refund of ${m.group(2)} ({m.group(1)}) has been issued."
            if m
            else "Your refund has been issued."
        )
    if "pending" in joined.lower() and "approval" in joined.lower():
        return "Your refund request is pending approval. Our billing team will review it and we will let you know the outcome."
    if "Not eligible" in joined:
        return "I'm sorry, that invoice is not eligible for a refund."
    return "I wasn't able to complete that, so I've asked a human colleague to follow up with you."


def check_reply(reply: str, tool_results: list[str]) -> Guarded:
    violations = []
    issued = any("Refund issued" in r for r in tool_results)
    if claims_success(reply) and not issued:
        violations.append("unverified_success")
    if asks_for_secret(reply):
        violations.append("secret_request")
    if PROMISE.search(reply):
        violations.append("promise")
    if not violations:
        return Guarded(reply)
    return Guarded(
        safe_reply(tool_results) if "secret_request" not in violations else CARD_REFUSAL, violations
    )
