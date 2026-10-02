"""Hardening for the Week 6 support system, found by red-teaming it (see the findings register):

  ownership      a customer may look up and refund only THEIR invoices; anything else answers exactly like an invoice that does not exist
                 (so the tool cannot be used to find out which ids exist)
  reason         the refund reason a human approver reads is customer-controlled text. Two options: "filter" caps it, strips links and withholds it
                 when it looks like an instruction to the approver (probabilistic: it recognises phrasings); "enum" keeps NO free text, only one of
                 a few categories chosen by keyword (structural: nothing the customer wrote can reach the approver)
  reply guard    the reply is rendered by the customer's client: no remote images, no links to hosts we do not own

All three plug into the two hooks of ``SupportSystem`` (``tool_wrapper`` and ``reply_filter``); with none of them the system behaves as before.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from common import guard  # noqa: E402
from common.chat import ToolCall  # noqa: E402
from common.tools import ToolRegistry, ToolResult  # noqa: E402

OWN_HOSTS = {"support.acme.example"}
INVOICE_TOOLS = ("lookup_invoice", "request_refund")
WITHHELD = "(the customer's explanation was withheld: it looked like an instruction)"


def normalise_invoice(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).upper()


REASON_CATEGORIES = (
    ("duplicate charge", r"twice|double|duplicate|again|two times"),
    ("unauthorised charge", r"fraud|unauthori[sz]ed|did ?n.t make|not me|stolen"),
    ("service not delivered", r"not working|not delivered|never received|outage|broken|down for"),
    ("accidental purchase", r"accident|by mistake|mistake|didn.t mean|wrong plan"),
)
OTHER = "other (the customer's text is not shown)"


def categorise_reason(reason: str) -> str:
    """One of a handful of categories, chosen by keyword. Whatever else the customer wrote is dropped, so it cannot carry anything to the approver."""
    for label, pattern in REASON_CATEGORIES:
        if re.search(pattern, str(reason), re.I):
            return label
    return OTHER


def clean_reason(reason: str, *, limit: int = 120) -> str:
    """What the approver is shown: one line, no links, no more than ``limit`` characters, and nothing that reads like an instruction to them."""
    text = re.sub(r"https?://\S+|//\S+", "[link removed]", str(reason))
    text = re.sub(r"[\x00-\x1f​-‏‪-‮⁠﻿\U000e0000-\U000e007f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if guard.detect(text).flagged or re.search(
        r"\b(?:approv\w*|pre-?cleared|pre-?authori[sz]ed|waived?)\b", text, re.I
    ):
        return WITHHELD
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


class HardenedTools(ToolRegistry):
    """Wraps a specialist's registry. Every call to an invoice tool is checked against the caller's identity BEFORE it runs; the arguments the
    approver will see are cleaned. Anything else passes through. The audit list is the evidence trail (tool, verdict), without arguments."""

    def __init__(
        self,
        inner: ToolRegistry,
        customer: str | None,
        owners: dict[str, str],
        *,
        ownership: bool = True,
        reason: str = "filter",
        audit: list[dict] | None = None,
    ):
        super().__init__(inner.tools(), compact=getattr(inner, "_compact", False))
        self._inner, self.customer, self.owners = inner, customer, owners
        self.ownership, self.reason = ownership, reason
        self.audit = audit if audit is not None else []

    def _rewrap(self, inner: ToolRegistry) -> HardenedTools:
        return HardenedTools(
            inner,
            self.customer,
            self.owners,
            ownership=self.ownership,
            reason=self.reason,
            audit=self.audit,
        )

    def compact(self) -> HardenedTools:  # the inherited versions would return an UNGUARDED copy
        return self._rewrap(self._inner.compact())

    def without(self, *names: str) -> HardenedTools:
        return self._rewrap(self._inner.without(*names))

    def execute(self, call: ToolCall) -> ToolResult:
        if call.name in INVOICE_TOOLS:
            invoice = normalise_invoice(call.args.get("invoice_id", ""))
            if self.ownership and (
                self.customer is None or self.owners.get(invoice) != self.customer
            ):
                self.audit.append(
                    {"tool": call.name, "verdict": "denied", "why": "not the caller's invoice"}
                )
                text = (
                    f"No invoice {call.args.get('invoice_id')!r}."
                    if call.name == "lookup_invoice"
                    else "Not eligible for a refund (invoice_not_found)."
                )
                return ToolResult(call.id, call.name, text, False)
            if call.name == "request_refund" and self.reason and "reason" in call.args:
                clean = categorise_reason if self.reason == "enum" else clean_reason
                call = dataclasses.replace(
                    call, args={**call.args, "reason": clean(call.args["reason"])}
                )
        return super().execute(call)


def tool_wrapper_for(
    identity: dict[str, str],
    owners: dict[str, str],
    *,
    ownership: bool = True,
    reason: str = "filter",
    audit: list[dict] | None = None,
) -> Callable[[ToolRegistry, str, str], ToolRegistry]:
    """``identity`` maps a conversation id to the AUTHENTICATED customer (set by the web layer from the session, never from message text).
    An unknown conversation has no customer and therefore owns nothing: the check fails closed."""

    def wrap(tools: ToolRegistry, conversation_id: str, agent: str) -> ToolRegistry:
        return HardenedTools(
            tools,
            identity.get(conversation_id),
            owners,
            ownership=ownership,
            reason=reason,
            audit=audit,
        )

    return wrap


def reply_filter(text: str) -> str:
    return guard.guard_output(
        text, guard.OutputPolicy(allowed_hosts=OWN_HOSTS, allow_images=False)
    ).text
