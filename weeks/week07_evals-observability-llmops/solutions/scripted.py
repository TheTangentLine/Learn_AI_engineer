"""Scripted models for Week 7 tests: a GOOD support system and named FAULTS to inject (each fault = one root cause).

``rules(**faults)`` returns FakeLLM rules. Faults (all default off):
  misroute_tech      triage sends technical questions to 'other'
  skip_tools         specialists answer in prose without calling any tool
  narrate            like skip_tools, but the prose announces the action ("I will look that up")
  skip_lookup        billing calls request_refund without looking the invoice up
  wrong_invoice      billing refunds an invoice id it made up
  refund_unpaid      billing requests a refund even when the invoice is open
  guess_invoice      billing guesses an invoice id instead of asking
  liar               billing claims a pending refund was approved and issued
  ignore_kb          tech calls search_kb but answers with something unrelated
  loop               billing calls lookup_invoice forever
"""

from __future__ import annotations

import json
import re

from common.fake import tool_calls


def triage(prompt, *, misroute_tech=False):
    low = prompt.split("<message>")[-1].lower()
    if re.search(r"real person|human|speak to someone|phone call|not a bot", low):
        cat = "human"
    elif re.search(
        r"\b(invoice|charged|refund|inv-\d+|billed|double|duplicate|payment|money back|reverse|approve)\b",
        low,
    ):
        cat = "billing"
    elif re.search(
        r"\b(exports?|error|reset|password|crash(es)?|down|login|0x5f|api|two-factor|formats|locked|email|app|outage|broken|stuck)\b",
        low,
    ):
        cat = "other" if misroute_tech else "technical"
    else:
        cat = "other"
    return json.dumps({"category": cat, "reason": "keywords"})


def specialist(**f):
    def decide(prompt, call):
        system = call.system or ""
        msgs = call.messages
        users = [m["content"] for m in msgs if m["role"] == "user"]
        idx = max(i for i, m in enumerate(msgs) if m["role"] == "user")
        now = msgs[idx]["content"]
        results = [m["content"] for m in msgs[idx + 1 :] if m["role"] == "tool"]
        invoice = next((x for u in reversed(users) for x in re.findall(r"INV-\d+", u)), None)
        if re.search(r"thanks", now, re.I):
            return "You're welcome!"
        if f.get("skip_tools"):
            return "Your refund should be fine; our team looks at it."
        if f.get("narrate"):
            return "I will look that up for you right away."
        if "billing support specialist" in system:
            if f.get("loop"):
                return tool_calls(("lookup_invoice", {"invoice_id": invoice or "INV-3001"}))
            if not invoice and f.get("guess_invoice"):
                invoice = "INV-3001"  # the fault: it GUESSES an id and carries on instead of asking
            if not invoice:
                return "Could you share the invoice number?"
            if f.get("wrong_invoice") and not results:
                return tool_calls(
                    ("request_refund", {"invoice_id": "INV-9999", "reason": "duplicate"})
                )
            if f.get("skip_lookup") and not results:
                return tool_calls(
                    ("request_refund", {"invoice_id": invoice, "reason": "duplicate charge"})
                )
            if results:
                last = results[-1]
                if "Refund issued" in last:
                    return f"Your refund {re.search(r'RF-[0-9]+', last).group()} has been issued."
                if "pending" in last:
                    return (
                        "Great news, your refund has been approved and issued!"
                        if f.get("liar")
                        else "Your refund is pending approval; our billing team will review it."
                    )
                if "Not eligible" in last:
                    return "Sorry, that invoice is not eligible for a refund."
                if "open" in last and not f.get("refund_unpaid"):
                    return "That invoice is still open, so there is nothing to refund yet."
                if "open" in last or "paid" in last:
                    return tool_calls(
                        ("request_refund", {"invoice_id": invoice, "reason": "duplicate charge"})
                    )
            return tool_calls(("lookup_invoice", {"invoice_id": invoice}))
        # technical
        if results:
            if f.get("ignore_kb"):
                return "Please try again later."
            last = results[-1]
            if last.startswith(
                "degraded"
            ):  # a service_status result (KB articles can mention the word too)
                return (
                    "The export service is degraded (the queue is backed up, ETA 2h); see KB-101."
                )
            return "Here is what I found: " + last
        if re.search(r"export", now, re.I) and re.search(
            r"down|outage|stuck|broken|problems|issue|going on|today|fails|failing", now, re.I
        ):
            return tool_calls(("service_status", {"service": "export"}))
        return tool_calls(("search_kb", {"query": now}))

    return decide


def rules(**faults):
    return [
        (
            r"You are the triage agent",
            lambda p: triage(p, misroute_tech=faults.get("misroute_tech", False)),
        ),
        (r"(?s).*", specialist(**faults)),
    ]
