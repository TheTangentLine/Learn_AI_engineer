"""Each specialist gets ONLY the tools it needs (least privilege)."""

from __future__ import annotations

from _weeks import load

from common.tools import ToolRegistry, tool

d5 = load("week06_frameworks-and-multi-agent", "day5_solution")

SERVICES = {
    "export": "degraded: the export queue is backed up (ETA 2h)",
    "login": "operational",
    "api": "operational",
    "mobile": "operational",
}


def billing_tools(system, conversation_id: str) -> ToolRegistry:
    @tool
    def lookup_invoice(invoice_id: str) -> str:
        """Look up an invoice by id (like INV-1234): returns its amount and whether it is paid or open.

        Args:
            invoice_id: The invoice id, e.g. INV-3001.
        """
        inv = d5.INVOICES.get(invoice_id.strip().upper())
        return (
            f"No invoice {invoice_id!r}."
            if inv is None
            else f"{invoice_id.upper()}: ${inv['amount']:.2f}, {inv['status']}"
        )

    @tool
    def request_refund(invoice_id: str, reason: str) -> str:
        """Request a refund for a PAID invoice. Small refunds are issued immediately; larger ones wait for a human.

        Args:
            invoice_id: The invoice id, e.g. INV-3001.
            reason: Why the customer wants the refund, in a few words.
        """
        invoice = invoice_id.strip().upper()
        r = system.refunds.start(
            f"{conversation_id}-{invoice}", invoice, reason
        )  # one request per (conversation, invoice)
        system.store.log(conversation_id, "refund_request", invoice=invoice, status=r["status"])
        if r["status"] == "refunded":
            return f"Refund issued: {r['refund_id']} (${r['refunded_amount']:.2f})."
        if r["status"] == "pending_approval":
            return "Refund pending human approval; the billing team will review it. Do NOT tell the customer it is approved."
        return f"Not eligible for a refund ({r['status']})."

    return ToolRegistry([lookup_invoice, request_refund, escalate_tool(system, conversation_id)])


def tech_tools(system, conversation_id: str) -> ToolRegistry:
    @tool
    def service_status(service: str) -> str:
        """Current status of a service: export, login, api or mobile.

        Args:
            service: The service name.
        """
        return SERVICES.get(
            service.strip().lower(), f"Unknown service {service!r}. Known: {', '.join(SERVICES)}."
        )

    @tool
    def search_kb(query: str) -> str:
        """Search the help-center articles. Returns the best matches with their text. Use it before answering how-to questions.

        Args:
            query: A few specific words, e.g. "export error 0x5F".
        """
        hits = system.kb.search(query)
        return (
            "\n".join(f"{i}: {title}. {body}" for i, title, body, _ in hits)
            or "No matching article."
        )

    return ToolRegistry([service_status, search_kb, escalate_tool(system, conversation_id)])


def escalate_tool(system, conversation_id: str):
    @tool
    def escalate_to_human(reason: str) -> str:
        """Hand the conversation to a human colleague (use when you cannot or must not resolve it yourself).

        Args:
            reason: One sentence for the colleague: what the customer needs and what you tried.
        """
        ticket = system.store.escalate(conversation_id, reason)
        return f"Escalated to a human colleague (escalation #{ticket}). Tell the customer a colleague will follow up."

    return escalate_to_human
