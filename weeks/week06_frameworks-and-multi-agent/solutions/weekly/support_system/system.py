"""The support system: triage -> billing / tech specialist, human approval for refunds, guards, durable state."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from common import llm, tracing
from common.agent import run_agent
from common.cache import ResponseCache
from common.chat import ToolCall

from .guards import CARD, check_reply, redact_cards, safe_reply
from .kb import KnowledgeBase
from .routing import rule_route
from .store import Store
from .tools import billing_tools, d5, tech_tools

RULES = (
    "Rules: tell the customer ONLY what your tool results say; never claim a refund or action was completed unless a tool says so; "
    "never promise outcomes; never ask for or repeat card numbers or passwords; ask for missing information (like an invoice id) before acting; "
    "if you cannot resolve it, use escalate_to_human. Keep replies to two or three sentences."
)
PROMPTS = {
    "billing": "You are a billing support specialist. You can look up invoices and request refunds. Request a refund only for a PAID invoice, after looking it up. "
    "If a refund is pending approval, say it is pending and that the billing team will review it. "
    + RULES,
    "tech": "You are a technical support specialist. Check service_status for outages and search_kb for how-to questions before answering. "
    "Quote the article id when you rely on it. " + RULES,
}
# The same jobs in about a third of the words (Week 7 Day 5: input tokens are paid on EVERY model call). The runtime
# guards still enforce the rules these prompts no longer spell out; whether a small model copes is a measurement.
LEAN_PROMPTS = {
    "billing": "Billing support. Look up the invoice, then request a refund only if it is paid. If approval is pending, say so. "
    "Say only what tools return. Ask for a missing invoice id. Escalate if stuck. Reply in 2-3 sentences.",
    "tech": "Technical support. Check service_status for outages and search_kb for how-to questions; quote the article id. "
    "Say only what tools return. Escalate if stuck. Reply in 2-3 sentences.",
}
PUBLIC_TOOLS = {
    "search_kb"
}  # answers built only from these are the same for every customer, so they may be cached
TRIAGE_EXAMPLES = (
    "\nExamples:\n"
    '- "My dashboard shows a 502 error since this morning" -> technical\n'
    '- "Why is my invoice higher than last month?" -> billing\n'
    '- "Can you connect me to a manager?" -> human\n'
    '- "Recommend a good pizza place" -> other\n'
    '- "I can\'t sign in after changing my phone" -> technical\n'
)  # deliberately UNLIKE any evaluation text: examples that appear in the test set would be leakage
TRIAGE_SYSTEM = (
    "You are the triage agent of a support team. Classify the customer's message: billing (charges, invoices, refunds), "
    "technical (errors, crashes, outages, how-to, login problems), human (the customer asks for a person or is angry or it is sensitive), "
    "other (anything not about our product)."
)
OUT_OF_SCOPE = "I can help with billing and technical questions about your account. What do you need help with?"
HUMAN_REPLY = "I've asked a human colleague to follow up with you."


class Route(BaseModel):
    category: Literal["billing", "technical", "human", "other"]
    reason: str = Field(description="one short sentence")


@dataclass
class Reply:
    text: str
    agent: str  # billing | tech | triage | human
    status: str = "ok"  # ok | escalated | out_of_scope | error
    violations: list[str] = field(default_factory=list)
    steps: int = 0
    cost_usd: float = 0.0
    calls: list[dict] = field(
        default_factory=list
    )  # the specialist's tool calls this turn: {name, args}
    cached: bool = False  # answered from the response cache: no specialist run, no model call
    trace_id: str | None = (
        None  # the trace of this turn (None when tracing is off): store it next to any feedback
    )


class SupportSystem:
    def __init__(self, db_path: str | Path, *, provider: str | None = None, model: str | None = None, clock=time.time, kb: KnowledgeBase | None = None,
                 guards: bool = True, max_steps: int = 6, max_cost_usd: float | None = None, drafter=None,
                 force_first_tool: str = "", triage_few_shot: bool = False, triage_mode: str = "llm", prefetch_invoice: bool = False,
                 lean: bool = False, hide_prefetched: bool = False, reply_from_tool: bool = False, cache: ResponseCache | None = None,
                 tool_wrapper=None, reply_filter=None):  # fmt: skip
        self.provider, self.model, self.guards, self.max_steps, self.max_cost_usd = (
            provider,
            model,
            guards,
            max_steps,
            max_cost_usd,
        )
        self.force_first_tool, self.triage_few_shot, self.triage_mode = (
            force_first_tool,
            triage_few_shot,
            triage_mode,
        )  # triage_mode: llm | rules | rules+llm
        self.prefetch_invoice = prefetch_invoice
        # cost levers (all off by default; each is measured in Week 7 Day 5)
        self.lean, self.hide_prefetched, self.reply_from_tool, self.cache = (
            lean,
            hide_prefetched,
            reply_from_tool,
            cache,
        )
        # security hooks (Week 8), both off by default: tool_wrapper(tools, conversation_id, agent) -> tools wraps every specialist's tools
        # (authorisation, argument rules); reply_filter(text) -> text sanitises the reply before the customer's client renders it
        self.tool_wrapper, self.reply_filter = tool_wrapper, reply_filter
        self.store = Store(db_path, clock=clock)
        self.refunds = d5.RefundService(db_path, clock=clock, drafter=drafter)
        self.kb = kb or KnowledgeBase()

    # ------------------------------------------------------------------ one customer message
    def handle(self, conversation_id: str, message: str) -> Reply:
        """One customer message. Wrapped in a root span (a no-op unless tracing is on) so every model call, tool call
        and guard decision of the turn hangs off one trace. Attributes: ids and outcomes, never the customer's text."""
        attrs = {
            tracing.OPERATION: "invoke_workflow",
            tracing.WORKFLOW_NAME: "support",
            tracing.CONVERSATION_ID: conversation_id,
        }
        with tracing.span("invoke_workflow support", attrs) as sp:
            reply = self._handle(conversation_id, message)
            reply.trace_id = tracing.current_trace_id()
            if reply.trace_id:
                self.store.log(conversation_id, "trace", trace_id=reply.trace_id)
            sp.set_attribute("app.reply.agent", reply.agent)
            sp.set_attribute("app.reply.status", reply.status)
            sp.set_attribute("app.reply.violations", reply.violations)
            sp.set_attribute(tracing.COST, reply.cost_usd)
            if reply.status == "escalated":
                tracing.set_error(sp, "escalated", "escalated to a human")
            return reply

    def _handle(self, conversation_id: str, message: str) -> Reply:
        notice = ""
        if CARD.search(message):  # the model never sees it and it is never stored
            message = redact_cards(message)
            notice = "For your security I removed the card number from your message; please don't share card numbers or passwords here. "
            self.store.log(conversation_id, "card_redacted")
        agent, history = self.store.get(conversation_id)
        spent = 0.0
        if agent is None:
            route = None
            asked_model = False
            with tracing.span("triage", {"app.triage.mode": self.triage_mode}) as tsp:
                if self.triage_mode in ("rules", "rules+llm"):
                    category, reason = rule_route(message)
                    if category is not None:
                        route = Route(category=category, reason=reason)
                    elif self.triage_mode == "rules":
                        route = Route(
                            category="other", reason=reason
                        )  # nothing matched and no model to ask
                if route is None:
                    asked_model = True
                    route, resp = llm.structured(
                        f"<message>\n{message}\n</message>",
                        Route,
                        system=TRIAGE_SYSTEM + (TRIAGE_EXAMPLES if self.triage_few_shot else ""),
                        provider=self.provider,
                        model=self.model,
                        max_tokens=200,
                    )
                    spent += resp.cost_usd
                tsp.set_attribute("app.triage.route", route.category)
                tsp.set_attribute("app.triage.asked_model", asked_model)
            self.store.log(conversation_id, "route", category=route.category, reason=route.reason)
            if route.category == "other":
                self.store.save(conversation_id, None, history)
                return Reply(notice + OUT_OF_SCOPE, "triage", "out_of_scope", cost_usd=spent)
            if route.category == "human":
                self.store.escalate(conversation_id, f"triage: {route.reason}")
                self.store.save(conversation_id, "human", history)
                return Reply(notice + HUMAN_REPLY, "human", "escalated", cost_usd=spent)
            agent = "billing" if route.category == "billing" else "tech"
        if agent == "human":
            return Reply(HUMAN_REPLY, "human", "escalated")
        if (
            self.max_cost_usd is not None
            and self._cost(conversation_id) + spent > self.max_cost_usd
        ):
            self.store.escalate(conversation_id, "cost budget reached")
            return Reply(HUMAN_REPLY, "human", "escalated", cost_usd=spent)

        prompt = (LEAN_PROMPTS if self.lean else PROMPTS)[agent]
        scope = f"{agent}|{hashlib.sha1(prompt.encode()).hexdigest()[:8]}|{self.model}"
        if (
            self.cache is not None and agent == "tech" and not history
        ):  # a first-turn how-to: same answer for everyone
            hit = self.cache.get(message, scope=scope)
            if hit is not None:
                self.store.log(conversation_id, "cache_hit", match=hit.kind)
                self.store.log(conversation_id, "turn", agent=agent, calls=[], cost=spent)
                self.store.save(
                    conversation_id,
                    agent,
                    [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": hit.value},
                    ],
                )
                return Reply(notice + hit.value, agent, "ok", [], 0, spent, [], cached=True)
        tools = (billing_tools if agent == "billing" else tech_tools)(self, conversation_id)
        if self.tool_wrapper is not None:
            tools = self.tool_wrapper(tools, conversation_id, agent)
        if self.lean:
            tools = tools.compact()
        prefetched: list[dict] = []
        if (
            self.prefetch_invoice and agent == "billing"
        ):  # verify in CODE what a weak model skips, and hand it the facts
            m = re.search(r"INV-\d+", message, re.I)
            if m:
                invoice = m.group().upper()
                result = tools.execute(
                    ToolCall("prefetch", "lookup_invoice", {"invoice_id": invoice})
                )
                message = f"{message}\n\n[Checked by the system: {result.content}]"
                prefetched = [{"name": "lookup_invoice", "args": {"invoice_id": invoice}}]
                self.store.log(conversation_id, "prefetch", invoice=invoice, result=result.content)
                if (
                    self.hide_prefetched
                ):  # the model has the facts: do not offer it the call it would repeat
                    tools = tools.without("lookup_invoice")
        run = run_agent(
            message,
            tools,
            agent_name=agent,
            system=prompt,
            provider=self.provider,
            model=self.model,
            messages=history,
            max_steps=self.max_steps,
            first_tool_choice=self._forced_tool(agent, message, history),
            stop_when=self._refund_outcome if self.reply_from_tool else None,
        )
        spent += run.cost_usd
        results = [r.content for r in run.results]
        if not run.ok or not run.answer:
            self.store.escalate(
                conversation_id, f"specialist {run.status}: {run.error or 'no answer'}"
            )
            self.store.log(conversation_id, "specialist_failed", agent=agent, status=run.status)
            self.store.save(conversation_id, agent, run.messages)
            return Reply(
                notice + HUMAN_REPLY, agent, "escalated", steps=len(run.steps), cost_usd=spent
            )
        guarded = check_reply(run.answer, results) if self.guards else None
        text = guarded.reply if guarded else run.answer
        violations = guarded.violations if guarded else []
        if self.reply_filter is not None:
            filtered = self.reply_filter(text)
            if filtered != text:
                violations = [*violations, "reply_filtered"]
                text = filtered
        for v in violations:
            self.store.log(conversation_id, "violation", violation=v, original=run.answer)
        if (
            self.cache is not None
            and not history
            and not violations
            and run.calls
            and {c.name for c in run.calls} <= PUBLIC_TOOLS
        ):  # only answers built from public knowledge: never a status that changes or a customer's own data
            self.cache.put(message, text, scope=scope)
        calls = prefetched + [{"name": c.name, "args": c.args} for c in run.calls]
        self.store.log(conversation_id, "turn", agent=agent, calls=calls, cost=spent)
        self.store.save(
            conversation_id, agent, run.messages + [{"role": "assistant", "content": text}]
        )
        return Reply(notice + text, agent, "ok", violations, len(run.steps), spent, calls)

    @staticmethod
    def _refund_outcome(run, step) -> str | None:
        """The refund tool's result IS the answer: build the customer reply from it in code instead of paying for another
        model call that would only rephrase it. Errors and every other tool still go back to the model."""
        for call, res in zip(step.calls, step.results, strict=True):
            if call.name == "request_refund" and not res.is_error:
                return safe_reply([res.content])
        return None

    def _forced_tool(self, agent: str, message: str, history: list[dict]) -> str | None:
        """Which tool (if any) to FORCE on the specialist's first step.
        ""    never force.   "always"  force some tool on every first step (even "thanks": a known trap).
        "smart"  only on the conversation's first specialist turn, and only when there is something to verify:
                 billing needs an invoice id in the message (then force lookup_invoice); tech always forces a tool."""
        if not self.force_first_tool:
            return None
        if self.force_first_tool == "always":
            return "required"
        if any(m.get("role") == "tool" for m in history):
            return None  # not the first specialist turn
        if agent == "billing":
            return "lookup_invoice" if re.search(r"INV-\d+", message, re.I) else None
        return "required"

    # ------------------------------------------------------------------ feedback from the customer
    FEEDBACK_REASONS = ("wrong", "unhelpful", "rude", "slow", "unsafe", "other")

    def feedback(
        self, conversation_id: str, rating: int, *, reason: str = "other", comment: str = ""
    ) -> dict:
        """Record a thumbs up (+1) or down (-1) on the conversation's latest reply. The free-text comment is customer
        input: card numbers are redacted and it is capped, like everything else we store. The latest trace id (if
        tracing was on) is attached, so a thumbs-down can be opened as a trace."""
        if rating not in (-1, 1):
            raise ValueError("rating must be +1 or -1")
        if reason not in self.FEEDBACK_REASONS:
            raise ValueError(f"reason must be one of {self.FEEDBACK_REASONS}")
        agent, history = self.store.get(conversation_id)
        if not history and agent is None and not self.store.events(conversation_id):
            raise ValueError(f"unknown conversation {conversation_id!r}")
        traces = self.store.events(conversation_id, "trace")
        record = {
            "rating": rating,
            "reason": reason,
            "comment": redact_cards(comment)[:500],
            "trace_id": traces[-1]["trace_id"] if traces else None,
            "turns": len(self.store.events(conversation_id, "turn")),
        }
        self.store.log(conversation_id, "feedback", **record)
        return record

    # ------------------------------------------------------------------ the human side
    def pending_approvals(self) -> list[dict]:
        return self.refunds.pending()

    def approve(self, conversation_id: str, invoice_id: str, decision: dict) -> dict:
        """A human decides a pending refund; the customer's conversation gets the outcome as the next assistant message."""
        result = self.refunds.decide(f"{conversation_id}-{invoice_id.strip().upper()}", decision)
        agent, history = self.store.get(conversation_id)
        history.append({"role": "assistant", "content": result["message"]})
        self.store.save(conversation_id, agent, history)
        self.store.log(
            conversation_id,
            "approval",
            invoice=invoice_id,
            outcome=result["status"],
            approver=decision.get("approver"),
        )
        return result

    def _cost(self, conversation_id: str) -> float:
        return sum(e.get("cost", 0.0) for e in self.store.events(conversation_id, "turn"))
