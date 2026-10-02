"""Week 6 Day 3 - Solution: the same two-agent support app on three SDKs.

Domain: a TRIAGE agent decides who should handle a message; a BILLING specialist (tool: lookup_invoice) or a TECH
specialist (tool: service_status) answers it. Three ways to express "triage, then a specialist":

  OpenAI Agents SDK   HANDOFF: control TRANSFERS to the specialist, which replies to the user directly.
                      Plus: an input guardrail (card numbers), run hooks (who handled what), SQLite sessions (memory).
  PydanticAI          DELEGATION: the triage agent calls the specialist as a TOOL and stays in charge of the reply.
                      Usage is shared across agents so one limit covers the whole conversation.
  Claude Agent SDK    SUBAGENTS: ``AgentDefinition``s the main agent can delegate to. Built and unit-tested, NOT run
                      (it needs the Claude Code CLI and Anthropic credentials).

The difference that matters is handoff (the specialist takes over the conversation) versus delegation (the parent keeps
it and receives a result): it decides who sees the history, who writes the final answer, and where guardrails apply.

  uv run python weeks/week06_frameworks-and-multi-agent/solutions/day3_solution.py       # local Qwen via LocalOpenAIServer
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # must be set before pydantic_ai is imported
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# module level on purpose: PydanticAI resolves tool annotations by NAME, so RunContext must be a module global
from pydantic_ai import RunContext  # noqa: E402

INVOICES = {
    "INV-1001": {
        "amount": 49.0,
        "status": "paid",
        "date": "2026-09-01",
        "customer": "ada@example.com",
    },
    "INV-1002": {
        "amount": 49.0,
        "status": "paid",
        "date": "2026-09-01",
        "customer": "ada@example.com",
    },  # a duplicate charge
    "INV-2001": {
        "amount": 120.0,
        "status": "open",
        "date": "2026-09-15",
        "customer": "bob@example.com",
    },
}
SERVICES = {
    "export": "degraded: the export queue is backed up (ETA 2h)",
    "login": "operational",
    "api": "operational",
}
CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")

TRIAGE_PROMPT = (
    "You are the triage agent of a support team. Decide who should handle the customer's message and hand off to "
    "that specialist: Billing for charges, invoices and refunds; Tech for errors, crashes and outages. Do not answer yourself."
)
BILLING_PROMPT = "You are a billing specialist. Use lookup_invoice to check an invoice before answering. You cannot approve refunds."
TECH_PROMPT = (
    "You are a technical specialist. Use service_status to check a service before answering."
)


def lookup_invoice(invoice_id: str) -> str:
    """Look up an invoice by id, e.g. INV-1001. Returns amount, status, date and customer."""
    inv = INVOICES.get(invoice_id.strip().upper())
    if not inv:
        return f"No invoice {invoice_id!r}. Known ids look like INV-1001."
    return f"{invoice_id.upper()}: ${inv['amount']:.2f}, {inv['status']}, {inv['date']}, {inv['customer']}"


def service_status(service: str) -> str:
    """Current status of a service: export, login or api."""
    return SERVICES.get(
        service.strip().lower(), f"Unknown service {service!r}. Known: {', '.join(SERVICES)}."
    )


@dataclass
class AppResult:
    framework: str
    answer: str = ""
    handled_by: str = ""  # the agent that wrote the final answer (handoff) or None for delegation
    path: list[str] = field(default_factory=list)  # agents/tools touched, in order
    blocked: str = ""  # a guardrail reason, if the input was refused
    requests: int = 0


# ============================================================================ OpenAI Agents SDK: handoffs


def build_agents_sdk_app(
    base_url: str,
    *,
    guardrail_parallel: bool = False,
    guardrail_delay_s: float = 0.0,
    events: list | None = None,
):
    from agents import (
        Agent,
        GuardrailFunctionOutput,
        OpenAIChatCompletionsModel,
        RunHooks,
        function_tool,
        input_guardrail,
        set_tracing_disabled,
    )
    from openai import AsyncOpenAI

    set_tracing_disabled(True)
    model = OpenAIChatCompletionsModel(
        model="local-qwen", openai_client=AsyncOpenAI(base_url=base_url, api_key="local")
    )

    @function_tool(name_override="lookup_invoice")
    def lookup_invoice_tool(invoice_id: str) -> str:
        """Look up an invoice by id, e.g. INV-1001. Returns amount, status, date and customer."""
        return lookup_invoice(invoice_id)

    @function_tool(name_override="service_status")
    def service_status_tool(service: str) -> str:
        """Current status of a service: export, login or api."""
        return service_status(service)

    @input_guardrail(name="no_card_numbers", run_in_parallel=guardrail_parallel)
    async def no_card_numbers(ctx, agent, user_input) -> GuardrailFunctionOutput:
        if guardrail_delay_s:
            await asyncio.sleep(guardrail_delay_s)  # a slow check (say, a classifier call)
        text = (
            user_input
            if isinstance(user_input, str)
            else " ".join(str(i.get("content", "")) for i in user_input if isinstance(i, dict))
        )
        found = bool(CARD.search(text))
        return GuardrailFunctionOutput(output_info={"card_number": found}, tripwire_triggered=found)

    billing = Agent(
        name="Billing",
        handoff_description="Charges, invoices, refunds.",
        instructions=BILLING_PROMPT,
        model=model,
        tools=[lookup_invoice_tool],
    )
    tech = Agent(
        name="Tech",
        handoff_description="Errors, crashes, outages.",
        instructions=TECH_PROMPT,
        model=model,
        tools=[service_status_tool],
    )
    triage = Agent(
        name="Triage",
        instructions=TRIAGE_PROMPT,
        model=model,
        handoffs=[billing, tech],
        input_guardrails=[no_card_numbers],
    )

    class Log(RunHooks):
        async def on_handoff(self, context, from_agent, to_agent):
            if events is not None:
                events.append(("handoff", from_agent.name, to_agent.name))

        async def on_tool_start(self, context, agent, tool):
            if events is not None:
                events.append(("tool", agent.name, tool.name))

    return triage, Log()


def run_agents_sdk(
    message: str,
    base_url: str,
    *,
    session=None,
    guardrail_parallel: bool = False,
    guardrail_delay_s: float = 0.0,
    max_turns: int = 6,
) -> AppResult:
    from agents import InputGuardrailTripwireTriggered, MaxTurnsExceeded, ModelBehaviorError, Runner

    events: list = []
    triage, hooks = build_agents_sdk_app(
        base_url,
        guardrail_parallel=guardrail_parallel,
        guardrail_delay_s=guardrail_delay_s,
        events=events,
    )
    out = AppResult("agents_sdk")
    try:
        result = Runner.run_sync(triage, message, hooks=hooks, session=session, max_turns=max_turns)
    except InputGuardrailTripwireTriggered as exc:
        out.blocked = exc.guardrail_result.guardrail.get_name()
        out.path = [e[-1] for e in events]
        return out
    except (MaxTurnsExceeded, ModelBehaviorError) as exc:
        out.blocked = f"error: {type(exc).__name__}"
        return out
    out.answer, out.handled_by = str(result.final_output or ""), result.last_agent.name
    out.path = ["Triage", *[e[2] if e[0] == "handoff" else e[2] for e in events]]
    out.requests = len(result.raw_responses)
    return out


# ============================================================================ PydanticAI: delegation


def build_pydantic_ai_app(base_url: str, *, share_usage: bool = True):
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    model = OpenAIChatModel(
        "local-qwen", provider=OpenAIProvider(base_url=base_url, api_key="local")
    )
    billing = Agent(model, system_prompt=BILLING_PROMPT, tools=[lookup_invoice])
    tech = Agent(model, system_prompt=TECH_PROMPT, tools=[service_status])
    triage = Agent(
        model,
        system_prompt=(
            "You are the triage agent of a support team. For billing questions call ask_billing; for errors, crashes "
            "and outages call ask_tech. Then reply to the customer using the specialist's answer."
        ),
    )
    path: list[str] = []

    @triage.tool
    async def ask_billing(ctx: RunContext[None], question: str) -> str:
        """Ask the billing specialist about charges, invoices or refunds. Returns their answer."""
        path.append("Billing")
        r = await billing.run(
            question, usage=ctx.usage
        )  # shared usage: ONE limit covers the whole conversation
        return str(r.output)

    @triage.tool
    async def ask_tech(ctx: RunContext[None], question: str) -> str:
        """Ask the technical specialist about errors, crashes or outages. Returns their answer."""
        path.append("Tech")
        r = await tech.run(question, usage=ctx.usage if share_usage else None)
        return str(r.output)

    return triage, path


def run_pydantic_ai(
    message: str, base_url: str, *, request_limit: int = 8, share_usage: bool = True
) -> AppResult:
    from pydantic_ai import UsageLimits
    from pydantic_ai.exceptions import UsageLimitExceeded

    triage, path = build_pydantic_ai_app(base_url, share_usage=share_usage)
    out = AppResult("pydantic_ai")
    try:
        result = triage.run_sync(message, usage_limits=UsageLimits(request_limit=request_limit))
    except UsageLimitExceeded as exc:
        out.blocked = f"limit: {str(exc)[:60]}"
        out.path = ["Triage", *path]
        return out
    out.answer, out.handled_by, out.path, out.requests = (
        str(result.output),
        "Triage",
        ["Triage", *path],
        result.usage.requests,
    )
    return out


# ============================================================================ Claude Agent SDK: subagents (built, NOT run)


def claude_agent_options(*, max_turns: int = 8, max_budget_usd: float = 0.50):
    """Options for a main agent with two subagents. Needs the Claude Code CLI + credentials to RUN; only built here."""
    from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, create_sdk_mcp_server
    from claude_agent_sdk import tool as sdk_tool

    @sdk_tool("lookup_invoice", "Look up an invoice by id, e.g. INV-1001.", {"invoice_id": str})
    async def invoice(args: dict) -> dict:
        return {"content": [{"type": "text", "text": lookup_invoice(args["invoice_id"])}]}

    @sdk_tool(
        "service_status", "Current status of a service: export, login or api.", {"service": str}
    )
    async def status(args: dict) -> dict:
        return {"content": [{"type": "text", "text": service_status(args["service"])}]}

    server = create_sdk_mcp_server("support", tools=[invoice, status])
    return ClaudeAgentOptions(
        system_prompt=TRIAGE_PROMPT,
        mcp_servers={"support": server},
        agents={
            "billing": AgentDefinition(
                description="Handles charges, invoices and refunds.",
                prompt=BILLING_PROMPT,
                tools=["mcp__support__lookup_invoice"],
            ),
            "tech": AgentDefinition(
                description="Handles errors, crashes and outages.",
                prompt=TECH_PROMPT,
                tools=["mcp__support__service_status"],
            ),
        },
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
    )


# ============================================================================ demo

MESSAGES = [
    "I was charged twice for invoice INV-1001, can you check it?",
    "The export screen keeps crashing, is something down?",
    "My card number is 4111 1111 1111 1111, please update it.",
]


def main() -> None:
    from agents import SQLiteSession

    from common.local_server import LocalOpenAIServer

    with LocalOpenAIServer() as srv:
        print("== OpenAI Agents SDK (handoff)")
        for m in MESSAGES:
            r = run_agents_sdk(m, srv.url)
            print(
                f"  {m[:48]:48s} -> handled_by={r.handled_by or '-':8s} path={r.path} blocked={r.blocked or '-'} answer={r.answer[:60]!r}"
            )
        print("\n== PydanticAI (delegation)")
        for m in MESSAGES[:2]:
            r = run_pydantic_ai(m, srv.url)
            print(
                f"  {m[:48]:48s} -> path={r.path} blocked={r.blocked or '-'} answer={r.answer[:60]!r}"
            )
        _ = SQLiteSession, asyncio


if __name__ == "__main__":
    main()
