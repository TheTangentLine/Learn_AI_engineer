"""Tests for Week 6 Day 3: handoff vs delegation, guardrails (blocking vs parallel), hooks, sessions, shared limits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))

import day3_solution as d3  # noqa: E402
from fake_llm_server import FakeLLMServer  # noqa: E402

BILLING_FLOW = [
    {"text": "", "tool_calls": [{"name": "transfer_to_billing", "args": {}}]},
    {"text": "", "tool_calls": [{"name": "lookup_invoice", "args": {"invoice_id": "INV-1001"}}]},
    {"text": "INV-1001 was charged once; INV-1002 is a duplicate."},
]
TECH_FLOW = [
    {"text": "", "tool_calls": [{"name": "transfer_to_tech", "args": {}}]},
    {"text": "", "tool_calls": [{"name": "service_status", "args": {"service": "export"}}]},
    {"text": "The export service is degraded; ETA 2 hours."},
]


@pytest.fixture()
def server():
    with FakeLLMServer() as srv:
        yield srv


def url(srv):
    return srv.url + "/v1"


def tool_names(request):
    return sorted(t["function"]["name"] for t in request["body"].get("tools", []))


# ----------------------------------------------------------------------------- the domain functions


def test_domain_tools_return_readable_results_and_errors():
    assert d3.lookup_invoice("inv-1001") == "INV-1001: $49.00, paid, 2026-09-01, ada@example.com"
    assert "No invoice 'INV-9'" in d3.lookup_invoice("INV-9") and "INV-1001" in d3.lookup_invoice(
        "INV-9"
    )
    assert d3.service_status(" Export ").startswith(
        "degraded"
    ) and "Known: export, login, api" in d3.service_status("sms")
    assert d3.INVOICES["INV-1001"]["amount"] == d3.INVOICES["INV-1002"]["amount"], (
        "the fixture holds a duplicate charge"
    )


def test_the_card_number_pattern():
    for hit in (
        "4111 1111 1111 1111",
        "4111-1111-1111-1111",
        "4111111111111111",
        "card 5500 0000 0000 0004 ok",
    ):
        assert d3.CARD.search(hit), hit
    for miss in ("INV-1001", "order 12345", "call 555 1234", "2026-09-01", "version 3.14.159"):
        assert not d3.CARD.search(miss), miss


# ----------------------------------------------------------------------------- OpenAI Agents SDK: handoff


def test_handoff_transfers_control_and_the_specialist_writes_the_answer(server):
    server.script = list(BILLING_FLOW)
    r = d3.run_agents_sdk("I was charged twice for INV-1001", url(server))
    assert r.answer == BILLING_FLOW[-1]["text"] and r.handled_by == "Billing" and r.requests == 3
    assert r.path == ["Triage", "Billing", "lookup_invoice"]
    triage_req, billing_req1, billing_req2 = server.requests
    assert tool_names(triage_req) == ["transfer_to_billing", "transfer_to_tech"], (
        "triage can only hand off"
    )
    assert tool_names(billing_req1) == ["lookup_invoice"], (
        "the specialist has its own tools, not the handoff tools"
    )
    assert "billing specialist" in json.dumps(billing_req1["body"]["messages"][0])
    assert "charged twice" in json.dumps(billing_req1["body"]["messages"]), (
        "the conversation history travels with the handoff"
    )


def test_handoff_to_the_other_specialist(server):
    server.script = list(TECH_FLOW)
    r = d3.run_agents_sdk("The export screen keeps crashing", url(server))
    assert (
        r.handled_by == "Tech"
        and r.path == ["Triage", "Tech", "service_status"]
        and "degraded" in r.answer
    )
    assert tool_names(server.requests[1]) == ["service_status"]


def test_a_handoff_to_an_agent_that_does_not_exist_is_reported_not_raised(server):
    server.script = [{"text": "", "tool_calls": [{"name": "transfer_to_sales", "args": {}}]}]
    r = d3.run_agents_sdk("hello", url(server))
    assert r.blocked == "error: ModelBehaviorError" and r.answer == ""


def test_a_triage_agent_that_answers_itself_is_not_handed_off(server):
    server.script = [{"text": "Hello! How can I help?"}]
    r = d3.run_agents_sdk("hi", url(server))
    assert r.handled_by == "Triage" and r.path == ["Triage"] and r.requests == 1


def test_run_hooks_see_the_handoff_and_the_tool_calls(server):
    from agents import Runner

    server.script = list(BILLING_FLOW)
    events = []
    triage, hooks = d3.build_agents_sdk_app(url(server), events=events)
    Runner.run_sync(triage, "charged twice INV-1001", hooks=hooks)
    assert events == [("handoff", "Triage", "Billing"), ("tool", "Billing", "lookup_invoice")]


# ----------------------------------------------------------------------------- guardrails


def test_a_card_number_is_blocked_and_a_clean_message_is_not(server):
    r = d3.run_agents_sdk("My card number is 4111 1111 1111 1111, please update it.", url(server))
    assert r.blocked == "no_card_numbers" and r.answer == "" and server.requests == []
    server.script = [{"text": "Hello!"}]
    ok = d3.run_agents_sdk("hello there", url(server))
    assert ok.blocked == "" and ok.answer == "Hello!"


def test_a_blocking_guardrail_spends_no_tokens_but_a_parallel_slow_one_does(server):
    msg = "My card is 4111 1111 1111 1111"
    blocking = d3.run_agents_sdk(msg, url(server), guardrail_parallel=False, guardrail_delay_s=0.4)
    assert blocking.blocked == "no_card_numbers" and len(server.requests) == 0
    parallel = d3.run_agents_sdk(msg, url(server), guardrail_parallel=True, guardrail_delay_s=0.4)
    assert parallel.blocked == "no_card_numbers", "still blocked ..."
    assert len(server.requests) == 1, (
        "... but the model call was already made: you paid for the blocked input"
    )


# ----------------------------------------------------------------------------- sessions


def test_a_session_carries_history_into_the_next_turn_and_survives_a_restart(server, tmp_path):
    from agents import SQLiteSession

    db = tmp_path / "chat.sqlite"
    server.script = [{"text": "Hello Ada!"}, {"text": "You told me your name is Ada."}]
    d3.run_agents_sdk("Hi, my name is Ada", url(server), session=SQLiteSession("s1", db))
    r = d3.run_agents_sdk(
        "What is my name?", url(server), session=SQLiteSession("s1", db)
    )  # a NEW session object: a 'restart'
    second = json.dumps(server.requests[1]["body"]["messages"])
    assert "my name is Ada" in second and "Hello Ada!" in second and "What is my name?" in second
    assert r.answer == "You told me your name is Ada."


def test_sessions_are_isolated_by_id(server, tmp_path):
    from agents import SQLiteSession

    db = tmp_path / "chat.sqlite"
    server.script = [{"text": "ok A"}, {"text": "ok B"}]
    d3.run_agents_sdk("secret of customer A", url(server), session=SQLiteSession("A", db))
    d3.run_agents_sdk("hello from B", url(server), session=SQLiteSession("B", db))
    assert "customer A" not in json.dumps(server.requests[1]["body"]["messages"])


# ----------------------------------------------------------------------------- PydanticAI: delegation


DELEGATION_FLOW = [
    {"text": "", "tool_calls": [{"name": "ask_billing", "args": {"question": "check INV-1001"}}]},
    {"text": "", "tool_calls": [{"name": "lookup_invoice", "args": {"invoice_id": "INV-1001"}}]},
    {
        "text": "INV-1001 was paid once."
    },  # the billing agent's answer, returned to triage as a TOOL RESULT
    {"text": "Good news: your invoice INV-1001 was charged only once."},  # triage writes the reply
]


def test_delegation_keeps_the_parent_in_charge_and_isolates_the_specialists_context(server):
    server.script = list(DELEGATION_FLOW)
    r = d3.run_pydantic_ai("I was charged twice for INV-1001", url(server))
    assert (
        r.handled_by == "Triage"
        and r.answer.startswith("Good news")
        and r.path == ["Triage", "Billing"]
        and r.requests == 4
    )
    sub_request = json.dumps(server.requests[1]["body"]["messages"])
    assert "check INV-1001" in sub_request and "charged twice" not in sub_request, (
        "the specialist saw only the question it was asked"
    )
    assert "billing specialist" in sub_request
    final_request = json.dumps(server.requests[3]["body"]["messages"])
    assert "INV-1001 was paid once." in final_request, (
        "the specialist's answer came back as a tool result"
    )


def test_one_usage_limit_covers_the_parent_and_the_delegated_agent(server):
    server.script = list(DELEGATION_FLOW)
    blocked = d3.run_pydantic_ai("charged twice INV-1001", url(server), request_limit=3)
    assert blocked.blocked.startswith("limit:") and blocked.answer == ""
    assert len(server.requests) <= 3, "the cap counted the sub-agent's requests too"
    server.requests.clear()
    server.script = list(DELEGATION_FLOW)
    assert d3.run_pydantic_ai("charged twice INV-1001", url(server), request_limit=4).blocked == ""


def test_pydantic_ai_2x_counts_a_delegated_agents_requests_even_without_passing_usage_explicitly(
    server,
):
    """Finding: in PydanticAI 2.x a nested run inside a tool INHERITS the parent's usage and limits. Older versions needed
    ``usage=ctx.usage``; passing it explicitly is harmless and documents the intent, but it is no longer what makes the cap work."""
    for share in (True, False):
        server.requests.clear()
        server.script = list(DELEGATION_FLOW)
        r = d3.run_pydantic_ai(
            "charged twice INV-1001", url(server), request_limit=3, share_usage=share
        )
        assert r.blocked.startswith("limit:") and len(server.requests) == 3, share
        server.requests.clear()
        server.script = list(DELEGATION_FLOW)
        ok = d3.run_pydantic_ai(
            "charged twice INV-1001", url(server), request_limit=10, share_usage=share
        )
        assert ok.blocked == "" and ok.requests == 4, (
            "the parent's reported usage includes the specialist's 2 requests"
        )


def test_handoff_and_delegation_differ_in_who_writes_the_final_answer(server):
    server.script = list(BILLING_FLOW)
    handoff = d3.run_agents_sdk("charged twice INV-1001", url(server))
    server.script = list(DELEGATION_FLOW)
    delegation = d3.run_pydantic_ai("charged twice INV-1001", url(server))
    assert (handoff.handled_by, delegation.handled_by) == ("Billing", "Triage")
    assert handoff.requests == 3 and delegation.requests == 4, (
        "delegation costs one extra model call: the parent re-writes the answer"
    )


# ----------------------------------------------------------------------------- Claude Agent SDK: subagents (built only)


def test_claude_agent_options_define_two_subagents_with_least_privilege_tools():
    opts = d3.claude_agent_options(max_turns=5, max_budget_usd=0.25)
    assert (
        set(opts.agents) == {"billing", "tech"}
        and opts.max_turns == 5
        and opts.max_budget_usd == 0.25
    )
    assert opts.agents["billing"].tools == ["mcp__support__lookup_invoice"] and opts.agents[
        "tech"
    ].tools == ["mcp__support__service_status"]
    assert (
        "billing specialist" in opts.agents["billing"].prompt
        and "technical specialist" in opts.agents["tech"].prompt
    )
    assert set(opts.mcp_servers) == {"support"} and "triage" in opts.system_prompt.lower()
