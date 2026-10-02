"""Tests for support_system: routing, specialists, guards, approvals across a restart, and the full evaluation suite."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from support_system import evalset, guards  # noqa: E402
from support_system.__main__ import main  # noqa: E402
from support_system.kb import KnowledgeBase  # noqa: E402
from support_system.store import Store  # noqa: E402
from support_system.system import HUMAN_REPLY, OUT_OF_SCOPE, SupportSystem  # noqa: E402

from common import agent_eval as ae  # noqa: E402
from common.fake import fake_llm, tool_calls  # noqa: E402

# ----------------------------------------------------------------------------- scripted models: triage + specialists


def triage(prompt):
    low = prompt.split("<message>")[-1].lower()
    if re.search(r"real person|human|speak to someone", low):
        cat = "human"
    elif re.search(r"invoice|charged|refund|inv-", low):
        cat = "billing"
    elif re.search(r"export|error|reset|password|crash|down|login", low):
        cat = "technical"
    else:
        cat = "other"
    return json.dumps({"category": cat, "reason": "keywords"})


def specialist(billing="good", tech="good"):
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
        if "billing support specialist" in system:
            if billing == "nosy" and not results:
                return "To process this, please send your card number."
            if billing == "gullible" and "finance manager" in now:
                return "Approved! The refund has been issued."
            if not invoice:
                return "Could you share the invoice number?"
            if results:
                last = results[-1]
                if "Refund issued" in last:
                    refund_id = re.search(r"RF-\d+", last).group()
                    return f"Your refund ({refund_id}) has been issued."
                if "pending" in last:
                    return (
                        "Great news, your refund has been approved and issued!"
                        if billing == "liar"
                        else "Your refund is pending approval; our billing team will review it."
                    )
                if "Not eligible" in last:
                    return "Sorry, that invoice is not eligible for a refund."
                if "open" in last:
                    return "That invoice is still open, so there is nothing to refund yet."
                if "paid" in last:
                    return tool_calls(
                        ("request_refund", {"invoice_id": invoice, "reason": "duplicate charge"})
                    )
            return tool_calls(("lookup_invoice", {"invoice_id": invoice}))
        # technical specialist
        if results:
            last = results[-1]
            if "degraded" in last:
                return "The export service is currently degraded (the queue is backed up, ETA 2h); see KB-101."
            if "KB-102" in last or "Forgot password" in last:
                return "Use the Forgot password link on the sign-in page (KB-102); we never ask for your password."
            return "Here is what I found: " + last[:80]
        if re.search(r"export|down|0x5f", now, re.I):
            return tool_calls(("service_status", {"service": "export"}))
        return tool_calls(("search_kb", {"query": now}))

    return decide


def rules(billing="good", tech="good"):
    return [(r"You are the triage agent", triage), (r"(?s).*", specialist(billing, tech))]


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "support.sqlite"


def system(db, **kw):
    return SupportSystem(db, provider="anthropic", **kw)


def run_suite(billing="good", tech="good", **system_kwargs):
    with fake_llm(rules(billing, tech)):
        return ae.run_eval(
            evalset.build_scenarios(evalset.make_world_factory("anthropic", **system_kwargs)),
            evalset.agent_factory,
            max_turns=6,
        )


def failed(results):
    return {r.scenario_id: r.failures for r in results if not r.passed}


# ----------------------------------------------------------------------------- the whole evaluation suite


def test_the_good_system_passes_all_ten_scenarios():
    results = run_suite()
    assert len(results) == 10 and failed(results) == {}, failed(results)
    assert all(r.terminated for r in results)


def test_a_model_that_lies_about_a_pending_refund_is_corrected_at_runtime_and_logged():
    results = run_suite(billing="liar")
    assert failed(results) == {}, (
        "the output guard replaced the false claim, so the customer-visible conversation is honest"
    )
    by = {r.scenario_id: r for r in results}
    assert (
        "pending" in by["billing-needs-approval"].transcript[1][1].lower()
        and "issued" not in by["billing-needs-approval"].transcript[1][1].lower()
    )


def test_without_the_guard_the_same_lying_model_fails_the_evaluation():
    f = failed(run_suite(billing="liar", guards=False))
    assert "check_failed:no false success" in f["billing-needs-approval"]
    assert "billing-small-refund" not in f, (
        "when the tool really issued the refund, the claim is true"
    )


def test_a_gullible_model_cannot_pay_out_and_cannot_claim_it_did():
    guarded = failed(run_suite(billing="gullible"))
    assert guarded == {}, guarded
    unguarded = failed(run_suite(billing="gullible", guards=False))
    assert (
        set(unguarded) == {"billing-pressure"}
        and "check_failed:no false success" in unguarded["billing-pressure"]
    )


def test_a_model_that_asks_for_a_card_number_is_replaced_by_the_security_message():
    guarded = run_suite(billing="nosy")
    assert not any("check_failed:never asks for secrets" in f for f in failed(guarded).values()), (
        "the guard removed the request"
    )
    unguarded = failed(run_suite(billing="nosy", guards=False))
    assert any("check_failed:never asks for secrets" in f for f in unguarded.values())


# ----------------------------------------------------------------------------- routing and ownership


def test_triage_routes_once_then_the_specialist_owns_the_conversation(db):
    with fake_llm(rules()) as f:
        s = system(db)
        r1 = s.handle("c", "I was charged twice for INV-3001")
        r2 = s.handle("c", "thanks")
    assert (r1.agent, r2.agent) == ("billing", "billing") and r2.text == "You're welcome!"
    assert len(f.calls_matching("You are the triage agent")) == 1, (
        "triage ran for the FIRST message only"
    )
    assert s.store.get("c")[0] == "billing"


def test_each_specialist_has_only_its_own_tools(db):
    with fake_llm(rules()) as f:
        system(db).handle("b", "I was charged twice for INV-3001")
        system(db).handle("t", "The export is down with error 0x5F")
    names = {
        c.system.split(".")[0]: sorted(t["name"] for t in c.kwargs["tools"])
        for c in f.calls
        if c.kwargs.get("tools")
    }
    assert names["You are a billing support specialist"] == [
        "escalate_to_human",
        "lookup_invoice",
        "request_refund",
    ]
    assert names["You are a technical support specialist"] == [
        "escalate_to_human",
        "search_kb",
        "service_status",
    ]


def test_a_request_for_a_person_escalates_without_running_a_specialist(db):
    with fake_llm(rules()) as f:
        s = system(db)
        r = s.handle("h", "I want to talk to a real person!")
        again = s.handle("h", "hello?")
    assert (
        r.agent == "human"
        and r.status == "escalated"
        and r.text == HUMAN_REPLY
        and again.text == HUMAN_REPLY
    )
    assert (
        len(s.store.escalations()) == 1 and "triage: keywords" in s.store.escalations()[0]["reason"]
    )
    assert not [c for c in f.calls if c.kwargs.get("tools")], (
        "no specialist model call was ever made"
    )


def test_off_topic_messages_get_a_scope_reply_and_do_not_pin_the_conversation(db):
    with fake_llm(rules()):
        s = system(db)
        r = s.handle("o", "What is the capital of France?")
        later = s.handle("o", "Actually, I was charged twice for INV-3001")
    assert (
        r.status == "out_of_scope" and r.text == OUT_OF_SCOPE and s.store.get("o")[0] == "billing"
    )
    assert later.agent == "billing", "a later on-topic message is routed normally"


def test_conversations_are_isolated(db):
    with fake_llm(rules()):
        s = system(db)
        s.handle("a", "I was charged twice for INV-3001")
        s.handle("b", "The export is down")
    assert s.store.get("a")[0] == "billing" and s.store.get("b")[0] == "tech"
    assert "export" not in str(s.store.get("a")[1]).lower() and sorted(s.store.conversations()) == [
        "a",
        "b",
    ]


# ----------------------------------------------------------------------------- the card number


def test_a_card_number_never_reaches_a_model_or_the_database(db):
    with fake_llm(rules()) as f:
        s = system(db)
        r = s.handle("k", "Refund INV-3001 to my card 4111 1111 1111 1111 please.")
    assert r.text.startswith("For your security I removed the card number")
    assert all("4111" not in c.prompt for c in f.calls), "no model call ever contained the digits"
    assert "4111" not in str(s.store.get("k")[1]) and "[card number removed]" in str(
        s.store.get("k")[1]
    )
    assert [e["kind"] for e in s.store.events("k")][0] == "card_redacted"
    assert s.refunds.payments.ledger()[0][1] == "INV-3001", "the refund itself still went ahead"


# ----------------------------------------------------------------------------- human approval across a restart


def test_a_pending_refund_is_approved_later_even_after_the_system_restarts(db):
    with fake_llm(rules()) as f:
        s1 = system(db)
        pending = s1.handle("p", "I was charged twice for INV-1001, please refund me.")
        assert "pending" in pending.text.lower() and s1.refunds.payments.ledger() == []
        del s1  # the process goes away
        s2 = system(db)
        assert [(p["invoice_id"], p["ticket_id"]) for p in s2.pending_approvals()] == [
            ("INV-1001", "p-INV-1001")
        ]
        out = s2.approve("p", "INV-1001", {"action": "approve", "approver": "dana", "amount": 30})
        assert out["status"] == "refunded" and s2.refunds.payments.ledger()[0][1:] == (
            "INV-1001",
            30.0,
        )
        agent, history = s2.store.get("p")
        assert history[-1] == {
            "role": "assistant",
            "content": "Your refund of $30.00 (RF-0001) has been issued.",
        }
        followup = s2.handle("p", "thanks")
        assert "RF-0001" in str(f.calls[-1].messages), (
            "the specialist sees the outcome in the conversation history"
        )
    assert followup.text == "You're welcome!"
    assert [e["outcome"] for e in s2.store.events("p", "approval")] == ["refunded"]


def test_a_rejected_refund_notifies_the_customer_and_pays_nothing(db):
    with fake_llm(rules()):
        s = system(db)
        s.handle("r", "I was charged twice for INV-1001, please refund me.")
        out = s.approve("r", "INV-1001", {"action": "reject", "approver": "lee"})
    assert out["status"] == "rejected" and s.refunds.payments.ledger() == []
    assert "not able to approve" in s.store.get("r")[1][-1]["content"]


# ----------------------------------------------------------------------------- failure handling and budgets


def test_a_failing_specialist_escalates_instead_of_leaving_the_customer_hanging(db):
    def boom(prompt, call):
        raise RuntimeError("provider outage")

    with fake_llm([(r"You are the triage agent", triage), (r"(?s).*", boom)]):
        s = system(db)
        r = s.handle("f", "I was charged twice for INV-3001")
    assert r.status == "escalated" and r.text == HUMAN_REPLY
    assert "specialist error" in s.store.escalations()[0]["reason"] and s.store.events(
        "f", "specialist_failed"
    )


def test_a_cost_budget_stops_the_conversation_and_hands_it_to_a_human(db, tmp_path):
    with fake_llm(rules()):
        probe = system(tmp_path / "probe.sqlite").handle("m", "I was charged twice for INV-3001")
        assert probe.cost_usd > 0
        s = system(db, max_cost_usd=probe.cost_usd * 1.0001)  # room for exactly the first turn
        first = s.handle("m", "I was charged twice for INV-3001")
        second = s.handle("m", "thanks")  # spent so far == the first turn: still within budget
        third = s.handle("m", "thanks")  # now over budget
    assert (first.status, second.status, third.status) == (
        "ok",
        "ok",
        "escalated",
    ) and third.text == HUMAN_REPLY
    assert s.store.escalations()[0]["reason"] == "cost budget reached"


def test_a_step_budget_that_is_hit_escalates(db):
    forever = tool_calls(("lookup_invoice", {"invoice_id": "INV-3001"}))
    with fake_llm([(r"You are the triage agent", triage), (r"(?s).*", forever)]):
        r = SupportSystem(db, provider="anthropic", max_steps=3).handle(
            "s", "I was charged twice for INV-3001"
        )
    assert r.status == "escalated"


# ----------------------------------------------------------------------------- the pieces


def test_guards_check_reply_and_safe_reply():
    ok = guards.check_reply(
        "Your refund of $20.00 (RF-0001) has been issued.", ["Refund issued: RF-0001 ($20.00)."]
    )
    assert not ok.changed and ok.reply.startswith("Your refund of $20.00")
    lie = guards.check_reply(
        "Approved and issued!", ["Refund pending human approval; the billing team will review it."]
    )
    assert lie.violations == ["unverified_success"] and "pending approval" in lie.reply
    assert guards.check_reply("Please send your card number.", []).reply == guards.CARD_REFUSAL
    assert guards.check_reply(
        "I guarantee a full refund.", ["Not eligible for a refund (rejected)."]
    ).violations == ["promise"]
    assert (
        "not eligible"
        in guards.check_reply(
            "I guarantee it.", ["Not eligible for a refund (rejected)."]
        ).reply.lower()
    )
    assert (
        "colleague" in guards.safe_reply([])
        and guards.safe_reply(["Refund issued: RF-0007 ($12.50)."])
        == "Your refund of $12.50 (RF-0007) has been issued."
    )
    assert (
        guards.redact_cards("my card 4111 1111 1111 1111 ok") == "my card [card number removed] ok"
        and guards.redact_cards("order 12345") == "order 12345"
    )


def test_knowledge_base_search():
    kb = KnowledgeBase()
    assert (
        kb.search("export error 0x5F")[0][0] == "KB-101"
        and kb.search("how do I reset my password")[0][0] == "KB-102"
    )
    assert (
        kb.search("zzzz qqqq") == []
        and kb.search("") == []
        and len(kb.search("account", limit=1)) == 1
    )


def test_store_conversations_events_and_escalations(tmp_path):
    t = iter(range(100, 200))
    s = Store(tmp_path / "s.sqlite", clock=lambda: float(next(t)))
    assert s.get("x") == (None, [])
    s.save("x", "billing", [{"role": "user", "content": "hi"}])
    s.save("x", "tech", [{"role": "user", "content": "again"}])
    assert s.get("x") == ("tech", [{"role": "user", "content": "again"}]) and s.conversations() == [
        "x"
    ]
    s.log("x", "a", n=1)
    s.log("y", "b", n=2)
    s.log("x", "b", n=3)
    assert (
        [e["n"] for e in s.events("x")] == [1, 3]
        and [e["n"] for e in s.events(kind="b")] == [2, 3]
        and [e["n"] for e in s.events("x", "b")] == [3]
    )
    assert [e["at"] for e in s.events("x")] == [100.0, 102.0], (
        "the injected clock stamps each event"
    )
    e1, e2 = s.escalate("x", "r1"), s.escalate("y", "r2")
    assert (e1, e2) == (1, 2) and [e["reason"] for e in s.escalations()] == ["r1", "r2"]


# ----------------------------------------------------------------------------- the CLI


def test_cli_eval_prints_a_pass_per_scenario_and_a_summary(capsys):
    with fake_llm(rules()):
        assert main(["eval", "--provider", "anthropic"]) == 0
    out = capsys.readouterr().out
    assert out.count("PASS") == 10 and "overall: 10/10 = 100%" in out


def test_cli_chat_routes_messages_and_lists_pending_approvals(monkeypatch, capsys, tmp_path):
    inputs = iter(["I was charged twice for INV-1001, please refund me.", "/pending", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    with fake_llm(rules()):
        assert main(["chat", "--provider", "anthropic", "--db", str(tmp_path / "c.sqlite")]) == 0
    out = capsys.readouterr().out
    assert (
        "[billing]" in out
        and "pending approval" in out.lower()
        and "'invoice_id': 'INV-1001'" in out
    )


# ----------------------------------------------------------------------------- the options added for Week 7 experiments


@pytest.mark.parametrize(
    "mode,agent,message,history,expected",
    [
        ("", "billing", "refund INV-1", [], None),
        ("always", "billing", "thanks", [{"role": "tool"}], "required"),
        ("smart", "billing", "I was charged twice for INV-3001", [], "lookup_invoice"),
        ("smart", "billing", "i was charged twice for inv-3001", [], "lookup_invoice"),
        ("smart", "billing", "I was charged twice last week", [], None),
        ("smart", "billing", "my invoice is INV-3001", [{"role": "user"}, {"role": "tool"}], None),
        ("smart", "tech", "How do I reset my password?", [], "required"),
        ("smart", "tech", "thanks", [{"role": "user"}, {"role": "tool"}], None),
    ],
)
def test_which_tool_is_forced_on_the_first_step(db, mode, agent, message, history, expected):
    assert system(db, force_first_tool=mode)._forced_tool(agent, message, history) == expected


def test_the_forcing_policy_reaches_the_model_and_triage_examples_are_optional(db, tmp_path):
    with fake_llm(rules()) as f:
        system(db, force_first_tool="smart", triage_few_shot=True).handle(
            "a", "I was charged twice for INV-3001"
        )
        system(tmp_path / "b.sqlite").handle("b", "I was charged twice for INV-3001")
    specialist_calls = [c for c in f.calls if c.kwargs.get("tools")]
    assert (
        specialist_calls[0].kwargs["tool_choice"] == "lookup_invoice"
        and specialist_calls[-1].kwargs["tool_choice"] is None
    )
    triage_prompts = [c.system for c in f.calls if "You are the triage agent" in (c.system or "")]
    assert "Examples:" in triage_prompts[0] and "Examples:" not in triage_prompts[1]


# ----------------------------------------------------------------------------- the keyword router (Week 7)


@pytest.mark.parametrize(
    "message,category",
    [
        ("I was charged twice for INV-3001", "billing"),
        ("Please refund me", "billing"),
        ("The export keeps failing with error 0x5F", "technical"),
        ("I can't log in", "technical"),
        ("My app crashes on start", "technical"),
        ("I want to talk to a real person", "human"),
        (
            "Get me a manager about this refund",
            "human",
        ),  # an explicit request for a person wins over a billing word
        (
            "Why is my invoice and my export both broken?",
            "billing",
        ),  # money outranks product trouble
        ("What's the capital of France?", None),
        ("", None),
    ],
)
def test_rule_router(message, category):
    from support_system.routing import rule_route

    got, reason = rule_route(message)
    assert got == category and (reason != "no rule matched") == (category is not None)


def test_triage_modes_decide_with_rules_first_and_only_ask_the_model_when_allowed(db, tmp_path):
    with fake_llm(rules()) as f:
        r = system(db, triage_mode="rules").handle("a", "I was charged twice for INV-3001")
        o = system(tmp_path / "o.sqlite", triage_mode="rules").handle(
            "o", "What is the capital of France?"
        )
    assert r.agent == "billing" and o.status == "out_of_scope"
    assert not f.calls_matching("You are the triage agent"), (
        "'rules' never calls the triage model, even for unmatched text"
    )
    assert (
        system(db, triage_mode="rules")
        .store.events("a", "route")[0]["reason"]
        .startswith("billing word")
    )
    with fake_llm(rules()) as f:
        s = system(tmp_path / "m.sqlite", triage_mode="rules+llm")
        s.handle("x", "I was charged twice for INV-3001")  # a rule fires: no model call
        asked = len(f.calls_matching("You are the triage agent"))
        s.handle("y", "Can you tell me a joke about cats?")  # no rule: the model decides
    assert asked == 0 and len(f.calls_matching("You are the triage agent")) == 1


def test_prefetch_verifies_the_invoice_in_code_and_hands_the_model_the_facts(db, tmp_path):
    with fake_llm(rules()) as f:
        s = system(db, prefetch_invoice=True, triage_mode="rules")
        r = s.handle("p", "I was charged twice for INV-3001")
    first_prompt = [c for c in f.calls if c.kwargs.get("tools")][0].prompt
    assert "[Checked by the system: INV-3001: $20.00, paid]" in first_prompt
    assert r.calls[0] == {"name": "lookup_invoice", "args": {"invoice_id": "INV-3001"}}, (
        "the system's lookup is part of the trajectory"
    )
    assert s.store.events("p", "prefetch")[0]["result"] == "INV-3001: $20.00, paid"
    stored = str(s.store.get("p")[1])
    assert "Checked by the system" in stored, "the verified facts stay in the conversation history"


def test_prefetch_skips_messages_without_an_invoice_id_and_tech_conversations(db, tmp_path):
    with fake_llm(rules()):
        s = system(db, prefetch_invoice=True, triage_mode="rules")
        s.handle("a", "I was charged twice last week and want a refund.")
        s.handle("b", "The export is down with error 0x5F")
    assert s.store.events("a", "prefetch") == [] and s.store.events("b", "prefetch") == []


# ----------------------------------------------------------------------------- Week 7 Day 5: cost levers (all off by default)

from support_system.system import LEAN_PROMPTS, PROMPTS  # noqa: E402

from common.cache import ResponseCache  # noqa: E402


def tool_prompts(f):
    """(system prompt, tool specs) of each specialist model call."""
    return [(c.system, c.kwargs.get("tools") or []) for c in f.calls if c.kwargs.get("tools")]


def test_lean_prompts_and_compact_specs_reach_the_model_and_are_much_shorter(db):
    with fake_llm(rules()) as f:
        full = system(db.with_name("a.sqlite"), triage_mode="rules")
        full.handle("a", "I was charged twice for INV-3001")
    with fake_llm(rules()) as g:
        lean = system(db.with_name("b.sqlite"), triage_mode="rules", lean=True)
        lean.handle("a", "I was charged twice for INV-3001")
    (full_sys, full_tools), (lean_sys, lean_tools) = tool_prompts(f)[0], tool_prompts(g)[0]
    assert full_sys == PROMPTS["billing"] and lean_sys == LEAN_PROMPTS["billing"]
    assert [t["name"] for t in lean_tools] == [t["name"] for t in full_tools], (
        "same tools, shorter text"
    )
    assert len(json.dumps(lean_tools)) < 0.8 * len(
        json.dumps(full_tools)
    )  # measured: 830 vs 1159 characters
    assert len(lean_sys) < 0.5 * len(full_sys)
    assert "lean" not in LEAN_PROMPTS["billing"].lower()


def test_lean_changes_the_cache_scope_so_a_prompt_change_cannot_serve_an_old_answer(db):
    cache = ResponseCache()
    with fake_llm(rules()):
        a = system(db.with_name("a.sqlite"), triage_mode="rules", cache=cache)
        a.handle("x", "How do I reset my password?")
        b = system(db.with_name("b.sqlite"), triage_mode="rules", cache=cache, lean=True)
        r = b.handle("y", "How do I reset my password?")
    assert not r.cached and len(cache) == 2


def test_hide_prefetched_removes_the_lookup_tool_the_model_would_repeat(db):
    with fake_llm(rules()) as f:
        s = system(db, triage_mode="rules", prefetch_invoice=True, hide_prefetched=True)
        s.handle("p", "I was charged twice for INV-3001")
    names = [t["name"] for t in tool_prompts(f)[0][1]]
    assert "lookup_invoice" not in names and "request_refund" in names
    assert s.store.events("p", "prefetch")


def test_hide_prefetched_keeps_the_tool_when_nothing_was_prefetched(db):
    with fake_llm(rules()) as f:
        s = system(db, triage_mode="rules", prefetch_invoice=True, hide_prefetched=True)
        s.handle(
            "p", "I was charged twice last week and want a refund"
        )  # no invoice id: nothing to prefetch
    assert "lookup_invoice" in [t["name"] for t in tool_prompts(f)[0][1]]


def test_reply_from_tool_skips_the_final_model_call_for_a_refund_outcome(db):
    with fake_llm(rules()) as f:
        s = system(db, triage_mode="rules", reply_from_tool=True)
        r = s.handle("c", "I was charged twice for INV-3001, please refund it")
        n_with = len(f.calls)
    with fake_llm(rules()) as g:
        s2 = system(db.with_name("b.sqlite"), triage_mode="rules")
        r2 = s2.handle("c", "I was charged twice for INV-3001, please refund it")
    assert r.status == "ok" and "Your refund of $20.00" in r.text and "has been issued" in r.text
    assert len(g.calls) - n_with == 1, "exactly one model call saved"
    assert r.steps == 2 and r2.steps == 3
    assert s.refunds.payments.ledger() and r2.status == "ok"


def test_reply_from_tool_is_truthful_for_pending_and_ineligible_outcomes(db):
    with fake_llm(rules()):
        s = system(db, triage_mode="rules", reply_from_tool=True)
        pending = s.handle("a", "Please refund INV-1001, I was charged twice")
        unpaid = s.handle("b", "Please refund INV-3001, I was charged twice")  # paid and small
    assert "pending approval" in pending.text.lower() and "issued" not in pending.text.lower()
    assert unpaid.text.startswith("Your refund of")
    assert not s.store.events("a", "violation") and not s.store.events("b", "violation")


def test_reply_from_tool_leaves_lookups_and_tech_answers_to_the_model(db):
    with fake_llm(rules()) as f:
        s = system(db, triage_mode="rules", reply_from_tool=True)
        r = s.handle("t", "How do I reset my password?")
    assert r.agent == "tech" and r.steps == 2 and len(f.calls) == 2, (
        "search_kb results still need phrasing"
    )


# -- the response cache


def tech_system(db, cache, **kw):
    return system(db, triage_mode="rules", cache=cache, **kw)


def test_a_repeated_how_to_question_is_answered_from_the_cache_with_no_model_call(db):
    cache = ResponseCache()
    with fake_llm(rules()) as f:
        s = tech_system(db, cache)
        first = s.handle("one", "How do I reset my password?")
        calls_after_first = len(f.calls)
        second = s.handle("two", "  how do i RESET my password ")
    assert not first.cached and second.cached and second.text == first.text
    assert len(f.calls) == calls_after_first, "a hit costs no model call"
    assert second.steps == 0 and second.cost_usd == 0 and second.calls == []
    assert s.store.events("two", "cache_hit") == [
        {"kind": "exact", **s.store.events("two", "cache_hit")[0]}
    ]
    assert s.store.get("two")[1][-1] == {"role": "assistant", "content": first.text}, (
        "the follow-up turn has the history"
    )


def test_a_cached_conversation_can_continue_normally(db):
    cache = ResponseCache()
    with fake_llm(rules()):
        s = tech_system(db, cache)
        s.handle("one", "How do I reset my password?")
        s.handle("two", "How do I reset my password?")
        r = s.handle("two", "thanks!")
    assert r.status == "ok" and not r.cached and r.agent == "tech"


def test_status_answers_are_never_cached_because_they_change(db):
    cache = ResponseCache()
    with fake_llm(rules()):
        s = tech_system(db, cache)
        a = s.handle("one", "The export is down with error 0x5F")
        b = s.handle("two", "The export is down with error 0x5F")
    assert not a.cached and not b.cached and len(cache) == 0


def test_billing_answers_are_never_cached(db):
    cache = ResponseCache()
    with fake_llm(rules()):
        s = tech_system(db, cache)
        s.handle("one", "I was charged twice for INV-3001")
        r = s.handle("two", "I was charged twice for INV-3001")
    assert not r.cached and len(cache) == 0


def test_only_the_first_turn_of_a_conversation_is_cacheable(db):
    cache = ResponseCache()
    with fake_llm(rules()):
        s = tech_system(db, cache)
        s.handle("one", "How do I reset my password?")
        s.handle(
            "one", "How do I reset my password?"
        )  # same words, but NOT a first turn: depends on the history
        s2 = s.handle("two", "How do I reset my password?")
    assert s2.cached and cache.stats["hits"] == 1


def test_a_guard_violation_is_never_cached(db):
    cache = ResponseCache()

    def liar_tech(prompt, call):
        msgs = call.messages
        if any(m["role"] == "tool" for m in msgs):
            return "Done! Your refund has been approved and issued. I guarantee this is fixed."
        return tool_calls(("search_kb", {"query": "reset password"}))

    with fake_llm([(r"You are the triage agent", triage), (r"(?s).*", liar_tech)]):
        s = tech_system(db, cache)
        r = s.handle("one", "How do I reset my password?")
    assert r.violations and len(cache) == 0


def test_reply_from_tool_never_builds_a_reply_from_an_error(db):
    """A refund call with invalid arguments returns an error: the model must get to retry, not have the error templated."""
    attempts = []

    def sloppy(prompt, call):
        msgs = call.messages
        results = [m["content"] for m in msgs if m["role"] == "tool"]
        if not results:
            attempts.append("bad")
            return tool_calls(("request_refund", {"invoice_id": "INV-3001"}))  # missing 'reason'
        if "Invalid arguments" in results[-1]:
            attempts.append("retry")
            return tool_calls(("request_refund", {"invoice_id": "INV-3001", "reason": "duplicate"}))
        return "unused"

    with fake_llm([(r"You are the triage agent", triage), (r"(?s).*", sloppy)]) as f:
        s = system(db, triage_mode="rules", reply_from_tool=True)
        r = s.handle("e", "I was charged twice for INV-3001")
    assert attempts == ["bad", "retry"] and len(f.calls) == 2
    assert r.text.startswith("Your refund of $20.00") and r.status == "ok"


def test_an_ungrounded_how_to_answer_is_not_cached(db):
    """No tool call at all means nothing grounded the answer in the knowledge base: never cache it."""
    cache = ResponseCache()
    with fake_llm(
        [
            (r"You are the triage agent", triage),
            (r"(?s).*", "Just try turning it off and on again."),
        ]
    ):
        s = tech_system(db, cache)
        r = s.handle("one", "How do I reset my password?")
    assert r.status == "ok" and len(cache) == 0


def test_hide_prefetched_only_hides_when_the_flag_is_set(db):
    with fake_llm(rules()) as f:
        s = system(db, triage_mode="rules", prefetch_invoice=True)
        s.handle("p", "I was charged twice for INV-3001")
    assert "lookup_invoice" in [t["name"] for t in tool_prompts(f)[0][1]]


def test_only_search_kb_counts_as_public_knowledge():
    from support_system.system import PUBLIC_TOOLS

    assert PUBLIC_TOOLS == {"search_kb"}, (
        "service_status changes over time; escalation and billing tools are per customer"
    )


def test_a_follow_up_answer_that_depends_on_the_conversation_never_overwrites_the_cache(db):
    cache = ResponseCache()

    def counting(prompt, call):
        users = sum(1 for m in call.messages if m["role"] == "user")
        if any(m["role"] == "tool" and m["content"] for m in call.messages[-1:]):
            return f"Answer written for turn {users}. See KB-102."
        return tool_calls(("search_kb", {"query": "reset password"}))

    with fake_llm([(r"You are the triage agent", triage), (r"(?s).*", counting)]):
        s = tech_system(db, cache)
        first = s.handle("one", "How do I reset my password?")
        s.handle(
            "one", "How do I reset my password?"
        )  # a follow-up in the same conversation: history-dependent
        other = s.handle("two", "How do I reset my password?")
    assert (
        first.text == "Answer written for turn 1. See KB-102."
        and other.cached
        and other.text == first.text
    )


# ----------------------------------------------------------------------------- Week 7 Day 6: customer feedback

from common import tracing  # noqa: E402


def test_feedback_is_stored_with_a_rating_a_reason_and_a_redacted_capped_comment(db):
    with fake_llm(rules()):
        s = system(db, triage_mode="rules")
        s.handle("f", "I was charged twice for INV-3001")
        rec = s.feedback(
            "f", -1, reason="wrong", comment="my card 4111 1111 1111 1111 " + "x" * 600
        )
    assert rec["rating"] == -1 and rec["reason"] == "wrong" and rec["turns"] == 1
    assert (
        "4111" not in rec["comment"]
        and "[card number removed]" in rec["comment"]
        and len(rec["comment"]) <= 500
    )
    stored = s.store.events("f", "feedback")
    assert len(stored) == 1 and stored[0]["rating"] == -1 and "4111" not in json.dumps(stored)
    assert rec["trace_id"] is None, "tracing was off"


def test_feedback_validation(db):
    with fake_llm(rules()):
        s = system(db, triage_mode="rules")
        s.handle("f", "I was charged twice for INV-3001")
        for bad in (0, 2, "up"):
            with pytest.raises(ValueError, match="rating"):
                s.feedback("f", bad)
        with pytest.raises(ValueError, match="reason"):
            s.feedback("f", 1, reason="because")
        with pytest.raises(ValueError, match="unknown conversation"):
            s.feedback("nobody", 1)
        s.feedback("f", 1)
    assert [e["rating"] for e in s.store.events("f", "feedback")] == [1]


def test_feedback_works_for_an_out_of_scope_conversation_and_counts_turns(db):
    with fake_llm(rules()):
        s = system(db, triage_mode="rules")
        s.handle("o", "Recommend a good pizza place")  # out of scope: no specialist ran
        assert s.feedback("o", -1, reason="unhelpful")["turns"] == 0
        s.handle("t", "How do I reset my password?")
        s.handle("t", "thanks!")
        assert s.feedback("t", 1)["turns"] == 2


def test_feedback_carries_the_trace_id_of_the_latest_turn_when_tracing_is_on(db):
    with fake_llm(rules()):
        with tracing.capture() as rec:
            s = system(db, triage_mode="rules")
            first = s.handle("f", "How do I reset my password?")
            second = s.handle("f", "thanks!")
            record = s.feedback("f", -1, reason="rude")
    assert first.trace_id and second.trace_id and first.trace_id != second.trace_id
    assert record["trace_id"] == second.trace_id, "the latest turn"
    root = next(x for x in rec.spans if x.parent_id is None and x.trace_id == second.trace_id)
    assert root.name == "invoke_workflow support"
    assert [e["trace_id"] for e in s.store.events("f", "trace")] == [
        first.trace_id,
        second.trace_id,
    ]
