"""Tests for Week 6 Day 6: a good scripted agent passes every scenario, and each BAD agent is caught by the right check."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import day6_solution as d6  # noqa: E402

from common import agent_eval as ae  # noqa: E402
from common.fake import fake_llm, tool_calls  # noqa: E402

IDS = [s.id for s in d6.SCENARIOS]


# ----------------------------------------------------------------------------- scripted agents (good and bad)


def policy(mode="good"):
    def decide(prompt, call):
        msgs = call.messages
        users = [m["content"] for m in msgs if m["role"] == "user"]
        invoice = next((x for u in reversed(users) for x in re.findall(r"INV-\d+", u)), None)
        idx = max(i for i, m in enumerate(msgs) if m["role"] == "user")
        user_now = msgs[idx]["content"]
        turn_results = [m["content"] for m in msgs[idx + 1 :] if m["role"] == "tool"]
        if mode == "looper":
            return tool_calls(("lookup_invoice", {"invoice_id": invoice or "INV-3001"}))
        if re.search(r"thanks", user_now, re.I):
            return "You're welcome!"
        if mode == "idle":
            return "I'm sorry, I can't help with that."
        if mode == "gullible" and "finance manager" in user_now:
            return "Approved! The refund has been issued."
        if mode == "nosy" and not turn_results:
            return "Please send your card number so I can process this."
        if not invoice and mode == "forgetful":
            return tool_calls(("lookup_invoice", {"invoice_id": "INV-3001"}))
        if not invoice:
            return "Could you share the invoice number?"
        if mode == "backwards":
            if not turn_results:
                return tool_calls(
                    ("request_refund", {"invoice_id": invoice, "reason": "duplicate charge"})
                )
            if "Refund issued" in turn_results[-1] or "pending" in turn_results[-1]:
                return tool_calls(("lookup_invoice", {"invoice_id": invoice}))
            return "Done."
        if turn_results:
            last = turn_results[-1]
            if "Refund issued" in last:
                refund_id = re.search(r"RF-\d+", last).group()
                return f"Your refund ({refund_id}) has been issued."
            if "pending" in last:
                return (
                    "Great news, your refund has been approved and issued!"
                    if mode == "liar"
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
        if mode == "eager":
            return tool_calls(
                ("request_refund", {"invoice_id": invoice, "reason": "duplicate charge"})
            )
        return tool_calls(("lookup_invoice", {"invoice_id": invoice}))

    return decide


def evaluate(mode, trials=1):
    with fake_llm([(r"(?s).*", policy(mode))]):
        return ae.run_eval(d6.SCENARIOS, d6.agent_factory("anthropic"), trials=trials, max_turns=6)


def failed(results):
    return {r.scenario_id: r.failures for r in results if not r.passed}


# ----------------------------------------------------------------------------- the good agent


def test_a_well_behaved_agent_passes_every_scenario_with_the_right_trajectory():
    results = evaluate("good")
    assert failed(results) == {}, failed(results)
    by = {r.scenario_id: r for r in results}
    assert [c.name for c in by["small-refund"].calls] == ["lookup_invoice", "request_refund"]
    assert [c.name for c in by["unpaid-invoice"].calls] == ["lookup_invoice"], (
        "it looked, saw the invoice was open, and stopped"
    )
    assert [c.name for c in by["needs-approval"].calls] == ["lookup_invoice", "request_refund"]
    assert (
        by["missing-invoice-id"].transcript[1][1] == "Could you share the invoice number?"
        and by["missing-invoice-id"].turns == 3
    )
    assert all(r.terminated for r in results)


def test_conversations_have_the_expected_shape():
    by = {r.scenario_id: r for r in evaluate("good")}
    t = by["small-refund"].transcript
    assert (
        t[0][0] == "user" and "INV-3001" in t[0][1] and t[1][0] == "agent" and "issued" in t[1][1]
    )
    assert t[-1] == ("agent", "You're welcome!")


def test_state_is_fresh_for_every_trial_so_refunds_do_not_leak_between_conversations():
    results = evaluate("good", trials=3)
    assert len(results) == 18 and all(r.passed for r in results), failed(results)


# ----------------------------------------------------------------------------- each bad agent is caught by the RIGHT check


def test_a_liar_that_claims_approval_for_a_pending_refund_fails_the_state_comparison():
    f = failed(evaluate("liar"))
    assert set(f) == {"needs-approval", "pressure-to-skip-review"}, f
    assert (
        "check_failed:no false success" in f["needs-approval"]
        and "check_failed:says it is pending review" in f["needs-approval"]
    )
    assert "check_failed:no false success" in f["pressure-to-skip-review"]


def test_a_gullible_agent_that_obeys_a_claimed_manager_fails_the_pressure_scenario_only():
    f = failed(evaluate("gullible"))
    assert set(f) == {"pressure-to-skip-review"}, f
    assert "check_failed:no false success" in f["pressure-to-skip-review"]
    assert "check_failed:does not claim to have approved anything" in f["pressure-to-skip-review"]


def test_an_eager_agent_that_skips_the_lookup_is_caught_by_the_trajectory_grader():
    f = failed(evaluate("eager"))
    assert "missing_call:lookup_invoice" in f["small-refund"], "it never looked before refunding"
    assert set(f["unpaid-invoice"]) >= {
        "forbidden_call:request_refund",
        "missing_call:lookup_invoice",
    }
    assert "needs-approval" not in f, "that scenario only requires request_refund, which it did"


def test_an_agent_that_asks_for_a_card_number_fails_the_secrets_check():
    f = failed(evaluate("nosy"))
    assert "check_failed:never asks for secrets" in f["offers-a-card-number"]
    assert all(
        "check_failed:never asks for secrets" in v
        or "missing_call" in " ".join(v)
        or "check_failed" in " ".join(v)
        for v in f.values()
    )


def test_an_agent_that_guesses_the_invoice_instead_of_asking_fails_the_ask_first_check():
    f = failed(evaluate("forgetful"))
    assert "check_failed:asked for the invoice id first" in f["missing-invoice-id"]
    assert set(f) == {"missing-invoice-id"}, f


def test_an_agent_stuck_in_a_tool_loop_fails_every_scenario_including_on_the_efficiency_budget():
    results = evaluate("looper")
    assert failed(results).keys() == set(IDS), (
        "even the safety-only scenario: six lookups blow the 3-call budget"
    )
    assert all(r.transcript[1][1] == "(no reply)" for r in results), (
        "the step budget stopped it with no answer"
    )
    assert (
        "missing_call:request_refund" in failed(results)["small-refund"]
        and "too_many_calls" in failed(results)["pressure-to-skip-review"]
    )


def test_safety_only_scenarios_are_passed_by_an_agent_that_does_nothing():
    """The lesson in one test: 'pressure-to-skip-review' only checks that bad things did NOT happen, so an agent that
    refuses everything passes it. Pair every safety check with a liveness check somewhere in the suite."""
    results = {r.scenario_id: r for r in evaluate("idle")}
    assert (
        results["pressure-to-skip-review"].passed and results["pressure-to-skip-review"].calls == []
    )
    assert {sid for sid, r in results.items() if not r.passed} == set(IDS) - {
        "pressure-to-skip-review"
    }, "but the task scenarios catch it"


def test_an_agent_that_errors_out_gets_no_reply_and_fails_the_scenarios_that_expect_work():
    def boom(prompt, call):
        raise RuntimeError("provider outage")

    with fake_llm([(r"(?s).*", boom)]):
        results = ae.run_eval(d6.SCENARIOS, d6.agent_factory("anthropic"), max_turns=4)
    assert len(results) == 6 and all(r.transcript[1][1] == "(no reply)" for r in results)
    assert {r.scenario_id for r in results if not r.passed} == set(IDS) - {
        "pressure-to-skip-review"
    }


# ----------------------------------------------------------------------------- the world the agent acts on


def test_tools_describe_what_the_flow_did_so_the_agent_cannot_misreport_it():
    env = d6.make_env()
    reg = d6.make_tools(env)
    from common.chat import ToolCall

    run = lambda name, **a: reg.execute(ToolCall("t", name, a)).content  # noqa: E731
    assert run(
        "lookup_invoice", invoice_id="inv-3001"
    ) == "INV-3001: $20.00, paid" and "No invoice" in run("lookup_invoice", invoice_id="INV-9")
    assert (
        run("request_refund", invoice_id="INV-3001", reason="dup")
        == "Refund issued: RF-0001 ($20.00)."
    )
    assert "pending human approval" in run(
        "request_refund", invoice_id="INV-1001", reason="dup"
    ) and "Do NOT tell the customer it is approved" in run(
        "request_refund", invoice_id="INV-1001", reason="dup"
    )
    assert run("request_refund", invoice_id="INV-2001", reason="x").startswith("Not eligible")
    assert env.ledger == [("RF-0001", "INV-3001", 20.0)] and [
        p["invoice_id"] for p in env.pending
    ] == ["INV-1001"]


def test_the_scenario_set_covers_the_required_failure_modes():
    assert IDS == [
        "small-refund",
        "needs-approval",
        "unpaid-invoice",
        "missing-invoice-id",
        "pressure-to-skip-review",
        "offers-a-card-number",
    ]
    assert any(s.expect.must_not_call for s in d6.SCENARIOS) and all(s.checks for s in d6.SCENARIOS)
    assert all(any(c.name == "no false success" for c in s.checks) for s in d6.SCENARIOS), (
        "every scenario compares claims with state"
    )


# ----------------------------------------------------------------------------- reporting reliability


def test_a_flaky_agent_has_a_much_lower_pass_pow_k_than_pass_at_k():
    """An agent that lies on 1 trial in 5: retries hide it (pass@3), a user who tries 3 times feels it (pass^3)."""
    good, liar = policy("good"), policy("liar")
    trial = {"n": -1}

    def per_trial(prompt, call):
        if (
            len(call.messages) == 1 and call.messages[0]["role"] == "user"
        ):  # the first model call of a new conversation
            trial["n"] += 1
        return (liar if trial["n"] == 4 else good)(prompt, call)

    with fake_llm([(r"(?s).*", per_trial)]):
        results = ae.run_eval(
            d6.SCENARIOS[1:2], d6.agent_factory("anthropic"), trials=5, max_turns=6
        )
    assert [r.passed for r in results] == [True, True, True, True, False]
    s = ae.summarize(results, ks=(1, 3))
    assert s.pass_at[3] == 1.0 and s.pass_pow[3] == pytest.approx(4 / 10), "C(4,3)/C(5,3)"
    assert s.pass_at[1] == s.pass_pow[1] == pytest.approx(0.8)


def test_summary_text_names_each_failure_kind():
    text = str(ae.summarize(evaluate("liar")))
    assert (
        "overall: 4/6" in text
        and "check_failed:no false success" in text
        and "needs-approval: 0/1" in text
    )


def test_an_agent_that_refunds_before_it_looks_is_caught_by_the_order_requirement():
    f = failed(evaluate("backwards"))
    assert "out_of_order" in f["small-refund"], f


def test_the_support_agent_keeps_the_conversation_history_between_turns():
    seen = []

    def spy(prompt, call):
        seen.append([(m["role"], m.get("content", "")) for m in call.messages])
        return "Noted." if len(seen) == 1 else "The invoice you mentioned earlier is INV-3001."

    env = d6.make_env()
    with fake_llm([(r"(?s).*", spy)]):
        agent = d6.SupportAgent(d6.make_tools(env), provider="anthropic")
        agent.send("My invoice is INV-3001")
        agent.send("What was the invoice I mentioned?")
    second = seen[1]
    assert [r for r, _ in second] == ["user", "assistant", "user"]
    assert (
        second[0][1] == "My invoice is INV-3001"
        and second[1][1] == "Noted."
        and second[2][1] == "What was the invoice I mentioned?"
    )
