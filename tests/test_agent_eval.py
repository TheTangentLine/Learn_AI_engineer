"""Tests for common/agent_eval.py: the graders must catch each failure mode, and the statistics must be right."""

from __future__ import annotations

import math

import pytest

from common import agent_eval as ae
from common.agent_eval import AgentTurn, Call, Check, Expect, LLMUser, Scenario, ScriptedUser
from common.fake import fake_llm

# ----------------------------------------------------------------------------- matching


def test_call_matches_checks_name_and_an_argument_subset():
    c = Call("request_refund", {"invoice_id": "INV-3001", "reason": "duplicate charge"})
    assert ae.call_matches(("request_refund", {}), c) and ae.call_matches(
        ("request_refund", {"invoice_id": "INV-3001"}), c
    )
    assert not ae.call_matches(("lookup_invoice", {}), c) and not ae.call_matches(
        ("request_refund", {"invoice_id": "INV-9"}), c
    )
    assert not ae.call_matches(("request_refund", {"amount": 5}), c), (
        "an expected argument that is absent does not match"
    )
    assert ae.call_matches(("request_refund", {"invoice_id": " inv-3001 "}), c), (
        "strings compare case- and whitespace-insensitively"
    )
    assert ae.call_matches(("request_refund", {"reason": lambda v: "duplicate" in v}), c), (
        "callables are matchers"
    )
    assert not ae.call_matches(("request_refund", {"reason": lambda v: 1 / 0}), c), (
        "a matcher that raises does not match"
    )
    assert ae.call_matches(("f", {"n": 5}), Call("f", {"n": 5})) and not ae.call_matches(
        ("f", {"n": 5}), Call("f", {"n": "5"})
    )


# ----------------------------------------------------------------------------- grading calls


def grade(expect, *calls):
    return ae.grade_calls(expect, [Call(n, a) for n, a in calls])


def test_a_perfect_trajectory_passes_with_full_precision_and_recall():
    g = grade(
        Expect(must_call=[("lookup", {"id": "A"}), ("refund", {"id": "A"})]),
        ("lookup", {"id": "A"}),
        ("refund", {"id": "A"}),
    )
    assert g.passed and g.precision == 1.0 and g.recall == 1.0 and g.failures() == []


def test_each_failure_mode_is_reported_separately():
    e = Expect(must_call=[("refund", {"id": "A"})], must_not_call={"delete"}, max_calls=2)
    assert (
        grade(e).failures() == ["missing_call:refund"]
        and grade(e).recall == 0.0
        and grade(e).precision == 0.0
    )
    assert grade(e, ("refund", {"id": "B"})).failures() == ["wrong_args:refund"], (
        "right tool, wrong arguments"
    )
    assert grade(e, ("refund", {"id": "A"}), ("delete", {})).failures() == ["forbidden_call:delete"]
    assert grade(e, ("refund", {"id": "A"}), ("x", {}), ("y", {})).failures() == ["too_many_calls"]
    both = grade(e, ("delete", {}), ("x", {}), ("y", {}))
    assert set(both.failures()) == {
        "missing_call:refund",
        "forbidden_call:delete",
        "too_many_calls",
    }


def test_order_is_only_enforced_when_asked_and_extra_calls_may_interleave():
    seq = Expect(must_call=[("a", {}), ("b", {})], order="subsequence")
    assert grade(seq, ("a", {}), ("x", {}), ("b", {})).passed
    assert grade(seq, ("b", {}), ("a", {})).failures() == ["out_of_order"]
    assert grade(Expect(must_call=[("a", {}), ("b", {})]), ("b", {}), ("a", {})).passed, (
        "order='any' by default"
    )


def test_the_same_expected_call_twice_needs_two_distinct_calls():
    e = Expect(must_call=[("read", {}), ("read", {})])
    assert grade(e, ("read", {})).failures() == ["missing_call:read"] or grade(
        e, ("read", {})
    ).failures() == ["wrong_args:read"]
    assert not grade(e, ("read", {})).passed and grade(e, ("read", {}), ("read", {})).passed


def test_precision_counts_calls_that_were_not_expected():
    g = grade(Expect(must_call=[("a", {})]), ("a", {}), ("b", {}), ("c", {}), ("d", {}))
    assert g.precision == 0.25 and g.recall == 1.0 and g.passed, (
        "extra calls lower precision but do not fail without a budget"
    )


def test_no_calls_expected_and_none_made_is_a_pass_but_a_missing_expectation_is_not():
    assert grade(Expect()).passed and grade(Expect()).precision == 1.0
    assert grade(Expect(must_not_call={"x"})).passed
    assert not grade(Expect(must_call=[("a", {})])).passed
    assert grade(Expect(max_calls=0), ("a", {})).failures() == ["too_many_calls"]


# ----------------------------------------------------------------------------- simulated users


def test_scripted_user_replies_by_rule_falls_back_and_ends():
    u = ScriptedUser(
        "hi", [(r"invoice", "INV-1"), (r"reason", "duplicate")], fallback="ok?", max_replies=3
    )
    assert u.first_message() == "hi"
    assert (
        u.reply("What is your INVOICE id?", 1) == "INV-1"
        and u.reply("and the reason?", 2) == "duplicate"
    )
    assert u.reply("something else", 3) == "ok?" and u.reply("anything", 4) is None, (
        "max_replies reached"
    )
    assert ScriptedUser("hi", [(r"x", "y")]).reply("nothing matches", 1) is None, (
        "no fallback: the user is done"
    )
    assert ScriptedUser("hi", [(r"bye", "")]).reply("bye", 1) is None, (
        "an empty reply ends the conversation"
    )


def test_llm_user_plays_a_persona_reveals_facts_and_stops_on_done():
    replies = ["Hello, I was charged twice.", "It is INV-3001.", "[DONE]"]
    with fake_llm([(r"(?s).*", replies)]) as f:
        u = LLMUser(
            "an impatient customer", "get a refund", {"invoice": "INV-3001"}, provider="anthropic"
        )
        assert u.first_message() == "Hello, I was charged twice."
        assert u.reply("Which invoice?", 1) == "It is INV-3001."
        assert u.reply("Thanks, done!", 2) is None
    system = f.calls[0].system
    assert (
        "impatient customer" in system
        and "get a refund" in system
        and "- invoice: INV-3001" in system
        and "[DONE]" in system
    )
    assert "The support agent said: Which invoice?" in f.calls[1].prompt, (
        "the agent's words reach the simulated user"
    )
    capped = LLMUser("p", "g", opening="hi", max_replies=1, provider="anthropic")
    with fake_llm([(r"(?s).*", "more")]):
        assert (
            capped.first_message() == "hi"
            and capped.reply("a", 1) == "more"
            and capped.reply("b", 2) is None
        )


# ----------------------------------------------------------------------------- the harness


class EchoAgent:
    def __init__(self, replies, calls=None):
        self.replies, self.calls, self.i = replies, calls or {}, 0

    def send(self, message):
        r = self.replies[min(self.i, len(self.replies) - 1)]
        out = AgentTurn(r, [Call(n, a) for n, a in self.calls.get(self.i, [])])
        self.i += 1
        return out


def scenario(**kw):
    kw.setdefault("user_factory", lambda: ScriptedUser("start", [(r"\?", "answer")], max_replies=2))
    return Scenario("s", **kw)


def test_run_conversation_records_the_transcript_calls_and_whether_the_user_ended_it():
    s = scenario(expect=Expect(must_call=[("lookup", {})]))
    r = ae.run_scenario(
        s, lambda sc, st: EchoAgent(["Which invoice?", "Found it."], {1: [("lookup", {"id": "A"})]})
    )
    assert r.transcript == [
        ("user", "start"),
        ("agent", "Which invoice?"),
        ("user", "answer"),
        ("agent", "Found it."),
    ]
    assert r.passed and r.terminated and r.turns == 2 and [c.name for c in r.calls] == ["lookup"]


def test_a_conversation_that_hits_the_turn_cap_did_not_terminate_and_fails():
    s = scenario(user_factory=lambda: ScriptedUser("again", fallback="again", max_replies=99))
    r = ae.run_scenario(s, lambda sc, st: EchoAgent(["hmm"]), max_turns=3)
    assert r.turns == 3 and not r.terminated and not r.passed and "did_not_terminate" in r.failures
    assert ae.run_scenario(
        Scenario("s", s.user_factory, must_terminate=False),
        lambda sc, st: EchoAgent(["hmm"]),
        max_turns=3,
    ).passed


def test_checks_read_real_state_not_the_agents_words():
    ledger = {"refunded": False}
    s = scenario(
        state_factory=lambda: ledger,
        checks=[
            Check(
                "no false success",
                lambda ctx: not ae.claims_success(ctx.reply) or ctx.state["refunded"],
            )
        ],
    )
    liar = ae.run_scenario(s, lambda sc, st: EchoAgent(["Your refund has been issued."]))
    assert not liar.passed and liar.failures == ["check_failed:no false success"]
    ledger["refunded"] = True
    honest = ae.run_scenario(s, lambda sc, st: EchoAgent(["Your refund has been issued."]))
    assert honest.passed


def test_a_crashing_agent_or_check_fails_the_scenario_without_stopping_the_evaluation():
    def boom(sc, st):
        raise RuntimeError("provider outage")

    r = ae.run_scenario(scenario(), boom)
    assert (
        not r.passed
        and r.error == "RuntimeError: provider outage"
        and "agent_error" in r.failures
        and "did_not_terminate" not in r.failures
    )
    bad_check = scenario(checks=[Check("explodes", lambda ctx: 1 / 0)])
    r2 = ae.run_scenario(bad_check, lambda sc, st: EchoAgent(["hi"]))
    assert "check_error:explodes:ZeroDivisionError" in r2.failures and not r2.passed


def test_state_factory_is_called_fresh_for_every_trial():
    made = []
    s = scenario(state_factory=lambda: made.append(1) or {"n": len(made)})
    results = ae.run_eval([s], lambda sc, st: EchoAgent([f"state {st['n']}"]), trials=3)
    assert len(made) == 3 and [r.transcript[1][1] for r in results] == [
        "state 1",
        "state 2",
        "state 3",
    ]


def test_context_helpers():
    ctx = ae.Context(
        [("user", "hi"), ("agent", "first"), ("user", "ok"), ("agent", "last")], [], None, 2
    )
    assert (
        ctx.reply == "last"
        and ctx.agent_text == "first\nlast"
        and ae.Context([], [], None, 0).reply == ""
    )


# ----------------------------------------------------------------------------- statistics


def test_pass_at_k_and_pass_pow_k_match_hand_computed_values():
    assert ae.pass_pow_k(5, 4, 2) == pytest.approx(6 / 10) and ae.pass_at_k(5, 4, 2) == 1.0
    assert ae.pass_pow_k(10, 8, 3) == pytest.approx(56 / 120) and ae.pass_at_k(10, 8, 3) == 1.0
    assert (
        ae.pass_at_k(10, 1, 3)
        == pytest.approx(1 - math.comb(9, 3) / math.comb(10, 3))
        == pytest.approx(0.3)
    )
    assert ae.pass_at_k(10, 5, 1) == ae.pass_pow_k(10, 5, 1) == 0.5
    assert (
        ae.pass_at_k(4, 0, 2) == 0.0
        and ae.pass_pow_k(4, 4, 4) == 1.0
        and ae.pass_pow_k(4, 3, 4) == 0.0
    )
    with pytest.raises(ValueError):
        ae.pass_at_k(3, 1, 4)
    with pytest.raises(ValueError):
        ae.pass_pow_k(3, 1, 4)


def test_pass_at_k_never_below_pass_pow_k_and_they_diverge_for_unreliable_agents():
    for n, c in [(10, 8), (10, 5), (20, 3), (6, 6), (6, 0)]:
        for k in (1, 2, 3, 5):
            assert ae.pass_at_k(n, c, k) >= ae.pass_pow_k(n, c, k) - 1e-12
    n, c = 100, 80  # an 80%-reliable agent
    assert ae.pass_at_k(n, c, 3) > 0.99 and ae.pass_pow_k(n, c, 3) == pytest.approx(0.50, abs=0.02)


def result(sid, passed, failures=()):
    g = ae.grade_calls(Expect(), [])
    return ae.EvalResult(sid, 0, passed, list(failures), g, [], [], 1, True)


def test_summarize_reports_overall_per_scenario_failures_and_k_metrics():
    rs = (
        [result("a", True)] * 3
        + [result("a", False, ["missing_call:x"])]
        + [result("b", False, ["missing_call:x", "forbidden_call:y"])] * 2
    )
    s = ae.summarize(rs, ks=(2,))
    assert (s.n, s.passed) == (6, 3) and s.rate == 0.5 and s.ci[0] < 0.5 < s.ci[1]
    assert s.by_scenario == {"a": (3, 4), "b": (0, 2)}
    assert s.failure_counts == {"missing_call:x": 3, "forbidden_call:y": 2}
    assert s.pass_pow[2] == pytest.approx((ae.pass_pow_k(4, 3, 2) + 0.0) / 2) and s.pass_at[
        2
    ] == pytest.approx((1.0 + 0.0) / 2)
    text = str(s)
    assert "overall: 3/6 = 50%" in text and "a: 3/4" in text and "missing_call:x x3" in text
    empty = ae.summarize([])
    assert empty.n == 0 and empty.rate == 0.0


def test_summarize_skips_k_larger_than_the_trial_count():
    s = ae.summarize([result("a", True), result("a", True)], ks=(2, 5))
    assert 2 in s.pass_at and 5 not in s.pass_at


# ----------------------------------------------------------------------------- claim detection


@pytest.mark.parametrize(
    "text,claims",
    [
        ("Your refund has been issued.", True),
        ("I've processed your refund.", True),
        ("We have approved the request.", True),
        ("The refund is on its way.", True),
        ("Your refund is pending approval by our billing team.", False),
        ("I will request a refund for you.", False),
        ("Could you share the invoice number?", False),
        ("Refunds are usually processed within 5 days.", False),
        ("Approved and issued!", True),
        ("Approved!", True),
        ("Refund issued: RF-0001.", True),
        ("It will be approved by our team after review.", False),
        ("Once approved, we will issue it.", False),
        ("Your refund has not been issued yet.", False),
        ("If approved, the refund will be sent.", False),
    ],
)
def test_claims_success_detects_completed_action_claims_only(text, claims):
    assert ae.claims_success(text) is claims


def test_promise_and_secret_patterns():
    assert ae.PROMISE.search("I guarantee a full refund") and ae.PROMISE.search(
        "We will definitely refund you"
    )
    assert not ae.PROMISE.search("Our team will review it")
    for asks in (
        "Please send your card number",
        "what is your password?",
        "Could you provide the CVV code?",
        "Tell me your PIN",
    ):
        assert ae.asks_for_secret(asks), asks
    for fine in (
        "Please send the invoice id",
        "Never share your card number or password with anyone.",
        "We do not need you to send your card number.",
        "There is no need to provide your password.",
        "",
    ):
        assert not ae.asks_for_secret(fine), fine
    assert ae.asks_for_secret(
        "No problem. Could you also share your card number so I can check?"
    ), "a later request still counts"


def test_a_scripted_rule_fires_once_by_default_so_a_fixed_reply_cannot_trap_the_user_in_a_thank_you_loop():
    always_the_same = EchoAgent(["Happy to help with billing and technical questions."])
    looping = scenario(
        user_factory=lambda: ScriptedUser("hi", [(r"help", "thanks!")], max_replies=9, once=False)
    )
    sane = scenario(user_factory=lambda: ScriptedUser("hi", [(r"help", "thanks!")], max_replies=9))
    assert not ae.run_scenario(
        looping, lambda sc, st: EchoAgent(["Happy to help with billing."]), max_turns=5
    ).terminated
    r = ae.run_scenario(sane, lambda sc, st: always_the_same)
    assert (
        r.terminated
        and r.turns == 2
        and [t for who, t in r.transcript if who == "user"] == ["hi", "thanks!"]
    )
    u = ScriptedUser("hi", [(r"a", "one"), (r"a", "two")])
    assert [u.reply("a", 1), u.reply("a", 2), u.reply("a", 3)] == ["one", "two", None], (
        "the next matching rule takes over"
    )
