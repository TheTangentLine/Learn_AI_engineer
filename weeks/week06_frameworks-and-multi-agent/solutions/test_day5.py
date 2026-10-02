"""Tests for Week 6 Day 5: pause/approve/resume, idempotent money movement, crashes, expiry, a real process restart."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import day5_solution as d5  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from common.fake import fake_llm  # noqa: E402


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "refunds.sqlite"


@pytest.fixture()
def clock():
    return Clock()


@pytest.fixture()
def svc(db, clock):
    return d5.RefundService(db, clock=clock)


def events(result):
    return [(a["actor"], a["event"]) for a in result["audit"]]


# ----------------------------------------------------------------------------- the three entrances


def test_a_small_refund_is_approved_automatically(svc):
    r = svc.start("T1", "INV-3001", "duplicate")
    assert (
        r["status"] == "refunded" and r["refunded_amount"] == 20.0 and r["refund_id"] == "RF-0001"
    )
    assert r["message"] == "Your refund of $20.00 (RF-0001) has been issued."
    assert events(r) == [
        ("system", "invoice_found"),
        ("policy", "auto_approved"),
        ("system", "refund_issued"),
        ("system", "customer_notified"),
    ]
    assert svc.payments.ledger() == [("RF-0001", "INV-3001", 20.0)] and svc.pending() == []


@pytest.mark.parametrize(
    "invoice,event", [("INV-9999", "invoice_not_found"), ("INV-2001", "invoice_not_paid")]
)
def test_missing_or_unpaid_invoices_are_rejected_without_a_human(svc, invoice, event):
    r = svc.start("T2", invoice, "x")
    assert r["status"] == "rejected" and svc.payments.ledger() == [] and svc.pending() == []
    assert events(r)[0] == ("system", event) and ("system", "closed_without_refund") in events(r)
    assert "not able to approve" in r["message"]


def test_a_larger_refund_pauses_and_nothing_is_paid_yet(svc, clock):
    r = svc.start("T3", "INV-1001", "charged twice")
    assert r["status"] == "pending_approval" and svc.payments.ledger() == []
    assert r["approval"] == {
        "ticket_id": "T3",
        "invoice_id": "INV-1001",
        "amount": 49.0,
        "reason": "charged twice",
        "requested_at": clock.now,
    }
    assert events(r) == [("system", "invoice_found"), ("policy", "needs_human")]
    assert svc.pending() == [
        {
            "ticket_id": "T3",
            "invoice_id": "INV-1001",
            "amount": 49.0,
            "reason": "charged twice",
            "requested_at": clock.now,
        }
    ]


def test_the_auto_limit_boundary(svc, monkeypatch):
    monkeypatch.setitem(d5.INVOICES, "INV-AT", {"amount": d5.AUTO_LIMIT, "status": "paid"})
    monkeypatch.setitem(d5.INVOICES, "INV-OVER", {"amount": d5.AUTO_LIMIT + 0.01, "status": "paid"})
    assert svc.start("A", "INV-AT", "x")["status"] == "refunded", (
        "exactly at the limit is automatic"
    )
    assert svc.start("B", "INV-OVER", "x")["status"] == "pending_approval"


# ----------------------------------------------------------------------------- the human decides


def test_approval_resumes_the_run_and_pays_once(svc):
    svc.start("T4", "INV-1001", "dup")
    r = svc.decide("T4", {"action": "approve", "approver": "dana", "note": "confirmed"})
    assert r["status"] == "refunded" and r["refunded_amount"] == 49.0
    assert events(r) == [
        ("system", "invoice_found"),
        ("policy", "needs_human"),
        ("dana", "approved"),
        ("system", "refund_issued"),
        ("system", "customer_notified"),
    ]
    assert svc.payments.ledger() == [("RF-0001", "INV-1001", 49.0)] and svc.pending() == []


def test_an_approver_can_approve_a_smaller_amount_but_never_a_larger_one(svc):
    svc.start("T5", "INV-1001", "dup")
    assert (
        svc.decide("T5", {"action": "approve", "approver": "dana", "amount": 30})["refunded_amount"]
        == 30
    )
    svc.start("T6", "INV-4001", "dup")
    assert (
        svc.decide("T6", {"action": "approve", "approver": "dana", "amount": 5000})[
            "refunded_amount"
        ]
        == 900.0
    ), "capped at the invoice amount"


def test_a_rejection_closes_the_ticket_with_no_payment(svc):
    svc.start("T7", "INV-1001", "dup")
    r = svc.decide("T7", {"action": "reject", "approver": "lee", "note": "not a duplicate"})
    assert (
        r["status"] == "rejected"
        and svc.payments.ledger() == []
        and ("lee", "rejected") in events(r)
    )
    assert "not able to approve" in r["message"] and svc.pending() == []


def test_an_expired_approval_never_pays_even_if_the_approver_says_yes(db, clock):
    svc = d5.RefundService(db, clock=clock)
    svc.start("T8", "INV-1001", "dup")
    clock.now += d5.APPROVAL_TTL_S + 1
    r = svc.decide("T8", {"action": "approve", "approver": "dana"})
    assert (
        r["status"] == "expired"
        and svc.payments.ledger() == []
        and ("system", "approval_expired") in events(r)
    )
    svc.start("T9", "INV-1001", "dup")
    clock.now += d5.APPROVAL_TTL_S  # exactly at the TTL is still valid
    assert svc.decide("T9", {"action": "approve", "approver": "dana"})["status"] == "refunded"


@pytest.mark.parametrize(
    "bad",
    [
        {"action": "maybe", "approver": "x"},
        {"action": "approve", "approver": ""},
        {"action": "approve", "approver": "x", "amount": -5},
        {"approver": "x"},
        "approve",
        None,
    ],
)
def test_a_malformed_decision_is_rejected_and_pays_nothing(svc, bad):
    svc.start("T10", "INV-1001", "dup")
    with pytest.raises((ValidationError, ValueError, TypeError)):
        svc.decide("T10", bad)
    assert svc.payments.ledger() == []
    assert svc.start("T10", "INV-1001", "dup")["status"] == "pending_approval", (
        "the request is still waiting for a VALID decision"
    )
    assert svc.decide("T10", {"action": "approve", "approver": "dana"})["status"] == "refunded"


def test_deciding_twice_or_on_an_unknown_ticket(svc):
    svc.start("T11", "INV-1001", "dup")
    first = svc.decide("T11", {"action": "approve", "approver": "dana"})
    again = svc.decide("T11", {"action": "approve", "approver": "someone-else"})
    assert (
        again["status"] == "refunded"
        and again["refund_id"] == first["refund_id"]
        and len(svc.payments.ledger()) == 1
    )
    assert ("someone-else", "approved") not in events(again), "the second click changed nothing"
    with pytest.raises(KeyError, match="no refund for ticket 'nope'"):
        svc.decide("nope", {"action": "approve", "approver": "x"})
    auto = svc.start("T12", "INV-3001", "x")
    assert (
        svc.decide("T12", {"action": "reject", "approver": "x"})["status"] == "refunded"
        and auto["refund_id"] == "RF-0002"
    )


def test_starting_the_same_ticket_again_never_restarts_it(svc):
    a = svc.start("T13", "INV-1001", "dup")
    b = svc.start("T13", "INV-1001", "a different reason")
    assert (
        b["status"] == "pending_approval"
        and b["approval"]["reason"] == "dup"
        and a["approval"] == b["approval"]
    )
    assert len(svc.pending()) == 1
    svc.decide("T13", {"action": "approve", "approver": "dana"})
    done = svc.start("T13", "INV-1001", "again")
    assert done["status"] == "refunded" and len(svc.payments.ledger()) == 1


# ----------------------------------------------------------------------------- the node body runs again on resume


def test_code_before_interrupt_runs_again_on_resume_so_it_must_be_idempotent(
    db, clock, monkeypatch
):
    calls = []
    real = d5._register_pending
    monkeypatch.setattr(d5, "_register_pending", lambda *a: calls.append(a[0]) or real(*a))
    svc = d5.RefundService(db, clock=clock)
    svc.start("T14", "INV-1001", "dup")
    assert len(calls) == 1
    svc.decide("T14", {"action": "approve", "approver": "dana"})
    assert len(calls) == 2, "the first half of the node ran AGAIN when the run resumed"
    with __import__("sqlite3").connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM approvals").fetchone()[0] == 1, (
            "but INSERT OR IGNORE kept it to one row"
        )


# ----------------------------------------------------------------------------- money moves once, whatever fails


def test_payments_are_idempotent_by_key(db):
    p = d5.Payments(db)
    a = p.refund("k1", "INV-1", 10.0)
    assert p.refund("k1", "INV-1", 10.0) == a and p.refund("k1", "INV-1", 99.0) == a, (
        "the first request wins; a retry cannot change the amount"
    )
    assert (
        p.refund("k2", "INV-2", 5.0) != a
        and len(p.ledger()) == 2
        and p.ledger()[0] == (a, "INV-1", 10.0)
    )


def test_concurrent_refunds_with_the_same_key_move_money_once(db):
    p = d5.Payments(db)
    ids = []
    lock = threading.Lock()

    def go():
        rid = p.refund("same-key", "INV-1", 10.0)
        with lock:
            ids.append(rid)

    threads = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(set(ids)) == 1 and len(p.ledger()) == 1


def test_a_crash_after_the_payment_but_before_the_checkpoint_does_not_pay_twice(db, monkeypatch):
    ticking = iter(
        range(1_000_000, 2_000_000)
    )  # a REAL clock moves on: an idempotency key built from it would change
    svc = d5.RefundService(db, clock=lambda: float(next(ticking)))
    real = svc.payments.refund
    state = {"crashed": False}

    def refund_then_crash(key, invoice, amount):
        rid = real(key, invoice, amount)  # the money DID move
        if not state["crashed"]:
            state["crashed"] = True
            raise ConnectionError("process died right after the payment call")
        return rid

    monkeypatch.setattr(svc.payments, "refund", refund_then_crash)
    svc.start("T15", "INV-1001", "dup")
    with pytest.raises(ConnectionError):
        svc.decide("T15", {"action": "approve", "approver": "dana"})
    assert len(svc.payments.ledger()) == 1, "paid once so far"
    r = svc.start(
        "T15", "INV-1001", "dup"
    )  # the worker retries: the thread continues from the crashed node
    assert (
        r["status"] == "refunded"
        and len(svc.payments.ledger()) == 1
        and r["refund_id"] == "RF-0001"
    )


def test_a_crash_in_a_later_node_resumes_without_repeating_the_payment(db, clock):
    boom = {"n": 0}

    def flaky_drafter(outcome, state):
        boom["n"] += 1
        if boom["n"] == 1:
            raise RuntimeError("message provider down")
        return d5.default_message(outcome, state)

    svc = d5.RefundService(db, clock=clock, drafter=flaky_drafter)
    with pytest.raises(RuntimeError):
        svc.start("T16", "INV-3001", "dup")
    assert len(svc.payments.ledger()) == 1
    r = svc.start("T16", "INV-3001", "dup")
    assert (
        r["status"] == "refunded"
        and r["message"].startswith("Your refund of $20.00")
        and len(svc.payments.ledger()) == 1
    )
    assert [e for e in events(r)].count(("system", "refund_issued")) == 1, (
        "the audit trail shows ONE refund"
    )


# ----------------------------------------------------------------------------- a real process restart


def test_a_refund_paused_in_one_process_is_approved_in_another(db):
    root = str(Path(__file__).parent)
    first = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {root!r})
        import day5_solution as d5
        svc = d5.RefundService({str(db)!r})
        print(json.dumps(svc.start("T-RESTART", "INV-4001", "charged twice")["status"]))
        """
    )
    second = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {root!r})
        import day5_solution as d5
        svc = d5.RefundService({str(db)!r})
        pending = [p["ticket_id"] for p in svc.pending()]
        r = svc.decide("T-RESTART", {{"action": "approve", "approver": "ops-bot", "note": "from another process", "amount": 450}})
        print(json.dumps({{"pending": pending, "status": r["status"], "amount": r["refunded_amount"]}}))
        """
    )
    a = subprocess.run([sys.executable, "-c", first], capture_output=True, text=True, timeout=120)
    assert a.returncode == 0, a.stderr
    assert json.loads(a.stdout.strip().splitlines()[-1]) == "pending_approval"
    assert d5.Payments(db).ledger() == [], "process 1 exited with the refund still waiting"
    b = subprocess.run([sys.executable, "-c", second], capture_output=True, text=True, timeout=120)
    assert b.returncode == 0, b.stderr
    assert json.loads(b.stdout.strip().splitlines()[-1]) == {
        "pending": ["T-RESTART"],
        "status": "refunded",
        "amount": 450.0,
    }
    assert d5.Payments(db).ledger() == [("RF-0001", "INV-4001", 450.0)]


def test_several_tickets_can_be_processed_at_once_by_separate_service_instances(db):
    results = {}

    def go(i):
        s = d5.RefundService(db)
        results[i] = s.start(f"P{i}", "INV-3001", "x")["status"]

    threads = [threading.Thread(target=go, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == dict.fromkeys(range(4), "refunded") and len(d5.Payments(db).ledger()) == 4


# ----------------------------------------------------------------------------- the model writes the words, never the outcome


def test_the_llm_drafter_rewrites_the_message_but_cannot_drop_or_change_a_fact(db, clock):
    ok = [
        (r"Rewrite this support message", "Good news! Your $20.00 refund (RF-0001) is on its way.")
    ]
    with fake_llm(ok):
        r = d5.RefundService(db, clock=clock, drafter=d5.llm_drafter("anthropic")).start(
            "L1", "INV-3001", "x"
        )
    assert r["message"] == "Good news! Your $20.00 refund (RF-0001) is on its way."
    with fake_llm(
        [(r"Rewrite this support message", "Good news! A refund is on its way.")]
    ):  # dropped the amount and the id
        r2 = d5.RefundService(
            db.with_name("r2.sqlite"), clock=clock, drafter=d5.llm_drafter("anthropic")
        ).start("L2", "INV-3001", "x")
    assert r2["message"] == "Your refund of $20.00 (RF-0001) has been issued.", (
        "fell back to the template"
    )
    assert d5._facts("Your refund of $20.50 (RF-0003) and $7") == ["$20.50", "RF-0003", "$7"]


def test_decision_model_validation():
    assert d5.Decision.model_validate({"action": "approve", "approver": "a"}).amount is None
    with pytest.raises(ValidationError):
        d5.Decision.model_validate({"action": "approve", "approver": "a", "amount": 0})


def test_the_graph_can_be_drawn():
    from langgraph.checkpoint.memory import MemorySaver

    g = (
        d5.build_refund_graph(MemorySaver(), d5.Payments(":memory:"), ":memory:")
        .get_graph()
        .draw_mermaid()
    )
    for node in ("lookup", "assess", "human_approval", "issue_refund", "reject", "notify"):
        assert node in g


# ----------------------------------------------------------------------------- the graph protects itself too


def graph_resume(svc, ticket, value):
    from langgraph.types import Command

    with svc._app() as app:
        return app.invoke(Command(resume=value), svc._cfg(ticket))


def test_a_malformed_decision_that_reaches_the_graph_is_asked_again_and_does_not_poison_the_thread(
    svc,
):
    svc.start("G1", "INV-1001", "dup")
    out = graph_resume(
        svc, "G1", {"action": "maybe", "approver": "x"}
    )  # bypasses the service's pre-validation
    assert (
        "__interrupt__" in out
        and "action: Input should be 'approve' or 'reject'"
        in out["__interrupt__"][-1].value["error"]
    )
    assert svc.payments.ledger() == []
    final = graph_resume(svc, "G1", {"action": "approve", "approver": "dana"})
    assert final["outcome"] == "refunded" and len(svc.payments.ledger()) == 1, (
        "a valid decision still works afterwards"
    )


def test_too_many_malformed_decisions_close_the_request_instead_of_asking_forever(svc):
    svc.start("G2", "INV-1001", "dup")
    out = None
    for _ in range(d5.MAX_DECISION_ATTEMPTS):
        out = graph_resume(svc, "G2", {"action": "nonsense"})
    assert out["outcome"] == "rejected" and "__interrupt__" not in out
    assert (
        ("system", "too_many_invalid_decisions") in events(out)
        and svc.payments.ledger() == []
        and svc.pending() == []
    )


def test_the_service_view_shows_the_validation_error_to_the_reviewer_ui(svc):
    svc.start("G3", "INV-1001", "dup")
    graph_resume(svc, "G3", {"action": "approve"})  # approver missing
    status = svc.start("G3", "INV-1001", "dup")
    assert status["status"] == "pending_approval" and "approver" in status["approval"]["error"]
