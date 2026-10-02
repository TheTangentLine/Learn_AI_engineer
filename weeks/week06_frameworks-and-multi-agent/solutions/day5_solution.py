"""Week 6 Day 5 - Solution: a refund flow that pauses for human approval and resumes after a restart.

The flow (a LangGraph graph with a SQLite checkpointer):

  lookup -> [invoice missing or not paid] -> reject
         -> assess -> [<= AUTO_LIMIT] ---------------------------> issue_refund -> notify
                   -> [needs a human] -> human_approval (INTERRUPT) -> approved -> issue_refund -> notify
                                                                    -> rejected / edited / expired -> ...

What makes it production-shaped rather than a demo:
  * the pause is DURABLE: the process can exit; days later another process resumes the thread
  * the money-moving step is IDEMPOTENT (a unique key per refund), because a node re-runs from its start after a failure
    and because ``interrupt()`` re-executes the code before it on resume
  * the approver decides with a VALIDATED, typed decision (approve / reject / approve a smaller amount) and an expiry
  * every step appends to an AUDIT trail stored in the state
  * a table of pending approvals lets a reviewer UI list what is waiting

  uv run python weeks/week06_frameworks-and-multi-agent/solutions/day5_solution.py        # a scripted walk-through
"""

from __future__ import annotations

import operator
import sqlite3
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402

from common import llm  # noqa: E402

MAX_DECISION_ATTEMPTS = (
    3  # malformed decisions the approver may submit before the request is closed
)
AUTO_LIMIT = 25.0  # refunds up to this are approved automatically
APPROVAL_TTL_S = 3 * 24 * 3600  # a pending approval expires after 3 days
INVOICES = {
    "INV-1001": {"amount": 49.0, "status": "paid"},
    "INV-1002": {"amount": 49.0, "status": "paid"},
    "INV-2001": {"amount": 120.0, "status": "open"},
    "INV-3001": {"amount": 20.0, "status": "paid"},
    "INV-4001": {"amount": 900.0, "status": "paid"},
}


# ----------------------------------------------------------------------------- the money-moving side effect


class Payments:
    """A fake payments API that is IDEMPOTENT: the same key always returns the same refund, and moves money once."""

    def __init__(self, db_path: str | Path):
        self.db = str(db_path)
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS refunds (key TEXT PRIMARY KEY, refund_id TEXT, invoice_id TEXT, amount REAL)"
            )

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def refund(self, key: str, invoice_id: str, amount: float) -> str:
        """Atomic: concurrent callers with the same key get the same refund id and money moves once."""
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO refunds (key, invoice_id, amount) VALUES (?, ?, ?)",
                (key, invoice_id, amount),
            )
            rowid, refund_id = c.execute(
                "SELECT rowid, refund_id FROM refunds WHERE key = ?", (key,)
            ).fetchone()
            if refund_id is None:
                refund_id = f"RF-{rowid:04d}"
                c.execute(
                    "UPDATE refunds SET refund_id = ? WHERE key = ? AND refund_id IS NULL",
                    (refund_id, key),
                )
            return c.execute("SELECT refund_id FROM refunds WHERE key = ?", (key,)).fetchone()[0]

    def ledger(self) -> list[tuple[str, str, float]]:
        with self._conn() as c:
            return [
                (r[0], r[1], r[2])
                for r in c.execute(
                    "SELECT refund_id, invoice_id, amount FROM refunds ORDER BY rowid"
                )
            ]


# ----------------------------------------------------------------------------- the decision a human makes


class Decision(BaseModel):
    action: Literal["approve", "reject"]
    approver: str = Field(min_length=1)
    note: str = ""
    amount: float | None = Field(
        default=None, gt=0, description="approve a SMALLER amount than requested"
    )


class RefundState(TypedDict, total=False):
    ticket_id: str
    invoice_id: str
    reason: str
    amount: float
    requested_at: float
    outcome: str  # refunded | rejected | expired
    refund_id: str
    refunded_amount: float
    message: str
    audit: Annotated[list[dict], operator.add]


def build_refund_graph(
    checkpointer,
    payments: Payments,
    db_path: str,
    *,
    clock: Callable[[], float] = time.time,
    drafter=None,
):
    """``drafter(outcome, state) -> str`` writes the customer message (a model call in real life)."""

    def note(actor: str, event: str, **kw) -> list[dict]:
        return [{"actor": actor, "event": event, **kw}]

    def lookup(state: RefundState) -> dict:
        inv = INVOICES.get(state["invoice_id"])
        if inv is None:
            return {"outcome": "rejected", "audit": note("system", "invoice_not_found")}
        if inv["status"] != "paid":
            return {
                "outcome": "rejected",
                "audit": note("system", "invoice_not_paid", status=inv["status"]),
            }
        return {
            "amount": inv["amount"],
            "audit": note("system", "invoice_found", amount=inv["amount"]),
        }

    def after_lookup(state: RefundState) -> str:
        return "reject" if state.get("outcome") == "rejected" else "assess"

    def assess(state: RefundState) -> dict:
        auto = state["amount"] <= AUTO_LIMIT
        # requested_at is set HERE, in a node that completes: anything written inside human_approval before interrupt()
        # is lost (that node never returns), so a timestamp made there would be recomputed on resume and never expire.
        return {
            "requested_at": clock(),
            "audit": note("policy", "auto_approved" if auto else "needs_human", limit=AUTO_LIMIT),
        }

    def after_assess(state: RefundState) -> str:
        return "issue_refund" if state["amount"] <= AUTO_LIMIT else "human_approval"

    def human_approval(state: RefundState) -> dict:
        # Everything BEFORE interrupt() runs again on resume, so it must be safe to repeat (idempotent writes only).
        requested_at = state["requested_at"]
        _register_pending(
            db_path,
            state["ticket_id"],
            state["invoice_id"],
            state["amount"],
            state["reason"],
            requested_at,
        )
        payload = {
            "ticket_id": state["ticket_id"],
            "invoice_id": state["invoice_id"],
            "amount": state["amount"],
            "reason": state["reason"],
            "requested_at": requested_at,
        }
        # A malformed decision must NOT be raised on after the resume value is accepted: LangGraph stores that value
        # and replays it, which poisons the thread (even a later VALID resume fails). So validate in a loop and ASK AGAIN.
        d = None
        last_error = ""
        for _attempt in range(MAX_DECISION_ATTEMPTS):
            raw = interrupt(payload if not last_error else {**payload, "error": last_error})
            try:
                d = Decision.model_validate(raw)
                break
            except ValidationError as exc:
                last_error = "; ".join(
                    f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
                )
        if d is None:
            _resolve_pending(db_path, state["ticket_id"])
            return {"outcome": "rejected", "audit": note("system", "too_many_invalid_decisions")}
        _resolve_pending(db_path, state["ticket_id"])
        if clock() - requested_at > APPROVAL_TTL_S:
            return {"outcome": "expired", "audit": note("system", "approval_expired")}
        if d.action == "reject":
            return {"outcome": "rejected", "audit": note(d.approver, "rejected", note=d.note)}
        amount = min(d.amount, state["amount"]) if d.amount else state["amount"]
        return {
            "amount": amount,
            "requested_at": requested_at,
            "outcome": "approved",
            "audit": note(d.approver, "approved", amount=amount, note=d.note),
        }

    def after_approval(state: RefundState) -> str:
        return "issue_refund" if state.get("outcome") == "approved" else "reject"

    def issue_refund(state: RefundState) -> dict:
        key = f"{state['ticket_id']}:{state['invoice_id']}"  # the idempotency key: a retry cannot pay twice
        refund_id = payments.refund(key, state["invoice_id"], state["amount"])
        return {
            "outcome": "refunded",
            "refund_id": refund_id,
            "refunded_amount": state["amount"],
            "audit": note("system", "refund_issued", refund_id=refund_id),
        }

    def reject(state: RefundState) -> dict:
        return {
            "outcome": state.get("outcome", "rejected"),
            "audit": note("system", "closed_without_refund"),
        }

    def notify(state: RefundState) -> dict:
        text = (drafter or default_message)(state["outcome"], state)
        return {"message": text, "audit": note("system", "customer_notified")}

    g = StateGraph(RefundState)
    for name, fn in (
        ("lookup", lookup),
        ("assess", assess),
        ("human_approval", human_approval),
        ("issue_refund", issue_refund),
        ("reject", reject),
        ("notify", notify),
    ):
        g.add_node(name, fn)
    g.add_edge(START, "lookup")
    g.add_conditional_edges("lookup", after_lookup, {"reject": "reject", "assess": "assess"})
    g.add_conditional_edges(
        "assess", after_assess, {"issue_refund": "issue_refund", "human_approval": "human_approval"}
    )
    g.add_conditional_edges(
        "human_approval", after_approval, {"issue_refund": "issue_refund", "reject": "reject"}
    )
    g.add_edge("issue_refund", "notify")
    g.add_edge("reject", "notify")
    g.add_edge("notify", END)
    return g.compile(checkpointer=checkpointer)


def default_message(outcome: str, state: RefundState) -> str:
    if outcome == "refunded":
        return f"Your refund of ${state['refunded_amount']:.2f} ({state['refund_id']}) has been issued."
    if outcome == "expired":
        return "We could not complete your refund request in time; please contact us again."
    return "We were not able to approve this refund. Please contact support if you have questions."


def llm_drafter(provider: str | None = None):
    """A model-written customer message, constrained to the facts of the decision (the model cannot change the outcome)."""

    def draft(outcome: str, state: RefundState) -> str:
        facts = default_message(outcome, state)
        prompt = f"Rewrite this support message in a warm, concise tone, two sentences at most, keeping every fact and number exactly:\n{facts}"
        text = llm.complete(prompt, provider=provider, max_tokens=120).text.strip()
        return (
            text if all(tok in text for tok in _facts(facts)) else facts
        )  # fall back if the model dropped a number

    return draft


def _facts(text: str) -> list[str]:
    import re

    return re.findall(r"\$\d+(?:\.\d+)?|RF-\d+", text)


# ----------------------------------------------------------------------------- the pending-approvals table (for a reviewer UI)


def _init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as c:
        c.execute(
            "CREATE TABLE IF NOT EXISTS approvals (ticket_id TEXT PRIMARY KEY, invoice_id TEXT, amount REAL, reason TEXT, requested_at REAL, resolved INTEGER DEFAULT 0)"
        )


def _register_pending(db_path, ticket_id, invoice_id, amount, reason, requested_at) -> None:
    with sqlite3.connect(
        db_path
    ) as c:  # INSERT OR IGNORE: idempotent, because this node body re-runs on resume
        c.execute(
            "INSERT OR IGNORE INTO approvals VALUES (?, ?, ?, ?, ?, 0)",
            (ticket_id, invoice_id, amount, reason, requested_at),
        )


def _resolve_pending(db_path, ticket_id) -> None:
    with sqlite3.connect(db_path) as c:
        c.execute("UPDATE approvals SET resolved = 1 WHERE ticket_id = ?", (ticket_id,))


# ----------------------------------------------------------------------------- the service a web app / worker would call


class RefundService:
    """Opens the database per call, so a NEW process (or a new instance) can pick up any thread: that is the point."""

    def __init__(
        self, db_path: str | Path, *, clock: Callable[[], float] = time.time, drafter=None
    ):
        self.db = str(db_path)
        self.clock = clock
        self.drafter = drafter
        _init_db(self.db)
        self.payments = Payments(self.db)

    @contextmanager
    def _app(self):
        with SqliteSaver.from_conn_string(self.db) as saver:
            yield build_refund_graph(
                saver, self.payments, self.db, clock=self.clock, drafter=self.drafter
            )

    @staticmethod
    def _cfg(ticket_id: str) -> dict:
        return {"configurable": {"thread_id": f"refund:{ticket_id}"}}

    @staticmethod
    def _summary(out: dict) -> dict:
        if "__interrupt__" in out:
            payload = out["__interrupt__"][-1].value
            return {
                "status": "pending_approval",
                "approval": payload,
                "audit": out.get("audit", []),
            }
        return {
            "status": out.get("outcome", "unknown"),
            "refund_id": out.get("refund_id"),
            "refunded_amount": out.get("refunded_amount"),
            "message": out.get("message"),
            "audit": out.get("audit", []),
        }

    def start(self, ticket_id: str, invoice_id: str, reason: str) -> dict:
        """Start (or fetch) a refund. Idempotent per ticket: a finished or paused ticket is not re-run."""
        with self._app() as app:
            cfg = self._cfg(ticket_id)
            snap = app.get_state(cfg)
            if snap.values:  # already started: never restart it
                interrupts = [i for t in snap.tasks for i in t.interrupts]
                if (
                    snap.next and not interrupts
                ):  # stopped by a CRASH (not a pause): continue where it stopped
                    return self._summary(app.invoke(None, cfg))
                return self._summary(
                    {**snap.values, **({"__interrupt__": interrupts} if interrupts else {})}
                )
            return self._summary(
                app.invoke(
                    {
                        "ticket_id": ticket_id,
                        "invoice_id": invoice_id,
                        "reason": reason,
                        "audit": [],
                    },
                    cfg,
                )
            )

    def decide(self, ticket_id: str, decision: dict) -> dict:
        """Resume a paused refund with a human's decision (validated inside the graph)."""
        valid = Decision.model_validate(decision) if isinstance(decision, dict) else None
        if (
            valid is None
        ):  # also guards a LangGraph 1.2 crash on Command(resume=None / a bare string)
            raise TypeError(
                f"a decision must be a dict like {{'action': 'approve', 'approver': 'dana'}}, got {type(decision).__name__}"
            )
        with self._app() as app:
            cfg = self._cfg(ticket_id)
            snap = app.get_state(cfg)
            if not snap.values:
                raise KeyError(f"no refund for ticket {ticket_id!r}")
            if not snap.next:
                return self._summary(
                    snap.values
                )  # already finished: a second click must not do anything
            return self._summary(
                app.invoke(Command(resume=valid.model_dump(exclude_none=True)), cfg)
            )

    def pending(self) -> list[dict]:
        with sqlite3.connect(self.db) as c:
            rows = c.execute(
                "SELECT ticket_id, invoice_id, amount, reason, requested_at FROM approvals WHERE resolved = 0 ORDER BY requested_at"
            ).fetchall()
        return [
            dict(
                zip(("ticket_id", "invoice_id", "amount", "reason", "requested_at"), r, strict=True)
            )
            for r in rows
        ]


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        svc = RefundService(Path(tmp) / "refunds.sqlite")
        for ticket, inv in (("T-1", "INV-3001"), ("T-2", "INV-1001"), ("T-3", "INV-2001")):
            r = svc.start(ticket, inv, "duplicate charge")
            print(ticket, inv, "->", r["status"], r.get("message") or r.get("approval"))
        print("pending:", [(p["ticket_id"], p["amount"]) for p in svc.pending()])
        r = svc.decide(
            "T-2",
            {
                "action": "approve",
                "approver": "dana",
                "note": "confirmed duplicate",
                "amount": 30.0,
            },
        )
        print("T-2 after approval ->", r["status"], r["message"])
        print("ledger:", svc.payments.ledger())


if __name__ == "__main__":
    main()
