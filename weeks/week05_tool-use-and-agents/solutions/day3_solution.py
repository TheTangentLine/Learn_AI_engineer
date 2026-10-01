"""Week 5 Day 3 - Solution: redesign badly designed tools and MEASURE the success-rate change.

The same capability (look up orders) is exposed two ways:

  BAD   search(q, f, n)  "Searches orders."   hidden mini-language in `f`, magic strings, opaque errors,
        get(id)          "Gets stuff."        overlaps with search, list_all() dumps everything, cryptic output
  GOOD  find_orders(customer_email, status, created_after, limit, newest_first)  enums, examples, pagination hints
        get_order(order_id)                  readable output, actionable errors

Experiments (a model call is greedy/deterministic, so "trials" are 40 different questions, not repeats):
  A. first-call accuracy: 5 question types x 8 phrasings, BAD vs GOOD, with bootstrap CIs and a paired test
  B. error recovery: the same opaque-vs-helpful error message, same tools; does the model fix its second call?
  C. context cost: how big are the results (characters), no model needed

  uv run python weeks/week05_tool-use-and-agents/solutions/day3_solution.py        # local Qwen-0.5B (slow, cached)
  LLM_PROVIDER=anthropic uv run python weeks/week05_tool-use-and-agents/solutions/day3_solution.py
"""

from __future__ import annotations

import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common import chat  # noqa: E402
from common.chat import ToolCall  # noqa: E402
from common.evalkit import bootstrap_ci, fmt_ci, paired_bootstrap  # noqa: E402
from common.tools import ToolRegistry, tool  # noqa: E402

Status = Literal["pending", "shipped", "delivered", "cancelled"]
STATUSES = ["pending", "shipped", "delivered", "cancelled"]
CUSTOMERS = ["alice@example.com", "bob@example.com", "carol@example.com", "dave@example.com"]
START = date(2026, 1, 1)
ID_RE = re.compile(r"\bA\d{4}\b")

# ----------------------------------------------------------------------------- the data


def build_db() -> list[dict]:
    """40 deterministic orders A1001..A1040. Statuses and totals come from a seeded RNG so that status is NOT
    correlated with the customer (a first version used a modular cycle and several question types had no answer)."""
    rng = random.Random(11)
    return [
        {
            "id": f"A{1000 + i}",
            "customer": CUSTOMERS[(i - 1) % 4],
            "status": rng.choice(STATUSES),
            "total": rng.randint(20, 199),
            "created": (START + timedelta(days=3 * i)).isoformat(),
        }
        for i in range(1, 41)
    ]


DB = build_db()


def fmt_good(o: dict) -> str:
    return (
        f"{o['id']} | {o['customer']} | {o['status']} | ${o['total']:.2f} | created {o['created']}"
    )


def fmt_bad(o: dict) -> str:
    return str(
        {
            "id": o["id"],
            "c": o["customer"],
            "s": STATUSES.index(o["status"]),
            "t": o["total"] * 100,
            "d": o["created"].replace("-", ""),
        }
    )


# ----------------------------------------------------------------------------- BAD tools

BAD_FILTER_HELP = (
    "Filters are key=value pairs separated by semicolons, e.g. 'status=shipped;after=2026-03-01;sort=desc'. "
    "Allowed keys: status (pending|shipped|delivered|cancelled), after (YYYY-MM-DD), sort (asc|desc)."
)


def make_bad_registry(db: list[dict] = DB, helpful_errors: bool = False) -> ToolRegistry:
    def bad_filter(f: str, reason: str) -> str:
        return (
            f"Error: bad filter {f!r} ({reason}). {BAD_FILTER_HELP}"
            if helpful_errors
            else "Error: bad filter"
        )

    @tool
    def search(q: str, f: str = "", n: int = 10) -> str:
        """Searches orders."""
        rows = [o for o in db if q.lower() in o["customer"] or q.upper() == o["id"] or not q]
        desc = False
        for part in [p for p in f.split(";") if p.strip()]:
            key, eq, value = part.partition("=")
            key, value = key.strip(), value.strip()
            if not eq:
                return bad_filter(f, f"{part!r} has no '='")
            if key == "status" and value in STATUSES:
                rows = [o for o in rows if o["status"] == value]
            elif key == "after":
                rows = [o for o in rows if o["created"] > value]
            elif key == "sort" and value in ("asc", "desc"):
                desc = value == "desc"
            else:
                return bad_filter(f, f"unknown filter {part!r}")
        rows = sorted(rows, key=lambda o: o["created"], reverse=desc)
        return "\n".join(fmt_bad(o) for o in rows[:n]) or "[]"

    @tool
    def get(id: str) -> str:
        """Gets stuff."""
        match = [o for o in db if o["id"] == id.upper()]
        return fmt_bad(match[0]) if match else "None"

    @tool
    def list_all() -> str:
        """Lists everything."""
        return "\n".join(fmt_bad(o) for o in db)

    return ToolRegistry([search, get, list_all])


# ----------------------------------------------------------------------------- GOOD tools


_ORDER_V1 = (
    'Results are oldest first unless newest_first is true, so use newest_first=true with a small limit for "most recent" questions.',
    "Sort newest first instead of oldest first.",
)
_ORDER_V2 = (
    "Results are newest first unless newest_first is false.",
    "Newest orders first (the default). Set false for oldest first.",
)


def make_good_registry(db: list[dict] = DB, newest_default: bool = False) -> ToolRegistry:
    """GOOD v1 sorts oldest first (the model never asked for newest); v2 makes the common case the default."""

    def find_orders(
        customer_email: str | None = None,
        status: Status | None = None,
        created_after: str | None = None,
        limit: int = 10,
        newest_first: bool = newest_default,
    ) -> str:
        """Find orders matching ALL the given filters. Use get_order instead when you already know the order id.

        Returns one order per line: id | customer | status | total | created date. {ORDER_NOTE}

        Args:
            customer_email: Only this customer's orders, e.g. "alice@example.com". Omit for all customers.
            status: Only orders in this status. Omit for any status.
            created_after: Only orders created strictly after this ISO date, e.g. "2026-03-01".
            limit: Maximum orders to return, 1-50. The reply says how many more matched.
            newest_first: {NEWEST_ARG}
        """
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        rows = list(db)
        if customer_email:
            rows = [o for o in rows if o["customer"] == customer_email.lower()]
        if status:
            rows = [o for o in rows if o["status"] == status]
        if created_after:
            try:
                date.fromisoformat(created_after)
            except ValueError:
                raise ValueError(
                    f"created_after must be an ISO date like 2026-03-01, got {created_after!r}"
                ) from None
            rows = [o for o in rows if o["created"] > created_after]
        rows = sorted(rows, key=lambda o: o["created"], reverse=newest_first)
        shown = rows[:limit]
        head = f"{len(shown)} of {len(rows)} matching orders" + (
            f" (raise limit to see the other {len(rows) - len(shown)})"
            if len(rows) > len(shown)
            else ""
        )
        return head + (
            ":\n" + "\n".join(fmt_good(o) for o in shown)
            if shown
            else ". Nothing matched; try fewer filters."
        )

    note, arg = _ORDER_V2 if newest_default else _ORDER_V1
    find_orders.__doc__ = (
        (find_orders.__doc__ or "").replace("{ORDER_NOTE}", note).replace("{NEWEST_ARG}", arg)
    )
    find_orders = tool(find_orders)

    @tool
    def get_order(order_id: str) -> str:
        """Get one order by its id, e.g. "A1007". Use find_orders to search by customer, status or date.

        Args:
            order_id: The order id: the letter A and four digits.
        """
        match = [o for o in db if o["id"] == order_id.strip().upper()]
        if not match:
            raise ValueError(
                f"no order with id {order_id!r}. Order ids look like A1001..A1040; use find_orders to search."
            )
        return fmt_good(match[0])

    return ToolRegistry([find_orders, get_order])


# ----------------------------------------------------------------------------- the questions


@dataclass
class Question:
    id: str
    kind: str
    text: str
    expected: list[str]  # ordered ids; scoring compares as a SET (order is a presentation matter)
    customer: str = ""
    status: str = ""


def _ids(rows) -> list[str]:
    return [o["id"] for o in rows]


def build_questions(db: list[dict] = DB) -> list[Question]:
    qs: list[Question] = []
    by_customer = [
        ("Which orders has {c} placed?", CUSTOMERS[0]), ("List every order from {c}.", CUSTOMERS[1]),
        ("Show me all of {c}'s orders.", CUSTOMERS[2]), ("What has {c} ordered so far?", CUSTOMERS[3]),
        ("I need the orders for {c}.", CUSTOMERS[1]), ("Pull up the order history of {c}.", CUSTOMERS[0]),
        ("Orders placed by {c}, please.", CUSTOMERS[3]), ("Find the orders belonging to {c}.", CUSTOMERS[2]),
    ]  # fmt: skip
    for i, (tpl, c) in enumerate(by_customer):
        qs.append(
            Question(
                f"cust-{i}",
                "customer",
                tpl.format(c=c),
                _ids(o for o in db if o["customer"] == c),
                customer=c,
            )
        )

    combos = [  # (customer, status) pairs that have at least two orders
        (CUSTOMERS[0], "shipped"), (CUSTOMERS[0], "cancelled"), (CUSTOMERS[1], "shipped"), (CUSTOMERS[1], "cancelled"),
        (CUSTOMERS[2], "pending"), (CUSTOMERS[2], "shipped"), (CUSTOMERS[3], "shipped"), (CUSTOMERS[3], "cancelled"),
    ]  # fmt: skip
    tpls = [
        "Show {c}'s {s} orders.",
        "Which {s} orders does {c} have?",
        "List the {s} orders for {c}.",
        "Any {s} orders from {c}?",
    ]
    for i, (c, s) in enumerate(combos):
        text = tpls[i % 4].format(c=c, s=s)
        qs.append(
            Question(
                f"cs-{i}",
                "customer+status",
                text,
                _ids(o for o in db if o["customer"] == c and o["status"] == s),
                customer=c,
                status=s,
            )
        )

    recent = [
        (3, "cancelled"),
        (2, "shipped"),
        (4, "pending"),
        (3, "delivered"),
        (2, "cancelled"),
        (5, "shipped"),
        (3, "pending"),
        (2, "delivered"),
    ]
    rtpls = [
        "What are the {n} most recent {s} orders?",
        "Show the latest {n} {s} orders.",
        "Give me the {n} newest {s} orders.",
    ]
    for i, (n, s) in enumerate(recent):
        rows = sorted(
            (o for o in db if o["status"] == s), key=lambda o: o["created"], reverse=True
        )[:n]
        qs.append(
            Question(f"recent-{i}", "recent", rtpls[i % 3].format(n=n, s=s), _ids(rows), status=s)
        )

    for i, oid in enumerate(
        ["A1007", "A1023", "A1040", "A1001", "A1015", "A1031", "A1002", "A1036"]
    ):
        tpl = [
            "What is the status of order {o}?",
            "Look up order {o}.",
            "Details for {o}, please.",
            "Tell me about order {o}.",
        ][i % 4]
        qs.append(Question(f"id-{i}", "by-id", tpl.format(o=oid), [oid]))

    after = [
        ("2026-03-01", "pending"),
        ("2026-02-15", "delivered"),
        ("2026-04-01", "cancelled"),
        ("2026-03-15", "shipped"),
        ("2026-04-20", "pending"),
        ("2026-04-10", "delivered"),
        ("2026-03-20", "cancelled"),
        ("2026-02-01", "shipped"),
    ]
    atpls = [
        "List {s} orders created after {d}.",
        "Which {s} orders were placed after {d}?",
        "Show {s} orders since {d} (exclusive).",
    ]
    for i, (d, s) in enumerate(after):
        qs.append(
            Question(
                f"after-{i}",
                "after-date",
                atpls[i % 3].format(s=s, d=d),
                _ids(o for o in db if o["status"] == s and o["created"] > d),
                status=s,
            )
        )
    return qs


def build_holdout(db: list[dict] = DB) -> list[Question]:
    """8 NEW 'most recent' questions with different wording and counts, written after v2 was designed.
    v2 was a response to v1's failures on the 40 questions above, so measuring it on those alone is optimistic."""
    items = [
        (2, "pending", "What were the last {n} {s} orders placed?"),
        (4, "cancelled", "Which {s} orders came in most recently? I need {n}."),
        (3, "shipped", "Give me the {n} latest {s} orders, newest first."),
        (2, "delivered", "Most recently created {s} orders (just {n}) please."),
        (5, "pending", "Show the {n} most recently placed {s} orders."),
        (3, "cancelled", "The {n} newest {s} orders?"),
        (4, "delivered", "Fetch the {n} latest {s} orders."),
        (2, "shipped", "What are the {n} most recent {s} orders?"),
    ]
    out = []
    for i, (n, s, tpl) in enumerate(items):
        rows = sorted(
            (o for o in db if o["status"] == s), key=lambda o: o["created"], reverse=True
        )[:n]
        out.append(Question(f"hold-{i}", "recent", tpl.format(n=n, s=s), _ids(rows), status=s))
    return out


# ----------------------------------------------------------------------------- scoring

SYSTEM = "You answer questions about customer orders. Look things up with the tools; do not guess."


@dataclass
class Outcome:
    q: Question
    calls: list[ToolCall]
    output: str
    n_errors: int
    ok: bool  # the executed calls returned exactly the expected orders
    said: str = ""  # what the model SAID: matters when it answers instead of retrying

    @property
    def called(self) -> bool:
        return bool(self.calls)


def score_calls(
    q: Question, registry: ToolRegistry, calls: list[ToolCall], said: str = ""
) -> Outcome:
    results = registry.execute_all(calls)
    text = "\n".join(r.content for r in results)
    returned = set(ID_RE.findall(text))
    return Outcome(
        q,
        calls,
        text,
        sum(r.is_error or r.content.startswith("Error") for r in results),
        bool(calls) and returned == set(q.expected),
        said,
    )


def first_call_eval(
    registry: ToolRegistry, questions: list[Question], provider: str | None
) -> list[Outcome]:
    out = []
    for q in questions:
        t = chat.turn(
            [{"role": "user", "content": q.text}],
            registry.specs(),
            system=SYSTEM,
            provider=provider,
        )
        out.append(score_calls(q, registry, t.tool_calls))
    return out


def wrong_first_call(q: Question) -> ToolCall:
    """A realistic mistake on the BAD tools: a colon instead of '=' in the filter."""
    return ToolCall("c0", "search", {"q": q.customer, "f": f"status:{q.status}"})


def error_recovery_eval(
    registry: ToolRegistry, questions: list[Question], provider: str | None
) -> list[Outcome]:
    """Show the model its own failing call and the error text, then score its SECOND call."""
    out = []
    for q in questions:
        first = wrong_first_call(q)
        res = registry.execute(first)
        msgs = [
            {"role": "user", "content": q.text},
            chat.assistant_message(
                chat.Turn("", [first], "tool_use", chat.Usage(), 0.0, 0.0, "", "")
            ),
            chat.tool_message(first, res.content, res.is_error or res.content.startswith("Error")),
        ]
        t = chat.turn(msgs, registry.specs(), system=SYSTEM, provider=provider)
        out.append(score_calls(q, registry, t.tool_calls, t.text))
    return out


def uses_fix(o: Outcome) -> bool:
    """Heuristic: the reply states the key=value syntax or a status=<value> pair (the model understood the error)."""
    return re.search(r"key=value|status=\w+", o.said) is not None


def rate(outcomes: list[Outcome], pick=lambda o: o.ok) -> list[float]:
    return [1.0 if pick(o) else 0.0 for o in outcomes]


def by_kind(outcomes: list[Outcome]) -> dict[str, tuple[int, int]]:
    kinds: dict[str, list[bool]] = {}
    for o in outcomes:
        kinds.setdefault(o.q.kind, []).append(o.ok)
    return {k: (sum(v), len(v)) for k, v in kinds.items()}


def lint_spec(spec: dict) -> list[str]:
    """A cheap static check of a tool definition: the things a reviewer would flag before any model sees it."""
    issues = []
    desc = spec["description"].strip()
    if len(desc) < 40:
        issues.append(
            f"description is only {len(desc)} characters: say what it does, returns and when NOT to use it"
        )
    props = spec["parameters"].get("properties", {})
    for name, p in props.items():
        if not p.get("description"):
            issues.append(f"parameter '{name}' has no description")
        if len(name) <= 1:
            issues.append(f"parameter '{name}' has a cryptic name")
        if name in ("status", "sort", "order", "mode", "type", "kind") and "enum" not in str(p):
            issues.append(f"parameter '{name}' looks like a closed set but has no enum")
    return issues


def context_cost() -> dict[str, int]:
    """Characters a result adds to the context for the same 'show me orders' intent."""
    bad, good = make_bad_registry(), make_good_registry()
    sizes = {
        "list_all (BAD)": len(bad.execute(ToolCall("x", "list_all", {})).content),
        "search n=10 (BAD)": len(
            bad.execute(ToolCall("x", "search", {"q": "alice@example.com"})).content
        ),
        "find_orders limit=10 (GOOD)": len(
            good.execute(
                ToolCall("x", "find_orders", {"customer_email": "alice@example.com"})
            ).content
        ),
        "find_orders limit=3 (GOOD)": len(
            good.execute(
                ToolCall("x", "find_orders", {"customer_email": "alice@example.com", "limit": 3})
            ).content
        ),
    }
    return sizes


def main() -> None:
    provider = os.environ.get("LLM_PROVIDER") or "local"
    questions = build_questions()
    print(f"provider={provider}  questions={len(questions)}\n")

    bad, good = make_bad_registry(), make_good_registry()
    a_bad = first_call_eval(bad, questions, provider)
    a_good = first_call_eval(good, questions, provider)
    print("== A. first-call accuracy (exactly the right orders returned)")
    for name, res in (("BAD ", a_bad), ("GOOD", a_good)):
        print(
            f"  {name}: success {fmt_ci(bootstrap_ci(rate(res)))}   called a tool {sum(o.called for o in res)}/{len(res)}   errors {sum(o.n_errors for o in res)}"
        )
        print(f"        by kind: {by_kind(res)}")
    p = paired_bootstrap(rate(a_good), rate(a_bad))
    print(
        f"  paired GOOD-BAD: diff {p['diff']:+.0%} [{p['ci_low']:+.0%}, {p['ci_high']:+.0%}] wins/losses/ties {p['wins']}/{p['losses']}/{p['ties']}\n"
    )

    cs = [q for q in questions if q.kind == "customer+status"]
    b_opaque = error_recovery_eval(make_bad_registry(), cs, provider)
    b_helpful = error_recovery_eval(make_bad_registry(helpful_errors=True), cs, provider)
    print("== B. error recovery on the BAD tools (second call after a failing first call)")
    for name, res in (
        ("opaque error  'Error: bad filter'       ", b_opaque),
        ("helpful error (syntax + allowed values)", b_helpful),
    ):
        print(
            f"  {name}: retried correctly {sum(o.ok for o in res)}/{len(res)}   "
            f"retried at all {sum(o.called for o in res)}/{len(res)}   "
            f"reply states the fix {sum(uses_fix(o) for o in res)}/{len(res)}"
        )
    print()

    v2 = first_call_eval(make_good_registry(newest_default=True), questions, provider)
    print("== D. GOOD v2 (newest first by default; the only change)")
    print(f"  success {fmt_ci(bootstrap_ci(rate(v2)))}   by kind: {by_kind(v2)}")
    p2 = paired_bootstrap(rate(v2), rate(a_good))
    print(
        f"  paired v2-v1: diff {p2['diff']:+.0%} [{p2['ci_low']:+.0%}, {p2['ci_high']:+.0%}] wins/losses/ties {p2['wins']}/{p2['losses']}/{p2['ties']}\n"
    )

    hold = build_holdout()
    h1 = first_call_eval(good, hold, provider)
    h2 = first_call_eval(make_good_registry(newest_default=True), hold, provider)
    print("== E. held-out 'most recent' questions (written after v2 was designed)")
    print(
        f"  GOOD v1: {sum(o.ok for o in h1)}/{len(hold)}   GOOD v2: {sum(o.ok for o in h2)}/{len(hold)}\n"
    )

    print("== static lint of the tool definitions")
    for name, reg in (("BAD", bad), ("GOOD", good)):
        for spec in reg.specs():
            print(f"  {name} {spec['name']}: {len(lint_spec(spec))} issue(s)")
    print()

    print("== C. context cost of one result (characters)")
    for k, v in context_cost().items():
        print(f"  {k:30s} {v:6d}")


if __name__ == "__main__":
    main()
