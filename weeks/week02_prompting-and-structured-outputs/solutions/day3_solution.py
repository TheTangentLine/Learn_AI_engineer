"""Week 2 Day 3 - Solution: typed extraction from messy emails, with validate-and-retry.

Layers of defence, in the order you should apply them:
  1. a precise Pydantic schema (types, enums, Optional = "may be absent", field descriptions)
  2. NATIVE structured output (the provider constrains decoding to the schema)   -> extract_native
  3. semantic validation the schema can't express (dates, cross-field rules)
  4. validate-and-retry: feed the *exact error* back to the model                -> extract_with_retry

  uv run python .../day3_solution.py             # real provider
  uv run python .../day3_solution.py --offline   # scripted fake model that fails first, then recovers
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common import llm  # noqa: E402
from common.fake import fake_llm  # noqa: E402

RECEIVED = date(2026, 10, 1)  # when the emails arrived (used for a semantic check)

# ----------------------------------------------------------------- schema


class Item(BaseModel):
    name: str = Field(description="Product name without the quantity, e.g. 'blue widgets'")
    quantity: int = Field(description="Number of units ordered", ge=1)


class Order(BaseModel):
    is_order: bool = Field(
        description="False if the email does not place an order (e.g. a cancellation)"
    )
    customer_name: str | None = Field(None, description="Full name of the customer, if stated")
    order_id: str | None = Field(None, description="Order reference such as 'A-1042', if stated")
    items: list[Item] = Field(default_factory=list)
    urgency: Literal["low", "normal", "high"] = Field(
        "normal", description="high if urgent/ASAP, low if the sender says there is no rush"
    )
    delivery_date: date | None = Field(
        None, description="Requested delivery date as ISO-8601 (YYYY-MM-DD)"
    )
    total_amount: float | None = Field(None, description="Total price as a number, if stated")
    currency: Literal["USD", "EUR", "GBP"] | None = Field(
        None, description="Currency of total_amount"
    )

    @model_validator(mode="after")
    def order_needs_core_fields(self) -> Order:
        if self.is_order and (not self.order_id or not self.customer_name or not self.items):
            raise ValueError("an order must have customer_name, order_id and at least one item")
        return self


def semantic_checks(o: Order) -> None:
    """Rules a JSON schema can't express. Raise ValueError with a message the model can act on."""
    if o.delivery_date and o.delivery_date < RECEIVED:
        raise ValueError(
            f"delivery_date {o.delivery_date} is before the email was received ({RECEIVED})"
        )
    if (o.total_amount is None) != (o.currency is None):
        raise ValueError("total_amount and currency must be given together (or both null)")


SYSTEM = (
    "You extract structured order data from customer emails. Use null for anything the email "
    "does not state; never guess. Normalise dates to YYYY-MM-DD."
)


def prompt_for(email: str) -> str:
    return f"<email>\n{email}\n</email>\n\nExtract the order. Today is {RECEIVED}."


# ----------------------------------------------------------------- extraction strategies


def parse_and_validate(text: str) -> Order:
    """Tolerant JSON parse (strip code fences) + schema + semantic validation."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    order = Order.model_validate_json(body)
    semantic_checks(order)
    return order


@dataclass
class Extraction:
    order: Order | None
    attempts: int
    errors: list[str] = field(default_factory=list)


def extract_with_retry(email: str, max_attempts: int = 3) -> Extraction:
    """Provider-independent: ask for JSON, validate, and send the error back on failure."""
    schema_hint = json.dumps(Order.model_json_schema())
    messages = [
        {
            "role": "user",
            "content": prompt_for(email) + f"\n\nReply with only JSON matching:\n{schema_hint}",
        }
    ]
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        reply = llm.complete(messages, system=SYSTEM, max_tokens=800).text
        try:
            return Extraction(parse_and_validate(reply), attempt, errors)
        except (ValidationError, ValueError) as exc:
            msg = str(exc)[:500]
            errors.append(msg)
            messages += [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": f"That output failed validation:\n{msg}\nFix it and reply "
                    "with only the corrected JSON.",
                },
            ]
    return Extraction(None, max_attempts, errors)


def extract_native(email: str, max_attempts: int = 2) -> Extraction:
    """Use the provider's schema-constrained output; still run semantic checks + retry once."""
    errors: list[str] = []
    messages = [{"role": "user", "content": prompt_for(email)}]
    for attempt in range(1, max_attempts + 1):
        try:
            order, _ = llm.structured(messages, Order, system=SYSTEM, max_tokens=800)
            semantic_checks(order)
            return Extraction(order, attempt, errors)
        except (ValidationError, ValueError) as exc:
            errors.append(str(exc)[:300])
            messages = messages + [
                {"role": "user", "content": f"Previous output was rejected: {errors[-1]}"}
            ]
    return Extraction(None, max_attempts, errors)


# ----------------------------------------------------------------- data

EMAILS: list[tuple[str, dict]] = [
    (
        "Hi team, it's Dana Whitfield. Order #A-1042 please: 3x blue widgets and 2 red gaskets. "
        "Need them by 2026-11-05. Total should come to $149.50. Thanks!",
        dict(
            is_order=True,
            customer_name="Dana Whitfield",
            order_id="A-1042",
            items=[("blue widgets", 3), ("red gaskets", 2)],
            urgency="normal",
            delivery_date="2026-11-05",
            total_amount=149.5,
            currency="USD",
        ),
    ),
    (
        "URGENT!!! Order B-7 - need 10 pallets of cement ASAP, ship to Leeds. - M. Okafor",
        dict(
            is_order=True,
            customer_name="M. Okafor",
            order_id="B-7",
            items=[("pallets of cement", 10)],
            urgency="high",
            delivery_date=None,
            total_amount=None,
            currency=None,
        ),
    ),
    (
        "Hello, ordering 1 desk lamp for Chen Wei (order C-2231). There's no rush; delivery by "
        "2026-12-15 would be great. EUR 45",
        dict(
            is_order=True,
            customer_name="Chen Wei",
            order_id="C-2231",
            items=[("desk lamp", 1)],
            urgency="low",
            delivery_date="2026-12-15",
            total_amount=45.0,
            currency="EUR",
        ),
    ),
    (
        "Sam Rivera here, re order D-900: please add 4 AA batteries and 1 charger. GBP 23.99 all in. "
        "Delivery date TBD.",
        dict(
            is_order=True,
            customer_name="Sam Rivera",
            order_id="D-900",
            items=[("AA batteries", 4), ("charger", 1)],
            urgency="normal",
            delivery_date=None,
            total_amount=23.99,
            currency="GBP",
        ),
    ),
    (
        "From: priya@acme.example\nOrder E-12\n- 2 x laptop sleeve\n- 5 x usb cable\nPriya Nair\n"
        "needed by Nov 20 2026",
        dict(
            is_order=True,
            customer_name="Priya Nair",
            order_id="E-12",
            items=[("laptop sleeve", 2), ("usb cable", 5)],
            urgency="normal",
            delivery_date="2026-11-20",
            total_amount=None,
            currency=None,
        ),
    ),
    (
        "this is Lena Kowalski. order F-3. 12 mugs. asap pls. total 60 USD",
        dict(
            is_order=True,
            customer_name="Lena Kowalski",
            order_id="F-3",
            items=[("mugs", 12)],
            urgency="high",
            delivery_date=None,
            total_amount=60.0,
            currency="USD",
        ),
    ),
    (
        "Order G-55 for Tomas Ruiz: 6 stools. Not urgent. Please deliver on the 3rd of January 2027. "
        "Quote total 310 USD.",
        dict(
            is_order=True,
            customer_name="Tomas Ruiz",
            order_id="G-55",
            items=[("stools", 6)],
            urgency="low",
            delivery_date="2027-01-03",
            total_amount=310.0,
            currency="USD",
        ),
    ),
    (
        "Hi, I'd like to cancel my earlier request, never mind! Thanks, Alex",
        dict(
            is_order=False,
            customer_name=None,
            order_id=None,
            items=[],
            urgency="normal",
            delivery_date=None,
            total_amount=None,
            currency=None,
        ),
    ),
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower()).rstrip("s")


FIELDS = [
    "is_order",
    "customer_name",
    "order_id",
    "items",
    "urgency",
    "delivery_date",
    "total_amount",
    "currency",
]


def field_matches(o: Order, gold: dict) -> dict[str, bool]:
    got_items = sorted((i.quantity, _norm(i.name)) for i in o.items)
    gold_items = sorted((q, _norm(n)) for n, q in gold["items"])
    out = {}
    for f in FIELDS:
        if f == "items":
            out[f] = got_items == gold_items
        elif f == "delivery_date":
            out[f] = (o.delivery_date.isoformat() if o.delivery_date else None) == gold[f]
        elif f == "customer_name" and gold[f]:
            out[f] = bool(o.customer_name) and _norm(o.customer_name) == _norm(gold[f])
        else:
            out[f] = getattr(o, f) == gold[f]
    return out


def evaluate(extract_fn, label: str) -> dict:
    n_first_try = n_final = 0
    field_hits = {f: 0 for f in FIELDS}
    for email, gold in EMAILS:
        ex = extract_fn(email)
        n_first_try += ex.attempts == 1 and ex.order is not None
        if ex.order:
            n_final += 1
            for f, ok in field_matches(ex.order, gold).items():
                field_hits[f] += ok
        else:
            print(f"  FAILED: {email[:40]!r} errors={ex.errors}")
    n = len(EMAILS)
    total_fields = n * len(FIELDS)
    acc = sum(field_hits.values()) / total_fields
    print(
        f"{label:22} first-try valid {n_first_try}/{n} | final valid {n_final}/{n} | "
        f"field accuracy {acc:.0%}"
    )
    for f, h in field_hits.items():
        if h < n:
            print(f"    field {f}: {h}/{n}")
    return {"first_try": n_first_try, "final": n_final, "accuracy": acc}


# ----------------------------------------------------------------- offline harness


def offline_model():
    """Scripted model: returns gold JSON, but botches the FIRST attempt on some emails."""
    gold_by_email = {e: g for e, g in EMAILS}

    def responder(prompt: str) -> str:
        email = next(e for e in gold_by_email if e in prompt)
        gold = gold_by_email[email]
        body = {**gold, "items": [{"name": n, "quantity": q} for n, q in gold["items"]]}
        if "failed validation" in prompt or "was rejected" in prompt:
            return json.dumps(body)  # recovered after seeing the error
        idx = [e for e, _ in EMAILS].index(email)
        if idx == 1:
            return "```json\n" + json.dumps(body) + "\n```"  # fenced: tolerated by the parser
        if idx == 2:
            return json.dumps({**body, "urgency": "not urgent"})  # invalid enum value
        if idx == 4:
            return json.dumps({**body, "customer_name": None})  # violates order_needs_core_fields
        if idx == 6:
            return json.dumps({**body, "delivery_date": "2025-01-03"})  # semantic: in the past
        return json.dumps(body)

    return responder


def main() -> None:
    if "--offline" in sys.argv:
        print(
            "*** OFFLINE: scripted model that botches the first attempt on 4 of 8 emails (fenced JSON is tolerated, 3 need a retry) ***"
        )
        with fake_llm([(r"(?s).*", offline_model())]) as fake:
            r = evaluate(extract_with_retry, "validate-and-retry")
            assert r["final"] == len(EMAILS) and r["accuracy"] == 1.0
            assert r["first_try"] == len(EMAILS) - 3, (
                "fenced JSON is tolerated; 3 others need a retry"
            )
            retried = fake.calls_matching("failed validation")
            assert len(retried) == 3 and "urgency" in retried[0].prompt
            print(f"retry prompts sent: {len(retried)} (each quotes the exact validation error)")
            r = evaluate(extract_native, "native + semantic retry")
            assert r["final"] == len(EMAILS)
        print("\nself-test passed")
        return
    print(f"provider: {llm.resolve()}")
    evaluate(extract_native, "native structured")
    evaluate(extract_with_retry, "prompt+validate+retry")


if __name__ == "__main__":
    main()
