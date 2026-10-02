"""The task of Week 10: turn a messy customer email into a typed ``Order`` (the Week 2 Day 3 schema), as compact JSON.

Order, parse_and_validate     the Week 2 schema and its checks (re-used, not re-written)
order_json(gold)              the exact JSON text a model must produce (compact, fixed key order)
score(reply, gold)            did the model produce valid JSON? a valid ORDER? which of the 8 fields are right?
HUMAN_EMAILS                  40 emails written by hand: the 10 of Week 2 plus 30 new ones, with gold labels. NEVER used for training.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "weeks/week07_evals-observability-llmops/solutions"))

from _weeks import load  # noqa: E402

_w2 = load("week02_prompting-and-structured-outputs", "day3_solution")
Order, Item = _w2.Order, _w2.Item
semantic_checks, field_matches, FIELDS = _w2.semantic_checks, _w2.field_matches, _w2.FIELDS
WEEK2_EMAILS: list[tuple[str, dict]] = _w2.EMAILS
RECEIVED = _w2.RECEIVED

SYSTEM_SHORT = "Extract the order from the email as JSON."
SCHEMA_PROMPT = (
    "You extract structured order data from customer emails. Reply with ONLY a JSON object with exactly these keys:\n"
    '"is_order" (true/false: false if the email does not place an order, e.g. a cancellation, question or complaint), '
    '"customer_name" (string or null), "order_id" (the order reference exactly as written in the email, or null), '
    '"items" (list of {"name": string, "quantity": integer}), '
    '"urgency" ("low" if there is no rush, "high" if urgent/ASAP, otherwise "normal"), '
    '"delivery_date" (YYYY-MM-DD or null), "total_amount" (number or null), "currency" ("USD", "EUR", "GBP" or null). '
    "Use null for anything the email does not state. Never guess."
)


def gold_to_order(gold: dict) -> Order:
    return Order(
        is_order=gold["is_order"],
        customer_name=gold["customer_name"],
        order_id=gold["order_id"],
        items=[Item(name=n, quantity=q) for n, q in gold["items"]],
        urgency=gold["urgency"],
        delivery_date=gold["delivery_date"],
        total_amount=gold["total_amount"],
        currency=gold["currency"],
    )


def order_json(gold: dict) -> str:
    """The training target: compact JSON, fixed key order, dates as ISO strings."""
    o = gold_to_order(gold)
    d = o.model_dump(mode="json")
    return json.dumps({k: d[k] for k in FIELDS}, separators=(",", ":"), ensure_ascii=False)


def parse_reply(text: str) -> Order:
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    o = Order.model_validate_json(body)
    semantic_checks(o)
    return o


@dataclass
class Score:
    valid_json: bool = False
    valid_order: bool = False  # parses, matches the schema AND passes the semantic checks
    fields: dict[str, bool] = field(default_factory=dict)
    error: str = ""

    @property
    def exact(self) -> bool:
        return self.valid_order and all(self.fields.values()) and len(self.fields) == len(FIELDS)

    @property
    def n_correct(self) -> int:
        return sum(self.fields.values()) if self.valid_order else 0


def score(reply: str, gold: dict) -> Score:
    s = Score()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
    try:
        json.loads(body)
        s.valid_json = True
    except ValueError as exc:
        s.error = f"not JSON: {str(exc)[:80]}"
        return s
    try:
        order = parse_reply(reply)
    except Exception as exc:  # noqa: BLE001 - pydantic and value errors alike are "not a valid order"
        s.error = f"invalid order: {str(exc).splitlines()[0][:100]}"
        return s
    s.valid_order = True
    s.fields = field_matches(order, gold)
    return s


def summarize(scores: list[Score]) -> dict:
    n = len(scores)
    fields = {f: sum(s.fields.get(f, False) and s.valid_order for s in scores) / n for f in FIELDS}
    return {
        "n": n,
        "valid_json": sum(s.valid_json for s in scores) / n,
        "valid_order": sum(s.valid_order for s in scores) / n,
        "exact": sum(s.exact for s in scores) / n,
        "field_accuracy": sum(s.n_correct for s in scores) / (n * len(FIELDS)),
        "fields": fields,
    }


# ----------------------------------------------------------------------------- the hand-written evaluation set


def _g(
    is_order=True, name=None, oid=None, items=(), urgency="normal", date=None, total=None, cur=None
):
    return dict(
        is_order=is_order,
        customer_name=name,
        order_id=oid,
        items=[tuple(i) for i in items],
        urgency=urgency,
        delivery_date=date,
        total_amount=total,
        currency=cur,
    )


NOT_AN_ORDER = _g(is_order=False)

NEW_EMAILS: list[tuple[str, dict]] = [
    (
        "Good morning,\n\nPlease process order H-410 for Jamal Brooks. We need 8 ergonomic chairs and 8 monitor arms. Delivery by 15 December 2026 please. Budget approved: £2,340.00.\n\nRegards,\nJamal",
        _g(
            name="Jamal Brooks",
            oid="H-410",
            items=[("ergonomic chairs", 8), ("monitor arms", 8)],
            date="2026-12-15",
            total=2340.0,
            cur="GBP",
        ),
    ),
    (
        "Quick one - Ines Carvalho, order J-77, 20 rolls of packing tape. Whenever is fine, no hurry at all.",
        _g(name="Ines Carvalho", oid="J-77", items=[("rolls of packing tape", 20)], urgency="low"),
    ),
    (
        "hey its Oliver. need 2 standing desks for order K-9001 ASAP!!! very urgent",
        _g(name="Oliver", oid="K-9001", items=[("standing desks", 2)], urgency="high"),
    ),
    (
        "Hello, could you tell me when order L-18 will arrive? It's been two weeks. Thanks, Grace Liu",
        NOT_AN_ORDER,
    ),
    (
        "Dear Sales,\nI would like to purchase 15 paper reams and 3 staplers for my office. My name is Beatriz Alves and the reference is M-502. Please deliver before 2 February 2027. We will pay 180 EUR upon delivery.\nThank you",
        _g(
            name="Beatriz Alves",
            oid="M-502",
            items=[("paper reams", 15), ("staplers", 3)],
            date="2027-02-02",
            total=180.0,
            cur="EUR",
        ),
    ),
    (
        "From: sven@nordic.example\nSubject: order N-6\n\n1 x conference table\n4 x whiteboard markers (black)\n\nSven Lindqvist\nneeded urgently - client visit on Monday",
        _g(
            name="Sven Lindqvist",
            oid="N-6",
            items=[("conference table", 1), ("whiteboard markers (black)", 4)],
            urgency="high",
        ),
    ),
    ("Please cancel order P-31, the customer changed their mind. - Ravi", NOT_AN_ORDER),
    (
        "Order Q-1203, Hannah Müller: 100 hex bolts M8, 100 nuts M8, 50 washers. Total 75 EUR.",
        _g(
            name="Hannah Müller",
            oid="Q-1203",
            items=[("hex bolts M8", 100), ("nuts M8", 100), ("washers", 50)],
            total=75.0,
            cur="EUR",
        ),
    ),
    (
        "Greetings! Chloe Park here with a small order (R-44): three cactus pots. Could they arrive by 9 Nov 2026? $27 total.",
        _g(
            name="Chloe Park",
            oid="R-44",
            items=[("cactus pots", 3)],
            date="2026-11-09",
            total=27.0,
            cur="USD",
        ),
    ),
    (
        "Thanks for the quick delivery last time!! Just wanted to say the lamps are great. -Mei",
        NOT_AN_ORDER,
    ),
    (
        "ORDER S-8\nCustomer: Diego Herrera\nItems: 2 office phones; 1 headset\nDeadline: 2026-10-30\nPriority: high\nAmount: 410.00 USD",
        _g(
            name="Diego Herrera",
            oid="S-8",
            items=[("office phones", 2), ("headset", 1)],
            urgency="high",
            date="2026-10-30",
            total=410.0,
            cur="USD",
        ),
    ),
    (
        "I'd like to order a replacement battery for my laptop, order id T-15, name is Zainab Khan. Nothing urgent.",
        _g(name="Zainab Khan", oid="T-15", items=[("replacement battery", 1)], urgency="low"),
    ),
    (
        "Hi, Tom Becker. U-300. 6 garden hoses and 2 sprinklers please. If you can get them here before 12 Dec 2026 that'd be perfect. Thx",
        _g(
            name="Tom Becker",
            oid="U-300",
            items=[("garden hoses", 6), ("sprinklers", 2)],
            date="2026-12-12",
        ),
    ),
    ("Can I get a quote for 500 flyers? No order yet.", NOT_AN_ORDER),
    (
        "Order V-72 / Ayesha Siddiqui / 40 notebooks / 40 pens / £96 / ASAP",
        _g(
            name="Ayesha Siddiqui",
            oid="V-72",
            items=[("notebooks", 40), ("pens", 40)],
            urgency="high",
            total=96.0,
            cur="GBP",
        ),
    ),
    (
        "Hello. This is Pierre Dubois, order W-5. I need 12 bottles of olive oil and 6 jars of capers. The total is 142,50 EUR. Delivery on the 22nd of November 2026.",
        _g(
            name="Pierre Dubois",
            oid="W-5",
            items=[("bottles of olive oil", 12), ("jars of capers", 6)],
            date="2026-11-22",
            total=142.5,
            cur="EUR",
        ),
    ),
    (
        "Hi! I need 2 yoga mats (order Y-81). Name: Lucy Tran. Delivery no later than January 3, 2027.",
        _g(name="Lucy Tran", oid="Y-81", items=[("yoga mats", 2)], date="2027-01-03"),
    ),
    (
        "Order Z-2 for Omar Haddad: 1 projector. Low priority. USD 899.",
        _g(
            name="Omar Haddad",
            oid="Z-2",
            items=[("projector", 1)],
            urgency="low",
            total=899.0,
            cur="USD",
        ),
    ),
    (
        "bulk order AA-17 from Fatima Zahra: 250 t-shirts (medium), 250 t-shirts (large). need by 1 Dec 2026, total 3,125 USD. this is urgent",
        _g(
            name="Fatima Zahra",
            oid="AA-17",
            items=[("t-shirts (medium)", 250), ("t-shirts (large)", 250)],
            urgency="high",
            date="2026-12-01",
            total=3125.0,
            cur="USD",
        ),
    ),
    (
        "Dear team, kindly note that we wish to place an order, reference AB-640, under the name Elena Petrova, for 9 desk organizers. Delivery date: 18 January 2027. Price agreed: EUR 117.",
        _g(
            name="Elena Petrova",
            oid="AB-640",
            items=[("desk organizers", 9)],
            date="2027-01-18",
            total=117.0,
            cur="EUR",
        ),
    ),
    ("Your invoice for order AC-3 has the wrong address. Please fix.", NOT_AN_ORDER),
    (
        "Jin Park - order AD-55 - 3 keyboards, 3 mice - GBP 150 - not urgent but before 14 Dec 2026 would be ideal",
        _g(
            name="Jin Park",
            oid="AD-55",
            items=[("keyboards", 3), ("mice", 3)],
            urgency="low",
            date="2026-12-14",
            total=150.0,
            cur="GBP",
        ),
    ),
    (
        "yo, Marcus again. order AE-9: just 1 coffee machine. need it ASAP, the old one died",
        _g(name="Marcus", oid="AE-9", items=[("coffee machine", 1)], urgency="high"),
    ),
    (
        "Hi, I'm ordering for Sofia Rossi (order AF-210): 2 wall clocks and 4 picture frames. Please send by 5 Jan 2027. The total with shipping should be $88.",
        _g(
            name="Sofia Rossi",
            oid="AF-210",
            items=[("wall clocks", 2), ("picture frames", 4)],
            date="2027-01-05",
            total=88.0,
            cur="USD",
        ),
    ),
    (
        "From: kofi@example.org\n\nOrder AG-4. Kofi Mensah. Five (5) tool boxes. EUR 230 total. Normal delivery is fine.",
        _g(name="Kofi Mensah", oid="AG-4", items=[("tool boxes", 5)], total=230.0, cur="EUR"),
    ),
    (
        "URGENT URGENT order AI-1 Karim Aziz 30 face masks need today",
        _g(name="Karim Aziz", oid="AI-1", items=[("face masks", 30)], urgency="high"),
    ),
    (
        "Regarding AJ-12: customer Ben Okoro orders 1 bike helmet and 1 bike lock. £64.99. They said take your time.",
        _g(
            name="Ben Okoro",
            oid="AJ-12",
            items=[("bike helmet", 1), ("bike lock", 1)],
            urgency="low",
            total=64.99,
            cur="GBP",
        ),
    ),
    ("Out of office until Monday, will respond then.", NOT_AN_ORDER),
    (
        "Hi, this is Amara Nwosu. For order AK-31 please send 14 candles and 2 diffusers by the 6th of December 2026. Total 96 GBP.",
        _g(
            name="Amara Nwosu",
            oid="AK-31",
            items=[("candles", 14), ("diffusers", 2)],
            date="2026-12-06",
            total=96.0,
            cur="GBP",
        ),
    ),
    (
        "order AL-8 - Henrik Berg - 7 snow shovels - rush please - total EUR 210",
        _g(
            name="Henrik Berg",
            oid="AL-8",
            items=[("snow shovels", 7)],
            urgency="high",
            total=210.0,
            cur="EUR",
        ),
    ),
]

HUMAN_EMAILS: list[tuple[str, dict]] = WEEK2_EMAILS + NEW_EMAILS

# three worked examples for the few-shot baseline: written separately and NOT part of the evaluation set
FEW_SHOT_EXAMPLES: list[tuple[str, dict]] = [
    (
        "Hi, Priyanka Rao here. Order BX-310: 4 desk lamps and 1 monitor stand. Please deliver by 10 Nov 2026. Total $212.40. Thanks!",
        _g(
            name="Priyanka Rao",
            oid="BX-310",
            items=[("desk lamps", 4), ("monitor stand", 1)],
            date="2026-11-10",
            total=212.4,
            cur="USD",
        ),
    ),
    (
        "URGENT - order BY-7, Luca Bianchi, 12 fire extinguishers, we need them ASAP",
        _g(name="Luca Bianchi", oid="BY-7", items=[("fire extinguishers", 12)], urgency="high"),
    ),
    (
        "Hello, I would like to change the delivery address of my last order. Could someone call me back? - Anna",
        NOT_AN_ORDER,
    ),
]


def check_gold(gold: dict) -> None:
    """Every gold label must itself be a valid order (a wrong answer key is worse than none)."""
    o = gold_to_order(gold)
    semantic_checks(o)
    if not gold["is_order"]:
        assert not gold["customer_name"] and not gold["order_id"] and not gold["items"]
