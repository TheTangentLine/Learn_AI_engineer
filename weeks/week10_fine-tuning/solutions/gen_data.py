"""Synthetic training data for order extraction: a template-based generator, defect injection, quality filters, near-duplicate removal, balancing.

A language model would normally write these emails (Day 2 shows the prompt); a PROGRAM is used here because it runs offline, costs nothing, and gives
labels that are correct by construction, which lets the lesson do something a real pipeline cannot: inject KNOWN label errors and measure whether the filters
catch them.

    raw = generate(1500, seed=0, defect_rate=0.08)       # list of Sample (email, gold, template, defect)
    kept, report = curate(raw, human_emails)             # filters, dedup, contamination check -> report of how many each stage removed
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta

# ----------------------------------------------------------------------------- vocabulary
# fmt: off

FIRST = ['Aiko', 'Marcus', 'Lucia', 'Noor', 'Priya', 'Tomas', 'Chen', 'Sam', 'Dana', 'Grace', 'Oliver', 'Beatriz', 'Sven', 'Ravi', 'Hannah', 'Chloe', 'Diego', 'Zainab', 'Pierre', 'Ayesha', 'Omar', 'Fatima', 'Elena', 'Jin', 'Sofia', 'Kofi', 'Karim', 'Ben', 'Amara', 'Henrik', 'Yusuf', 'Marta', 'Ivan', 'Leila', 'Tariq', 'Greta', 'Mateo', 'Ananya', 'Felix', 'Nadia', 'Hugo', 'Mina', 'Jonas', 'Rosa', 'Dmitri', 'Ingrid', 'Samuel', 'Camille', 'Rahul', 'Elif', 'Pablo', 'Wei', 'Aaliyah', 'Lars', 'Imani', 'Rin', 'Joao', 'Kavya', 'Nikolai', 'Salma']
LAST = ['Tanaka', 'Webb', 'Fernandez', 'Haddad', 'Rao', 'Ruiz', 'Wei', 'Rivera', 'Whitfield', 'Liu', 'Brooks', 'Alves', 'Lindqvist', 'Patel', 'Muller', 'Park', 'Herrera', 'Khan', 'Dubois', 'Siddiqui', 'Zahra', 'Petrova', 'Rossi', 'Mensah', 'Aziz', 'Okoro', 'Nwosu', 'Berg', 'Demir', 'Kowalski', 'Novak', 'Haas', 'Silva', 'Moreau', 'Ibrahim', 'Larsen', 'Costa', 'Sharma', 'Weber', 'Farouk', 'Laurent', 'Sato', 'Becker', 'Marino', 'Volkov', 'Holm', 'Mbeki', 'Girard', 'Iyer', 'Yilmaz', 'Vega', 'Chang', 'Bello', 'Nilsson', 'Okafor', 'Abe', 'Pereira', 'Menon']
ITEMS = ['desk lamps', 'monitor stands', 'paper reams', 'staplers', 'office chairs', 'whiteboard markers', 'usb cables', 'laptop sleeves', 'coffee mugs', 'notebooks', 'ballpoint pens', 'water bottles', 'tool boxes', 'hex bolts', 'steel washers', 'garden hoses', 'sprinklers', 'yoga mats', 'wall clocks', 'picture frames', 'cactus pots', 'candles', 'diffusers', 'snow shovels', 'bike helmets', 'bike locks', 'keyboards', 'wireless mice', 'headsets', 'projectors', 'extension cords', 'door mats', 'storage bins', 'shelf brackets', 'paint rollers', 'work gloves', 'safety goggles', 'face masks', 't-shirts', 'tote bags', 'flash drives', 'webcams', 'desk fans', 'space heaters', 'kettles', 'toasters', 'cutting boards', 'olive oil bottles', 'jars of honey', 'rolls of packing tape', 'pallets of cement', 'sacks of flour', 'garden stools', 'ergonomic chairs', 'monitor arms', 'standing desks', 'filing cabinets', 'label printers', 'spare batteries', 'phone chargers', 'AA batteries', 'light bulbs', 'smoke detectors', 'fire extinguishers', 'first aid kits', 'folding tables', 'conference chairs', 'projector screens', 'badge holders']
SINGULAR_OK = ['projector', 'kettle', 'toaster', 'webcam', 'headset', 'keyboard', 'coffee machine', 'conference table', 'office phone', 'standing desk', 'label printer', 'space heater', 'desk lamp', 'monitor stand', 'bike helmet', 'filing cabinet']
NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
HIGH_CUES = ['ASAP', 'urgent', 'this is urgent', 'needed urgently', 'rush please', 'as soon as possible', 'very urgent', 'URGENT!!!', 'need it today', 'top priority']
LOW_CUES = ['no rush', 'not urgent', 'whenever is fine', 'take your time', 'no hurry at all', 'nothing urgent', 'low priority', 'no rush at all']
GREETINGS = ['Hi,', 'Hello,', 'Good morning,', 'Dear team,', 'Hi team,', 'Hey,', 'Greetings,', 'Dear Sales,', 'Hello there,', '']
CLOSINGS = ['Thanks!', 'Thank you', 'Regards', 'Best', 'Cheers', 'Many thanks', 'Kind regards', 'Thx', '', 'Thanks in advance']
FILLER = ['Hope you are well.', 'Following our call.', 'As discussed.', 'Quick one.', 'Please confirm receipt.', 'Let me know if you need anything else.', 'Sorry for the short notice.', 'Same address as last time.']
SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£"}
CODE_WORD = {"USD": "USD", "EUR": "EUR", "GBP": "GBP"}
WORD_FOR = {"USD": "dollars", "EUR": "euros", "GBP": "pounds"}
EARLIEST = date(
    2026, 10, 2
)  # the Week 2 emails were "received" on 2026-10-01: a delivery date must be after it


# ----------------------------------------------------------------------------- one email


@dataclass
class Sample:
    email: str
    gold: dict
    template: str
    defect: str = ""  # "" for a clean sample; otherwise what was corrupted in the LABEL (known by construction)
    meta: dict = field(default_factory=dict)


def _fmt_date(d: date, style: int) -> str:
    m = MONTHS[d.month - 1]
    day = d.day
    suffix = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return [
        d.isoformat(),
        f"{day} {m} {d.year}",
        f"{m[:3]} {day}, {d.year}",
        f"{m} {day}, {d.year}",
        f"the {day}{suffix} of {m} {d.year}",
        f"{day} {m[:3]} {d.year}",
        f"{m} {day}{suffix}, {d.year}",
    ][style % 7]


def _fmt_money(amount: float, cur: str, style: int, rng: random.Random) -> str:
    whole = amount == int(amount) and rng.random() < 0.5
    num = f"{int(amount)}" if whole else f"{amount:.2f}"
    if amount >= 1000 and style % 2 == 0:
        head, _, tail = num.partition(".")
        num = f"{int(head):,}" + (f".{tail}" if tail else "")
    if style % 6 == 5 and cur == "EUR" and not whole:  # the continental decimal comma
        num = num.replace(",", "").replace(".", ",")
        return f"{num} EUR"
    return (
        [
            f"{SYMBOL[cur]}{num}",
            f"{CODE_WORD[cur]} {num}",
            f"{num} {CODE_WORD[cur]}",
            f"{CODE_WORD[cur]} {num}",
            f"{num} {CODE_WORD[cur]}",
            f"{num} {CODE_WORD[cur]}",
        ][style % 6]
        if cur != "EUR" or style % 6 != 5
        else f"{num} EUR"
    )


def _qty(q: int, rng: random.Random) -> str:
    r = rng.random()
    if q in NUMBER_WORDS and r < 0.2:
        return NUMBER_WORDS[q]
    if r < 0.25 and q > 1:
        return f"{q}"
    return str(q)


def _item_text(name: str, q: int, rng: random.Random, style: str) -> str:
    n = _qty(q, rng)
    if style == "x":
        return f"{n} x {name}"
    if style == "plain":
        return f"{n} {name}"
    return f"{name} ({n})" if rng.random() < 0.4 else f"{n} {name}"


def make_gold(rng: random.Random, *, order: bool = True) -> dict:
    if not order:
        return dict(
            is_order=False,
            customer_name=None,
            order_id=None,
            items=[],
            urgency="normal",
            delivery_date=None,
            total_amount=None,
            currency=None,
        )
    n_items = rng.choices([1, 2, 3, 4], [40, 35, 18, 7])[0]
    names = rng.sample(ITEMS, n_items)
    items = [
        (n, rng.choice([1, 2, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 40, 50, 100, 250]))
        for n in names
    ]
    if n_items == 1 and rng.random() < 0.15:  # a single countable thing, written in the singular
        items = [(rng.choice(SINGULAR_OK), 1)]
    urgency = rng.choices(["low", "normal", "high"], [20, 55, 25])[0]
    delivery = None
    if rng.random() < 0.5:
        delivery = (EARLIEST + timedelta(days=rng.randint(3, 260))).isoformat()
    total = cur = None
    if rng.random() < 0.55:
        total = round(
            rng.choice([rng.uniform(8, 120), rng.uniform(100, 900), rng.uniform(500, 5000)]),
            rng.choice([0, 2, 2, 2]),
        )
        total = float(total)
        cur = rng.choice(["USD", "EUR", "GBP"])
    letters = rng.choice(
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "J",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "AA",
            "AB",
            "AC",
            "AD",
            "BX",
            "BY",
        ]
    )
    oid = f"{letters}-{rng.choice([rng.randint(1, 99), rng.randint(100, 999), rng.randint(1000, 9999)])}"
    return dict(
        is_order=True,
        customer_name=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        order_id=oid,
        items=items,
        urgency=urgency,
        delivery_date=delivery,
        total_amount=total,
        currency=cur,
    )


def _urgency_phrase(urgency: str, rng: random.Random) -> str:
    return (
        rng.choice(HIGH_CUES)
        if urgency == "high"
        else rng.choice(LOW_CUES)
        if urgency == "low"
        else ""
    )


def render_order(gold: dict, rng: random.Random) -> tuple[str, str, dict]:
    """(email text, template id, the label this email supports). Four layouts: prose, list, form, terse. When an email gives only the customer's FIRST name the
    label is that first name (a label must be recoverable from the text: asking a model to produce a surname the email does not contain teaches it to invent)."""
    kind = rng.choices(["prose", "list", "form", "terse"], [38, 22, 14, 26])[0]
    name, oid = gold["customer_name"], gold["order_id"]
    items = gold["items"]
    urgency = _urgency_phrase(gold["urgency"], rng)
    date_text = (
        _fmt_date(date.fromisoformat(gold["delivery_date"]), rng.randrange(7))
        if gold["delivery_date"]
        else ""
    )
    money = (
        _fmt_money(gold["total_amount"], gold["currency"], rng.randrange(6), rng)
        if gold["total_amount"] is not None
        else ""
    )
    greet, closing, filler = (
        rng.choice(GREETINGS),
        rng.choice(CLOSINGS),
        rng.choice(FILLER) if rng.random() < 0.3 else "",
    )
    if kind == "prose":
        item_style = rng.choice(["plain", "paren"])
        item_list = [_item_text(n, q, rng, item_style) for n, q in items]
        listed = (
            item_list[0]
            if len(item_list) == 1
            else ", ".join(item_list[:-1]) + " and " + item_list[-1]
        )
        first_only = rng.random() < 0.12
        shown = name.split()[0] if first_only else name
        intro = rng.choice([f"it's {shown}.", f"This is {shown}.", f"{shown} here.", f"My name is {shown}."])
        if first_only:
            gold = {**gold, "customer_name": shown}
        ask = rng.choice(
            [
                "Order {id} please: {items}.",
                "I'd like to order {items} (order {id}).",
                "Please process order {id} for {items}.",
                "For order {id} we need {items}.",
                "Placing an order, ref {id}: {items}.",
            ]
        ).format(id=oid, items=listed)
        parts = [greet, filler, intro, ask]
        if date_text:
            parts.append(
                rng.choice(
                    [
                        f"Please deliver by {date_text}.",
                        f"Need them by {date_text}.",
                        f"Delivery date: {date_text}.",
                        f"Could they arrive before {date_text}?",
                    ]
                )
            )
        if money:
            parts.append(
                rng.choice(
                    [
                        f"Total {money}.",
                        f"Total should come to {money}.",
                        f"We will pay {money}.",
                        f"Price agreed: {money}.",
                        f"{money} in total.",
                    ]
                )
            )
        if urgency:
            parts.append(
                urgency[0].upper() + urgency[1:] + ("." if urgency[-1] not in ".!" else "")
            )
        parts += [closing, shown if rng.random() < 0.4 else ""]
        return " ".join(p for p in parts if p).strip(), "prose", gold
    if kind == "list":
        lines = [f"From: {name.split()[0].lower()}@example.com", f"Order {oid}"]
        lines += [f"- {_item_text(n, q, rng, 'x')}" for n, q in items]
        lines.append(name)
        tail = [t for t in (f"needed by {date_text}" if date_text else "", money, urgency) if t]
        if tail:
            lines.append(rng.choice([" - ", ", ", "\n"]).join(tail))
        return "\n".join(lines), "list", gold
    if kind == "form":
        lines = [
            f"ORDER {oid}",
            f"Customer: {name}",
            "Items: " + "; ".join(_item_text(n, q, rng, "plain") for n, q in items),
        ]
        if date_text:
            lines.append(f"Deadline: {date_text}")
        if urgency:
            lines.append(f"Priority: {'high' if gold['urgency'] == 'high' else 'low'}")
        if money:
            lines.append(f"Amount: {money}")
        return "\n".join(lines), "form", gold
    sep = rng.choice([" / ", " - ", ", ", ". "])
    bits = [f"order {oid}", name] + [_item_text(n, q, rng, "plain") for n, q in items]
    if money:
        bits.append(money)
    if date_text:
        bits.append(f"by {date_text}")
    if urgency:
        bits.append(urgency)
    return sep.join(bits), "terse", gold


# fmt: off
NON_ORDER_TEMPLATES = ["Hello, could you tell me when order {id} will arrive? It's been {n} weeks. Thanks, {name}", 'Please cancel {id}, our customer has changed their mind. Regards {first}', 'Could you quote me a price for {n} {item}? We are not ordering yet.', 'Thank you for the fast delivery! The {item} work really well. {first}', 'The invoice attached to {id} shows an old billing address, can you correct it? {name}', "I'm away until next week and will reply to your message when I am back.", 'Hi, I need to update the shipping address on my previous purchase. Please call me. {first}', 'Do you sell {item} in other colours? Just asking. {name}', 'I never received a confirmation for {id}. Is it still being processed? {first}', 'We are unhappy with the {item} we got. Quality was poor. Regards, {name}', 'Hi, what are your opening hours over the holidays? - {first}', 're: {id} - please ignore my previous email, I made a mistake. {first}', 'Could you send me your latest catalogue? {name}', 'Unsubscribe me from your mailing list please. {first}', 'How long does standard shipping usually take to the north of the country? {first}', "Is {item} back in stock yet? Haven't heard anything since March. {name}"]
# fmt: on


def render_non_order(rng: random.Random) -> str:
    return rng.choice(NON_ORDER_TEMPLATES).format(
        id=f"{rng.choice('ABCDEFGHJKLMNP')}-{rng.randint(1, 9999)}",
        n=rng.choice([2, 3, 4, 50, 100, 500]),
        name=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        first=rng.choice(FIRST),
        item=rng.choice(ITEMS),
    )


def add_noise(text: str, rng: random.Random, protected: list[str]) -> str:
    """Casual-email noise that never touches a value the label depends on: double spaces, a dropped final punctuation mark, an occasional capital letter lost."""
    if rng.random() < 0.15:
        text = text.replace(". ", ".  ", 1)
    if rng.random() < 0.15 and text.endswith((".", "!")):
        text = text[:-1]
    if rng.random() < 0.1:
        for filler in ("Thanks", "Hello", "Hi", "Please"):
            if filler in text and not any(filler in p for p in protected):
                text = text.replace(filler, filler.lower(), 1)
                break
    return text


# ----------------------------------------------------------------------------- label defects (known errors injected on purpose)

DEFECTS = [
    "wrong_order_id",
    "wrong_quantity",
    "invented_item",
    "wrong_urgency",
    "wrong_currency",
    "date_in_the_past",
    "dropped_item",
    "wrong_name",
]


def inject_defect(gold: dict, rng: random.Random) -> tuple[dict, str]:
    """Corrupt the LABEL of an order (the email stays correct), the way a sloppy generator or a tired annotator would. Returns (bad gold, defect name)."""
    g = {**gold, "items": list(gold["items"])}
    options = list(DEFECTS)
    if g["total_amount"] is None:
        options.remove("wrong_currency")
    if len(g["items"]) < 2:
        options.remove("dropped_item")
    kind = rng.choice(options)
    if kind == "wrong_order_id":
        g["order_id"] = g["order_id"][:-1] + str((int(g["order_id"][-1]) + 3) % 10)
    elif kind == "wrong_quantity":
        i = rng.randrange(len(g["items"]))
        n, q = g["items"][i]
        g["items"][i] = (n, q + rng.choice([1, 2, 5]))
    elif kind == "invented_item":
        g["items"].append(
            (
                rng.choice([x for x in ITEMS if x not in {n for n, _ in g["items"]}]),
                rng.randint(1, 5),
            )
        )
    elif kind == "wrong_urgency":
        g["urgency"] = rng.choice([u for u in ("low", "normal", "high") if u != g["urgency"]])
    elif kind == "wrong_currency":
        g["currency"] = rng.choice([c for c in ("USD", "EUR", "GBP") if c != g["currency"]])
    elif kind == "date_in_the_past":
        g["delivery_date"] = "2026-03-15"
    elif kind == "dropped_item":
        g["items"].pop(rng.randrange(len(g["items"])))
    else:
        first, _, last = g["customer_name"].partition(" ")
        g["customer_name"] = f"{rng.choice([f for f in FIRST if f != first])} {last}"
    return g, kind


def generate(
    n: int, seed: int = 0, *, defect_rate: float = 0.0, non_order_rate: float = 0.14
) -> list[Sample]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < non_order_rate:
            email = render_non_order(rng)
            out.append(Sample(add_noise(email, rng, []), make_gold(rng, order=False), "non_order"))
            continue
        gold = make_gold(rng)
        email, template, gold = render_order(gold, rng)
        email = add_noise(email, rng, [gold["customer_name"], gold["order_id"]])
        defect = ""
        label = gold
        if rng.random() < defect_rate:
            label, defect = inject_defect(gold, rng)
        out.append(Sample(email, label, template, defect))
    return out


# ----------------------------------------------------------------------------- quality filters


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _stem(word: str) -> str:
    w = word.lower()
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def find_dates(text: str) -> set[str]:
    """Every date the email states, as ISO strings, for the formats the generator (and most real emails) use."""
    found = set()
    month = "|".join(m for m in MONTHS) + "|" + "|".join(m[:3] for m in MONTHS)
    for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", text):
        found.add(m.group(0))
    pats = [
        rf"(\d{{1,2}})(?:st|nd|rd|th)? (?:of )?({month}),? (\d{{4}})",
        rf"({month}) (\d{{1,2}})(?:st|nd|rd|th)?,? (\d{{4}})",
    ]
    for i, pat in enumerate(pats):
        for m in re.finditer(pat, text, re.I):
            d, mo, y = (
                (m.group(1), m.group(2), m.group(3))
                if i == 0
                else (m.group(2), m.group(1), m.group(3))
            )
            idx = next(
                k for k, name in enumerate(MONTHS) if name.lower().startswith(mo[:3].lower())
            )
            try:
                found.add(date(int(y), idx + 1, int(d)).isoformat())
            except ValueError:
                pass
    return found


def money_values(text: str) -> set[float]:
    vals = set()
    for m in re.finditer(
        r"(?<![\w-])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)(?![\w-])", text
    ):
        raw = m.group(1)
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw):
            vals.add(float(raw.replace(",", "")))
        elif re.fullmatch(r"\d+,\d{2}", raw):
            vals.add(float(raw.replace(",", ".")))
        else:
            vals.add(float(raw))
    return vals


def unexplained_numbers(text: str, gold: dict) -> list[str]:
    """Numbers (digits or number words) the email states that NOTHING in the label accounts for. The grounding checks only prove that what the label says
    is in the email; this is the converse, and it is what catches an OMITTED item: its quantity is still in the text with no item to explain it."""
    rest = re.sub(re.escape(gold["order_id"] or "\0"), " ", text, flags=re.I)
    for iso in find_dates(text):
        _ = iso
    month = "|".join(MONTHS) + "|" + "|".join(m[:3] for m in MONTHS)
    rest = re.sub(r"\d{4}-\d{2}-\d{2}", " ", rest)
    rest = re.sub(
        rf"\d{{1,2}}(?:st|nd|rd|th)? (?:of )?(?:{month}),? \d{{4}}", " ", rest, flags=re.I
    )
    rest = re.sub(rf"(?:{month}) \d{{1,2}}(?:st|nd|rd|th)?,? \d{{4}}", " ", rest, flags=re.I)
    rest = (
        re.sub(
            r"[$€£]\s?\d+(?:[.,]\d+)*(?:\s*(?:USD|EUR|GBP))?|(?:USD|EUR|GBP)\s*\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*\s*(?:USD|EUR|GBP)",
            " ",
            rest,
            flags=re.I,
        )
        if gold["total_amount"] is not None
        else rest
    )
    quantities = [q for _, q in gold["items"]]
    left = list(quantities)
    out = []
    for token in re.findall(r"(?<![\w-])\d+(?![\w-])", rest):
        if int(token) in left:
            left.remove(int(token))
        else:
            out.append(token)
    for word in re.findall(r"[a-z]+", rest.lower()):
        k = next(
            (n for n, w in NUMBER_WORDS.items() if w == word and w != "one"), None
        )  # "one" is too common in ordinary prose ("quick one", "this one")
        if k is not None:
            if k in left:
                left.remove(k)
            else:
                out.append(word)
    return out


def failures(sample: Sample, received: date = date(2026, 10, 1)) -> list[str]:
    """Why a sample's label cannot be right, by checking it against the email's own text. Empty list = passes every check."""
    g, text = sample.gold, sample.email
    low, flat = text.lower(), _norm(text)
    bad: list[str] = []
    if not g["is_order"]:
        if g["customer_name"] or g["order_id"] or g["items"]:
            bad.append("non_order_has_fields")
        return bad
    if _norm(g["order_id"]) not in flat:
        bad.append("order_id_not_in_email")
    if _norm(g["customer_name"]) not in flat and _norm(g["customer_name"].split()[0]) not in flat:
        bad.append("name_not_in_email")
    words = set(re.findall(r"[a-z0-9]+", low))
    stems = {_stem(w) for w in words}
    nums = {int(x) for x in re.findall(r"\d+", text)} | {
        k for k, w in NUMBER_WORDS.items() if w in words
    }
    for item, q in g["items"]:
        toks = [_stem(t) for t in re.findall(r"[a-z0-9]+", item.lower())]
        if not all(t in stems for t in toks):
            bad.append("item_not_in_email")
        if q not in nums and not (q == 1 and ({"a", "an"} & words)):
            bad.append("quantity_not_in_email")
    dates = find_dates(text)
    if g["delivery_date"]:
        if g["delivery_date"] not in dates:
            bad.append("date_not_in_email")
        elif date.fromisoformat(g["delivery_date"]) < received:
            bad.append("date_before_received")
    elif dates:
        bad.append("email_has_a_date_label_has_none")
    if g["total_amount"] is not None:
        if g["total_amount"] not in money_values(text):
            bad.append("total_not_in_email")
        has = {
            "USD": ["$", "usd", "dollar"],
            "EUR": ["€", "eur", "euro"],
            "GBP": ["£", "gbp", "pound"],
        }[g["currency"]]
        if not any(h in low for h in has):
            bad.append("currency_not_in_email")
    if unexplained_numbers(text, g):
        bad.append("number_in_email_not_in_label")
    lowc = any(c.lower() in low for c in LOW_CUES) or "priority: low" in low
    unnegated = low
    for c in LOW_CUES:  # "nothing urgent" and "not urgent" contain the word "urgent": take the low-priority phrases out before looking for urgency
        unnegated = unnegated.replace(c.lower(), " ")
    high = any(c.lower() in unnegated for c in HIGH_CUES) or "priority: high" in low
    if g["urgency"] == "high" and not high:
        bad.append("urgency_high_without_cue")
    if g["urgency"] == "low" and not lowc:
        bad.append("urgency_low_without_cue")
    if g["urgency"] == "normal" and (high or lowc):
        bad.append("urgency_normal_but_cue_present")
    return sorted(set(bad))


# ----------------------------------------------------------------------------- near-duplicates


def shingles(text: str, k: int = 5) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i : i + k]) for i in range(max(1, len(words) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def minhash(sh: set[str], n_perm: int = 64) -> tuple[int, ...]:
    """A signature whose agreement rate estimates Jaccard similarity: for each of n_perm salts, the smallest hash of any shingle."""
    return tuple(
        min(
            int.from_bytes(hashlib.blake2b(f"{salt}|{s}".encode(), digest_size=8).digest(), "big")
            for s in sh
        )
        for salt in range(n_perm)
    )


def near_duplicate_pairs(
    texts: list[str], threshold: float = 0.8, n_perm: int = 64, bands: int = 16
) -> set[tuple[int, int]]:
    """LSH over MinHash signatures: texts that share ANY band of the signature become candidate pairs, which are then checked with the exact Jaccard.
    Avoids comparing every pair (n squared) while finding almost all pairs above the threshold."""
    sh = [shingles(t) for t in texts]
    sig = [minhash(s, n_perm) for s in sh]
    rows = n_perm // bands
    buckets: dict[tuple, list[int]] = {}
    for i, s in enumerate(sig):
        for b in range(bands):
            buckets.setdefault((b, s[b * rows : (b + 1) * rows]), []).append(i)
    pairs = set()
    for members in buckets.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                if jaccard(sh[i], sh[j]) >= threshold:
                    pairs.add((i, j))
    return pairs


def dedup(samples: list[Sample], threshold: float = 0.8) -> tuple[list[Sample], int, int]:
    """Remove exact (normalised) duplicates, then one of every near-duplicate pair. Returns (kept, exact removed, near removed)."""
    seen, unique = set(), []
    for s in samples:
        key = _norm(s.email)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    pairs = near_duplicate_pairs([s.email for s in unique], threshold)
    drop = {j for _, j in pairs}
    return [s for i, s in enumerate(unique) if i not in drop], len(samples) - len(unique), len(drop)


def ngram_overlap(a: str, b: str, n: int = 8) -> bool:
    wa, wb = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
    ga = {tuple(wa[i : i + n]) for i in range(len(wa) - n + 1)}
    return any(tuple(wb[i : i + n]) in ga for i in range(len(wb) - n + 1))


def decontaminate(
    samples: list[Sample], reference_texts: list[str], n: int = 8
) -> tuple[list[Sample], int]:
    """Drop samples that share an n-word sequence with any text that must stay unseen (the evaluation set)."""
    kept = [s for s in samples if not any(ngram_overlap(s.email, r, n) for r in reference_texts)]
    return kept, len(samples) - len(kept)


@dataclass
class CurationReport:
    generated: int
    stages: list[tuple[str, int]]  # (stage name, samples removed by it)
    kept: int
    defects_injected: int
    defects_caught: int
    defects_missed: int
    false_rejections: int  # clean samples the filters removed
    failure_counts: Counter = field(default_factory=Counter)


def curate(
    raw: list[Sample], reference_texts: list[str] = ()
) -> tuple[list[Sample], CurationReport]:
    """Run the pipeline in the order a real one would: validity checks against the email, exact and near-duplicate removal, decontamination."""
    injected = sum(bool(s.defect) for s in raw)
    stage_counts, fail_counts = [], Counter()
    passing = []
    for s in raw:
        f = failures(s)
        if f:
            fail_counts.update(f)
        else:
            passing.append(s)
    stage_counts.append(("label fails a check against the email text", len(raw) - len(passing)))
    kept, n_exact, n_near = dedup(passing)
    stage_counts += [
        ("exact duplicate", n_exact),
        ("near duplicate (MinHash + Jaccard >= 0.8)", n_near),
    ]
    kept, n_contaminated = decontaminate(kept, list(reference_texts))
    stage_counts.append(("shares an 8-word sequence with an evaluation email", n_contaminated))
    caught = sum(1 for s in raw if s.defect and failures(s))
    survivors_defective = sum(1 for s in kept if s.defect)
    clean_rejected = sum(1 for s in raw if not s.defect and failures(s))
    return kept, CurationReport(
        len(raw),
        stage_counts,
        len(kept),
        injected,
        caught,
        survivors_defective,
        clean_rejected,
        fail_counts,
    )


# ----------------------------------------------------------------------------- balance and splits


def distribution(samples: list[Sample]) -> dict[str, dict[str, float]]:
    n = len(samples)
    pct = lambda c: {k: v / n for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}  # noqa: E731
    return {
        "is_order": pct(Counter(s.gold["is_order"] for s in samples)),
        "urgency": pct(Counter(s.gold["urgency"] for s in samples if s.gold["is_order"])),
        "currency": pct(Counter(s.gold["currency"] for s in samples if s.gold["is_order"])),
        "has_date": pct(
            Counter(s.gold["delivery_date"] is not None for s in samples if s.gold["is_order"])
        ),
        "n_items": pct(Counter(len(s.gold["items"]) for s in samples if s.gold["is_order"])),
        "template": pct(Counter(s.template for s in samples)),
    }


def split(
    samples: list[Sample], n_dev: int, n_test: int, seed: int = 0
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """train / dev / test by a seeded shuffle (the pipeline's output order carries no information)."""
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    pick = lambda idx: [samples[i] for i in idx]  # noqa: E731
    return pick(order[n_dev + n_test :]), pick(order[:n_dev]), pick(order[n_dev : n_dev + n_test])
