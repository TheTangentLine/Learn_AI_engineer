"""A labelled synthetic PII set: sentences with embedded personal data of every type, plus hard negatives that merely LOOK like it.

All values are generated (valid checksums where a checksum exists), none are real. The set is seeded and deterministic, so a detector's
precision and recall are reproducible numbers, not impressions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

FIRST = [
    "Alice",
    "Bob",
    "Carla",
    "Dmitri",
    "Elena",
    "Farid",
    "Grace",
    "Hiro",
    "Ines",
    "Jamal",
    "Katya",
    "Liam",
    "Mei",
    "Nadia",
    "Omar",
    "Priya",
]
LAST = [
    "Johnson",
    "Nguyen",
    "Okafor",
    "Petrov",
    "Rossi",
    "Silva",
    "Tanaka",
    "Weber",
    "Haddad",
    "Larsen",
    "Kim",
    "Moreau",
]
STREETS = [
    "Baker Street",
    "Elm Court",
    "Maple Avenue",
    "Oak Road",
    "Pine Lane",
    "Cedar Drive",
    "Harbor Boulevard",
    "Willow Way",
]
CITIES = ["London", "Springfield", "Austin", "Leeds", "Portland"]
DOMAINS = ["example.com", "mail.example.org", "corp.example.co.uk", "acme.example"]


@dataclass
class Item:
    text: str
    gold: list[tuple[str, str]] = field(default_factory=list)  # (type, exact value)


def luhn_complete(prefix: str) -> str:
    total = 0
    for i, ch in enumerate(reversed(prefix)):
        d = int(ch) * (2 if i % 2 == 0 else 1)
        total += d - 9 if d > 9 else d
    return prefix + str((10 - total % 10) % 10)


def iban_with_check(country: str, bban: str) -> str:
    numeric = "".join(str(int(c, 36)) for c in bban + country + "00")
    return f"{country}{98 - int(numeric) % 97:02d}{bban}"


def values(rng: random.Random) -> dict[str, str]:
    first, last = rng.choice(FIRST), rng.choice(LAST)
    brand = rng.choice(["4", "51", "37", "6011"])
    n = 15 if brand == "37" else 16
    card = luhn_complete(
        brand + "".join(rng.choice("0123456789") for _ in range(n - 1 - len(brand)))
    )
    if brand == "37":
        card_fmt = f"{card[:4]} {card[4:10]} {card[10:]}"
    else:
        sep = rng.choice([" ", "-", ""])
        card_fmt = sep.join(card[i : i + 4] for i in range(0, 16, 4))
    area, grp, ser = rng.randint(100, 665), rng.randint(1, 99), rng.randint(1, 9999)
    cc = rng.choice(["+1", "+44", "+84", "+49"])
    phone = rng.choice(
        [
            f"+1 ({rng.randint(200, 999)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
            f"{rng.randint(200, 999)}-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
            f"{cc} {rng.randint(10, 99)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}",
            f"{rng.randint(200, 999)}.{rng.randint(200, 999)}.{rng.randint(1000, 9999)}",
        ]
    )
    alnum = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return {
        "PERSON": f"{first} {last}",
        "EMAIL": f"{first.lower()}.{last.lower()}{rng.randint(1, 99)}@{rng.choice(DOMAINS)}",
        "CREDIT_CARD": card_fmt,
        "US_SSN": f"{area}-{grp:02d}-{ser:04d}",
        "PHONE": phone,
        "IBAN": rng.choice(
            [
                iban_with_check("DE", "".join(rng.choice("0123456789") for _ in range(18))),
                iban_with_check(
                    "GB", "WEST" + "".join(rng.choice("0123456789") for _ in range(14))
                ),
                iban_with_check("FR", "".join(rng.choice("0123456789") for _ in range(23))),
            ]
        ),
        "IP_ADDRESS": f"{rng.choice([10, 172, 192, 203])}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
        "API_KEY": rng.choice(
            [
                "sk-ant-api03-" + "".join(rng.choice(alnum) for _ in range(40)),
                "AKIA" + "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567") for _ in range(16)),
                "ghp_" + "".join(rng.choice(alnum) for _ in range(36)),
            ]
        ),
        "CREDENTIALS": "".join(rng.choice(alnum) for _ in range(rng.randint(8, 14))),
        "ADDRESS": f"{rng.randint(1, 999)} {rng.choice(STREETS)}, {rng.choice(CITIES)}",
        "DATE_OF_BIRTH": f"{rng.randint(1950, 2005)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "SECRET_TOKEN": "".join(rng.choice(alnum) for _ in range(40)),
    }


TEMPLATES = {
    "PERSON": ["Hi, my name is {v}.", "Regards, {v}", "Please ask Dr. {last} to call back."],
    "EMAIL": ["Please write to {v} about this.", "Contact: {v}", "My address is {v}, thanks."],
    "CREDIT_CARD": [
        "I paid with card {v} yesterday.",
        "Card number: {v}.",
        "The charge on {v} is wrong.",
    ],
    "US_SSN": ["My SSN is {v}.", "Social security number {v} on file."],
    "PHONE": ["Call me on {v}.", "Phone: {v}", "You can reach me at {v} after 5."],
    "IBAN": ["Refund to IBAN {v} please.", "My bank account {v} changed."],
    "IP_ADDRESS": ["The request came from {v} at noon.", "Blocked IP {v}."],
    "API_KEY": ["Our key is {v} (do not share).", "Authorization uses {v} in the header."],
    "CREDENTIALS": ["Login failed with password={v} again.", "config: api_key = '{v}'"],
    "ADDRESS": ["Ship it to {v}.", "I live at {v}."],
    "DATE_OF_BIRTH": ["DOB: {v}", "I was born on {v} in the north."],
    "SECRET_TOKEN": ["session token {v} expired.", "Here is the token {v} you asked for."],
}

NEGATIVES = [
    "Invoice INV-3001 for $49.00 was paid on 2024-05-17.",
    "Order 12345678 shipped; tracking number 1Z999AA10123456784.",
    "Version 2.31.0 needs Python 3.12 or newer.",
    "Error 5003 means a temporary outage; retry in 30 seconds.",
    "The limit is 100 requests per minute (HTTP 429).",
    "Meeting 10:30-11:45 in room 4B on floor 2.",
    "Refund RF-0007 ($20.00) was issued; reference 1234-5678.",
    "Coordinates 51.5074, -0.1278 are the city centre.",
    "Commit a94a8fe5ccb19ba61c4c0873d391e987982fbbd3 fixed it.",
    "The file digest is 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08.",
    "Order number 4111111111111112 was cancelled.",
    "Part number 123-45-6789x is back in stock.",
    "Build 10.0.0.1 of the library is deprecated, see v1.2.3.4.",
    "Call the support line on weekdays between 9 and 17 UTC.",
    "Ticket 98765432 was escalated to the billing team.",
    "The server is 999.300.1.1 in the old documentation, which is invalid.",
    "Chapter 12, page 345, paragraph 6789.",
    "The meeting is on 2024-05-17 at 14:00.",
    "Use the search tool and the filters to narrow results.",
    "Settings > Security > Rotate keys (every 90 days).",
    "The new office opens in Springfield next spring.",
    "Our product Acme Cloud supports CSV and JSON exports.",
    "The maximum upload is 25 MB; larger files use multipart uploads.",
    "Timestamp 1714521600 is in seconds since the epoch.",
    "Latency was 120ms at p50 and 480ms at p95.",
    "The pull request fixes issue #4417 and closes #4418.",
    "Item SKU AB-12345-XY is out of stock.",
    "Tracking: 940011189922334455667788 (carrier code 9400).",
    "She said the show starts at 7.30pm in Hall 3.",
    "Flight BA2490 departs from Gate 22 at 08:55.",
]


def build_dataset(seed: int = 0, per_type: int = 10) -> list[Item]:
    rng = random.Random(seed)
    items: list[Item] = []
    for ptype, templates in TEMPLATES.items():
        for _ in range(per_type):
            v = values(rng)
            val = v[ptype]
            tmpl = rng.choice(templates)
            text = tmpl.format(v=val, last=val.split()[-1] if ptype == "PERSON" else "")
            gold_value = val.split()[-1] if "Dr. {last}" in tmpl else val
            items.append(Item(text, [(ptype, gold_value)]))
    for neg in NEGATIVES:
        items.append(Item(neg, []))
    # multi-PII sentences: a realistic support message
    for _ in range(per_type):
        v = values(rng)
        text = f"Hi, my name is {v['PERSON']}. Please refund {v['CREDIT_CARD']} and email {v['EMAIL']} or call {v['PHONE']}."
        items.append(
            Item(
                text,
                [
                    ("PERSON", v["PERSON"]),
                    ("CREDIT_CARD", v["CREDIT_CARD"]),
                    ("EMAIL", v["EMAIL"]),
                    ("PHONE", v["PHONE"]),
                ],
            )
        )
    rng.shuffle(items)
    return items


# Fake credentials with the SHAPE of real vendor tokens, assembled at import time so no literal in the source looks like a secret to a
# repository's push-protection scanner (the first version of this file was blocked by one).
SLACK_LIKE = "xo" + "xb-" + "263594206564-2343594206564-" + "AbCdEfGhIjKlMnOpQrStUvWx"
STRIPE_LIKE = "sk-" + "live-" + "4eC39HqLyjWDarjtT1zdp7dc"

# A messier, human-written set: phrasing and formats the generator above does not produce. Written AFTER the detector and not tuned on:
# its score is the honest estimate, the generated set's is a ceiling. (type, exact value) pairs; some gold values the detector is not built to find.
REALISTIC = [
    Item("Thanks so much, Priya Sharma", [("PERSON", "Priya Sharma")]),
    Item(
        "Hi, Alice Johnson here, billing question about my last invoice.",
        [("PERSON", "Alice Johnson")],
    ),
    Item("This was handled by Carlos Mendez in the Madrid office.", [("PERSON", "Carlos Mendez")]),
    Item("Nguyễn Văn An asked me to forward this.", [("PERSON", "Nguyễn Văn An")]),
    Item("my name is bob smith, lowercase because I typed on my phone", [("PERSON", "bob smith")]),
    Item("Please call me on (415)555-0132 tonight.", [("PHONE", "(415)555-0132")]),
    Item("My number is 415 555 0132.", [("PHONE", "415 555 0132")]),
    Item("Reach me at 0049 151 2345678 or by mail.", [("PHONE", "0049 151 2345678")]),
    Item("WhatsApp me: +84901234567", [("PHONE", "+84901234567")]),
    Item("tel +44 (0)20 7946 0958", [("PHONE", "+44 (0)20 7946 0958")]),
    Item("The card 4111  1111 1111 1111 was declined.", [("CREDIT_CARD", "4111  1111 1111 1111")]),
    Item("Card 4111.1111.1111.1111 expires 12/27.", [("CREDIT_CARD", "4111.1111.1111.1111")]),
    Item("I typed 4111111111111111 and it failed.", [("CREDIT_CARD", "4111111111111111")]),
    Item("my amex 378282246310005 please", [("CREDIT_CARD", "378282246310005")]),
    Item("write to alice [at] example [dot] com", [("EMAIL", "alice [at] example [dot] com")]),
    Item("Reply to Alice.Johnson@Example.COM today", [("EMAIL", "Alice.Johnson@Example.COM")]),
    Item("cc: ops-team+alerts@sub.example.org", [("EMAIL", "ops-team+alerts@sub.example.org")]),
    Item("My SSN is 123 45 6789.", [("US_SSN", "123 45 6789")]),
    Item("SSN: 123456789", [("US_SSN", "123456789")]),
    Item(
        "Send it to Flat 3, 22 Acacia Avenue, Bristol",
        [("ADDRESS", "Flat 3, 22 Acacia Avenue, Bristol")],
    ),
    Item("I live at Rue de Rivoli 10, Paris", [("ADDRESS", "Rue de Rivoli 10, Paris")]),
    Item(
        "Deliver to 1600 Pennsylvania Avenue, Washington",
        [("ADDRESS", "1600 Pennsylvania Avenue, Washington")],
    ),
    Item("I was born on 12 March 1985.", [("DATE_OF_BIRTH", "12 March 1985")]),
    Item("My birthday is March 12th, 1985.", [("DATE_OF_BIRTH", "March 12th, 1985")]),
    Item("DOB 12/03/1985", [("DATE_OF_BIRTH", "12/03/1985")]),
    Item("IBAN: gb82 west 1234 5698 7654 32", [("IBAN", "gb82 west 1234 5698 7654 32")]),
    Item("Refund to DE89 3704 0044 0532 0130 00 please", [("IBAN", "DE89 3704 0044 0532 0130 00")]),
    Item("The server at 10.1.2.3:8080 timed out.", [("IP_ADDRESS", "10.1.2.3")]),
    Item("My password is Tr0ub4dor&3 and I cannot log in.", [("CREDENTIALS", "Tr0ub4dor&3")]),
    Item("use secret: s3cr3t-pa55-w0rd for the demo", [("CREDENTIALS", "s3cr3t-pa55-w0rd")]),
    Item(
        f"Our key is {STRIPE_LIKE}",
        [("API_KEY", STRIPE_LIKE)],
    ),
    Item(
        f"token {SLACK_LIKE}",
        [("API_KEY", SLACK_LIKE)],
    ),
    Item(
        "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk",
        [
            (
                "API_KEY",
                "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk",
            )
        ],
    ),
    Item("Hello team, nothing personal in this one.", []),
    Item("Invoice INV-3001, order 4417, version 2.31.0.", []),
    Item("See https://docs.acme.example/retries for details.", []),
    Item("It costs $1,234.56 per month, billed on the 15th.", []),
    Item("Call 911 in an emergency; our line is open 9-17.", []),
    Item("The cat sat on the mat at 10:30:45.", []),
]
