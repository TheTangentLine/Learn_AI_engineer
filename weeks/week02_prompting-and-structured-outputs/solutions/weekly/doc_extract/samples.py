"""Generate deterministic sample PDFs (with ground truth) so the whole pipeline is testable."""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

VENDORS = [
    "Northwind Supplies",
    "Bluebird Logistics",
    "Kestrel Print Co",
    "Orchard & Vine Ltd",
    "Helios Energy",
    "Tidewater Tools",
]
MERCHANTS = ["Corner Bakery", "Metro Pharmacy", "Greenline Books", "Fuel Stop 24"]
PEOPLE = ["Aiko Tanaka", "Marcus Webb", "Lucia Fernandez", "Noor Haddad"]
ROLES = ["Backend Engineer", "Data Analyst", "Product Designer", "Support Lead"]
ITEMS = [
    ("Paper reams", 4.5),
    ("Toner cartridge", 62.0),
    ("Shipping crate", 18.25),
    ("Safety gloves", 7.8),
    ("LED panel", 41.0),
    ("Desk organiser", 12.4),
]
SYMBOL = {"USD": "$", "EUR": "EUR ", "GBP": "GBP "}


def _fmt_date(d: date, style: int) -> str:
    return [d.isoformat(), d.strftime("%d %b %Y"), d.strftime("%m/%d/%Y")][style % 3]


def _write_pdf(path: Path, lines: list[str]) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    y = 800
    for line in lines:
        c.setFont("Helvetica-Bold" if line.isupper() else "Helvetica", 11)
        c.drawString(60, y, line)
        y -= 18
    c.save()


def make_invoice(rng: random.Random, i: int, out: Path) -> dict:
    cur = rng.choice(["USD", "EUR", "GBP"])
    d = date(2026, 8, 1) + timedelta(days=rng.randint(0, 40))
    due = d + timedelta(days=30)
    items = [(n, rng.randint(1, 9), p) for n, p in rng.sample(ITEMS, rng.randint(2, 4))]
    subtotal = round(sum(q * p for _, q, p in items), 2)
    rate = rng.choice([0.0, 0.1, 0.2])
    tax = round(subtotal * rate, 2)
    total = round(subtotal + tax, 2)
    vendor, number = rng.choice(VENDORS), f"INV-2026-{100 + i:04d}"
    label = rng.choice(["Invoice No.", "Invoice #", "Inv. Ref"])
    sym, style = SYMBOL[cur], rng.randint(0, 2)
    lines = [
        vendor,
        "INVOICE",
        f"{label}: {number}",
        f"Date: {_fmt_date(d, style)}",
        f"Payment due: {_fmt_date(due, style)}",
        f"Currency: {cur}",
        "Bill to: Acme Corp",
        "",
        "Description | Qty | Unit price | Amount",
    ]
    lines += [f"{n} | {q} | {sym}{p:.2f} | {sym}{q * p:.2f}" for n, q, p in items]
    lines += [
        "",
        f"Subtotal: {sym}{subtotal:.2f}",
        f"Tax ({int(rate * 100)}%): {sym}{tax:.2f}",
        f"TOTAL DUE: {sym}{total:.2f}",
        f"Doc ref DOC-{out.stem}",
    ]
    _write_pdf(out, lines)
    return {
        "doc_type": "invoice",
        "data": {
            "vendor": vendor,
            "invoice_number": number,
            "invoice_date": d.isoformat(),
            "due_date": due.isoformat(),
            "currency": cur,
            "line_items": [{"description": n, "quantity": q, "unit_price": p} for n, q, p in items],
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
        },
    }


def make_receipt(rng: random.Random, i: int, out: Path) -> dict:
    d = date(2026, 9, 1) + timedelta(days=rng.randint(0, 25))
    items = [(n, p) for n, p in rng.sample(ITEMS, rng.randint(1, 3))]
    total = round(sum(p for _, p in items), 2)
    merchant = rng.choice(MERCHANTS)
    method_text, method = rng.choice(
        [("VISA ****4242", "card"), ("CASH", "cash"), ("Apple Pay", "other")]
    )
    lines = [merchant, "RECEIPT", f"Date: {_fmt_date(d, rng.randint(0, 2))}"]
    lines += [f"{n}  ${p:.2f}" for n, p in items]
    lines += [f"TOTAL ${total:.2f}", f"Paid with: {method_text}", f"Thank you! DOC-{out.stem}"]
    _write_pdf(out, lines)
    return {
        "doc_type": "receipt",
        "data": {
            "merchant": merchant,
            "purchase_date": d.isoformat(),
            "total": total,
            "payment_method": method,
            "item_count": len(items),
        },
    }


def make_offer(rng: random.Random, i: int, out: Path) -> dict:
    name, role = PEOPLE[i % len(PEOPLE)], rng.choice(ROLES)
    start = date(2026, 11, 1) + timedelta(days=rng.randint(0, 60))
    amount, period = rng.choice([(96000, "yearly"), (7500, "monthly"), (84500, "yearly")])
    cur = rng.choice(["USD", "EUR", "GBP"])
    per_word = "per year" if period == "yearly" else "per month"
    lines = [
        "Harbor Analytics Ltd",
        "OFFER OF EMPLOYMENT",
        f"Dear {name},",
        f"We are delighted to offer you the position of {role}.",
        f"Your start date will be {_fmt_date(start, rng.randint(0, 2))}.",
        f"Your gross salary will be {SYMBOL[cur]}{amount:,.2f} {per_word}.",
        "Please sign and return this letter within 7 days.",
        f"Doc ref DOC-{out.stem}",
    ]
    _write_pdf(out, lines)
    return {
        "doc_type": "offer_letter",
        "data": {
            "candidate_name": name,
            "position": role,
            "start_date": start.isoformat(),
            "salary_amount": float(amount),
            "salary_period": period,
            "currency": cur,
        },
    }


def generate(out_dir: Path, seed: int = 11) -> dict[str, dict]:
    """Create 12 PDFs (5 invoices, 4 receipts, 3 offer letters); returns {filename: gold}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    plan = (
        [("invoice", make_invoice)] * 5
        + [("receipt", make_receipt)] * 4
        + [("offer", make_offer)] * 3
    )
    gold: dict[str, dict] = {}
    for n, (_, fn) in enumerate(plan):
        path = out_dir / f"{n:02d}.pdf"
        gold[path.name] = fn(rng, n, path)
    (out_dir / "gold.json").write_text(json.dumps(gold, indent=2))
    return gold
