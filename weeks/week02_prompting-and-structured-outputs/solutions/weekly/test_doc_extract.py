"""Offline tests for doc_extract: real PDFs + real pypdf, scripted model."""

from __future__ import annotations

import asyncio
import json

import pytest
from doc_extract import samples
from doc_extract.offline_model import make_rules
from doc_extract.pipeline import parse_json, pdf_text, process_file, process_folder
from doc_extract.report import build_report, values_match
from doc_extract.schemas import Invoice, Receipt
from pydantic import ValidationError
from reportlab.pdfgen import canvas

from common.fake import fake_llm


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    folder = tmp_path_factory.mktemp("docs")
    return folder, samples.generate(folder)


def run_offline(folder, gold, extra_rules=None):
    rules = (extra_rules or []) + make_rules(gold)
    with fake_llm(rules) as fake:
        records = asyncio.run(process_folder(folder, concurrency=3))
    return records, fake


# ------------------------------------------------------------------ real PDF text extraction


def test_samples_are_real_pdfs_with_extractable_text(corpus):
    folder, gold = corpus
    assert len(gold) == 12 and len(list(folder.glob("*.pdf"))) == 12
    text = pdf_text(folder / "00.pdf")
    assert gold["00.pdf"]["data"]["vendor"] in text and "DOC-00" in text
    assert gold["00.pdf"]["data"]["invoice_number"] in text


def test_blank_pdf_goes_to_review_not_extraction(tmp_path):
    p = tmp_path / "blank.pdf"
    c = canvas.Canvas(str(p))
    c.showPage()
    c.save()
    with fake_llm() as fake:
        rec = asyncio.run(process_file(p))
    assert rec.status == "needs_review" and "OCR" in rec.reason and fake.calls == []


# ------------------------------------------------------------------ parsing and schemas


def test_parse_json_is_tolerant():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here you go: {"a": 1} Hope that helps.') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("no json here")


def invoice_dict(**over):
    d = {
        "vendor": "V",
        "invoice_number": "1",
        "invoice_date": "2026-08-01",
        "due_date": "2026-08-31",
        "currency": "USD",
        "line_items": [{"description": "a", "quantity": 2, "unit_price": 5.0}],
        "subtotal": 10.0,
        "tax": 1.0,
        "total": 11.0,
    }
    return {**d, **over}


def test_invoice_validators():
    Invoice.model_validate(invoice_dict())
    with pytest.raises(ValidationError, match="line items sum"):
        Invoice.model_validate(invoice_dict(subtotal=12.0, total=13.0))
    with pytest.raises(ValidationError, match="!= total"):
        Invoice.model_validate(invoice_dict(total=99.0))
    with pytest.raises(ValidationError, match="due_date"):
        Invoice.model_validate(invoice_dict(due_date="2026-07-01"))
    with pytest.raises(ValidationError):
        Receipt.model_validate(
            {
                "merchant": "m",
                "purchase_date": "2026-09-01",
                "total": 5,
                "payment_method": "visa",
                "item_count": 1,
            }
        )


def test_values_match_semantics():
    assert values_match(10.004, 10.0) and not values_match(10.2, 10.0)
    assert values_match("Aiko  Tanaka", "aiko tanaka") and not values_match("Aiko", "Marcus")
    assert values_match(None, None) and not values_match(None, "x")
    a = [
        {"description": "A b", "quantity": 1, "unit_price": 2.0},
        {"description": "c", "quantity": 3, "unit_price": 1.5},
    ]
    assert values_match(list(reversed(a)), a) and not values_match(a[:1], a)


# ------------------------------------------------------------------ end to end (scripted model)


def test_every_pipeline_branch_with_injected_faults(corpus):
    folder, gold = corpus
    records, fake = run_offline(folder, gold)
    by = {r.file: r for r in records}
    assert [r.file for r in records] == sorted(by), "results must come back in file order"
    assert sum(r.status == "ok" for r in records) == 10
    # validator caught three different first-attempt faults, retry fixed each
    for name in ("01.pdf", "06.pdf", "09.pdf"):
        assert by[name].status == "ok" and by[name].attempts == 2 and len(by[name].errors) == 1
    assert "!= total" in by["01.pdf"].errors[0]
    # misclassification -> wrong schema -> three failed attempts -> status failed
    assert by["04.pdf"].status == "failed" and by["04.pdf"].attempts == 3
    # low classifier confidence is gated BEFORE any extraction call
    assert by["11.pdf"].status == "needs_review" and by["11.pdf"].attempts == 0
    # untouched documents succeed first try
    assert by["00.pdf"].attempts == 1 and by["00.pdf"].errors == []
    # the retry prompts quote the exact error text
    retries = fake.calls_matching("failed validation")
    assert len(retries) == 3 + 2  # three single retries + two more for the misclassified doc
    assert any("!= total" in c.prompt for c in retries), (
        "retry prompt must quote the validator's message"
    )


def test_report_numbers_and_silent_error_visibility(corpus):
    folder, gold = corpus
    records, _ = run_offline(folder, gold)
    text, m = build_report(records, gold)
    assert m["status"] == {"ok": 10, "needs_review": 1, "failed": 1}
    assert m["classification"] == pytest.approx(11 / 12) and m["retried"] == 4
    assert m["exact_docs"] == 9, "10 ok docs minus the silently wrong vendor"
    assert "`03.pdf`: vendor" in text, (
        "an error no validator can catch must still surface in the report"
    )
    assert "`04.pdf` **failed**" in text and "`11.pdf` **needs_review**" in text


def test_one_crashing_document_does_not_kill_the_batch(corpus):
    folder, gold = corpus
    records, _ = run_offline(
        folder, gold, extra_rules=[(r"DOC-05", RuntimeError("provider exploded"))]
    )
    by = {r.file: r for r in records}
    assert by["05.pdf"].status == "failed" and "provider exploded" in by["05.pdf"].reason
    assert sum(r.status == "ok" for r in records) == 9


def test_gold_json_is_valid_for_every_schema(corpus):
    """The sample generator's own ground truth must satisfy the schemas (else the demo is rigged)."""
    from doc_extract.schemas import SCHEMAS

    _, gold = corpus
    for g in gold.values():
        SCHEMAS[g["doc_type"]].model_validate(json.loads(json.dumps(g["data"])))
