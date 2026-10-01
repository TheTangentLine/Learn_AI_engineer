"""The pipeline: PDF text -> classify (gated) -> extract with validate-and-retry -> Record."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError
from pypdf import PdfReader

from common import llm

from .schemas import SCHEMAS, Classification, sanity_date

CONFIDENCE_FLOOR = 0.7
MAX_ATTEMPTS = 3
MAX_CHARS = 12_000  # never silently truncate: documents longer than this are flagged instead

CLASSIFY_SYSTEM = (
    "You classify business documents. Types: invoice (a bill from a vendor), receipt (proof of a "
    "completed purchase), offer_letter (a job offer to a candidate). Use 'unknown' for anything else."
)
EXTRACT_SYSTEM = (
    "You extract structured data from a {kind}. Use only what the document states; normalise dates to "
    "YYYY-MM-DD and amounts to plain numbers (no currency symbols or thousands separators). "
    "Reply with only a JSON object."
)


@dataclass
class Record:
    file: str
    status: str = "failed"  # ok | needs_review | failed
    doc_type: str | None = None
    confidence: float | None = None
    data: dict[str, Any] | None = None
    attempts: int = 0
    errors: list[str] = field(default_factory=list)
    reason: str = ""
    cost_usd: float = 0.0
    seconds: float = 0.0


def pdf_text(path: Path) -> str:
    return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages).strip()


def parse_json(text: str) -> Any:
    """Tolerant parse: strip code fences and any prose around the outermost JSON object."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = body.find("{"), body.rfind("}")
    return json.loads(body[start : end + 1]) if start != -1 and end > start else json.loads(body)


async def _ask(record: Record, messages: list[dict], system: str) -> str:
    resp = await llm.acomplete(messages, system=system, max_tokens=1500)
    record.cost_usd += resp.cost_usd
    return resp.text


async def classify(record: Record, text: str) -> Classification:
    prompt = (
        f"<document>\n{text[:2500]}\n</document>\n\nClassify the document. Reply with only JSON: "
        '{"doc_type": "invoice|receipt|offer_letter|unknown", "confidence": 0.0-1.0}'
    )
    messages = [{"role": "user", "content": prompt}]
    for _ in range(2):
        reply = await _ask(record, messages, CLASSIFY_SYSTEM)
        try:
            return Classification.model_validate(parse_json(reply))
        except (ValidationError, ValueError) as exc:
            messages += [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": f"That failed validation: {str(exc)[:300]}. "
                    "Reply with only the corrected JSON.",
                },
            ]
    return Classification(doc_type="unknown", confidence=0.0)


def validate(schema: type[BaseModel], reply: str) -> BaseModel:
    obj = schema.model_validate(parse_json(reply))
    for value in obj.model_dump().values():  # plausibility of every date field
        if hasattr(value, "year"):
            sanity_date(value)
    return obj


async def extract(record: Record, text: str, doc_type: str) -> BaseModel | None:
    schema = SCHEMAS[doc_type]
    kind = doc_type.replace("_", " ")
    prompt = (
        f"<document>\n{text}\n</document>\n\nExtract the data as JSON matching this JSON Schema:\n"
        f"{json.dumps(schema.model_json_schema())}"
    )
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        record.attempts = attempt
        reply = await _ask(record, messages, EXTRACT_SYSTEM.format(kind=kind))
        try:
            return validate(schema, reply)
        except (ValidationError, ValueError) as exc:
            msg = str(exc)[:600]
            record.errors.append(msg)
            messages += [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": f"That output failed validation:\n{msg}\n"
                    "Fix it and reply with only the corrected JSON.",
                },
            ]
    return None


async def process_file(path: Path) -> Record:
    rec, t0 = Record(path.name), time.perf_counter()
    try:
        text = pdf_text(path)
        if not text:
            rec.reason = "no extractable text (scanned image? needs OCR)"
            rec.status = "needs_review"
            return rec
        if len(text) > MAX_CHARS:
            rec.reason, rec.status = (
                f"document longer than {MAX_CHARS} chars; needs chunking",
                "needs_review",
            )
            return rec
        cls = await classify(rec, text)
        rec.doc_type, rec.confidence = cls.doc_type, cls.confidence
        if cls.doc_type == "unknown" or cls.confidence < CONFIDENCE_FLOOR:
            rec.status, rec.reason = (
                "needs_review",
                f"classifier: {cls.doc_type} @ {cls.confidence:.2f}",
            )
            return rec
        obj = await extract(rec, text, cls.doc_type)
        if obj is None:
            rec.status, rec.reason = "failed", f"no valid output after {MAX_ATTEMPTS} attempts"
        else:
            rec.status, rec.data = "ok", json.loads(obj.model_dump_json())
    except Exception as exc:  # one bad document must never kill the batch
        rec.status, rec.reason = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        rec.seconds = time.perf_counter() - t0
    return rec


async def process_folder(folder: Path, concurrency: int = 4) -> list[Record]:
    sem = asyncio.Semaphore(concurrency)

    async def bounded(p: Path) -> Record:
        async with sem:
            return await process_file(p)

    return list(await asyncio.gather(*(bounded(p) for p in sorted(folder.glob("*.pdf")))))
