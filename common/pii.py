"""Detecting and removing personal data and secrets, with VALIDATORS so that a number that merely looks like a card is not redacted
and a real one is not missed.

    from common import pii

    spans = pii.detect("Mail alice@example.com, card 4111 1111 1111 1111")        # [Span(EMAIL ...), Span(CREDIT_CARD ...)]
    pii.redact(text)                                                                # "Mail [EMAIL], card [CREDIT_CARD]"
    ps = pii.Pseudonymizer(); safe = ps.pseudonymize(text); ps.restore(model_reply)  # the model sees <EMAIL_1>, the user sees alice@...
    pii.scrub({"password": "x", "note": text})                                      # for logs, spans, JSON payloads

What is detected (and how reliably; every row is measured in Week 8 Day 5 on a labelled set):
  structured and checkable   credit cards (Luhn + a brand prefix), IBANs (mod 97 + length), US SSNs (area/group/serial rules), e-mail
                             addresses, IP addresses, JWTs, private keys, vendor API keys (known prefixes), ``password=...`` assignments,
                             credentials inside URLs, phone numbers (shape + digit count + not a date)
  heuristic                  names after "my name is" or a title, street addresses, dates of birth after a label, long high-entropy tokens
Names and addresses in free text are the weak spot of any regex approach: a real deployment adds a trained NER model (for example
Presidio with spaCy) and still measures recall on its own data. Nothing here pretends otherwise.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# ----------------------------------------------------------------------------- validators


def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", number)]
    if not 12 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


CARD_PREFIX = re.compile(
    r"^(?:4\d{12}(?:\d{3}){0,2}|5[1-5]\d{14}|2(?:2[2-9]\d|[3-6]\d\d|7[01]\d|720)\d{12}|3[47]\d{13}|6(?:011|5\d\d|4[4-9]\d)\d{12,15}|3(?:0[0-5]|[68]\d)\d{11}|35(?:2[89]|[3-8]\d)\d{12})$"
)

IBAN_LENGTHS = {
    "AT": 20,
    "BE": 16,
    "CH": 21,
    "DE": 22,
    "ES": 24,
    "FR": 27,
    "GB": 22,
    "IE": 22,
    "IT": 27,
    "NL": 18,
    "PL": 28,
    "PT": 25,
    "SE": 24,
    "NO": 15,
    "DK": 18,
    "FI": 18,
    "LU": 20,
    "CZ": 24,
}


def iban_valid(value: str) -> bool:
    s = re.sub(r"\s", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    if IBAN_LENGTHS.get(s[:2], len(s)) != len(s):
        return False
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return int(numeric) % 97 == 1


def ssn_valid(area: str, group: str, serial: str) -> bool:
    """US SSN rules: no area 000, 666 or 900-999; no group 00; no serial 0000."""
    return (
        area not in ("000", "666")
        and not area.startswith("9")
        and group != "00"
        and serial != "0000"
    )


def entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


# ----------------------------------------------------------------------------- spans and detection


@dataclass(frozen=True)
class Span:
    type: str
    start: int
    end: int
    text: str
    score: float

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end


PRIORITY = [
    "PRIVATE_KEY",
    "JWT",
    "API_KEY",
    "CREDENTIALS",
    "CREDIT_CARD",
    "IBAN",
    "US_SSN",
    "URL_CREDENTIALS",
    "EMAIL",
    "IP_ADDRESS",
    "PHONE",
    "DATE_OF_BIRTH",
    "ADDRESS",
    "PERSON",
    "SECRET_TOKEN",
]

EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+-]{0,63}@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24}\b"
)
CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
SSN = re.compile(r"(?<![\w-])(\d{3})-(\d{2})-(\d{4})(?![\w-])")
SSN_CONTEXT = re.compile(r"(?i)(?:ssn|social security(?: number)?)\D{0,12}(\d{9})(?!\d)")
PHONE = re.compile(
    r"(?<![\w.\-/])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?|\d{2,4}[ .-])\d{3,4}[ .-]?\d{3,4}(?![\w\-/])|(?<![\w.\-/])\+\d{8,15}(?!\w)"
)
PHONE_CONTEXT = re.compile(
    r"(?i)(?:phone|tel|telephone|mobile|cell|call|whatsapp|fax)\W{0,6}(\+?\d[\d ()./-]{6,17}\d)"
)
IPV4 = re.compile(
    r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\w.]*\w)"
)
IPV6 = re.compile(r"(?<![\w:])(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{0,4}(?![\w:])")
JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b")
PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)", re.S
)
API_KEY = re.compile(
    r"\b(?:sk-ant-[A-Za-z0-9_-]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|xox[abprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35}|hf_[A-Za-z0-9]{30,}|glpat-[A-Za-z0-9_-]{20,})\b|(?i:bearer)\s+[A-Za-z0-9._~+/=-]{24,}"
)
CREDENTIALS = re.compile(
    r"(?i)\b(?:api[_-]?key|apikey|secret(?:[_-]?key)?|access[_-]?token|auth[_-]?token|token|passwd|password|pwd)\b\s*[:=]\s*['\"]?([^\s'\",;]{6,})"
)
URL_CREDS = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://([^/\s:@]+:[^/\s@]+)@")
HIGH_ENTROPY = re.compile(r"(?<![\w+/=-])[A-Za-z0-9+/_-]{32,}={0,2}(?![\w+/=-])")
DOB = re.compile(
    r"(?i)\b(?:dob|d\.o\.b\.?|date of birth|birth ?date|born(?: on)?)\b\W{0,4}((?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})|(?:[A-Za-z]{3,9}\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4})|(?:\d{1,2}(?:st|nd|rd|th)? [A-Za-z]{3,9},? \d{4}))"
)
NAME_AFTER = re.compile(
    r"\b(?:[Mm]y name is|[Mm]y name's|[Ii] am|[Ii]'m|[Tt]his is|[Nn]ame:|[Ss]igned,?|[Rr]egards,?|[Ss]incerely,?)\s+((?:[A-Z][a-z]+|[A-Z]\.)(?:\s+(?:[A-Z][a-z]+|[A-Z]\.)){1,2})\b"
)
NAME_TITLE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")
ADDRESS = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd|Court|Ct|Way|Place|Pl)\b\.?(?:,?\s+(?:[A-Z][a-z]+\s?){1,3},?)?(?:\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?)?"
)

NOT_NAMES = {
    "Not",
    "Sorry",
    "Happy",
    "Glad",
    "Here",
    "Just",
    "Using",
    "Looking",
    "Writing",
    "Trying",
    "Having",
    "Getting",
    "Unable",
    "Please",
    "Hoping",
}


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _find_cards(text: str) -> list[Span]:
    out = []
    for m in CARD.finditer(text):
        d = _digits(m.group(0))
        if 13 <= len(d) <= 19 and luhn_valid(d) and len(set(d)) > 1:
            out.append(
                Span(
                    "CREDIT_CARD",
                    m.start(),
                    m.end(),
                    m.group(0),
                    0.95 if CARD_PREFIX.match(d) else 0.6,
                )
            )
    return out


def _find_ibans(text: str) -> list[Span]:
    return [
        Span("IBAN", m.start(), m.end(), m.group(0), 0.95)
        for m in IBAN.finditer(text)
        if iban_valid(m.group(0))
    ]


def _find_ssns(text: str) -> list[Span]:
    out = [
        Span("US_SSN", m.start(), m.end(), m.group(0), 0.9)
        for m in SSN.finditer(text)
        if ssn_valid(*m.groups())
    ]
    for m in SSN_CONTEXT.finditer(text):
        d = m.group(1)
        if ssn_valid(d[:3], d[3:5], d[5:]):
            out.append(Span("US_SSN", m.start(1), m.end(1), d, 0.85))
    return out


def _find_phones(text: str) -> list[Span]:
    out = []
    for m in PHONE.finditer(text):
        raw = m.group(0)
        d = _digits(raw)
        if (
            len(set(d)) < 3
        ):  # 1111-1111: not a number anyone has (the shape already limits the digit count to 8-15)
            continue
        if re.fullmatch(
            r"\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}[-./]\d{2,4}", raw.strip()
        ):
            continue  # a date
        out.append(Span("PHONE", m.start(), m.end(), raw, 0.8 if re.search(r"[+(]", raw) else 0.65))
    for m in PHONE_CONTEXT.finditer(text):
        d = _digits(m.group(1))
        if 8 <= len(d) <= 15 and len(set(d)) >= 3:
            out.append(Span("PHONE", m.start(1), m.end(1), m.group(1), 0.7))
    return out


def _find_ips(text: str) -> list[Span]:
    out = []
    for m in IPV4.finditer(text):
        before = text[max(0, m.start() - 12) : m.start()].lower()
        if re.search(r"\b(?:version|ver\.?|v|release|build|py|node|python)\s*$", before):
            continue
        out.append(Span("IP_ADDRESS", m.start(), m.end(), m.group(0), 0.8))
    for m in IPV6.finditer(text):
        try:
            ipaddress.IPv6Address(m.group(0))
        except ValueError:
            continue
        out.append(Span("IP_ADDRESS", m.start(), m.end(), m.group(0), 0.8))
    return out


def _find_secrets(text: str) -> list[Span]:
    out = [
        Span("PRIVATE_KEY", m.start(), m.end(), m.group(0), 0.99)
        for m in PRIVATE_KEY.finditer(text)
    ]
    out += [Span("JWT", m.start(), m.end(), m.group(0), 0.9) for m in JWT.finditer(text)]
    out += [Span("API_KEY", m.start(), m.end(), m.group(0), 0.95) for m in API_KEY.finditer(text)]
    out += [
        Span("CREDENTIALS", m.start(1), m.end(1), m.group(1), 0.85)
        for m in CREDENTIALS.finditer(text)
        if not m.group(1).startswith(("[", "<", "{", "*"))
    ]
    out += [
        Span("URL_CREDENTIALS", m.start(1), m.end(1), m.group(1), 0.9)
        for m in URL_CREDS.finditer(text)
    ]
    for m in HIGH_ENTROPY.finditer(text):
        s = m.group(0)
        if (
            entropy(s) > 4.2 and re.search(r"\d", s) and re.search(r"[A-Za-z]", s)
        ):  # (a hex digest never gets here: its alphabet caps the entropy at 4.0)
            out.append(Span("SECRET_TOKEN", m.start(), m.end(), s, 0.6))
    return out


def _find_people_and_places(text: str) -> list[Span]:
    out = []
    for m in NAME_AFTER.finditer(text):
        name = m.group(1)
        if name.split()[0] in NOT_NAMES:
            continue
        out.append(Span("PERSON", m.start(1), m.end(1), name, 0.6))
    for m in NAME_TITLE.finditer(text):
        out.append(Span("PERSON", m.start(1), m.end(1), m.group(1), 0.6))
    out += [Span("ADDRESS", m.start(), m.end(), m.group(0), 0.6) for m in ADDRESS.finditer(text)]
    out += [
        Span("DATE_OF_BIRTH", m.start(1), m.end(1), m.group(1), 0.8) for m in DOB.finditer(text)
    ]
    return out


def _resolve(spans: list[Span]) -> list[Span]:
    """Overlaps go to the higher-priority type, then the higher score, then the longer span."""
    rank = {t: i for i, t in enumerate(PRIORITY)}
    kept: list[Span] = []
    for s in sorted(
        spans, key=lambda s: (rank.get(s.type, 99), -s.score, -(s.end - s.start), s.start)
    ):
        if not any(s.overlaps(k) for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: s.start)


def detect(text: str, *, types: set[str] | None = None, min_score: float = 0.5) -> list[Span]:
    """Personal data and secrets in ``text``: non-overlapping spans in order, each with a confidence in (0, 1]."""
    if not text:
        return []
    spans = [Span("EMAIL", m.start(), m.end(), m.group(0), 0.95) for m in EMAIL.finditer(text)]
    spans += (
        _find_cards(text)
        + _find_ibans(text)
        + _find_ssns(text)
        + _find_phones(text)
        + _find_ips(text)
        + _find_secrets(text)
        + _find_people_and_places(text)
    )
    spans = [s for s in spans if s.score >= min_score and (types is None or s.type in types)]
    return _resolve(spans)


# ----------------------------------------------------------------------------- redaction


def _mask_card(s: str, keep_last: int) -> str:
    d = _digits(s)
    return ("*" * (len(d) - keep_last) + d[-keep_last:]) if keep_last else "[CREDIT_CARD]"


def redact(
    text: str,
    *,
    mode: str = "mask",
    types: set[str] | None = None,
    min_score: float = 0.5,
    keep_card_last: int = 0,
    key: bytes | None = None,
) -> str:
    """Replace every detected span. Modes: ``mask`` ([EMAIL]), ``tag`` (<EMAIL>), ``hash`` (<EMAIL:3fa9c1>, a keyed hash so the same value
    gets the same tag without being recoverable). ``keep_card_last`` keeps the last digits of a card (support staff recognise a card by them)."""
    out, pos = [], 0
    key = key or _DEFAULT_KEY
    for s in detect(text, types=types, min_score=min_score):
        out.append(text[pos : s.start])
        if mode == "mask":
            out.append(
                _mask_card(s.text, keep_card_last)
                if s.type == "CREDIT_CARD" and keep_card_last
                else f"[{s.type}]"
            )
        elif mode == "tag":
            out.append(f"<{s.type}>")
        elif mode == "hash":
            out.append(
                f"<{s.type}:{hmac.new(key, s.text.encode(), hashlib.sha256).hexdigest()[:6]}>"
            )
        else:
            raise ValueError("mode must be 'mask', 'tag' or 'hash'")
        pos = s.end
    out.append(text[pos:])
    return "".join(out)


_DEFAULT_KEY = os.urandom(16)


@dataclass
class Pseudonymizer:
    """Reversible redaction for the LLM boundary: the model sees ``<EMAIL_1>``, the user gets ``alice@example.com`` back. The same value
    always maps to the same token inside one session, different values never share one, and ``restore`` only reverses tokens THIS
    session issued. The vault lives in memory: it is the sensitive part, keep it out of logs, spans and the database."""

    min_score: float = 0.5
    vault: dict[str, str] = field(default_factory=dict)  # token -> original
    _by_value: dict[tuple[str, str], str] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def token_for(self, ptype: str, value: str) -> str:
        """The token for one value (issued on first sight, the same afterwards)."""
        key = (ptype, value)
        if key not in self._by_value:
            self._counts[ptype] = self._counts.get(ptype, 0) + 1
            token = f"<{ptype}_{self._counts[ptype]}>"
            self._by_value[key] = token
            self.vault[token] = value
        return self._by_value[key]

    def pseudonymize(self, text: str, *, skip_types: set[str] | None = None) -> str:
        out, pos = [], 0
        for s in detect(text, min_score=self.min_score):
            if skip_types and s.type in skip_types:
                continue
            out.append(text[pos : s.start])
            out.append(self.token_for(s.type, s.text))
            pos = s.end
        out.append(text[pos:])
        return "".join(out)

    def restore(self, text: str) -> str:
        return re.sub(r"<[A-Z_]+_\d+>", lambda m: self.vault.get(m.group(0), m.group(0)), text)


# ----------------------------------------------------------------------------- structured data, logs and the LLM boundary

SENSITIVE_KEYS = re.compile(
    r"(?i)(?:pass(?:word|wd)?|secret|token|api[_-]?key|authorization|cookie|session|credential|private[_-]?key|ssn|card(?:[_-]?number)?|cvv|iban)"
)


def scrub(obj: Any, *, mode: str = "mask") -> Any:
    """Recursively redact strings in dicts, lists and tuples; values under keys that name a secret (password, token, ...) are replaced whole."""
    if isinstance(obj, str):
        return redact(obj, mode=mode)
    if isinstance(obj, dict):
        return {
            k: (
                "[REDACTED]"
                if isinstance(k, str) and SENSITIVE_KEYS.search(k) and v not in (None, "", False)
                else scrub(v, mode=mode)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [scrub(v, mode=mode) for v in obj]
    if isinstance(obj, tuple):
        return tuple(scrub(v, mode=mode) for v in obj)
    return obj


def leaks(text: str, values: list[str]) -> list[str]:
    """Which of the known sensitive ``values`` still appear in ``text`` (case-insensitive, ignoring spaces and dashes)? The test oracle for 'is the PII gone?'"""
    flat = re.sub(r"[\s\-.]", "", text.lower())
    return [v for v in values if re.sub(r"[\s\-.]", "", v.lower()) in flat]


def redaction_filter():
    """A ``logging.Filter`` that redacts the message and its arguments: attach it to a handler so no module has to remember to."""
    import logging

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.msg = redact(str(record.msg))
            if record.args:
                record.args = tuple(
                    redact(a) if isinstance(a, str) else a
                    for a in (record.args if isinstance(record.args, tuple) else (record.args,))
                )
            return True

    return _Filter()


def shield(llm_call, pseudonymizer: Pseudonymizer | None = None):
    """Wrap ``llm_call(prompt, **kw) -> str`` so the model never sees raw personal data: the prompt is pseudonymised on the way in and the
    reply is restored on the way out (what the user sees contains their own values; what the provider and your logs see does not)."""

    def wrapped(prompt: str, *args, **kw):
        ps = pseudonymizer or Pseudonymizer()
        reply = llm_call(ps.pseudonymize(prompt), *args, **kw)
        return ps.restore(reply)

    return wrapped
