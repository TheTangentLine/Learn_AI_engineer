"""Machine-check a cited report. A report is only as good as its weakest citation, so we verify, per sentence:

  bad citation   [n] where n is not a page the agent actually fetched
  uncited        a factual sentence with no citation at all
  unsupported    the cited pages do not contain the sentence's words (lexical support < 0.4), or one of its numbers
                 is not backed: percentages, decimals and numbers over 20 must appear in the cited pages; small
                 integers (so common that presence proves nothing) must appear in the best-matching passage
  weak           support between 0.4 and 0.6
  ok             otherwise

Lexical support is a CHEAP proxy, not entailment: it catches fabricated numbers and off-topic citations, and it
passes a sentence that reuses the right words with the wrong meaning. Use an NLI model or a judge for more
(Week 4, Day 2); the check below is the always-on safety net.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from common.judges import lexical_support

from .tools import Source

CITE = re.compile(r"\[(\d+)\]")
NUMBER = re.compile(r"(?<![\w@.])\d[\d,]*(?:\.\d+)?%?(?!\w)")  # not digits inside words: BM25, hit@1, week5
OK_AT, WEAK_AT = 0.6, 0.4
SMALL_INT_MAX = 20
NEAR_MARGIN, NEAR_FLOOR = 0.15, 0.4


def _norm_number(x: str) -> str:
    return x.rstrip("%").replace(",", "").rstrip(".")


def is_small_int(x: str) -> bool:
    """A bare integer up to 20: so common on any page that its mere presence proves nothing."""
    n = _norm_number(x)
    return n.isdigit() and int(n) <= SMALL_INT_MAX and not x.endswith("%") and "," not in x


def passages(text: str) -> list[str]:
    """Lines of a page, with long lines split into sentences: the units a claim is compared against."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 300:
            out += [p for p in re.split(r"(?<=[.!?])\s+", line) if p]
        elif line:
            out.append(line)
    return out


def best_passages(
    context: str, claim: str, margin: float = NEAR_MARGIN, floor: float = NEAR_FLOOR
) -> list[str]:
    """The passages that match the claim's WORDS best (numbers are ignored so they cannot pick their own evidence)."""
    words_only = NUMBER.sub(" ", claim)
    scored = [(lexical_support(p, words_only), p) for p in passages(context)]
    if not scored:
        return []
    top = max(score for score, _ in scored)
    return [p for score, p in scored if score >= max(floor, top - margin)]


def missing_numbers(plain: str, context: str) -> list[str]:
    """Numbers in a claim that its cited pages do not back. Distinctive numbers (percentages, decimals, thousands,
    anything over 20) must appear anywhere in the pages; small integers must appear in the best-matching passage."""
    page_numbers = {_norm_number(x) for x in NUMBER.findall(context)}
    near_numbers = {
        _norm_number(x) for p in best_passages(context, plain) for x in NUMBER.findall(p)
    }
    missing = []
    for x in NUMBER.findall(plain):
        pool = near_numbers if is_small_int(x) else page_numbers
        if _norm_number(x) not in pool:
            missing.append(x)
    return missing


@dataclass
class Claim:
    sentence: str
    cites: list[int]
    status: str
    support: float | None = None
    missing_numbers: list[str] = field(default_factory=list)


@dataclass
class Verification:
    claims: list[Claim]
    unused_sources: list[int]

    def count(self, status: str) -> int:
        return sum(c.status == status for c in self.claims)

    @property
    def ok(self) -> bool:
        return bool(self.claims) and all(c.status in ("ok", "weak") for c in self.claims)

    @property
    def cited_fraction(self) -> float:
        return sum(bool(c.cites) for c in self.claims) / len(self.claims) if self.claims else 0.0

    def summary(self) -> str:
        parts = [
            f"{self.count(s)} {s}"
            for s in ("ok", "weak", "unsupported", "bad-citation", "uncited")
            if self.count(s)
        ]
        return f"{len(self.claims)} claims: " + (", ".join(parts) or "none")


def body_of(report: str) -> str:
    """The report without its trailing '## Sources' section (the harness writes that one, not the model)."""
    return re.split(r"\n#{1,3}\s*Sources\b", "\n" + report, maxsplit=1, flags=re.I)[0].strip()


def claim_sentences(report: str, min_words: int = 6) -> list[str]:
    out = []
    in_fence = False
    for line in body_of(report).splitlines():
        line = line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line or line.startswith(("#", "|")):
            continue
        line = re.sub(r"^[-*]\s+|^\d+\.\s+", "", line)
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", line):
            s = s.strip()
            if len(CITE.sub("", s).split()) >= min_words:
                out.append(s)
    return out


def verify_report(report: str, sources: list[Source]) -> Verification:
    by_n = {s.n: s for s in sources}
    claims: list[Claim] = []
    used: set[int] = set()
    for sentence in claim_sentences(report):
        cites = [int(m) for m in CITE.findall(sentence)]
        used.update(cites)
        plain = CITE.sub("", sentence)
        if not cites:
            claims.append(Claim(sentence, [], "uncited"))
            continue
        if any(n not in by_n for n in cites):
            claims.append(Claim(sentence, cites, "bad-citation"))
            continue
        context = "\n".join(by_n[n].text for n in cites)
        support = lexical_support(context, plain)
        missing = missing_numbers(plain, context)
        status = (
            "unsupported" if missing or support < WEAK_AT else ("weak" if support < OK_AT else "ok")
        )
        claims.append(Claim(sentence, cites, status, round(support, 2), missing))
    return Verification(claims, sorted(set(by_n) - used))


def render_final(report: str, sources: list[Source]) -> str:
    """The model's body plus a Sources section built from what was REALLY fetched (the model cannot invent URLs)."""
    lines = [f"[{s.n}] {s.title}: {s.url}" for s in sources]
    return (
        body_of(report)
        + "\n\n## Sources\n"
        + ("\n".join(lines) if lines else "(no sources were opened)")
        + "\n"
    )
