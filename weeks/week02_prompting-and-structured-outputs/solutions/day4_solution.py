"""Week 2 Day 4 - Solution: workflow patterns as plain code.

A support-ticket pipeline that uses four of the five classic patterns, no framework:

  1. ROUTING             classify the ticket -> pick a specialised prompt
  2. PROMPT CHAINING     route -> (gate) -> draft -> review
  3. EVALUATOR-OPTIMISER a reviewer critiques the draft against a checklist; revise until it passes
  4. PARALLELISATION     summarise many threads concurrently with a bounded semaphore

Everything is traced (step, seconds, tokens) so you can see where time and money go.

  uv run python .../day4_solution.py --offline    # scripted fake model, asserts the control flow
  uv run python .../day4_solution.py              # your provider on the sample tickets
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common import llm  # noqa: E402
from common.fake import fake_llm  # noqa: E402

CONFIDENCE_FLOOR = 0.6  # below this we don't trust the router and escalate to a human

# ----------------------------------------------------------------- 1. routing


class Route(BaseModel):
    category: Literal["billing", "technical", "account", "other"] = Field(
        description="billing = charges/refunds/invoices; technical = bugs/outages/how-to; "
        "account = login/password/profile; other = anything else, or unclear"
    )
    confidence: float = Field(description="0 to 1: how sure you are", ge=0, le=1)
    reason: str = Field(description="One short sentence")


ROUTER_SYSTEM = "You classify customer support tickets. Pick exactly one category."

SPECIALISTS = {
    "billing": "You are a billing support specialist. You can explain charges and invoices. "
    "You cannot promise or approve refunds; say the billing team will review.",
    "technical": "You are a technical support specialist. Give concrete troubleshooting steps and "
    "ask for logs or error messages if needed.",
    "account": "You are an account support specialist. Never ask for a password. Point to the "
    "official password-reset flow for access problems.",
}

DRAFT_RULES = (
    "Write a reply of at most 90 words: friendly, specific, and ending with one clear next "
    "step for the customer."
)


def route(ticket: str) -> Route:
    r, _ = llm.structured(
        f"<ticket>\n{ticket}\n</ticket>", Route, system=ROUTER_SYSTEM, max_tokens=300
    )
    return r


# ----------------------------------------------------------------- 3. evaluator-optimiser


class Review(BaseModel):
    passes: bool
    problems: list[str] = Field(
        default_factory=list, description="Concrete fixes needed; empty if passes"
    )


CHECKLIST = [
    "Does not promise a refund, credit or compensation",
    "Does not ask for a password or full card number",
    "Contains exactly one clear next step",
    "Is at most 90 words",
]

REVIEW_SYSTEM = "You review support replies against a checklist. Be strict and specific."


def review(ticket: str, reply: str) -> Review:
    checklist = "\n".join(f"- {c}" for c in CHECKLIST)
    prompt = (
        f"<ticket>\n{ticket}\n</ticket>\n<reply>\n{reply}\n</reply>\n<checklist>\n{checklist}\n"
        "</checklist>\nDoes the reply satisfy every checklist item? List concrete problems."
    )
    r, _ = llm.structured(prompt, Review, system=REVIEW_SYSTEM, max_tokens=400)
    return r


# ----------------------------------------------------------------- 2. the chain


@dataclass
class Step:
    name: str
    seconds: float
    tokens: int


@dataclass
class Result:
    ticket: str
    category: str | None = None
    reply: str | None = None
    revisions: int = 0
    escalated: bool = False
    reason: str = ""
    trace: list[Step] = field(default_factory=list)


def _timed(result: Result, name: str, fn):
    t0, before = time.perf_counter(), llm.SESSION.usage.total_tokens
    out = fn()
    result.trace.append(
        Step(name, time.perf_counter() - t0, llm.SESSION.usage.total_tokens - before)
    )
    return out


def draft(ticket: str, category: str, feedback: list[str] | None = None) -> str:
    fix = ""
    if feedback:
        fix = (
            "\n<problems_to_fix>\n" + "\n".join(f"- {p}" for p in feedback) + "\n</problems_to_fix>"
        )
    prompt = f"<ticket>\n{ticket}\n</ticket>{fix}\n\n{DRAFT_RULES}"
    return llm.complete(prompt, system=SPECIALISTS[category], max_tokens=400).text.strip()


def handle_ticket(ticket: str, max_revisions: int = 2) -> Result:
    res = Result(ticket)
    r = _timed(res, "route", lambda: route(ticket))
    res.category = r.category

    # GATE: a cheap programmatic check between steps. Don't spend tokens on a doubtful route.
    if r.category == "other" or r.confidence < CONFIDENCE_FLOOR:
        res.escalated, res.reason = True, f"router: {r.category} @ {r.confidence:.2f} ({r.reason})"
        return res

    reply = _timed(res, "draft", lambda: draft(ticket, r.category))
    for _ in range(max_revisions + 1):
        verdict = _timed(res, "review", lambda reply=reply: review(ticket, reply))
        if verdict.passes:
            res.reply = reply
            return res
        if res.revisions == max_revisions:
            break
        res.revisions += 1
        reply = _timed(res, "revise", lambda v=verdict: draft(ticket, r.category, v.problems))
    res.escalated, res.reason = True, f"reply failed review after {max_revisions} revisions"
    return res


# ----------------------------------------------------------------- 4. parallelisation


async def summarise_many(threads: list[str], concurrency: int = 4) -> list[str]:
    """Fan out one summarisation call per thread, bounded by a semaphore; order is preserved."""
    sem = asyncio.Semaphore(concurrency)

    async def one(thread: str) -> str:
        async with sem:
            r = await llm.acomplete(
                f"<thread>\n{thread}\n</thread>\n\nSummarise this support thread "
                "in one sentence for a manager.",
                max_tokens=150,
            )
            return r.text.strip()

    return list(await asyncio.gather(*(one(t) for t in threads)))


# ----------------------------------------------------------------- demo data / offline model

TICKETS = [
    "I was charged twice for my subscription this month, please fix this.",
    "The app crashes with error 0x5F when I open the export screen.",
    "I can't log in anymore, it says my account is locked.",
    "Do you have a office in Berlin? I'd love to visit.",
]


def offline_responder_rules():
    def router(prompt: str) -> str:
        lowered = prompt.lower()
        cat, conf = "other", 0.4
        for keyword, c in (("charged", "billing"), ("crash", "technical"), ("log in", "account")):
            if keyword in lowered:
                cat, conf = c, 0.93
        return json.dumps({"category": cat, "confidence": conf, "reason": "keyword match"})

    def reviewer(prompt: str) -> str:
        bad = "guarantee a full refund" in prompt
        return json.dumps(
            {
                "passes": not bad,
                "problems": ["Promises a refund; say billing will review instead"] if bad else [],
            }
        )

    def drafter(prompt: str) -> str:
        if "<problems_to_fix>" in prompt:
            return "Thanks for flagging this. Our billing team will review the duplicate charge; please reply with your invoice number."
        if "billing support specialist" in prompt:
            return (
                "Sorry about that! We will guarantee a full refund today."  # violates the checklist
            )
        return "Thanks for reaching out. Please try restarting the app and send us the error log."

    return [
        (r"You classify customer support tickets", router),
        (r"You review support replies", reviewer),
        (
            r"Summarise this support thread",
            lambda p: "SUMMARY: " + p.split("<thread>")[1].strip()[:25].strip(),
        ),
        (r"(?s).*", drafter),
    ]


def run_offline() -> None:
    print("*** OFFLINE: scripted model; asserts the control flow, says nothing about quality ***")
    with fake_llm(offline_responder_rules()) as fake:
        results = [handle_ticket(t) for t in TICKETS]
        for r in results:
            steps = " > ".join(s.name for s in r.trace)
            print(
                f"- {r.category:<9} escalated={r.escalated!s:<5} revisions={r.revisions}  [{steps}]"
            )
        billing, tech, account, other = results

        assert billing.category == "billing" and billing.revisions == 1 and not billing.escalated
        assert "guarantee" not in billing.reply and "billing team" in billing.reply
        assert [s.name for s in billing.trace] == ["route", "draft", "review", "revise", "review"]
        assert tech.revisions == 0 and [s.name for s in tech.trace] == ["route", "draft", "review"]
        assert other.escalated and other.reply is None, (
            "gate must stop doubtful routes before drafting"
        )
        assert [s.name for s in other.trace] == ["route"]
        drafts = fake.calls_matching("billing support specialist")
        assert len(drafts) == 2, "billing prompt used for the draft and the revision only"
        assert not fake.calls_matching("technical support specialist.*Berlin")

        threads = [f"thread number {i}: customer asks about order {i}" for i in range(6)]
        out = asyncio.run(summarise_many(threads, concurrency=2))
        assert out == [f"SUMMARY: {t[:25].strip()}" for t in threads], "order preserved"
    print("\nself-test passed")


def main() -> None:
    if "--offline" in sys.argv:
        return run_offline()
    print(f"provider: {llm.resolve()}")
    for t in TICKETS:
        r = handle_ticket(t)
        print(
            f"\nTICKET: {t}\n  category={r.category} revisions={r.revisions} escalated={r.escalated} {r.reason}"
        )
        if r.reply:
            print(f"  REPLY: {r.reply}")
        print("  trace:", ", ".join(f"{s.name} {s.seconds:.1f}s/{s.tokens}tok" for s in r.trace))
    summaries = asyncio.run(summarise_many([f"{t}\n(agent: looking into it)" for t in TICKETS]))
    print("\nSUMMARIES:\n" + "\n".join(f"- {s}" for s in summaries))


if __name__ == "__main__":
    main()
