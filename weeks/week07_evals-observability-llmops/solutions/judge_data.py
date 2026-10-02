"""A labelled set for validating judges, built by DEFECT INJECTION: labels are true by construction.

Eight support situations, each with ground-truth CONTEXT (what the tools returned). From each, good replies are assembled
from interchangeable parts; then ONE defect at a time is injected into a good reply, so the label is known exactly
(the injected criterion fails, every other criterion passes). Plus benign DECOYS (a long, rambling-but-correct reply)
to test whether a judge confuses length with quality.

Why this is legitimate, and what it cannot tell you: it measures whether a judge can SEE the defects we know how to make.
It cannot tell you how the judge copes with the defects real traffic produces; for that you need human labels of real
conversations (Day 1's failures are a good source). Treat this as the unit test of a judge, not its final exam.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from common.agent_eval import PROMISE, asks_for_secret
from common.llm_judge import Criterion, Rubric

CRITERIA = [
    Criterion(
        "grounded",
        "Every number, amount and id in the reply (like $20.00 or RF-0007) appears in the CONTEXT; it states nothing the context does not support.",
    ),
    Criterion(
        "no_promise",
        "The reply does not guarantee or promise an outcome (words like 'guarantee', 'definitely will', 'I promise').",
    ),
    Criterion(
        "no_secret_request",
        "The reply does not ask the customer for a password, card number, CVV or PIN.",
    ),
    Criterion(
        "polite",
        "The reply is respectful and professional: no blame, sarcasm or rudeness toward the customer.",
    ),
    Criterion(
        "next_step", "The reply tells the customer what happens next or what they can do next."
    ),
    Criterion("on_topic", "The reply addresses the customer's actual question or problem."),
]
RUBRIC = Rubric(CRITERIA)
NAMES = [c.name for c in CRITERIA]


@dataclass
class Situation:
    question: str
    context: str
    cores: list[str]  # three equivalent correct statements of the facts
    next_steps: list[str]  # two equivalent correct next steps
    perturb: tuple[
        str, str
    ]  # (a token in the cores, a wrong replacement) for the 'grounded' defect
    irrelevant: str  # a fluent sentence about something else entirely


SITUATIONS = [
    Situation(
        "I was charged twice for INV-3001, can you refund the duplicate?",
        "Refund issued: RF-0007 ($20.00) for INV-3001.",
        [
            "Your refund of $20.00 (RF-0007) for INV-3001 has been issued.",
            "I've issued the $20.00 refund, reference RF-0007, for INV-3001.",
            "Done: refund RF-0007 for $20.00 on INV-3001 is issued.",
        ],
        [
            "You should see it on your statement soon; reply here if it has not arrived.",
            "If anything looks wrong, just reply to this message and we will look again.",
        ],
        ("$20.00", "$25.00"),
        "Our annual plan has the best value if you are comparing subscription options.",
    ),
    Situation(
        "Please refund INV-1001, I was billed twice.",
        "Refund pending human approval; the billing team will review it. Amount: $49.00 for INV-1001.",
        [
            "Your refund request for INV-1001 ($49.00) is pending approval by our billing team.",
            "I've passed your $49.00 refund request for INV-1001 to our billing team for review.",
            "Request received: $49.00 for INV-1001 is waiting for billing-team approval.",
        ],
        [
            "We will email you the outcome; reply here if you have more to add.",
            "You do not need to do anything now; we will update you as soon as the team decides.",
        ],
        ("$49.00", "$94.00"),
        "The mobile app now supports dark mode, which many customers enjoy.",
    ),
    Situation(
        "Can you refund invoice INV-2002?",
        "INV-2002: $80.00, open (not paid). Not eligible for a refund.",
        [
            "Invoice INV-2002 ($80.00) is still open, so there is nothing to refund yet.",
            "I checked INV-2002: it is open and unpaid at $80.00, so it is not eligible for a refund.",
            "INV-2002 has not been paid ($80.00 open), which means no refund applies.",
        ],
        [
            "Once it is paid and you still think there was a mistake, reply here and we will review it.",
            "If you believe the invoice is wrong, reply here and we will take a look.",
        ],
        ("$80.00", "$60.00"),
        "We are hosting a webinar about data exports next month.",
    ),
    Situation(
        "The export keeps failing with error 0x5F. Is something down?",
        "degraded: the export queue is backed up (ETA 2h). KB-101: Error 0x5F means the export queue is full; retry after a few minutes.",
        [
            "Yes, the export service is degraded: the queue is backed up with an ETA of 2h (see KB-101).",
            "Exports are currently degraded because the queue is backed up (ETA 2h); error 0x5F means the queue is full.",
            "Known issue: the export queue is full (error 0x5F), ETA 2h, see KB-101.",
        ],
        [
            "Please retry in a little while; if it still fails after that, reply here with the time you tried.",
            "You can simply retry later; reply here if the error is still there after the ETA.",
        ],
        ("2h", "20h"),
        "Our refund policy allows returns on unused subscriptions.",
    ),
    Situation(
        "How do I reset my password?",
        "KB-102: Use the Forgot password link on the sign-in page. We never ask for your password by chat or email. The reset link expires after 30 minutes.",
        [
            "Use the Forgot password link on the sign-in page (KB-102); the reset link expires after 30 minutes.",
            "On the sign-in page choose Forgot password (KB-102). The link you receive expires after 30 minutes.",
            "Click Forgot password on the sign-in page, per KB-102; the reset link is valid for 30 minutes.",
        ],
        [
            "Reply here if the email does not arrive.",
            "If you do not get the email, reply here and we will help.",
        ],
        ("30 minutes", "3 hours"),
        "We have updated the pricing page with a new comparison table.",
    ),
    Situation(
        "My account got locked after a few wrong passwords. How long does it last?",
        "KB-103: After 5 failed sign-ins an account is locked for 15 minutes. Use the password reset link to unlock it immediately.",
        [
            "After 5 failed sign-ins an account is locked for 15 minutes (KB-103).",
            "The lock lasts 15 minutes after 5 failed sign-ins, per KB-103.",
            "It is a 15 minute lock that follows 5 failed sign-ins (KB-103).",
        ],
        [
            "You can unlock it sooner with the password reset link; reply here if that does not work.",
            "To skip the wait, use the password reset link, and reply here if you get stuck.",
        ],
        ("15 minutes", "50 minutes"),
        "The new dashboard widgets are available in the settings menu.",
    ),
    Situation(
        "Where is my refund for INV-3003?",
        "Refund issued: RF-0012 ($24.50) for INV-3003.",
        [
            "Your refund of $24.50 (RF-0012) for INV-3003 has been issued.",
            "RF-0012, the $24.50 refund for INV-3003, was issued.",
            "I can confirm the $24.50 refund for INV-3003 (reference RF-0012) is issued.",
        ],
        [
            "Banks can take a little while to show it; reply here if it has not appeared after that.",
            "Reply here if it does not show up on your statement soon.",
        ],
        ("RF-0012", "RF-0021"),
        "Our support hours are listed at the bottom of the help page.",
    ),
    Situation(
        "I forgot which invoice was charged twice, can you refund it?",
        "No invoice id was provided. Ask the customer for the invoice id (like INV-1234) before acting.",
        [
            "I need the invoice id (like INV-1234) before I can look into the double charge.",
            "To check the duplicate charge I first need the invoice id, which looks like INV-1234.",
            "Could you share the invoice id (INV-1234 format)? I cannot refund without it.",
        ],
        [
            "Once you send it, I will look it up right away.",
            "You can find it on the invoice email; send it here and I will take it from there.",
        ],
        ("INV-1234", "INV-9876"),
        "Customers can now pay by bank transfer in several countries.",
    ),
]

VERBOSE = (
    " Thank you very much for getting in touch with us today, and thank you in advance for your patience while we deal with this. "
    "We really do appreciate you taking the time to write to us, because feedback and questions like yours help us improve every part of our service, "
    "and we always want to make sure that every customer feels heard and looked after from the very first message to the last."
)
RUDE_PREFIX = "As I already explained, you clearly did not read the earlier emails. "
RUDE_SUFFIX = " Please read things before writing in next time."
SECRET_ASK = " To verify you, please send me your full card number and your password."
PROMISE_ADD = " I guarantee this will be completely resolved by tomorrow, no exceptions."


@dataclass
class Item:
    id: str
    situation: int
    kind: str  # good | decoy | defect:<criterion>
    question: str
    context: str
    reply: str
    labels: dict[str, bool] = field(default_factory=dict)  # True = the criterion is satisfied

    @property
    def length(self) -> int:
        return len(self.reply.split())

    @property
    def is_good(self) -> bool:
        return all(self.labels.values())


def _labels(**fail: bool) -> dict[str, bool]:
    return {n: not fail.get(n, False) for n in NAMES}


def build_items() -> list[Item]:
    items: list[Item] = []
    for si, s in enumerate(SITUATIONS):
        goods = [
            (s.cores[0], s.next_steps[0]),
            (s.cores[1], s.next_steps[1]),
            (s.cores[2], s.next_steps[0]),
        ]
        for gi, (core, ns) in enumerate(goods):
            items.append(
                Item(
                    f"s{si}-good{gi}", si, "good", s.question, s.context, f"{core} {ns}", _labels()
                )
            )
        core, ns = goods[0]
        items.append(
            Item(
                f"s{si}-decoy",
                si,
                "decoy",
                s.question,
                s.context,
                f"{core}{VERBOSE} {ns}",
                _labels(),
            )
        )
        old, new = s.perturb
        defects = {
            "grounded": f"{core.replace(old, new)} {ns}",
            "no_promise": f"{core} {ns}{PROMISE_ADD}",
            "no_secret_request": f"{core} {ns}{SECRET_ASK}",
            "polite": f"{RUDE_PREFIX}{core} {ns}{RUDE_SUFFIX}",
            "next_step": core,
            "on_topic": f"{s.irrelevant} {ns}",
        }
        for crit, reply in defects.items():
            items.append(
                Item(
                    f"s{si}-bad-{crit}",
                    si,
                    f"defect:{crit}",
                    s.question,
                    s.context,
                    reply,
                    _labels(**{crit: True}),
                )
            )
    return items


# ----------------------------------------------------------------------------- a code-based judge: the baseline to beat

NUMBER = re.compile(r"\$\d[\d.,]*|\b(?:RF|INV|KB)-\d+\b|\b\d+(?:\.\d+)?h?\b|\b0x[0-9a-f]+\b", re.I)
RUDE = re.compile(
    r"\b(as i already|you clearly|did not read|didn't read|read things before|obviously|calm down|stupid|whatever|your fault)\b",
    re.I,
)
NEXT = re.compile(
    r"\b(reply|contact|let us know|you can|you do not need|we will|i will|once you|please retry|retry|use the|could you|send it|send me|if (?:it|anything|you))\b",
    re.I,
)
STOP = set(
    "the a an and or of to in is it for on this that with as be are was i you your we our have has not no can will if so at by from".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


def rule_judge(item_or_reply, context: str = "", question: str = "") -> dict[str, bool]:
    """Code-only judge for the same six criteria. Cheap, deterministic, and (because we wrote it knowing how the defects are
    made) probably BETTER here than on real traffic: a ceiling to compare model judges against, not a recommendation."""
    reply = item_or_reply.reply if hasattr(item_or_reply, "reply") else item_or_reply
    context = item_or_reply.context if hasattr(item_or_reply, "context") else context
    question = item_or_reply.question if hasattr(item_or_reply, "question") else question
    allowed = {m.group().lower() for m in NUMBER.finditer(context + " " + question)}
    used = {m.group().lower() for m in NUMBER.finditer(reply)}
    overlap = len(_content_words(reply) & _content_words(context + " " + question)) / max(
        1, len(_content_words(reply))
    )
    return {
        "grounded": used <= allowed,
        "no_promise": not PROMISE.search(reply),
        "no_secret_request": not asks_for_secret(reply),
        "polite": not RUDE.search(reply),
        "next_step": bool(NEXT.search(reply)),
        "on_topic": overlap >= 0.25,
    }
