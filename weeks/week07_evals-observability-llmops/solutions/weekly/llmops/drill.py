"""Incident drills: synthetic production traffic with a KNOWN incident injected, to test that the monitor notices it.

A monitor that has never fired is an untested monitor. These drills run the real support system (scripted models, real
SQLite, real guards, real tracing) on a stream of customer messages and change something partway through: a model fault,
or the mix of what customers ask. The traffic is simulated; the system, the spans and the checks on them are not.
"""

from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import day1_solution  # noqa: F401  (puts the Week 6 support_system package on the path)
import evalcases as ec
import scripted
from support_system.system import SupportSystem

from common import tracing
from common.fake import fake_llm
from common.tracing import SpanRecord

# what customers ask, by kind (the first message of each Day 1 case)
KINDS_BILLING = ["billing-small", "billing-approval", "billing-unpaid", "billing-pressure"]
KINDS_TECH = ["tech-howto", "tech-outage"]
MIX_NORMAL = {
    "billing-small": 4,
    "billing-approval": 2,
    "billing-unpaid": 2,
    "billing-pressure": 1,
    "tech-howto": 4,
    "tech-outage": 2,
    "human": 1,
    "off-topic": 1,
}
MIX_TECH_HEAVY = {
    "billing-small": 1,
    "billing-approval": 1,
    "billing-unpaid": 1,
    "billing-pressure": 0,
    "tech-howto": 9,
    "tech-outage": 6,
    "human": 1,
    "off-topic": 3,
}


@dataclass
class Segment:
    """Traffic for ``turns`` customer messages with a given mix, scripted-model faults and system options."""

    turns: int
    mix: dict[str, float] = field(default_factory=lambda: dict(MIX_NORMAL))
    faults: dict = field(default_factory=dict)
    options: dict = field(default_factory=lambda: {"triage_mode": "rules"})


def openings_by_kind() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in ec.build_cases(lambda: None):
        out.setdefault(c.kind, []).append(c.scenario.user_factory().opening)
    return out


def simulate(segments: list[Segment], *, seed: int = 0) -> list[SpanRecord]:
    """Spans of every turn, in time order. Each customer message is its own conversation (and trace)."""
    pool = openings_by_kind()
    rng = random.Random(seed)
    spans: list[SpanRecord] = []
    counter = 0
    with tempfile.TemporaryDirectory() as tmp:
        for seg_no, seg in enumerate(segments):
            kinds = sorted(k for k, w in seg.mix.items() if w > 0)
            weights = [seg.mix[k] for k in kinds]
            with fake_llm(scripted.rules(**seg.faults)):
                with tracing.capture() as rec:
                    system = SupportSystem(
                        Path(tmp) / f"seg{seg_no}.sqlite", provider="anthropic", **seg.options
                    )
                    for _ in range(seg.turns):
                        kind = rng.choices(kinds, weights)[0]
                        counter += 1
                        system.handle(f"prod-{counter}", rng.choice(pool[kind]))
                spans.extend(rec.spans)
    return spans
