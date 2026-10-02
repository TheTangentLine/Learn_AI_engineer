"""Root-cause diagnosis of failed conversations: the step between 'it failed' and 'what do we fix'.

A failed case usually trips SEVERAL assertions that share one root cause (a misrouted ticket also 'misses' the tool
call it never had the chance to make). ``diagnose`` walks a decision list from the EARLIEST point of failure in the
pipeline (routing) to the latest (the final reply) and returns the first cause that explains the case. Anything it cannot
explain lands in ``other``: that bucket is what a human reads next, and its size tells you whether the taxonomy is complete.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

NARRATION = re.compile(
    r"\b(i will|i'll|let me|i am going to|i'm going to|we will|you can use|you could use|please use|to do this|first,? i)\b",
    re.I,
)  # intent to act, not mere capability ("I can help")

CAUSES = [
    "crash",
    "misroute",
    "specialist_stuck_or_failed",
    "narrated_instead_of_acting",
    "skipped_tools",
    "skipped_verification",
    "invented_arguments",
    "acted_on_ineligible_input",
    "did_not_ask_for_missing_info",
    "hallucinated_success",
    "ignored_tool_result",
    "conversation_loop",
    "other",
]


@dataclass
class Diagnosis:
    cause: str
    evidence: str = ""


def diagnose(
    failures: list[str],
    *,
    kind: str,
    expected_route: str,
    actual_route: str | None,
    calls: list[str],
    agent_text: str,
    error: str = "",
    specialist_failed: str = "",
) -> Diagnosis:
    """The earliest explanation that fits. ``calls`` are the specialist's tool names; ``actual_route`` is triage's category."""
    f = set(failures)
    if "agent_error" in f:
        return Diagnosis("crash", error)
    if actual_route is not None and _route_class(actual_route) != _route_class(expected_route):
        return Diagnosis("misroute", f"expected {expected_route}, got {actual_route}")
    if specialist_failed:  # the specialist ran out of steps, got stuck repeating a call, or errored: the system escalated
        return Diagnosis("specialist_stuck_or_failed", specialist_failed)
    if "did_not_terminate" in f:
        return Diagnosis("conversation_loop")
    wants_calls = any(x.startswith(("missing_call", "wrong_args")) for x in f)
    if wants_calls and not calls:
        return Diagnosis(
            "narrated_instead_of_acting" if NARRATION.search(agent_text) else "skipped_tools",
            agent_text[:80],
        )
    if any(x.startswith("forbidden_call") for x in f):
        return Diagnosis("acted_on_ineligible_input")
    if any(
        x == "check_failed:asked for the invoice id first" for x in f
    ):  # asking comes BEFORE acting: the earlier cause
        return Diagnosis("did_not_ask_for_missing_info")
    if any(x.startswith("wrong_args") for x in f):
        return Diagnosis("invented_arguments")
    if "missing_call:lookup_invoice" in f:
        return Diagnosis("skipped_verification")
    if "check_failed:no false success" in f or "check_failed:ids are grounded" in f:
        return Diagnosis("hallucinated_success")
    if (
        any(
            x.startswith(
                ("check_failed:uses what the article says", "check_failed:mentions the degraded")
            )
            for x in f
        )
        and calls
    ):
        return Diagnosis("ignored_tool_result")
    return Diagnosis("other", ", ".join(sorted(f)))


def _route_class(category: str) -> str:
    return {"technical": "technical", "tech": "technical"}.get(category, category)


@dataclass
class Pareto:
    rows: list[tuple[str, int, float, dict[str, int]]] = field(
        default_factory=list
    )  # cause, count, share, by kind
    total_failures: int = 0
    total_cases: int = 0

    def __str__(self) -> str:
        lines = [
            f"{self.total_failures} failures in {self.total_cases} cases",
            f"{'cause':32s} {'n':>3s} {'share':>6s}  kinds",
        ]
        for cause, n, share, kinds in self.rows:
            lines.append(
                f"{cause:32s} {n:3d} {share:6.0%}  "
                + ", ".join(f"{k}x{v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
            )
        return "\n".join(lines)


def pareto(causes_by_case: list[tuple[str, str]], total_cases: int) -> Pareto:
    """``causes_by_case``: (kind, cause) for every FAILED case. Sorted by frequency: fix from the top."""
    counts = Counter(c for _, c in causes_by_case)
    n = len(causes_by_case)
    rows = []
    for cause, count in counts.most_common():
        kinds = Counter(k for k, c in causes_by_case if c == cause)
        rows.append((cause, count, count / n if n else 0.0, dict(kinds)))
    return Pareto(rows, n, total_cases)
