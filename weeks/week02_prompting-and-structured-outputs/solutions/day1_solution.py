"""Week 2 Day 1 - Solution: bad vs. good prompts, measured.

Three tasks, each with a "bad" and a "good" prompt template, 6 gold-labelled inputs, and a
code-based checker. The harness prints pass rate before/after per task and lists the good
prompt's failures so you can iterate.

  uv run python weeks/week02_prompting-and-structured-outputs/solutions/day1_solution.py            # real model
  uv run python weeks/week02_prompting-and-structured-outputs/solutions/day1_solution.py --offline  # harness self-test

``--offline`` swaps in a scripted fake model whose "sloppiness" depends on whether the prompt
contains an explicit ``<output_format>`` block. That is circular on purpose: it proves the
*harness* separates good from bad runs, it says nothing about real models.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common import llm  # noqa: E402
from common.fake import fake_llm  # noqa: E402

# ----------------------------------------------------------------- prompts

ACTIONS_BAD = Template("Get the action items from this: $text")
ACTIONS_GOOD = Template("""\
You turn meeting notes into a task list for the team's tracker.

<rules>
- An action item is a concrete task that a named person or group commits to do or is asked to do.
- Decisions, discussion and opinions are not action items.
- Write each item as: "- <owner>: <task>" (use "team" if no owner is named).
</rules>

<notes>
$text
</notes>

<output_format>
Reply with only the bullet lines, one per action item, each starting with "- ".
If the notes contain no action items, reply with exactly: NONE
</output_format>""")

PRIORITY_BAD = Template("What priority is this ticket? $text")
PRIORITY_GOOD = Template("""\
You triage support tickets for a payments company.

<priority_definitions>
P1: service down, data loss, a security issue, or money moved incorrectly
P2: a feature is broken but a workaround exists, or many users are inconvenienced
P3: how-to questions, cosmetic issues, feature requests
</priority_definitions>

<examples>
<example><ticket>Where can I download my invoice?</ticket><answer>P3</answer></example>
<example><ticket>The export button does nothing, but I can copy the table by hand.</ticket><answer>P2</answer></example>
<example><ticket>Customers are being charged twice since this morning.</ticket><answer>P1</answer></example>
</examples>

<ticket>
$text
</ticket>

<output_format>
Reply with only P1, P2 or P3. Your answer is parsed by a script, so add no other text.
</output_format>""")

SUMMARY_BAD = Template("Summarize: $text")
SUMMARY_GOOD = Template("""\
You write one-line summaries of product reviews for a dashboard.

<review>
$text
</review>

Summarise the review in at most 20 words. Keep the single most important product aspect the
reviewer mentions (for example: battery, screen, price, support).

<output_format>
Reply with the summary sentence only: no preamble such as "Here is" or "Summary:", no quotes.
</output_format>""")

# ----------------------------------------------------------------- data + checkers


@dataclass
class Case:
    text: str
    gold: object  # int (bullets), str (label) or str (key term)


def check_actions(out: str, gold: int) -> bool:
    out = out.strip()
    if gold == 0:
        return out == "NONE"
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return len(lines) == gold and all(ln.startswith("- ") for ln in lines)


def check_priority(out: str, gold: str) -> bool:
    return out.strip() == gold


def check_summary(out: str, gold: str) -> bool:
    out = out.strip()
    words = out.split()
    preamble = re.match(r"(?i)^(here|summary|this review|the review|sure)\b", out)
    return 0 < len(words) <= 20 and not preamble and gold.lower() in out.lower()


@dataclass
class Task:
    name: str
    bad: Template
    good: Template
    cases: list[Case]
    check: Callable[[str, object], bool]
    max_tokens: int


TASKS = [
    Task(
        "actions",
        ACTIONS_BAD,
        ACTIONS_GOOD,
        [
            Case(
                "Sam will send the budget by Friday. We agreed the launch date stays. "
                "Priya to book the venue.",
                2,
            ),
            Case("Discussion about Q3 numbers. No decisions were made.", 0),
            Case(
                "Lee: I'll fix the login bug today. Ana: I'll review Lee's PR tomorrow. "
                "Tom: I'll update the docs.",
                3,
            ),
            Case("Great meeting everyone, thanks for the energy!", 0),
            Case(
                "The team agreed to move standup to 10am. Jo will tell facilities about the room.",
                1,
            ),
            Case(
                "We reviewed the roadmap. Maria will draft the PRD. Dev must confirm the vendor "
                "quote by Tuesday.",
                2,
            ),
        ],
        check_actions,
        400,
    ),
    Task(
        "priority",
        PRIORITY_BAD,
        PRIORITY_GOOD,
        [
            Case("The whole checkout page returns a 500 error for every customer.", "P1"),
            Case("Someone logged into my account from another country and moved my savings.", "P1"),
            Case("The CSV export is broken but I can copy the data from the screen.", "P2"),
            Case("Search is very slow for all users on the EU cluster since the update.", "P2"),
            Case("How do I change the colour theme of my dashboard?", "P3"),
            Case("Could you add a dark mode some day? The logo also looks a little blurry.", "P3"),
        ],
        check_priority,
        100,
    ),
    Task(
        "summary",
        SUMMARY_BAD,
        SUMMARY_GOOD,
        [
            Case(
                "I bought this phone two weeks ago and the battery lasts nearly three days of "
                "heavy use. The camera is fine, nothing special, and the case feels a bit cheap "
                "but honestly the battery alone makes it worth the price.",
                "battery",
            ),
            Case(
                "The monitor arrived with a dead pixel in the middle of the screen. Support "
                "replaced it quickly, but the screen quality is otherwise excellent and the colours "
                "are accurate.",
                "screen",
            ),
            Case(
                "Way too expensive for what it is. The vacuum works okay but at this price I "
                "expected a lot more. I'd wait for a sale; the price is the main problem.",
                "price",
            ),
            Case(
                "I had a problem with my order and customer support fixed it in ten minutes, "
                "friendly and effective. The product itself is average, but their support was "
                "outstanding.",
                "support",
            ),
            Case(
                "The headphones sound great but the battery died after two hours which makes "
                "them useless on a flight. Battery is the dealbreaker for me.",
                "battery",
            ),
            Case(
                "Beautiful screen, sharp and bright even outdoors. Everything else about the "
                "tablet is mediocre, but the screen is why I would recommend it.",
                "screen",
            ),
        ],
        check_summary,
        200,
    ),
]


# ----------------------------------------------------------------- harness


def run_task(task: Task, version: str) -> list[tuple[Case, str, bool]]:
    template = task.bad if version == "bad" else task.good
    rows = []
    for case in task.cases:
        out = llm.complete(template.substitute(text=case.text), max_tokens=task.max_tokens).text
        rows.append((case, out, task.check(out, case.gold)))
    return rows


def self_test_checkers() -> None:
    """Every checker must accept a known-good and reject a known-bad string."""
    assert check_actions("- Sam: send budget\n- Priya: book venue", 2)
    assert not check_actions("Here are the items:\n- a\n- b", 2)
    assert check_actions("NONE", 0) and not check_actions("- nothing", 0)
    assert check_priority("P1", "P1") and not check_priority("P1.", "P1")
    assert check_priority(" P2\n", "P2") and not check_priority("It is P2", "P2")
    assert check_summary("Battery lasts three days; camera is average.", "battery")
    assert not check_summary("Here is a summary: battery is great.", "battery")
    assert not check_summary("word " * 25 + "battery", "battery")
    assert not check_summary("Great phone overall.", "battery")


def main(offline: bool) -> None:
    self_test_checkers()
    print("checkers validated (good accepted, bad rejected)\n")
    gold_by_text = {c.text: c.gold for t in TASKS for c in t.cases}

    def offline_model(prompt: str) -> str:
        """Scripted stand-in: a 'sloppy' model unless the prompt has an explicit output format."""
        text = next(t for t in gold_by_text if t in prompt)
        gold = gold_by_text[text]
        strict = "<output_format>" in prompt
        if isinstance(gold, int):  # actions
            body = "NONE" if gold == 0 else "\n".join(f"- team: task {i + 1}" for i in range(gold))
            return body if strict else "Sure! Here are the action items I found:\n" + body
        if gold in ("P1", "P2", "P3"):
            return gold if strict else f"I'd say this is probably a {gold} given the impact."
        return (
            f"{gold.capitalize()} is the standout point."
            if strict
            else f"Here is a summary of the review: the {gold} was discussed at length. " * 2
        )

    ctx = fake_llm([(r"(?s).*", offline_model)]) if offline else None
    if ctx:
        ctx.__enter__()
        print(
            "*** OFFLINE: scripted fake model - this checks the harness, not any real model ***\n"
        )
    try:
        print(f"{'task':10} {'bad prompt':>12} {'good prompt':>13}")
        failures = []
        totals = {"bad": [0, 0], "good": [0, 0]}
        for task in TASKS:
            line = f"{task.name:10}"
            for version in ("bad", "good"):
                rows = run_task(task, version)
                ok = sum(r[2] for r in rows)
                totals[version][0] += ok
                totals[version][1] += len(rows)
                line += f" {ok}/{len(rows)} ({ok / len(rows):.0%})".rjust(
                    13 if version == "bad" else 14
                )
                if version == "good":
                    failures += [(task.name, c.text, o) for c, o, passed in rows if not passed]
            print(line)
        for v in ("bad", "good"):
            print(f"overall {v}: {totals[v][0]}/{totals[v][1]}")
        print(f"\ngood-prompt failures: {len(failures)}")
        for name, text, out in failures:
            print(f"  [{name}] input={text[:50]!r}... output={out[:80]!r}")
        assert totals["good"][0] >= totals["bad"][0], "good prompts should not lose to bad ones"
        if offline:
            assert totals["good"][0] == totals["good"][1] and totals["bad"][0] < totals["bad"][1]
            print("\nharness self-test passed")
    finally:
        if ctx:
            ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main("--offline" in sys.argv)
