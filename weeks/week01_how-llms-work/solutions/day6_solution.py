"""Week 1 Day 6 - Solution: a model-selection benchmark (quality / latency / cost).

Runs every candidate model on three small tasks that are graded by CODE (no LLM judge, so the
scores are reproducible):

  classify     sentiment labels                       exact match
  math         multi-step word problems               numeric match on a final ``ANSWER: n`` line
  constraints  instruction-following with hard rules  per-item programmatic checkers

and prints accuracy, mean latency, total cost and *cost per correct answer* per model, then
recommends the cheapest model that clears an accuracy bar on each task.

Usage
-----
  uv run python weeks/week01_how-llms-work/solutions/day6_solution.py          # real models
  uv run python weeks/week01_how-llms-work/solutions/day6_solution.py --mock   # no keys needed:
                                                                               # tests the harness
  CANDIDATES="anthropic:claude-opus-5,anthropic:claude-haiku-4-5,openai:gpt-6-luna" ...

Real runs cost a few cents. The task sets are intentionally small - with ~8-10 items per task a
single item moves accuracy by 10-12 points, so treat results as a smoke test, not a leaderboard
(Week 7 covers building evals you can trust).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # reuse run_batch from Day 5
sys.path.insert(0, str(HERE.parents[2]))  # make `common` importable

from day5_solution import run_batch  # noqa: E402

# ----------------------------------------------------------------- the tasks


@dataclass
class Item:
    task: str
    prompt: str
    check: Any  # callable(text) -> bool


def _label(expected: str):
    return lambda text: text.strip().lower().strip(".!\"' ") == expected


def _number(expected: float):
    def check(text: str) -> bool:
        m = re.search(r"ANSWER:\s*\$?(-?[\d,]*\.?\d+)", text)
        return bool(m) and abs(float(m.group(1).replace(",", "")) - expected) < 1e-6

    return check


def _json_keys(keys: set[str]):
    def check(text: str) -> bool:
        try:
            body = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
            return set(json.loads(body)) == keys
        except (ValueError, TypeError):
            return False

    return check


def _bullets(n: int):
    def check(text: str) -> bool:
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        return len(lines) == n and all(ln.startswith("- ") for ln in lines)

    return check


def _word(length: int, *, starts: str | None = None, upper: bool = False):
    """Reply must be one alphabetic word of a given length (optionally prefix / UPPERCASE)."""

    def check(text: str) -> bool:
        w = text.strip().strip(".")
        return (
            w.isalpha()
            and len(w) == length
            and (starts is None or w.lower().startswith(starts))
            and (not upper or w.isupper())
        )

    return check


CLASSIFY_PROMPT = (
    "Classify the sentiment of this review as exactly one word: positive, negative or neutral.\n"
    "Review: {t}"
)
MATH_SUFFIX = "\nThink step by step, then finish with a final line of the form `ANSWER: <number>`."

ITEMS: list[Item] = [
    *[
        Item("classify", CLASSIFY_PROMPT.format(t=t), _label(lbl))
        for t, lbl in [
            ("Absolutely loved it, would buy again.", "positive"),
            ("Broke after two days. Total waste of money.", "negative"),
            ("It arrived on Tuesday in a brown box.", "neutral"),
            ("Not bad, but not great either; does the job I guess.", "neutral"),
            ("It works. Nothing special, nothing wrong with it.", "neutral"),
            ("I can't believe how much I hate the new update.", "negative"),
            ("Customer service resolved everything within minutes. Impressive!", "positive"),
            ("Meh.", "neutral"),
            ("Worst. Purchase. Ever. Never again.", "negative"),
            ("Exceeded every expectation I had.", "positive"),
        ]
    ],
    *[
        Item("math", q + MATH_SUFFIX, _number(a))
        for q, a in [
            (
                "A shop sells pens at $1.25 each. Maria buys 12 pens and pays with a $20 bill. "
                "How many dollars of change does she get?",
                5.0,
            ),
            (
                "A train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. What is its average "
                "speed in km/h for the whole trip?",
                75.0,
            ),
            (
                "A tank holds 240 litres. It is 35% full. After adding 60 litres, what percent "
                "full is it (give the percentage as a number)?",
                60.0,
            ),
            (
                "Tom is twice as old as Sam. In 6 years the sum of their ages will be 54. How old "
                "is Tom now?",
                28.0,
            ),
            ("What is 9.11 minus 9.9? Give the exact decimal.", -0.79),
            (
                "A recipe needs 3/4 cup of sugar for 12 cookies. How many cups of sugar for 40 "
                "cookies?",
                2.5,
            ),
            (
                "Revenue grew 20% in year one and fell 20% in year two, starting from $1000. "
                "What is the revenue after year two in dollars?",
                960.0,
            ),
            ("How many positive integers less than 100 are divisible by 6 or 15?", 19.0),
        ]
    ],
    *[
        Item("constraints", p, chk)
        for p, chk in [
            (
                "Write exactly 3 bullet points about rivers. Each line must start with '- '. "
                "No other text.",
                _bullets(3),
            ),
            (
                "Describe the sea in one sentence using only lowercase letters and no punctuation "
                "at all.",
                lambda t: (
                    t.strip() == t.strip().lower()
                    and not re.search(r"[^\w\s]", t.strip())
                    and len(t.strip()) > 10
                ),
            ),
            (
                "Return a JSON object with exactly the keys name and age (no other keys) for a "
                "fictional person. Output only the JSON.",
                _json_keys({"name", "age"}),
            ),
            (
                "Reply with a single English word that has exactly 7 letters and starts with 'b'. "
                "Nothing else.",
                _word(7, starts="b"),
            ),
            (
                "Write one sentence about coffee that does not contain the letter 'e'.",
                lambda t: "e" not in t.lower() and len(t.strip()) > 8,
            ),
            (
                "Give exactly 4 bullet points about sleep, each starting with '- ' and each at most "
                "6 words long. No other text.",
                lambda t: (
                    _bullets(4)(t)
                    and all(len(ln[2:].split()) <= 6 for ln in t.strip().splitlines() if ln.strip())
                ),
            ),
            (
                "Output the word 'banana' repeated exactly 5 times separated by single spaces, "
                "nothing else.",
                lambda t: t.strip().lower() == " ".join(["banana"] * 5),
            ),
            (
                "Respond with an uppercase word of exactly 5 letters. Nothing else.",
                _word(5, upper=True),
            ),
        ]
    ],
]

TASKS = ["classify", "math", "constraints"]
ACCURACY_BAR = 0.8

# ----------------------------------------------------------------- candidates & calling


def get_candidates(mock: bool) -> list[tuple[str, str]]:
    if mock:
        return [("mock", "mock-large"), ("mock", "mock-small"), ("mock", "mock-tiny")]
    env = os.getenv("CANDIDATES")
    if env:
        return [tuple(c.strip().split(":", 1)) for c in env.split(",") if c.strip()]  # type: ignore[misc]
    from common.llm import CHEAP_MODELS, DEFAULT_MODELS, available_providers

    out: list[tuple[str, str]] = []
    for p in available_providers():
        out.append((p, DEFAULT_MODELS[p]))
        if CHEAP_MODELS[p] != DEFAULT_MODELS[p]:
            out.append((p, CHEAP_MODELS[p]))
    return out


# Harness self-test: fake models whose accuracy/price/latency we control.
MOCK_SPECS = {  # accuracy, USD per call, latency seconds
    "mock-large": (0.97, 0.0100, 0.05),
    "mock-small": (0.85, 0.0010, 0.03),
    "mock-tiny": (0.55, 0.0001, 0.01),
}


async def call_model(candidate: tuple[str, str], item: Item) -> dict[str, Any]:
    provider, model = candidate
    if provider == "mock":
        acc, price, lat = MOCK_SPECS[model]
        rng = random.Random(f"{model}|{item.prompt}")
        await asyncio.sleep(lat * rng.uniform(0.8, 1.2))
        correct = rng.random() < acc
        return {
            "text": "__CORRECT__" if correct else "__WRONG__",
            "cost": price,
            "latency": lat,
            "mock_correct": correct,
        }
    from common.llm import acomplete

    r = await acomplete(item.prompt, provider=provider, model=model, max_tokens=3000)
    return {"text": r.text, "cost": r.cost_usd, "latency": r.latency_s, "stop": r.stop_reason}


def grade(item: Item, out: dict[str, Any]) -> bool:
    if "mock_correct" in out:
        return out["mock_correct"]
    return bool(item.check(out["text"]))


# ----------------------------------------------------------------- benchmark


async def benchmark(candidates: list[tuple[str, str]], items: list[Item]) -> dict:
    results: dict = {}
    for cand in candidates:
        stats = await run_batch(
            items,
            lambda it, c=cand: call_model(c, it),
            concurrency=4,
            max_retries=3,
            base_delay=1.0,
            attempt_timeout=120,
            seed=0,
        )
        per_task: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for it, r in zip(items, stats.results, strict=True):
            t = per_task[it.task]
            if r.ok:
                t["correct"].append(grade(it, r.value))
                t["cost"].append(r.value["cost"])
                t["latency"].append(r.value["latency"])
            else:
                t["correct"].append(False)  # an error is a wrong answer; we also keep the error
                t["errors"].append(r.error)
        results[cand] = per_task
    return results


def summarize(results: dict) -> None:
    print(f"\n{'model':32} {'task':12} {'acc':>5} {'lat(s)':>7} {'cost($)':>9} {'$/correct':>10}")
    print("-" * 80)
    for cand, per_task in results.items():
        for task in TASKS:
            t = per_task[task]
            n_ok = sum(t["correct"])
            acc = n_ok / len(t["correct"])
            cost = sum(t["cost"])
            lat = statistics.mean(t["latency"]) if t["latency"] else float("nan")
            per_correct = cost / n_ok if n_ok else float("inf")
            err = f"  ({len(t['errors'])} errors)" if t["errors"] else ""
            print(
                f"{':'.join(cand):32} {task:12} {acc:>5.0%} {lat:>7.2f} {cost:>9.5f} "
                f"{per_correct:>10.5f}{err}"
            )
        print()

    print(f"Recommendation (cheapest model with accuracy >= {ACCURACY_BAR:.0%}):")
    for task in TASKS:
        eligible = []
        for cand, per_task in results.items():
            t = per_task[task]
            if sum(t["correct"]) / len(t["correct"]) >= ACCURACY_BAR:
                eligible.append((sum(t["cost"]), ":".join(cand)))
        pick = (
            min(eligible)[1] if eligible else "none cleared the bar - use a stronger model/prompt"
        )
        print(f"  {task:12} -> {pick}")


async def main(mock: bool) -> None:
    candidates = get_candidates(mock)
    if not candidates:
        raise SystemExit("No providers configured: add a key to .env or run with --mock.")
    print(
        f"Candidates: {[':'.join(c) for c in candidates]}  |  {len(ITEMS)} items "
        f"({', '.join(f'{t}={sum(i.task == t for i in ITEMS)}' for t in TASKS)})"
    )
    results = await benchmark(candidates, ITEMS)
    summarize(results)

    if mock:  # harness assertions: ordering must follow the controlled accuracies/prices
        acc = {c[1]: sum(r for t in TASKS for r in results[c][t]["correct"]) for c in results}
        assert acc["mock-large"] > acc["mock-tiny"], "large should beat tiny"
        total = {c[1]: sum(sum(results[c][t]["cost"]) for t in TASKS) for c in results}
        assert total["mock-large"] > total["mock-small"] > total["mock-tiny"]
        print("\nharness self-test passed")


if __name__ == "__main__":
    asyncio.run(main(mock="--mock" in sys.argv))
