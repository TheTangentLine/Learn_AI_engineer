"""Week 2 Day 2 - Solution: self-consistency (sample several reasoning paths, majority-vote).

The voting core is backend-independent: it only needs ``sample_fn(question, k) -> list[str]``.

  uv run python .../day2_solution.py                 # local Qwen2.5-0.5B (no key; ~4 min on CPU)
  uv run python .../day2_solution.py --backend api   # your configured provider (costs a few cents)
  uv run python .../day2_solution.py --offline       # logic self-test with a scripted sampler

Reported: greedy accuracy (local only), mean *single-sample* accuracy, majority-vote accuracy,
and accuracy by agreement level (does "5 of 5 agree" really mean "more likely correct"?).
"""

from __future__ import annotations

import asyncio
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SYSTEM = "Solve the problem. Think step by step, then end with a final line: ANSWER: <number>"

PROBLEMS: list[tuple[str, float]] = [
    ("Tom has 3 apples, eats 2, then buys 5 more. How many apples does he have?", 6),
    (
        "A pen costs $2 and a notebook costs $3. How much do 4 pens and 2 notebooks cost in dollars?",
        14,
    ),
    ("A train leaves at 2:00 pm and arrives at 5:30 pm. How many minutes is the trip?", 210),
    (
        "There are 24 students. Half of them walk to school. Of the rest, a third take a bus. "
        "How many take a bus?",
        4,
    ),
    ("A rectangle is 7 m by 5 m. What is its perimeter in metres?", 24),
    ("Sara reads 12 pages a day. How many pages does she read in 2 weeks?", 168),
    ("A shirt costs $40 and is 25% off. What is the sale price in dollars?", 30),
    (
        "If 5 workers build a wall in 12 days, how many days do 10 workers need, working at "
        "the same rate?",
        6,
    ),
    ("A bakery makes 48 muffins and sells 3/4 of them. How many are left?", 12),
    ("Ben is 4 years older than Amy. Amy is twice as old as Cal. Cal is 5. How old is Ben?", 14),
]

# ----------------------------------------------------------------- answer extraction

_NUM = r"-?\$?\s?(?:\d[\d,]*\.?\d*|\.\d+)"
_PATTERNS = [
    rf"ANSWER:\s*\**\s*({_NUM})",
    r"\\boxed\{\s*\\?\$?\s*(-?[\d,]*\.?\d+)",
    rf"[Ff]inal [Aa]nswer\s*(?:is)?:?\s*\**\s*({_NUM})",
    rf"[Tt]he answer is:?\s*\**\s*({_NUM})",
    rf"\b[Aa]nswer:\s*\**\s*({_NUM})",
]


def _to_float(s: str) -> float | None:
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_number(text: str) -> float | None:
    """Pull the final numeric answer out of free text.

    Small models don't reliably obey "end with ANSWER: n", so we try several conventions
    (ANSWER:, \\boxed{}, 'Final Answer', 'the answer is') and fall back to the last number.
    Real systems should use structured output instead (Day 3); this is the pragmatic fallback.
    """
    for pat in _PATTERNS:
        found = re.findall(pat, text)
        if found:
            val = _to_float(found[-1])
            if val is not None:
                return val
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return _to_float(nums[-1].rstrip(".")) if nums else None


# ----------------------------------------------------------------- the voting core


@dataclass
class Vote:
    answers: list[float | None]
    winner: float | None
    agreement: float  # share of samples that voted for the winner (0..1)


def vote(answers: list[float | None]) -> Vote:
    """Majority vote over parsed answers; unparseable samples (None) don't vote.

    Ties are broken by first appearance, which is deterministic and unbiased for i.i.d. samples.
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return Vote(answers, None, 0.0)
    counts = Counter(valid)
    best = max(counts.values())
    winner = next(a for a in valid if counts[a] == best)
    return Vote(answers, winner, best / len(answers))


def self_consistency(sample_fn: Callable[[str, int], list[str]], question: str, k: int) -> Vote:
    return vote([extract_number(t) for t in sample_fn(question, k)])


def evaluate(sample_fn, problems, k: int, greedy_fn=None) -> dict:
    singles, votes, greedy_hits, rows = [], [], [], []
    for q, gold in problems:
        v = self_consistency(sample_fn, q, k)
        singles.append(sum(a == gold for a in v.answers) / len(v.answers))
        votes.append(v.winner == gold)
        if greedy_fn:
            greedy_hits.append(extract_number(greedy_fn(q)) == gold)
        rows.append((gold, v))
    by_agreement: dict[str, list[bool]] = {"unanimous": [], "majority": [], "split": []}
    for gold, v in rows:
        bucket = "unanimous" if v.agreement == 1 else "majority" if v.agreement > 0.5 else "split"
        by_agreement[bucket].append(v.winner == gold)
    return {
        "single_sample_acc": sum(singles) / len(singles),
        "vote_acc": sum(votes) / len(votes),
        "greedy_acc": sum(greedy_hits) / len(greedy_hits) if greedy_hits else None,
        "by_agreement": {k_: (sum(v_), len(v_)) for k_, v_ in by_agreement.items()},
        "rows": rows,
    }


def report(name: str, res: dict, k: int) -> None:
    print(f"\n=== {name} (k={k} samples/question, {len(res['rows'])} questions) ===")
    if res["greedy_acc"] is not None:
        print(f"greedy (T=0) accuracy        : {res['greedy_acc']:.0%}")
    print(
        f"mean single-sample accuracy  : {res['single_sample_acc']:.0%}   <- what one call gives you"
    )
    print(f"majority-vote accuracy       : {res['vote_acc']:.0%}   <- {k} calls, {k}x the cost")
    print("accuracy by agreement level  :", end=" ")
    print(", ".join(f"{b}: {c}/{n}" for b, (c, n) in res["by_agreement"].items()))
    for gold, v in res["rows"]:
        mark = "ok " if v.winner == gold else "BAD"
        print(f"  {mark} gold={gold:<6g} votes={v.answers} -> {v.winner} (agree {v.agreement:.0%})")


# ----------------------------------------------------------------- backends


def local_backend():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()

    def render(q: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def generate(q: str, n: int, sample: bool) -> list[str]:
        ids = tok([render(q)] * n, return_tensors="pt", padding=True)
        kw = (
            dict(do_sample=True, temperature=0.8, top_p=1.0, top_k=0)
            if sample
            else {"do_sample": False}
        )
        out = model.generate(**ids, max_new_tokens=200, pad_token_id=tok.eos_token_id, **kw)
        return [tok.decode(o[ids["input_ids"].shape[1] :], skip_special_tokens=True) for o in out]

    torch.manual_seed(0)
    return (lambda q, k: generate(q, k, True)), (lambda q: generate(q, 1, False)[0])


def api_backend():
    from common import llm

    async def many(q: str, k: int) -> list[str]:
        rs = await asyncio.gather(
            *(llm.acomplete(q, system=SYSTEM, max_tokens=1500) for _ in range(k))
        )
        return [r.text for r in rs]

    # Note: we don't pass `temperature`: several current models reject it. Independent calls
    # already differ because the provider samples by default.
    return (lambda q, k: asyncio.run(many(q, k))), None


# ----------------------------------------------------------------- offline self-test


def offline_selftest() -> None:
    import random

    print(
        "*** OFFLINE: scripted sampler (each sample is right with p=0.6) - tests the voting "
        "logic only ***"
    )
    rng = random.Random(0)
    gold_by_q = dict(PROBLEMS)

    def sampler(q: str, k: int) -> list[str]:
        gold = gold_by_q[q]
        return [
            f"... so ANSWER: {gold if rng.random() < 0.6 else gold + rng.choice([-2, -1, 1, 3])}"
            for _ in range(k)
        ]

    # extractor unit tests
    assert extract_number("so ANSWER: 14") == 14
    assert extract_number("Total is \\( \\$8 + \\$6 = \\$14 \\)\n\\[ \\boxed{14} \\]") == 14
    assert extract_number("Final Answer: 9") == 9
    assert extract_number("Therefore, the answer is 1,250.") == 1250
    assert extract_number("it costs $2.50 each") == 2.5  # last-number fallback
    assert extract_number("no digits here") is None
    # voting unit tests
    assert vote([6, 6, 9, None, 6]).winner == 6 and vote([6, 6, 9, None, 6]).agreement == 0.6
    assert vote([1, 2]).winner == 1  # tie -> first seen
    assert vote([None, None]).winner is None
    res = evaluate(sampler, PROBLEMS * 4, k=7)
    report("scripted sampler", res, 7)
    assert res["vote_acc"] > res["single_sample_acc"], "voting must beat a single p=0.6 sampler"
    print("\nself-test passed")


def main() -> None:
    if "--offline" in sys.argv:
        return offline_selftest()
    k = 5
    if "--backend" in sys.argv and sys.argv[sys.argv.index("--backend") + 1] == "api":
        sample_fn, greedy_fn = api_backend()
        name = "API model"
    else:
        sample_fn, greedy_fn = local_backend()
        name = "local Qwen2.5-0.5B-Instruct"
    report(name, evaluate(sample_fn, PROBLEMS, k, greedy_fn), k)


if __name__ == "__main__":
    main()
