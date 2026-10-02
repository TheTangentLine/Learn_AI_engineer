"""Week 8 Day 2 - Solution: a prompt-injection lab against the Week 3 RAG bot.

Four goals (print a token, leak a hidden code, exfiltrate it through an image URL, plant a falsehood), twelve techniques, two
channels (the user's own message, and a poisoned document the bot retrieves): 96 attacks. Success is decided by code on canaries,
never by a judge. Three models are put behind the same bot: a robust one (the floor), an obedient one (the ceiling and a check
that the oracles work) and the real local Qwen2.5-0.5B.

  uv run python weeks/week08_security-and-guardrails/solutions/day2_solution.py            # scripted models only (offline)
  uv run python weeks/week08_security-and-guardrails/solutions/day2_solution.py qwen       # + the real model (slow once, then cached)
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import targets as T  # noqa: E402

from common import redteam as rt  # noqa: E402

CANARY = rt.Canary.make(seed=7)
GOAL_IDS = [
    "say_token",
    "leak_secret",
    "exfil_url",
    "false_fact",
]  # tool_misuse needs an agent: Day 4
CHANNELS = ("user", "document")


def attacks() -> list[rt.Attack]:
    return rt.build_attacks(CANARY, goals=GOAL_IDS, channels=CHANNELS)


def run_model(name: str, model: T.Model) -> tuple[list[rt.Result], dict]:
    target = T.RagTarget(model, CANARY)
    return rt.run(target, attacks(), CANARY), T.golden_accuracy(target)


def fmt_table(rows: list[dict], by: str) -> str:
    lines = [f"{by:<18}{'attacks':>8}{'attempt':>9}{'success':>9}{'ASR':>7}  95% interval"]
    for r in rows:
        lines.append(
            f"{r[by]:<18}{r['attacks']:>8}{r['attempted']:>9}{r['succeeded']:>9}{r['asr']:>7.0%}  [{r['asr_ci'][0]:.0%}, {r['asr_ci'][1]:.0%}]"
        )
    return "\n".join(lines)


def report(name: str, results: list[rt.Result], golden: dict, examples: int = 3) -> str:
    o = rt.overall(results)
    lines = [
        f"=== {name}: {o['attacks']} attacks, {o['reached']} reached the model, {o['attempted']} attempted, {o['succeeded']} succeeded: ASR {o['asr']:.0%} [{o['asr_ci'][0]:.0%}, {o['asr_ci'][1]:.0%}]",
        f"    normal use: {golden['correct']}/{golden['n']} golden questions answered correctly",
        "",
        fmt_table(rt.asr_table(results, "goal"), "goal"),
        "",
        fmt_table(rt.asr_table(results, "channel"), "channel"),
        "",
        fmt_table(rt.asr_table(results, "technique"), "technique"),
        "",
        fmt_table(rt.asr_table(results, "family"), "family"),
    ]
    wins = [r for r in results if r.succeeded][:examples]
    for r in wins:
        lines.append(f"\n  e.g. {r.attack.id}: {r.output[:200]!r}")
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    models = {"robust (scripted)": T.robust_model, "obedient (scripted)": T.obedient_model(CANARY)}
    if "qwen" in argv:
        models["Qwen2.5-0.5B (real)"] = T.qwen_model()
    for name, model in models.items():
        results, golden = run_model(name, model)
        print(report(name, results, golden), "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
