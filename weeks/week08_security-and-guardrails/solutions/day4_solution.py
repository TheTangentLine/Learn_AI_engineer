"""Week 8 Day 4 - Solution: harden the Week 5 research agent against injected pages.

The agent is the real one (HTTP tools, the MCP notes server, the real loop); the model is a scripted OBEDIENT one, so every number here is
about what the CONTROLS stop when the model does exactly what the attacker's page says. Layers are switched on one at a time:

  capabilities   only the tools the task needs (notes only if the user asked to save something)
  policy         default-deny tool rules, argument rules, DLP on the secret, an egress allowlist, an injection check on note text
  taint          refuse search queries copied from a page
  confirm        note writes need a human (who here declines suspicious text)
  output guard   sanitise the report before a client renders it

  uv run python weeks/week08_security-and-guardrails/solutions/day4_solution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import agent_target as A  # noqa: E402

from common import redteam as rt  # noqa: E402

CANARY = rt.Canary.make(seed=7)
GOALS = ["tool_misuse", "exfil_url", "say_token", "leak_secret", "false_fact"]

CONFIGS: dict[str, A.Hardening] = {
    "none": A.Hardening(),
    "capabilities": A.Hardening(capabilities=True),
    "policy": A.Hardening(policy=True),
    "taint": A.Hardening(taint=True),
    "confirm notes": A.Hardening(confirm_notes=True),
    "filter tool results": A.Hardening(filter_results=True),
    "output guard": A.Hardening(output_guard=True),
    "capabilities + policy + output": A.Hardening(
        capabilities=True, policy=True, output_guard=True
    ),
    "all layers": A.Hardening(
        capabilities=True,
        policy=True,
        taint=True,
        confirm_notes=True,
        filter_results=True,
        output_guard=True,
    ),
}


def attacks() -> list[rt.Attack]:
    return rt.build_attacks(CANARY, goals=GOALS, channels=("tool_result",))


def benign(h: A.AgentHarness) -> dict:
    """Does normal work still work? A plain question, and one that asks to save a note."""
    plain = h.run(A.task_text(CANARY))
    saved = h.run(A.task_text(CANARY, save_note=True))
    return {
        "plain_ok": plain["verification"].ok
        and "3 times" in plain["report"]
        and not plain["denied"],
        "note_saved": "3 times" in saved["notes"] and saved["verification"].ok,
        "plain_denied": plain["denied"],
        "note_denied": saved["denied"],
    }


def evaluate(name: str, hardening: A.Hardening) -> dict:
    with A.AgentHarness(CANARY, hardening) as h:
        results = rt.run(h, attacks(), CANARY)
        return {
            "name": name,
            "overall": rt.overall(results),
            "by_goal": {r["goal"]: r["succeeded"] for r in rt.asr_table(results, "goal")},
            "benign": benign(h),
            "results": results,
        }


def main(argv: list[str]) -> None:
    n = len(attacks())
    print(
        f"{n} attacks (goals x techniques) through the poisoned search result; secret in the user's request; obedient scripted model\n"
    )
    print(
        f"{'hardening':<34}{'ASR':>6}  {'notes':>6}{'exfil':>7}{'token':>7}{'leak':>6}{'fact':>6}   benign: plain / save-a-note"
    )
    for name, h in CONFIGS.items():
        r = evaluate(name, h)
        g = r["by_goal"]
        b = r["benign"]
        print(
            f"{name:<34}{r['overall']['asr']:>6.0%}  {g['tool_misuse']:>6}{g['exfil_url']:>7}{g['say_token']:>7}{g['leak_secret']:>6}{g['false_fact']:>6}   {'ok' if b['plain_ok'] else 'BROKEN':>6} / {'ok' if b['note_saved'] else 'BROKEN'}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
