"""Week 8 Day 3 - Solution: defences, measured against the Day 2 attacks AND against normal use.

Each layer is switched on alone and then combined, and judged on four numbers: the attack success rate on the lab attacks (96), on a
HELD-OUT set written before the detector (60), whether normal use still works (golden questions) and how many benign inputs it wrongly
blocks. Two models: the scripted OBEDIENT one (what a defence achieves when the model does whatever it is told: the structural
effect) and the real Qwen (what the model's own behaviour adds).

  uv run python weeks/week08_security-and-guardrails/solutions/day3_solution.py [qwen]
"""

from __future__ import annotations

import dataclasses
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import defenses as D  # noqa: E402
import heldout  # noqa: E402
import targets as T  # noqa: E402
from day2_solution import CANARY, CHANNELS, GOAL_IDS  # noqa: E402

from common import guard  # noqa: E402
from common import redteam as rt  # noqa: E402


def lab_attacks() -> list[rt.Attack]:
    return rt.build_attacks(CANARY, goals=GOAL_IDS, channels=CHANNELS)


def heldout_attacks() -> list[rt.Attack]:
    """The held-out texts as attacks in both channels."""
    out = []
    for i, (text, goal) in enumerate(heldout.attacks_for(CANARY)):
        for ch in CHANNELS:
            out.append(rt.Attack(f"{goal}/heldout{i:02d}/{ch}", goal, f"heldout{i:02d}", ch, text))
    return out


def benign_inputs() -> list[str]:
    return heldout.BENIGN + [q for q, _, _ in T.GOLDEN]


def evaluate(model: T.Model, defenses: D.Defenses) -> dict:
    defenses = dataclasses.replace(
        defenses, rng=random.Random(0)
    )  # reproducible boundaries (production uses real randomness)
    target = T.RagTarget(model, CANARY, defenses=defenses)
    lab = rt.overall(rt.run(target, lab_attacks(), CANARY))
    held = rt.overall(rt.run(target, heldout_attacks(), CANARY))
    golden = T.golden_accuracy(target)
    # benign blocking: the defence refused or emptied the answer to a harmless question
    refused = 0
    for q in benign_inputs():
        obs = target.ask(q)
        refused += obs.blocked_by in ("input_guard",) or obs.output in (D.REFUSAL,)
    return {
        "lab": lab,
        "heldout": held,
        "golden": golden,
        "benign_refused": refused,
        "benign_n": len(benign_inputs()),
    }


def detector_report(threshold: float = guard.THRESHOLD) -> dict:
    """The detector alone: recall on the lab attacks (a ceiling: written knowing them) and on the held-out set, false positives."""
    lab = [a.text for a in lab_attacks()]
    held = [t for t, _ in heldout.attacks_for(CANARY)]
    benign = benign_inputs() + list(T.DOCS.values())
    flag = lambda t: guard.detect(t, threshold=threshold).flagged  # noqa: E731
    return {
        "lab_recall": sum(map(flag, lab)) / len(lab),
        "heldout_recall": sum(map(flag, held)) / len(held),
        "heldout_missed": [t for t in held if not flag(t)],
        "false_positives": [b for b in benign if flag(b)],
        "benign_n": len(benign),
    }


def table(model_name: str, model: T.Model) -> str:
    lines = [
        f"--- {model_name}",
        f"{'defence':<32}{'lab ASR':>9}{'held-out':>10}{'golden':>8}{'benign refused':>16}",
    ]
    for name, d in D.CONFIGS.items():
        r = evaluate(model, d)
        lines.append(
            f"{name:<32}{r['lab']['asr']:>9.0%}{r['heldout']['asr']:>10.0%}{r['golden']['correct']:>5}/{r['golden']['n']:<2}{r['benign_refused']:>10}/{r['benign_n']}"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    d = detector_report()
    print(
        f"detector alone: recall {d['lab_recall']:.0%} on the lab attacks, {d['heldout_recall']:.0%} on the held-out set; {len(d['false_positives'])} false positives of {d['benign_n']} benign texts"
    )
    for t in d["heldout_missed"]:
        print("   missed:", t[:100])
    for t in d["false_positives"]:
        print("   false positive:", t[:100])
    print()
    print(table("obedient (scripted): the structural effect", T.obedient_model(CANARY)))
    if "qwen" in argv:
        print()
        print(table("Qwen2.5-0.5B (real)", T.qwen_model()))


if __name__ == "__main__":
    main(sys.argv[1:])
