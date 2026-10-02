"""Week 8 Day 5 - Solution: PII detection, redaction middleware, logging hygiene and retention.

1. MEASURE   the detector on a generated labelled set (a ceiling) and on a messier human-written set (the honest estimate)
2. PROTECT   the Week 6 support system with ``PrivateSupport`` and check that no raw personal data reaches the database or the traces
3. RETAIN    purge old data, erase one conversation, and say what that does not cover

uv run python weeks/week08_security-and-guardrails/solutions/day5_solution.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import piidata  # noqa: E402

from common import pii  # noqa: E402
from common.abtest import wilson  # noqa: E402


def score(items: list[piidata.Item], *, strict_type: bool = True) -> dict:
    """Span-level scoring: a gold value is FOUND if a detected span of the right type (or ANY type when ``strict_type`` is false:
    for redaction it matters that the text is removed, not what the span is called) overlaps it. Detected spans that match no gold value,
    including every span in a hard negative, are false positives."""
    tp, fn, fp = defaultdict(int), defaultdict(int), defaultdict(int)
    missed, false_pos = [], []
    for it in items:
        spans = pii.detect(it.text)
        used: set[int] = set()
        for ptype, value in it.gold:
            at = it.text.find(value)
            hit = None
            for i, s in enumerate(spans):
                if i in used or (strict_type and s.type != ptype):
                    continue
                if min(s.end, at + len(value)) > max(s.start, at):
                    hit = i
                    break
            if hit is None:
                fn[ptype] += 1
                missed.append((ptype, value))
            else:
                tp[ptype] += 1
                used.add(hit)
        for i, s in enumerate(spans):
            if i not in used:
                fp[s.type] += 1
                false_pos.append((s.type, s.text, it.text))
    types = sorted(set(tp) | set(fn) | set(fp))
    rows = {}
    for t in types:
        n_gold = tp[t] + fn[t]
        rows[t] = {
            "tp": tp[t],
            "fn": fn[t],
            "fp": fp[t],
            "recall": tp[t] / n_gold if n_gold else None,
            "precision": tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else None,
            "recall_ci": wilson(tp[t], n_gold) if n_gold else None,
        }
    total_tp, total_fn, total_fp = sum(tp.values()), sum(fn.values()), sum(fp.values())
    return {
        "by_type": rows,
        "tp": total_tp,
        "fn": total_fn,
        "fp": total_fp,
        "recall": total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
        "precision": total_tp / (total_tp + total_fp) if total_tp + total_fp else 1.0,
        "recall_ci": wilson(total_tp, total_tp + total_fn),
        "missed": missed,
        "false_positives": false_pos,
    }


def fmt(name: str, r: dict) -> str:
    lines = [
        f"--- {name}: recall {r['recall']:.0%} [{r['recall_ci'][0]:.0%}, {r['recall_ci'][1]:.0%}] ({r['tp']} of {r['tp'] + r['fn']}), precision {r['precision']:.0%}, {r['fp']} false positives"
    ]
    lines.append(f"{'type':<15}{'found':>6}{'missed':>8}{'false+':>8}")
    for t, v in r["by_type"].items():
        lines.append(f"{t:<15}{v['tp']:>6}{v['fn']:>8}{v['fp']:>8}")
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    generated = score(piidata.build_dataset(0, per_type=10))
    realistic = score(piidata.REALISTIC)
    any_type = score(piidata.REALISTIC, strict_type=False)
    print(
        fmt("generated set (written with the detector's patterns in mind: a ceiling)", generated),
        "\n",
    )
    print(fmt("human-written realistic set (the honest estimate)", realistic), "\n")
    print(
        f"realistic set, ANY detected span counts (for redaction the label does not matter): recall {any_type['recall']:.0%} ({any_type['tp']} of {any_type['tp'] + any_type['fn']})\n"
    )
    print("missed in the realistic set:")
    for t, v in realistic["missed"]:
        print(f"   {t:<14}{v}")
    print("false positives in the realistic set:")
    for t, s, text in realistic["false_positives"]:
        print(f"   {t:<14}{s!r} in {text[:60]!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
