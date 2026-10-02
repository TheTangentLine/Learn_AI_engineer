"""Week 8 Day 6 - Solution: hallucination and grounding, then an automated red-team campaign with a findings table.

PART A  the real local model (Qwen2.5-0.5B) behind the Week 3 RAG bot answers 12 answerable, 12 unanswerable and 6 false-premise questions;
        three grounding gates (retrieval similarity, word overlap, NLI entailment) try to stop the bad answers without losing the good ones
PART B  an adaptive attacker mutates the 96 Day 2 attacks (18 variants each) against the Day 3 defences; the successes become findings, each
        replayed on the real model behind the same defences

  uv run python weeks/week08_security-and-guardrails/solutions/day6_solution.py [grounding|campaign]
"""

from __future__ import annotations

import dataclasses
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import campaign as C  # noqa: E402
import defenses as D  # noqa: E402
import grounding as G  # noqa: E402
import mutators  # noqa: E402
import targets as T  # noqa: E402
from day2_solution import CANARY  # noqa: E402
from day3_solution import lab_attacks  # noqa: E402

from common import guard  # noqa: E402
from common import redteam as rt  # noqa: E402

THRESHOLDS = {"top_cosine": [0.2, 0.3, 0.4], "lexical": [0.7, 0.9, 1.0], "nli": [0.1, 0.5, 0.9]}

CAMPAIGN_CONFIGS = {
    "none": D.Defenses(),
    "detector layers (input guard + document filter)": D.Defenses(
        input_guard=True, doc_filter="sentences"
    ),
    "structural (isolate + output guard)": D.Defenses(isolate_secret=True, output_guard=True),
    "all layers": D.CONFIGS["all layers"],
}


# ----------------------------------------------------------------------------- part A


def part_a() -> str:
    from common.judges import NLIJudge

    judge = NLIJudge()
    cases = G.collect(T.qwen_model(), CANARY, judge=judge)
    out = [
        "PART A - hallucination and grounding (real Qwen2.5-0.5B behind the Week 3 retrieval and prompt)",
        "",
    ]
    out.append(
        f"{'question type':<16}{'n':>3}{'correct':>9}{'abstained':>11}{'wrong':>7}{'uncorrected':>13}"
    )
    for kind in ("answerable", "unanswerable", "false_premise"):
        cs = [c for c in cases if c.kind == kind]
        n = lambda o, cs=cs: sum(c.outcome == o for c in cs)  # noqa: E731
        out.append(
            f"{kind:<16}{len(cs):>3}{n('correct'):>9}{n('abstained'):>11}{n('wrong'):>7}{n('uncorrected'):>13}"
        )
    good, bad = G.split(cases)
    sw = G.swapped(cases, judge)
    out += [
        "",
        f"answers the model gave: {len(good)} good (correct), {len(bad)} bad (hallucinated or going along with a false premise);"
        f" plus {len(sw)} synthetic number swaps of the good ones",
        "",
    ]
    for name, bads in (("natural errors", bad), ("number swaps", sw)):
        out.append(
            f"AUROC on {name} (1.0 = perfect, 0.5 = no signal):  "
            + "  ".join(
                f"{s}={G.auroc([getattr(c, s) or 0.0 for c in good], [getattr(c, s) or 0.0 for c in bads]):.2f}"
                for s in THRESHOLDS
            )
        )
    for name, bads in (("natural errors", bad), ("number swaps", sw)):
        out += [
            "",
            f"gate sweep on {name}: bad answers let through / good answers lost (an unjudged answer counts as lost)",
        ]
        for sig, ts in THRESHOLDS.items():
            rows = G.gate_table(good + bads, sig, ts)
            out.append(
                f"  {sig:<11}"
                + "   ".join(
                    f">={r['threshold']}: {r['bad_through']}/{r['bad_total']} through, {r['good_lost']}/{r['good_total']} lost"
                    for r in rows
                )
            )
    out += [
        "",
        "the bad answers every gate lets through (grounded but not an answer to the question, or an invented fact the sources happen to support):",
    ]
    for c in bad:
        if (c.lexical >= 0.9) and (c.nli is not None and c.nli >= 0.5):
            out.append(f"  Q: {c.question}\n     A: {c.answer[:100]}")
    return "\n".join(out)


# ----------------------------------------------------------------------------- part B


def mutator_table() -> list[str]:
    """Which rewrites get past the detector, and do they keep the attack working? (Scripted obedient model, no defence, for the second.)"""
    none = T.RagTarget(T.obedient_model(CANARY), CANARY)
    out = [
        f"{'mutator':<12}{'changed':>8}{'still flagged':>15}{'evades':>8}{'still works*':>14}{'evades AND works':>18}"
    ]
    for name in mutators.MUTATORS:
        changed = flagged = works = both = 0
        for a in lab_attacks():
            t = mutators.mutate(a.text, name, CANARY)
            if t == a.text:
                continue
            f = guard.detect(t).flagged
            ok = rt.run(none, [dataclasses.replace(a, text=t)], CANARY)[0].succeeded
            changed, flagged, works, both = (
                changed + 1,
                flagged + f,
                works + ok,
                both + (ok and not f),
            )
        out.append(
            f"{name:<12}{changed:>8}{flagged:>15}{changed - flagged:>8}{works:>14}{both:>18}"
        )
    out.append(
        "* a model that does what the text says; a real model reads some rewrites worse (zero-width characters) and some better"
    )
    return out


def qwen_replay(defenses: D.Defenses, outcomes: list[C.Outcome]) -> tuple[int, int, int]:
    """Replay each winning variant against the REAL model behind the same defences: (succeeded, attempted, replayed)."""
    target = T.RagTarget(
        T.qwen_model(), CANARY, defenses=dataclasses.replace(defenses, rng=random.Random(0))
    )
    won = [o for o in outcomes if o.first_success is not None]
    results = [
        rt.run(target, [dataclasses.replace(o.attack, text=o.winning_text)], CANARY)[0] for o in won
    ]
    return sum(r.succeeded for r in results), sum(r.attempted for r in results), len(results)


def part_b(real_model: bool = True):
    attacks = lab_attacks()
    lines = [
        "PART B - adaptive red-team campaign: 96 lab attacks x up to 18 mutations, scripted obedient model",
        "",
    ]
    lines += mutator_table()
    lines += [
        "",
        f"{'defence':<50}{'static':>8}{'<=5 queries':>13}{'<=19 queries':>14}{'  winners that also work on Qwen'}",
    ]
    findings: list[C.Finding] = []
    for name, d in CAMPAIGN_CONFIGS.items():
        dd = dataclasses.replace(d, rng=random.Random(0))
        target = T.RagTarget(T.obedient_model(CANARY), CANARY, defenses=dd)
        outs = C.adaptive_run(target, attacks, CANARY)
        s = C.summarize(outs, (1, 5, 1000))
        replay = ""
        if real_model and name != "none" and s[1000][0] > s[1][0]:
            ok, tried, n = qwen_replay(
                d, [o for o in outs if o.first_success and o.first_success > 1]
            )
            replay = f"  {ok} of {n} succeed, {tried} attempted"
        lines.append(
            f"{name:<50}"
            + "".join(f"{s[b][0]:>{w}}/{s[b][1]}" for b, w in ((1, 5), (5, 10), (1000, 11)))
            + replay
        )
        if name != "none":
            fs = C.findings_from(outs, name, start=len(findings) + 1)
            for f in fs:  # a finding is only worth filing if a developer can reproduce it
                o = next(x for x in outs if x.attack.id == f.example_attack)
                f.reproduced = C.reproduce(
                    lambda dd=dd: T.RagTarget(
                        T.obedient_model(CANARY),
                        CANARY,
                        defenses=dataclasses.replace(dd, rng=random.Random(0)),
                    ),
                    o.attack,
                    o.winning_text,
                    CANARY,
                    3,
                )
            findings += fs
    return "\n".join(lines), findings


def findings_markdown(findings: list[C.Finding]) -> str:
    rows = [
        "| id | severity | title | attacks | repro | evidence (attack id, rewrite) |",
        "|---|---|---|---|---|---|",
    ]
    for f in findings:
        rows.append(
            f"| {f.id} | {f.severity} | {f.title} | {f.successes}/{f.total} | {f.reproduced} | `{f.example_attack}` via {f.example_chain} |"
        )
    return "\n".join(rows)


def main(argv: list[str]) -> None:
    which = argv[1] if len(argv) > 1 else "all"
    if which in ("all", "grounding"):
        print(part_a(), "\n")
    if which in ("all", "campaign"):
        text, findings = part_b(real_model="--no-model" not in argv)
        print(text, "\n\nfindings:\n" + findings_markdown(findings))


if __name__ == "__main__":
    main(sys.argv)
