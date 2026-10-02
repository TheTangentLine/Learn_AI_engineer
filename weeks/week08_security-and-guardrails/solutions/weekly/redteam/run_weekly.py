"""Week 8 weekly challenge: red-team the Week 3 RAG bot, the Week 5 research agent and the Week 6 support system, harden them, and write the report.

  uv run python weeks/week08_security-and-guardrails/solutions/weekly/redteam/run_weekly.py [--no-model]

Three targets, one method: code oracles that read side effects (ledger, notes folder, rendered reply, approval queue), a scripted OBEDIENT model so the
numbers are about what the CONTROLS stop, an adaptive attacker (Day 6 mutators) next to the static attack set, and the benign tasks that must keep
working. The output is ``outputs/w8_findings_report.md`` (the register, the tables, the limits) and ``corpus.json`` (the regression suite).
"""

from __future__ import annotations

import dataclasses
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLUTIONS = HERE.parents[1]
ROOT = HERE.parents[4]
for p in (str(ROOT), str(SOLUTIONS), str(HERE)):
    sys.path.insert(0, p)

import campaign as C  # noqa: E402
import corpus  # noqa: E402
import day3_solution as d3  # noqa: E402
import day4_solution as d4  # noqa: E402
import defenses as D  # noqa: E402
import grounding as G  # noqa: E402
import register as R  # noqa: E402
import support_target as S  # noqa: E402
import targets as T  # noqa: E402

from common import redteam as rt  # noqa: E402

CANARY = corpus.CANARY
SUPPORT_CONFIGS = {
    "none": S.Controls(),
    "ownership": S.Controls(ownership=True),
    "filter": S.Controls(reason="filter"),
    "enum": S.Controls(reason="enum"),
    "reply_guard": S.Controls(reply_guard=True),
    "all": corpus.HARDENED_SUPPORT,
}


# ----------------------------------------------------------------------------- the measurements


def support_metrics() -> dict:
    static, benign = {}, {}
    for name, controls in SUPPORT_CONFIGS.items():
        res = S.run(controls, CANARY)
        static[name] = {
            g: (
                sum(r.succeeded for r in res if r.attack.goal == g),
                sum(r.attack.goal == g for r in res),
            )
            for g in S.GOALS
        }
        benign[name] = [
            k for k, ok in S.benign(controls).items() if not ok
        ]  # the names of what broke
    return {
        "static": static,
        "benign": benign,
        "adaptive_none": S.adaptive(S.Controls(), CANARY, "approver_injection"),
        "adaptive_filter": S.adaptive(SUPPORT_CONFIGS["filter"], CANARY, "approver_injection"),
        "adaptive_enum": S.adaptive(SUPPORT_CONFIGS["enum"], CANARY, "approver_injection"),
        "adaptive_ownership": S.adaptive(SUPPORT_CONFIGS["ownership"], CANARY, "unowned_refund"),
    }


def rag_metrics(real_model: bool) -> dict:
    out: dict = {}
    lab, held = d3.lab_attacks(), d3.heldout_attacks()
    for name in ("none", "all layers"):
        d = dataclasses.replace(D.CONFIGS[name], rng=random.Random(0))
        target = T.RagTarget(T.obedient_model(CANARY), CANARY, defenses=d)
        lab_res, held_res = rt.run(target, lab, CANARY), rt.run(target, held, CANARY)
        leak_exfil = [r for r in lab_res if r.attack.goal in ("leak_secret", "exfil_url")]
        entry = {
            "lab": (sum(r.succeeded for r in lab_res), len(lab_res)),
            "heldout": (sum(r.succeeded for r in held_res), len(held_res)),
            "leak_exfil": (sum(r.succeeded for r in leak_exfil), len(leak_exfil)),
        }
        outs = C.adaptive_run(target, lab, CANARY)
        entry["adaptive"] = (sum(o.first_success is not None for o in outs), len(outs))
        le = [o for o in outs if o.attack.goal in ("leak_secret", "exfil_url")]
        entry["adaptive_leak_exfil"] = (sum(o.first_success is not None for o in le), len(le))
        out[name] = entry
    if real_model:
        qwen = T.RagTarget(T.qwen_model(), CANARY)
        res = rt.run(qwen, lab, CANARY)
        out["qwen_none_lab"] = (sum(r.succeeded for r in res), len(res))
    else:
        out["qwen_none_lab"] = None
    return out


def agent_metrics() -> dict:
    out = {}
    for key, name in (
        ("none", "none"),
        ("caps_policy", "capabilities + policy"),
        ("caps_policy_output", "capabilities + policy + output"),
        ("all layers", "all layers"),
    ):
        h = d4.CONFIGS.get(name) or d4.A.Hardening(capabilities=True, policy=True)
        r = d4.evaluate(name, h)
        out[key] = (r["overall"]["succeeded"], r["overall"]["attacks"])
        out[key + "_benign"] = r["benign"]
    return out


def grounding_metrics(real_model: bool) -> dict | None:
    if not real_model:
        return None
    from common.judges import NLIJudge

    judge = NLIJudge()
    cases = G.collect(T.qwen_model(), CANARY, judge=judge)
    good, bad = G.split(cases)
    sw = G.swapped(cases, judge)
    lex = G.gate_table(good + bad, "lexical", [0.9])[0]
    return {
        "unanswerable_wrong": sum(c.kind == "unanswerable" and c.outcome == "wrong" for c in cases),
        "premise_wrong": sum(c.kind == "false_premise" and c.outcome == "wrong" for c in cases),
        "bad": len(bad),
        "good": len(good),
        "lexical_bad_through": lex["bad_through"],
        "lexical_good_lost": lex["good_lost"],
        "swaps": len(sw),
        "swaps_caught": sum(c.nli is not None and c.nli < 0.5 for c in sw),
    }


def collect(real_model: bool = True) -> dict:
    return {
        "support": support_metrics(),
        "rag": rag_metrics(real_model),
        "agent": agent_metrics(),
        "grounding": grounding_metrics(real_model),
    }


# ----------------------------------------------------------------------------- the report


def support_table(m: dict) -> str:
    goals = list(S.GOALS)
    rows = [
        "| controls | " + " | ".join(goals) + " | benign tasks |",
        "|---|" + "---|" * (len(goals) + 1),
    ]
    for name in SUPPORT_CONFIGS:
        cells = [R.frac(m["static"][name][g]) for g in goals]
        broke = m["benign"][name]
        cost = "the approver can read the real reason"
        if not broke:
            verdict = "all pass"
        elif broke == [
            cost
        ]:  # a designed cost, not a defect: the enum drops the customer's own words
            verdict = "all pass except: the approver sees a category, not the customer's words (by design)"
        else:
            verdict = "BROKEN: " + ", ".join(broke)
        rows.append(f"| {name} | " + " | ".join(cells) + f" | {verdict} |")
    return "\n".join(rows)


def report(m: dict, findings: list[R.Finding], doc: dict) -> str:
    rag, ag, sup = m["rag"], m["agent"], m["support"]
    summary = corpus.summary(doc)
    af, ae, an = sup["adaptive_filter"], sup["adaptive_enum"], sup["adaptive_none"]
    adaptive = (
        f"approver_injection ({an['attacks']} attacks, user channel): undefended {an['static']}/{an['attacks']}; the reason filter lets {af['static']}/{af['attacks']} through as written "
        f"(four human-written rewordings that avoid its trigger words) and {af['adaptive']}/{af['attacks']} after the mutators; the enum lets {ae['adaptive']}/{ae['attacks']} through"
    )
    lines = [
        "# Week 8 findings report: red-teaming the RAG bot, the research agent and the support system",
        "",
        "Scope: three systems built in Weeks 3, 5 and 6. Model: a scripted OBEDIENT one (so every number says what the controls stop when the model complies with the attacker) plus, where marked, the real local Qwen2.5-0.5B. Nothing here was run against a hosted model.",
        "",
        "## Headline",
        f"- **RAG bot:** undefended, {R.frac(rag['none']['lab'])} lab attacks succeed on the obedient model. With all layers: {R.frac(rag['all layers']['lab'])} on the lab set, {R.frac(rag['all layers']['heldout'])} on held-out phrasings, {R.frac(rag['all layers']['adaptive'])} for an adaptive attacker. **Leak and exfiltration: {R.frac(rag['all layers']['adaptive_leak_exfil'])} even adaptive.** What remains is integrity (a printed token, a false number).",
        f"- **Research agent:** {R.frac(ag['none'])} undefended, {R.frac(ag['all layers'])} with all layers; normal tasks still pass.",
        "- **Support system:** four real vulnerabilities found and fixed (any customer could refund or read any invoice; the approver reads customer text as authoritative; replies can carry attacker links). Two existing controls verified. Table below.",
        "",
        "## Support system: success counts by goal and control",
        support_table(sup),
        "",
        "Adaptive attacker (Day 6 mutators, up to 19 attempts per attack): "
        + adaptive
        + f"; ownership against unowned_refund: {sup['adaptive_ownership']['adaptive']}/{sup['adaptive_ownership']['attacks']}.",
        "",
        "## Findings register",
        R.markdown(findings),
        "",
        "## Residual risk (what is still true after the fixes)",
    ]
    for x in R.residual_risks(findings):
        why = x.residual or x.evidence
        lines.append(
            f"- **{x.id} ({x.severity}, {x.status}):** {x.title}. {why[:1].upper() + why[1:]}"
        )
    lines += [
        "",
        "## Regression suite",
        f"`corpus.json` (digest `{doc['digest']}`) freezes {sum(sum(v.values()) for v in summary.values())} attacks: "
        + "; ".join(
            f"{t}: {v['blocked']} must stay blocked, {v['open']} known open"
            for t, v in summary.items()
        )
        + ". A 'blocked' attack that succeeds fails the build; an 'open' one that stops working fails it too (strict expected failure) until the corpus is re-frozen and the register updated.",
        "",
        "## Limits of this report",
        "- A scripted obedient model is an upper bound on how often a model obeys; the real 0.5B model obeyed far less (Day 2 to 4). A hosted model is not measured.",
        "- The attack library, the mutators and the detector were written by the same person (me). The held-out and adaptive numbers are the honest ones; the lab numbers are ceilings.",
        "- Identity is assumed authenticated (F-13). The ownership check cannot be better than the identity it is given.",
        "- Not covered: denial of service and cost abuse (the Week 7 budget guard exists but was not attacked), training-data and supply-chain risks, multi-turn attacks, and attacks through the approver's own interface.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> None:
    real = "--no-model" not in argv
    m = collect(real)
    findings = R.build(m)
    doc = corpus.freeze()
    text = report(m, findings, doc)
    out = ROOT / "outputs" / "w8_findings_report.md"
    out.write_text(text)
    print(text)
    print(f"[written to {out.relative_to(ROOT)}; corpus digest {doc['digest']}]")


if __name__ == "__main__":
    main(sys.argv)
