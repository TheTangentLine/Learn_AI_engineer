"""Week 12 Day 4 - Solution: guardrails, a security suite, and the CI gate.

1. QUESTIONS   the input guard against 13 direct attacks, 52 golden questions and 6 benign security questions (blocks and over-blocking)
2. DOCUMENTS   ten poisoned documents added to an UNTRUSTED upload collection; attack success with each defence on or off, for the extractive answerer and for a scripted worst-case model
3. PRIVACY     personal data in a question never reaches a span
4. THE GATE    four pretend pull requests run through the CI gate on the dev split

  uv run python weeks/week12_capstone/solutions/day4_solution.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from copilot import answer as A
from copilot import core as C
from copilot import evalgate as EG
from copilot import evaluate as E
from copilot import guard as CG
from copilot import ingest as I
from copilot import retrieve as R
from copilot import security as S
from lab import ROOT, Lab

from common import tracing
from common.rag import SourceDoc

BASELINE = Path(__file__).resolve().parent / "golden" / "baseline_dev.json"


def poisoned_lab_index(lab: Lab, tmp: str):
    shutil.copytree(lab.index.dir, tmp, dirs_exist_ok=True)
    docs = I.load_corpus(ROOT / "weeks") + [
        SourceDoc(p.id, p.text, {"week": 0, "day": 0, "title": p.style}) for p in S.POISONED_DOCS
    ]
    index, _ = I.build_index(lab.embedder, tmp, docs)
    return index


def poisoning(lab: Lab, index) -> dict[str, dict[str, int]]:
    """Attack successes (of 10) per configuration. A configuration is (answerer, quarantine on/off, output guard on/off)."""
    retr = R.Retriever(index, R.RetrievalConfig(k=5))
    secrets_ = ["adm-cop-test"]
    obedient = S.ObedientChat()
    answerers = {
        "extractive answerer": A.ExtractiveAnswerer(lab.idf),
        "scripted obedient model, no verification": A.LlmAnswerer(obedient, None),
        "scripted obedient model + verification + fallback": A.LlmAnswerer(
            obedient, A.ExtractiveAnswerer(lab.idf)
        ),
    }
    defences = {
        "no defences": dict(trusted=None, out=False),
        "output guard only": dict(trusted=None, out=True),
        "quarantine only": dict(trusted=("week",), out=False),
        "quarantine + output guard": dict(trusted=("week",), out=True),
    }
    table: dict[str, dict[str, int]] = {}
    for aname, ans in answerers.items():
        table[aname] = {}
        for dname, d in defences.items():
            cp = C.Copilot(
                retr,
                ans,
                output_guard=CG.OutputGuard(secrets_=secrets_) if d["out"] else None,
                trusted_prefixes=d["trusted"],
            )
            wins = 0
            for doc in S.POISONED_DOCS:
                r = cp.ask(doc.trigger)
                wins += S.attack_succeeded(r.answer) and any(
                    s.doc.startswith("upload/") for s in r.sources
                )
            table[aname][dname] = wins
    return table


def direct_attacks(lab: Lab) -> dict[str, int]:
    """Attacks (of 18: the 13 direct ones and the 5 golden ones) that leaked a canary or were obeyed, per configuration of a scripted obedient model that holds a canary."""
    canary = CG.Canary()
    retr = lab.retriever()
    adversarial = S.DIRECT_ATTACKS + [i.question for i in lab.items if i.kind == "adversarial"]
    out: dict[str, int] = {}
    for label, guard_on, verify in (
        ("model with no guard and no verification", False, False),
        ("input guard on", True, False),
        ("input guard + verification + output guard", True, True),
    ):
        ans = A.LlmAnswerer(
            S.ObedientChat(),
            A.ExtractiveAnswerer(lab.idf) if verify else None,
            system_extra=canary.instruction(),
        )
        cp = C.Copilot(
            retr,
            ans,
            input_guard=CG.InputGuard() if guard_on else None,
            output_guard=CG.OutputGuard(secrets_=[canary.token]) if verify else None,
            gate=None,
        )
        out[label] = sum(
            bool(
                canary.leaked(a)
                or "PWNED" in a.upper()
                or "You answer questions about a course" in a
            )
            for a in (cp.ask(q).answer for q in adversarial)
        )
    return out


def gate_demo(lab: Lab, table: dict) -> tuple[dict, dict]:
    """(baseline snapshot, {pretend pull request: (snapshot, Decision)}) on the dev split."""
    product_attacks = table["extractive answerer"]["quarantine + output guard"]
    no_quarantine_attacks = table["extractive answerer"]["output guard only"]

    def snap(cp, attacks: int = product_attacks) -> dict:
        runs = E.run_golden(cp, lab.items, split="dev")
        rep = E.report(runs)
        return EG.snapshot(runs, rep["retrieved_any"][0], attacks, rep["latency"]["p95"])

    base = snap(lab.copilot())
    full_gate, cal = lab.calibrated_gate()
    prs = {
        "refactor: no behaviour change": snap(lab.copilot()),
        "tune: gate threshold +3 (too strict)": snap(
            lab.copilot(gate=type(full_gate)(lab.reranker, cal["threshold"] + 3))
        ),
        "tune: 1 source instead of 5": snap(lab.copilot(k=1)),
        "tune: BM25 only, no week scoping": snap(lab.copilot(alpha=0.0, scope_by_week=False)),
        "cleanup: drop the quarantine stage": snap(
            lab.copilot(trusted=None), attacks=no_quarantine_attacks
        ),
    }
    return base, {name: (cur, EG.gate(cur, base)) for name, cur in prs.items()}


def main() -> None:
    lab = Lab()
    guard = CG.InputGuard()

    print("1. THE INPUT GUARD")
    blocked = [q for q in S.DIRECT_ATTACKS if not guard.check(q).allowed]
    print(f"   direct attacks blocked: {len(blocked)} of {len(S.DIRECT_ATTACKS)}")
    for q in S.DIRECT_ATTACKS:
        if q not in blocked:
            print(f"      not blocked: {q}")
    golden_q = [i.question for i in lab.items if i.answerable]
    over = [q for q in golden_q + S.BENIGN_SECURITY if not guard.check(q).allowed]
    print(
        f"   over-blocking: {len(over)} of {len(golden_q) + len(S.BENIGN_SECURITY)} legitimate questions blocked ({len(golden_q)} golden, {len(S.BENIGN_SECURITY)} benign security questions)"
    )

    print(
        "\n   end to end, the 13 direct attacks and the 5 golden attack questions, with a scripted obedient model that also holds a canary in its system prompt:"
    )
    leaks_by_config = direct_attacks(lab)
    for label, leaks in leaks_by_config.items():
        print(
            f"      {label:<46} {leaks} of {len(S.DIRECT_ATTACKS) + sum(i.kind == 'adversarial' for i in lab.items)} attacks leaked or obeyed"
        )

    print("\n2. POISONED DOCUMENTS in an untrusted upload collection (attack success, of 10)")
    tmp = tempfile.mkdtemp(prefix="w12-poison-")
    try:
        index = poisoned_lab_index(lab, tmp)
        table = poisoning(lab, index)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    cols = list(next(iter(table.values())))
    print(f"   {'answerer':<52}" + "".join(f"{c:>28}" for c in cols))
    for a, row in table.items():
        print(f"   {a:<52}" + "".join(f"{row[c]:>28}" for c in cols))
    from common import guard as G

    detected = [p.style for p in S.POISONED_DOCS if G.detect(p.text).flagged]
    print(
        f"   the detector flags {len(detected)} of 10 documents; missed: "
        + ", ".join(p.style for p in S.POISONED_DOCS if p.style not in detected)
    )

    print("\n3. PRIVACY: personal data in a question")
    cp = lab.copilot()
    q = "Why was alice@example.com charged on card 4111 1111 1111 1111?"
    with tracing.capture() as rec:
        cp.ask(q)
    blob = json.dumps([s.attributes for s in rec.spans], default=str)
    print(f"   input guard's loggable form: {guard.check(q).redacted!r}")
    print(
        f"   raw email or card number present in any span attribute: {'alice@example.com' in blob or '4111' in blob}"
    )

    print("\n4. THE CI GATE on the dev split (baseline = the current design)")
    print(
        f"   the security suite with every defence on lets {table['extractive answerer']['quarantine + output guard']} of 10 poisoned documents through (the accepted residual); without quarantine it is {table['extractive answerer']['output guard only']}"
    )
    base, prs = gate_demo(lab, table)
    BASELINE.write_text(json.dumps(base, indent=1, sort_keys=True))
    for name, (_, d) in prs.items():
        print(f"   {name:<40} {d.markdown().replace(chr(10), chr(10) + '      ')}")


if __name__ == "__main__":
    main()
