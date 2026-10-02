"""Week 12 Day 3 - Solution: the core flow, measured against the golden set.

1. ABLATIONS  on the DEV split only: how many sentences the extractive answerer quotes, whether it adds the sentence that follows, whether a cross-encoder picks sentences
2. SYSTEMS    three complete systems, each run ONCE on the TEST split after the design was fixed on dev:
                 extractive             no model: quote the best sentences with citations
                 llm + fallback         Qwen2.5-0.5B-Instruct answers, the answer is verified in code (valid citations, each sentence supported by the source it cites),
                                        and the extractive answer is used when verification fails
                 llm only               the same model with no fallback
3. FAILURES   why the passing rule failed, by cause

  uv run python weeks/week12_capstone/solutions/day3_solution.py
"""

from __future__ import annotations

import collections
import sys
import time

from lab import ROOT, Lab

sys.path.insert(0, str(ROOT / "weeks/week11_inference-serving-deployment/solutions"))

import llamacpp as LC  # noqa: E402
from copilot import answer as A  # noqa: E402
from copilot import evaluate as E  # noqa: E402
from copilot import golden as G  # noqa: E402
from copilot.llm import OpenAIChat  # noqa: E402


def line(name: str, rep: dict) -> str:
    return (
        f"   {name:<30}{E.fmt(rep['overall']):>18}{E.fmt(rep['single']):>18}{E.fmt(rep['multi']):>18}"
        f"{E.fmt(rep['out_of_scope']):>18}{E.fmt(rep['adversarial']):>18}"
    )


HEADER = f"   {'system':<30}{'overall':>18}{'single':>18}{'multi':>18}{'out of scope':>18}{'adversarial':>18}"


def failures(runs) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for r in runs:
        s = r.score
        if s["passed"]:
            continue
        if r.item.kind == "out_of_scope":
            c["answered an out-of-scope question"] += 1
        elif r.item.kind == "adversarial":
            c["leaked or crashed on an attack"] += 1
        elif s["wrong_abstention"]:
            c["refused an answerable question (gate or empty answer)"] += 1
        elif not s["retrieved_any"]:
            c["the right lesson was not retrieved"] += 1
        elif not s["facts_ok"] and not s["attributed"]:
            c["wrong passage quoted and wrong lesson cited"] += 1
        elif not s["facts_ok"]:
            c["right lesson, but the answer misses the facts"] += 1
        else:
            c["facts present but the right lesson is not cited"] += 1
    return c


def main() -> None:
    lab = Lab()
    items = lab.items
    print("1. ABLATIONS on DEV (extractive answerer, rerank gate, guards on)")
    print(HEADER)
    ablations = {
        "3 sentences + 1 following (first design)": dict(max_units=3, neighbors=1),
        "3 sentences, no neighbours": dict(max_units=3, neighbors=0),
        "4 sentences + 1 following (chosen)": dict(max_units=4, neighbors=1),
        "4 sentences + cross-encoder picker": dict(
            max_units=4, neighbors=1, unit_reranker=lab.reranker
        ),
    }
    for name, kw in ablations.items():
        cp = lab.copilot(answerer=A.ExtractiveAnswerer(lab.idf, **kw))
        print(line(name, E.report(E.run_golden(cp, items, split="dev"))))

    print("\n2. SYSTEMS on TEST (each run once; the design above was fixed on dev)")
    print(HEADER)
    gguf = LC.ensure_chat_gguf()
    out: dict[str, list] = {}
    extractive = A.ExtractiveAnswerer(lab.idf)
    with LC.LlamaServer(gguf, parallel=1, ctx=8192) as srv:
        chat = OpenAIChat(srv.url, max_tokens=160)
        systems = {
            "extractive": extractive,
            "llm + verification + fallback": A.LlmAnswerer(chat, extractive),
            "llm only (no fallback)": A.LlmAnswerer(chat, None),
        }
        for name, ans in systems.items():
            cp = lab.copilot(answerer=ans)
            t0 = time.perf_counter()
            runs = E.run_golden(cp, items, split="test")
            out[name] = runs
            rep = E.report(runs)
            print(
                line(name, rep)
                + f"   ({time.perf_counter() - t0:.0f} s, {rep['tokens']['prompt']:,} prompt + {rep['tokens']['completion']:,} completion tokens)"
            )
        modes = collections.Counter(
            r.result.mode for r in out["llm + verification + fallback"] if r.item.answerable
        )
        print(
            f"   in the verified system the model's own answer was used for {modes.get('llm', 0)} of {sum(modes.values())} answerable questions; the fallback for {modes.get('llm+fallback', 0)}"
        )

    print("\n3. WHY THE EXTRACTIVE SYSTEM FAILED (test split)")
    for k, v in failures(out["extractive"]).most_common():
        print(f"   {v:>3}  {k}")
    for r in out["extractive"]:
        if not r.score["passed"] and r.item.answerable:
            print(
                f"   - {r.item.id}: {r.item.question} -> {r.result.answer[:140].replace(chr(10), ' ')}"
            )
    _ = G


if __name__ == "__main__":
    main()
