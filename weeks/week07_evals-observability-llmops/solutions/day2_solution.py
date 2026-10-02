"""Week 7 Day 2 - Solution: validating LLM judges (rubrics, pairwise comparison, calibration).

1. The labelled set (judge_data.py): 80 replies, labels true BY CONSTRUCTION (defect injection), eight situations.
2. A GROUP split by situation: dev situations are for tuning the rubric, test situations are touched once. (Splitting by
   reply would put near-duplicates of a situation on both sides: leakage.)
3. Pointwise judges: a rule-based judge (code) and an LLM judge (two rubric variants, chosen on DEV), calibrated
   per criterion on TEST: accuracy with a CI, kappa, TPR/TNR, leniency, verbosity bias.
4. Pairwise: (good, defective) pairs with a known winner, asked in BOTH orders: flips, position bias, accuracy.

uv run python weeks/week07_evals-observability-llmops/solutions/day2_solution.py run llm       # local Qwen, cached
uv run python weeks/week07_evals-observability-llmops/solutions/day2_solution.py report
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import judge_data as jd  # noqa: E402

from common import llm_judge as lj  # noqa: E402

OUT = ROOT / "outputs"


def group_split(items, dev_fraction: float = 0.6, seed: int = 0):
    """Split BY SITUATION so no situation appears on both sides (replies within one share facts and phrasing)."""
    groups = sorted({i.situation for i in items})
    random.Random(seed).shuffle(groups)
    cut = round(len(groups) * dev_fraction)
    dev_groups = set(groups[:cut])
    return [i for i in items if i.situation in dev_groups], [
        i for i in items if i.situation not in dev_groups
    ]


# ----------------------------------------------------------------------------- pointwise


def llm_judge_fn(rubric: lj.Rubric, provider: str | None, model: str | None = None):
    def judge(item) -> dict[str, bool]:
        return lj.judge_pointwise(
            rubric,
            item.reply,
            context=item.context,
            question=item.question,
            provider=provider,
            model=model,
        )

    return judge


def score(judge_fn, items) -> tuple[dict[str, list[bool]], dict[str, list[bool]]]:
    truth = {n: [] for n in jd.NAMES}
    pred = {n: [] for n in jd.NAMES}
    for item in items:
        verdict = judge_fn(item)
        for n in jd.NAMES:
            truth[n].append(item.labels[n])
            pred[n].append(bool(verdict[n]))
    return truth, pred


def reports(judge_fn, items) -> dict[str, lj.JudgeReport]:
    truth, pred = score(judge_fn, items)
    return lj.calibrate_rubric(truth, pred)


def mean_kappa(rs: dict[str, lj.JudgeReport]) -> float:
    return sum(r.kappa for r in rs.values()) / len(rs)


def bias_summary(judge_fn, items) -> dict[str, float]:
    truth, pred = score(judge_fn, items)
    lengths = [i.length for i in items]
    return {n: lj.verbosity_bias(lengths, truth[n], pred[n]) for n in jd.NAMES}


def format_reports(rs: dict[str, lj.JudgeReport]) -> str:
    return (
        "\n".join(f"  {n:18s} {r}" for n, r in rs.items()) + f"\n  mean kappa {mean_kappa(rs):.2f}"
    )


# ----------------------------------------------------------------------------- pairwise


def build_pairs(items, seed: int = 0):
    """(good, defective, winner_label) with the position randomised, plus (decoy vs good) pairs that should be TIES."""
    rnd = random.Random(seed)
    by_sit: dict[int, list] = {}
    for i in items:
        by_sit.setdefault(i.situation, []).append(i)
    pairs = []
    for sit, group in by_sit.items():
        good = next(i for i in group if i.kind == "good")
        decoy = next(i for i in group if i.kind == "decoy")
        for bad in (i for i in group if i.kind.startswith("defect")):
            a_is_good = rnd.random() < 0.5
            a, b = (good, bad) if a_is_good else (bad, good)
            pairs.append(
                {
                    "situation": sit,
                    "kind": bad.kind,
                    "a": a.reply,
                    "b": b.reply,
                    "question": good.question,
                    "truth": "A" if a_is_good else "B",
                }
            )
        a, b = (decoy, good) if rnd.random() < 0.5 else (good, decoy)
        pairs.append(
            {
                "situation": sit,
                "kind": "decoy-vs-good",
                "a": a.reply,
                "b": b.reply,
                "question": good.question,
                "truth": "tie",
                "longer": "A" if a is decoy else "B",
            }
        )
    return pairs


def run_pairwise(pairs, provider, model=None):
    return [
        lj.judge_pairwise(p["question"], p["a"], p["b"], provider=provider, model=model)
        for p in pairs
    ]


def pairwise_report(pairs, results) -> dict:
    defect = [(p, r) for p, r in zip(pairs, results, strict=True) if p["truth"] != "tie"]
    decoy = [(p, r) for p, r in zip(pairs, results, strict=True) if p["truth"] == "tie"]
    correct = sum(r.winner == p["truth"] for p, r in defect)
    wrong = sum(r.winner not in (p["truth"], "tie") for p, r in defect)
    longer_wins = sum(r.winner == p["longer"] for p, r in decoy)
    return {
        "n_defect_pairs": len(defect),
        "correct": correct,
        "tie": sum(r.winner == "tie" for p, r in defect),
        "wrong": wrong,
        "accuracy_when_decided": correct / (correct + wrong) if correct + wrong else float("nan"),
        "bias": lj.position_bias(results),
        "n_decoy_pairs": len(decoy),
        "longer_wins": longer_wins,
        "decoy_ties": sum(r.winner == "tie" for p, r in decoy),
    }


# ----------------------------------------------------------------------------- run / report (the slow, cached part)

RUBRICS = {"plain": lj.Rubric(jd.CRITERIA), "reasoning": lj.Rubric(jd.CRITERIA, reasoning=True)}


def run_llm(provider: str, model: str | None) -> dict:
    items = jd.build_items()
    dev, test = group_split(items)
    out: dict = {
        "split": {"dev": [i.id for i in dev], "test": [i.id for i in test]},
        "pointwise": {},
    }
    for name, rubric in RUBRICS.items():
        judge = llm_judge_fn(rubric, provider, model)
        verdicts = {
            i.id: judge(i) for i in items
        }  # every item once; dev and test are views of this
        out["pointwise"][name] = verdicts
    pairs = build_pairs(items)
    results = run_pairwise(pairs, provider, model)
    out["pairs"] = pairs
    out["pairwise"] = [asdict(r) for r in results]
    return out


def evaluate_saved(data: dict):
    items = {i.id: i for i in jd.build_items()}
    dev = [items[i] for i in data["split"]["dev"]]
    test = [items[i] for i in data["split"]["test"]]
    lines = []

    def saved_judge(name):
        return lambda item: data["pointwise"][name][item.id]

    lines.append("== pointwise: rubric variant chosen on DEV")
    dev_scores = {}
    for name in data["pointwise"]:
        rs = reports(saved_judge(name), dev)
        dev_scores[name] = mean_kappa(rs)
        lines.append(f"{name}: dev mean kappa {dev_scores[name]:.2f}")
    best = max(dev_scores, key=dev_scores.get)
    lines.append(f"chosen on dev: {best}")
    lines.append("== TEST (touched once), LLM judge:")
    lines.append(format_reports(reports(saved_judge(best), test)))
    lines.append("== TEST, rule-based judge (code):")
    lines.append(format_reports(reports(jd.rule_judge, test)))
    lines.append(
        "== verbosity bias of the LLM judge on TEST (within-class length/pass correlation): "
        + json.dumps({k: round(v, 2) for k, v in bias_summary(saved_judge(best), test).items()})
    )
    from common.llm_judge import PairwiseResult

    results = [PairwiseResult(**r) for r in data["pairwise"]]
    rep = pairwise_report(data["pairs"], results)
    lines.append(
        "== pairwise (both orders): "
        + json.dumps(
            {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rep.items() if k != "bias"}
        )
    )
    lines.append(
        "   position bias: " + json.dumps({k: round(v, 2) for k, v in rep["bias"].items()})
    )
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    path = OUT / "w7d2_llm.json"
    if argv[:2] == ["run", "llm"]:
        from common import llm
        from common.local_server import LocalOpenAIServer

        with LocalOpenAIServer() as srv:
            os.environ["OLLAMA_BASE_URL"] = srv.url
            llm._ollama.cache_clear()
            data = run_llm("ollama", "local-qwen")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(data))
        print(f"saved {path}")
    elif argv[:1] == ["report"]:
        print(evaluate_saved(json.loads(path.read_text())))
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
