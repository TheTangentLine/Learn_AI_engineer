"""Week 4 Day 2 - Solution: validate three faithfulness judges against human labels.

Judges:  lexical support | NLI entailment (DeBERTa cross-encoder) | small LLM judge (local Qwen, Yes/No)
Data:    18 contexts x (1 faithful + 1 unfaithful answer) = 36 hand-labelled examples
Method:  tune a decision threshold on DEV pairs, report on TEST pairs (no leakage), with bootstrap
         intervals and Cohen's kappa; then break recall down by failure type.

  uv run python weeks/week04_advanced-rag-and-evaluation/solutions/day2_solution.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from faithfulness_data import items  # noqa: E402

from common.evalkit import bootstrap_ci, cohens_kappa, fmt_ci  # noqa: E402
from common.judges import NLIJudge, lexical_support  # noqa: E402
from common.local_llm import LocalChat  # noqa: E402

JUDGE_SYSTEM = "You check whether an answer is fully supported by a context. Reply with exactly one word: Yes or No."


def llm_judge(chat: LocalChat, context: str, answer: str) -> int:
    """1 = judged faithful. Defaults to 'Yes' on unparseable output (a common, dangerous LLM-judge default)."""
    out = (
        chat(
            JUDGE_SYSTEM,
            f"<context>\n{context}\n</context>\n<answer>\n{answer}\n</answer>\n\n"
            "Is every claim in the answer supported by the context?",
            max_new_tokens=4,
        )
        .strip()
        .lower()
    )
    return 0 if out.startswith("no") else 1


def best_threshold(scores: list[float], labels: list[int]) -> float:
    """Threshold on a 'higher = more faithful' score that maximises accuracy (ties: the larger cut)."""
    cands = sorted(set(scores))
    cuts = [(a + b) / 2 for a, b in zip(cands, cands[1:], strict=False)] + [cands[0] - 1e-9]
    best = max(
        cuts,
        key=lambda t: (
            sum((s >= t) == bool(label) for s, label in zip(scores, labels, strict=True)),
            t,
        ),
    )
    return best


def metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Quality as a HALLUCINATION DETECTOR: the positive class is 'unfaithful' (label 0)."""
    tp = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred, strict=True))
    fp = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred, strict=True))
    fn = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred, strict=True))
    acc = [float(t == p) for t, p in zip(y_true, y_pred, strict=True)]
    return {
        "acc": bootstrap_ci(acc),
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "kappa": cohens_kappa(y_true, y_pred) if len(set(y_true) | set(y_pred)) > 1 else 0.0,
    }


def main() -> None:
    data = items()
    dev = [d for d in data if d["split"] == "dev"]
    test = [d for d in data if d["split"] == "test"]
    print(
        f"{len(data)} labelled examples: dev {len(dev)} (tune), test {len(test)} (report); "
        f"{sum(d['label'] == 0 for d in data)} unfaithful\n"
    )

    nli = NLIJudge()
    chat = LocalChat()
    scores: dict[str, dict[int, float]] = {"lexical support": {}, "NLI (min claim entailment)": {}}
    llm_pred: dict[int, int] = {}
    for i, d in enumerate(data):
        scores["lexical support"][i] = lexical_support(d["context"], d["answer"])
        scores["NLI (min claim entailment)"][i] = nli.faithfulness(d["context"], d["answer"]).score
        llm_pred[i] = llm_judge(chat, d["context"], d["answer"])

    idx = {
        "dev": [i for i, d in enumerate(data) if d["split"] == "dev"],
        "test": [i for i, d in enumerate(data) if d["split"] == "test"],
    }
    preds: dict[str, dict[int, int]] = {}
    print(
        f"{'judge':<28}{'threshold':>10}{'dev acc':>9}   TEST: {'accuracy [95% CI]':>22}{'catches bad':>13}{'precision':>10}{'kappa':>7}"
    )
    for name, sc in scores.items():
        tau = best_threshold([sc[i] for i in idx["dev"]], [data[i]["label"] for i in idx["dev"]])
        preds[name] = {i: int(sc[i] >= tau) for i in sc}
        dev_acc = np.mean([preds[name][i] == data[i]["label"] for i in idx["dev"]])
        m = metrics([data[i]["label"] for i in idx["test"]], [preds[name][i] for i in idx["test"]])
        print(
            f"{name:<28}{tau:>10.2f}{dev_acc:>9.0%}   {'':>6}{fmt_ci(m['acc']):>22}{m['recall']:>13.0%}{m['precision']:>10.0%}{m['kappa']:>7.2f}"
        )
    preds["LLM judge (Qwen 0.5B Yes/No)"] = llm_pred
    m = metrics([data[i]["label"] for i in idx["test"]], [llm_pred[i] for i in idx["test"]])
    print(
        f"{'LLM judge (Qwen 0.5B Yes/No)':<28}{'(none)':>10}{'':>9}   {'':>6}{fmt_ci(m['acc']):>22}{m['recall']:>13.0%}{m['precision']:>10.0%}{m['kappa']:>7.2f}"
    )
    always = metrics([data[i]["label"] for i in idx["test"]], [1] * len(idx["test"]))
    print(
        f"{'baseline: always faithful':<28}{'':>10}{'':>9}   {'':>6}{fmt_ci(always['acc']):>22}{always['recall']:>13.0%}{always['precision']:>10.0%}{always['kappa']:>7.2f}"
    )

    print(
        "\nRecall of UNFAITHFUL answers by failure type (all 18 unfaithful examples; thresholds from dev):"
    )
    types = sorted({d["type"] for d in data if d["label"] == 0})
    print(f"{'judge':<30}" + "".join(f"{t:>15}" for t in types))
    for name, p in preds.items():
        row = defaultdict(list)
        for i, d in enumerate(data):
            if d["label"] == 0:
                row[d["type"]].append(p[i] == 0)
        print(f"{name:<30}" + "".join(f"{sum(row[t])}/{len(row[t])}".rjust(15) for t in types))

    print("\nDisagreements worth reading (NLI judge on the full set):")
    nli_p = preds["NLI (min claim entailment)"]
    for i, d in enumerate(data):
        if nli_p[i] != d["label"]:
            kind = "missed hallucination" if d["label"] == 0 else "false alarm"
            print(
                f"  [{kind}] ({d['type']}) score {scores['NLI (min claim entailment)'][i]:.2f}: {d['answer'][:90]!r}"
            )


if __name__ == "__main__":
    main()
