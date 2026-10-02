"""LLM-as-judge, with the parts that make it trustworthy: binary rubric criteria, position-debiased pairwise comparison,
and CALIBRATION against labels (agreement, error rates, bias checks).

    from common.llm_judge import Criterion, Rubric, judge_pointwise, judge_pairwise, calibrate

    rubric = Rubric([Criterion("grounded", "Every number and id in the reply appears in the CONTEXT."),
                     Criterion("no_promise", "The reply does not guarantee an outcome.")])
    verdict = judge_pointwise(rubric, reply, context="Refund issued: RF-0007 ($20.00).", provider="anthropic")
    verdict["grounded"]            # True = the criterion is satisfied

    winner = judge_pairwise("Which reply is more helpful?", a, b)          # "A" | "B" | "tie", each order checked
    report = calibrate(human_labels, judge_labels)                          # kappa, TPR/TNR, accuracy with a CI

Design rules (each is tested):
  * BINARY criteria ("is X true of the reply?"), not 1-10 scales: scales invite drift, ties and false precision.
  * One criterion = one question; the judge answers all of them in one JSON object, so a defect is attributable.
  * A pairwise verdict that CHANGES when you swap the order is not a verdict: it is reported as a tie plus a flip.
  * A judge is a model like any other: it has an error rate, so MEASURE it on labelled data before using its scores,
    on a split it was not tuned on, and report agreement beyond chance (kappa), not just accuracy.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import create_model

from . import llm
from .evalkit import bootstrap_ci, cohens_kappa

# ----------------------------------------------------------------------------- pointwise rubric judging


@dataclass(frozen=True)
class Criterion:
    name: str  # a short identifier (valid as a JSON key): "grounded"
    question: str  # phrased so that TRUE means the response is GOOD on this criterion


@dataclass
class Rubric:
    criteria: list[Criterion]
    instructions: str = (
        "You are a strict evaluator of customer-support replies. Judge only what is written."
    )
    reasoning: bool = (
        False  # ask for a short "reasoning" string BEFORE the verdicts (often helps; costs tokens)
    )

    def names(self) -> list[str]:
        return [c.name for c in self.criteria]

    def system(self) -> str:
        lines = "\n".join(f"- {c.name}: {c.question}" for c in self.criteria)
        first = (
            'First write a one-sentence "reasoning", then the verdicts. ' if self.reasoning else ""
        )
        keys = (
            'exactly the criterion names, preceded by "reasoning"'
            if self.reasoning
            else "exactly the criterion names"
        )
        return (
            f"{self.instructions}\nFor EACH criterion below answer true if it is satisfied, false if it is not. {first}"
            f"Answer with one JSON object whose keys are {keys}.\nCriteria:\n{lines}"
        )

    def prompt(self, response: str, context: str = "", question: str = "") -> str:
        parts = []
        if question:
            parts.append(f"<customer_message>\n{question}\n</customer_message>")
        if context:
            parts.append(f"<context>\n{context}\n</context>")
        parts.append(f"<reply>\n{response}\n</reply>")
        return "\n".join(parts)

    def schema(self):
        fields: dict[str, Any] = {"reasoning": (str, ...)} if self.reasoning else {}
        fields.update({c.name: (bool, ...) for c in self.criteria})
        return create_model("Verdict", **fields)


def judge_pointwise(
    rubric: Rubric,
    response: str,
    *,
    context: str = "",
    question: str = "",
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 200,
) -> dict[str, bool]:
    verdict, _ = llm.structured(
        rubric.prompt(response, context, question),
        rubric.schema(),
        system=rubric.system(),
        provider=provider,
        model=model,
        max_tokens=max_tokens,
    )
    return {name: bool(getattr(verdict, name)) for name in rubric.names()}


# ----------------------------------------------------------------------------- pairwise, position-debiased


@dataclass
class PairwiseResult:
    winner: str  # "A" | "B" | "tie"
    flipped: bool  # the two orders disagreed
    first_order: (
        str  # the verdict when (a, b) was shown in that order, in terms of the ORIGINAL a/b
    )
    second_order: str


def _ask_pair(question: str, first: str, second: str, provider, model, criterion: str) -> str:
    """The raw verdict ("first" | "second" | "tie") for the responses in the order shown."""
    from pydantic import BaseModel

    class Pick(BaseModel):
        winner: str  # "first" | "second" | "tie"

    system = (
        f"You compare two replies to the same customer message. Decide which one is better on this criterion: {criterion} "
        'Answer with a JSON object {"winner": "first"}, {"winner": "second"} or {"winner": "tie"}. Do not prefer a reply for being longer.'
    )
    prompt = f"<customer_message>\n{question}\n</customer_message>\n<first>\n{first}\n</first>\n<second>\n{second}\n</second>"
    pick, _ = llm.structured(
        prompt, Pick, system=system, provider=provider, model=model, max_tokens=60
    )
    w = pick.winner.strip().lower()
    return w if w in ("first", "second", "tie") else "tie"


def judge_pairwise(
    question: str,
    a: str,
    b: str,
    *,
    criterion: str = "which reply is more helpful and correct",
    provider: str | None = None,
    model: str | None = None,
) -> PairwiseResult:
    """Ask in BOTH orders. A winner needs to win in both; a verdict that follows the position is a tie with ``flipped``."""
    r1 = _ask_pair(question, a, b, provider, model, criterion)  # a is first
    r2 = _ask_pair(question, b, a, provider, model, criterion)  # b is first
    v1 = {"first": "A", "second": "B", "tie": "tie"}[r1]
    v2 = {"first": "B", "second": "A", "tie": "tie"}[r2]  # translate back to the original labels
    if v1 == v2:
        return PairwiseResult(v1, False, v1, v2)
    return PairwiseResult("tie", v1 != "tie" and v2 != "tie", v1, v2)


def position_bias(results: Sequence[PairwiseResult]) -> dict[str, float]:
    """How often the verdict followed the POSITION rather than the content, over a set of pairs."""
    n = len(results)
    if not n:
        return {"n": 0, "flip_rate": 0.0, "first_bias": 0.0}
    flipped = [r for r in results if r.flipped]
    # when the orders disagree and each order picked its FIRST slot, the judge is position-biased towards "first"
    first_slot = sum(1 for r in flipped if r.first_order == "A" and r.second_order == "B")
    return {
        "n": n,
        "flip_rate": len(flipped) / n,
        "first_bias": first_slot / len(flipped) if flipped else 0.0,
    }


# ----------------------------------------------------------------------------- calibration against labels


@dataclass
class JudgeReport:
    n: int
    accuracy: float
    accuracy_ci: tuple[float, float]
    kappa: float
    tpr: float  # of the truly GOOD items, the share the judge passed
    tnr: float  # of the truly BAD items, the share the judge failed (the number that matters for catching defects)
    precision: float
    base_rate: float  # share of truly good items
    judge_pass_rate: float
    confusion: dict[str, int] = field(default_factory=dict)

    @property
    def leniency(self) -> float:
        """Judge pass rate minus the true pass rate: positive = too generous."""
        return self.judge_pass_rate - self.base_rate

    def __str__(self) -> str:
        return (
            f"n={self.n} accuracy {self.accuracy:.0%} [{self.accuracy_ci[0]:.0%}-{self.accuracy_ci[1]:.0%}] kappa {self.kappa:.2f} "
            f"TPR {self.tpr:.0%} TNR {self.tnr:.0%} (judge passes {self.judge_pass_rate:.0%}, truth {self.base_rate:.0%})"
        )


def calibrate(truth: Sequence[bool], predicted: Sequence[bool]) -> JudgeReport:
    """Agreement of a judge with labels. ``True`` = good / criterion satisfied."""
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have the same length")
    n = len(truth)
    if n == 0:
        raise ValueError("no items")
    c = Counter(zip(map(bool, truth), map(bool, predicted), strict=True))
    tp, fn, fp, tn = c[(True, True)], c[(True, False)], c[(False, True)], c[(False, False)]
    correct = [1.0 if t == p else 0.0 for t, p in zip(truth, predicted, strict=True)]
    acc, lo, hi = bootstrap_ci(correct)
    kappa = (
        cohens_kappa([bool(t) for t in truth], [bool(p) for p in predicted])
        if len(set(truth)) > 1 or len(set(predicted)) > 1
        else 0.0
    )
    return JudgeReport(
        n, acc, (lo, hi), kappa,
        tp / (tp + fn) if tp + fn else 0.0, tn / (tn + fp) if tn + fp else 0.0, tp / (tp + fp) if tp + fp else 0.0,
        (tp + fn) / n, (tp + fp) / n, {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    )  # fmt: skip


def calibrate_rubric(
    truth: dict[str, Sequence[bool]], predicted: dict[str, Sequence[bool]]
) -> dict[str, JudgeReport]:
    """One report per criterion (an overall 'accuracy' hides which criterion the judge cannot see)."""
    return {name: calibrate(truth[name], predicted[name]) for name in truth}


def verbosity_bias(
    lengths: Sequence[int], truth: Sequence[bool], predicted: Sequence[bool]
) -> float:
    """Does the judge like long answers FOR NO REASON? The correlation between length and the judge's pass decision,
    computed within each true class (so a real quality-length relationship does not count), averaged.
    Near 0 = no length preference; positive = longer answers get passed more often than their true quality warrants."""
    corrs = []
    for cls in (True, False):
        idx = [i for i, t in enumerate(truth) if bool(t) == cls]
        if len(idx) < 3:
            continue
        xs, ys = [float(lengths[i]) for i in idx], [1.0 if predicted[i] else 0.0 for i in idx]
        corrs.append(_pearson(xs, ys))
    return sum(corrs) / len(corrs) if corrs else 0.0


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return (
        0.0
        if sx == 0 or sy == 0
        else sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / (sx * sy)
    )


# ----------------------------------------------------------------------------- a judge you can trust enough to use


def split_labelled(
    items: Sequence[Any],
    dev_fraction: float = 0.6,
    seed: int = 0,
    key: Callable[[Any], Any] = lambda x: x,
) -> tuple[list, list]:
    """Deterministic dev/test split of labelled items. Tune the rubric on DEV; report agreement on TEST, once."""
    import random

    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    cut = round(len(items) * dev_fraction)
    dev = [items[i] for i in sorted(order[:cut])]
    test = [items[i] for i in sorted(order[cut:])]
    return dev, test
