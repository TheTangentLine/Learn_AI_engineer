"""Faithfulness judges: does an ANSWER follow from its CONTEXT?

    from common.judges import NLIJudge, lexical_support
    judge = NLIJudge()                                  # local DeBERTa NLI cross-encoder (~70M params)
    r = judge.faithfulness(context, answer)             # claim-by-claim entailment
    r.score, r.contradicted, [(c.text, c.entailment) for c in r.claims]

Two cheap judges (no LLM calls):
* ``lexical_support``  - share of the answer's content words found in the context. Fast, but blind to
  swapped subjects, flipped negations and changed relations: a corrupted answer reuses the same words.
* ``NLIJudge``         - natural-language inference: for each claim, P(context entails claim) vs
  P(contradicts). Catches contradictions and swaps; slower; needs a model; length-limited (512 tokens),
  so long contexts are scored in sliding windows of sentences.

LLM-as-judge (Week 4 Day 2) is the third option: most flexible, most expensive, and the one that must be
*validated against human labels* before you trust it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MODEL = "cross-encoder/nli-deberta-v3-xsmall"  # labels: 0 contradiction, 1 entailment, 2 neutral

_STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "and",
    "in",
    "is",
    "it",
    "for",
    "on",
    "that",
    "this",
    "you",
    "are",
    "be",
    "or",
    "as",
    "with",
    "by",
    "at",
    "from",
    "not",
    "can",
    "your",
    "they",
    "which",
    "will",
    "so",
}


def split_claims(answer: str, min_words: int = 3) -> list[str]:
    """Sentences of an answer, minus citation markers like [1]; fragments under `min_words` are dropped."""
    text = re.sub(r"\s*\[\d+\]", "", answer)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -*\t") for p in parts if len(p.split()) >= min_words]


def lexical_support(context: str, answer: str) -> float:
    """Share of the answer's content words that occur in the context (1.0 = every word appears)."""
    words = [
        w
        for w in re.findall(r"[a-z0-9]+", re.sub(r"\[\d+\]", " ", answer.lower()))
        if w not in _STOP
    ]
    if not words:
        return 0.0
    ctx = set(re.findall(r"[a-z0-9]+", context.lower()))
    return sum(w in ctx for w in words) / len(words)


def windows(context: str, size: int = 3, step: int = 2) -> list[str]:
    """Overlapping windows of `size` sentences, so a long context fits the model's length limit."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", context) if s.strip()]
    if len(sents) <= size:
        return [context.strip()]
    return [" ".join(sents[i : i + size]) for i in range(0, len(sents) - size + step, step)]


@dataclass
class Claim:
    text: str
    entailment: float
    contradiction: float
    neutral: float


@dataclass
class Faithfulness:
    score: float  # min over claims of P(entailment); 1.0 = every claim supported
    contradicted: bool  # some claim is more likely contradicted than entailed
    claims: list[Claim] = field(default_factory=list)


def aggregate(claims: list[Claim]) -> Faithfulness:
    """An answer is only as faithful as its WEAKEST claim."""
    if not claims:
        return Faithfulness(0.0, False, [])
    return Faithfulness(
        min(c.entailment for c in claims),
        any(c.contradiction > c.entailment and c.contradiction > 0.5 for c in claims),
        claims,
    )


class NLIJudge:
    def __init__(self, model: str = MODEL):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model, dtype=torch.float32
        ).eval()
        labels = {v.lower(): k for k, v in self._model.config.id2label.items()}
        self._idx = (labels["contradiction"], labels["entailment"], labels["neutral"])
        self._cache: dict[tuple[str, str], tuple[float, float, float]] = {}

    def classify(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        """(contradiction, entailment, neutral) probabilities for premise -> hypothesis."""
        key = (premise, hypothesis)
        if key not in self._cache:
            batch = self._tok(
                premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
            )
            with self._torch.no_grad():
                p = self._model(**batch).logits.softmax(-1)[0]
            self._cache[key] = tuple(float(p[i]) for i in self._idx)  # type: ignore[assignment]
        return self._cache[key]

    def faithfulness(self, context: str, answer: str) -> Faithfulness:
        wins = windows(context)
        claims = []
        for text in split_claims(answer):
            probs = [self.classify(w, text) for w in wins]
            claims.append(
                Claim(
                    text,
                    max(p[1] for p in probs),
                    max(p[0] for p in probs),
                    max(p[2] for p in probs),
                )
            )
        return aggregate(claims)
