"""The product's core flow: question in, a cited, guarded, accounted-for answer out.

    guard the input -> (cache) -> retrieve -> quarantine untrusted sources -> gate on relevance -> answer -> guard the output -> verify citations

    copilot = Copilot(retriever, ExtractiveAnswerer(idf), gate=RerankGate(reranker, tau), input_guard=InputGuard(), output_guard=OutputGuard())
    r = copilot.ask("What is HNSW?")
    r.answer, r.sources, r.cited, r.abstained, r.timings

Every stage is optional and replaceable, every stage is timed, and every stage opens a span (``common.tracing``; a no-op unless a tracer is configured). The pipeline never raises on a model or
retriever failure: it returns a ``Result`` with ``error`` set and an honest refusal, because a product that crashes on a bad day is worse than one that says it could not answer.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

from common import tracing  # noqa: E402
from common.cache import ResponseCache  # noqa: E402

from .answer import IDK, Draft, cited_numbers  # noqa: E402
from .gate import GateDecision  # noqa: E402
from .guard import BLOCKED_INPUT, InputGuard, OutputGuard, scrub_sources  # noqa: E402
from .llm import ChatError  # noqa: E402
from .retrieve import Retrieval, Retriever, Source  # noqa: E402

TRACE_STAGES = ("input_guard", "retrieve", "quarantine", "gate", "answer", "output_guard")


@dataclass
class Result:
    question: str
    answer: str
    abstained: bool = False
    blocked: bool = False
    sources: list[Source] = field(
        default_factory=list
    )  # what the answer may cite and the UI shows; empty when the gate refused
    retrieved: list[Source] = field(
        default_factory=list
    )  # everything retrieval returned (after quarantine), kept for diagnosis
    cited: list[int] = field(
        default_factory=list
    )  # source numbers the answer cites (all valid: invalid ones are removed or the answer is refused)
    mode: str = ""
    gate: GateDecision | None = None
    flags: list[str] = field(default_factory=list)  # guard and verification events, in order
    quarantined: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""
    cached: bool = False
    request_id: str = ""
    trace_id: str = ""

    @property
    def cited_docs(self) -> list[str]:
        by_n = {s.n: s.doc for s in self.sources}
        return [by_n[n] for n in self.cited if n in by_n]

    @property
    def seconds(self) -> float:
        return sum(self.timings.values())


class Copilot:
    def __init__(
        self,
        retriever: Retriever,
        answerer,
        *,
        gate=None,
        input_guard: InputGuard | None = None,
        output_guard: OutputGuard | None = None,
        cache: ResponseCache | None = None,
        scope: str = "copilot-v1",
        trusted_prefixes: tuple[str, ...] = ("week",),
    ):
        self.retriever, self.answerer, self.gate = retriever, answerer, gate
        self.input_guard, self.output_guard = input_guard, output_guard
        self.cache, self.scope, self.trusted_prefixes = cache, scope, trusted_prefixes

    # ------------------------------------------------------------------ one question
    def ask(self, question: str, *, request_id: str = "") -> Result:
        rid = request_id or uuid.uuid4().hex[:12]
        res = Result(question, "", request_id=rid)
        with tracing.span(
            "copilot.ask", {"app.request_id": rid, "app.question_chars": len(question or "")}
        ):
            res.trace_id = tracing.current_trace_id() or ""
            try:
                self._run(question, res)
            except ChatError as exc:  # the model server failed: say so, do not crash, do not invent
                res.error = f"model: {exc}"
                res.answer, res.abstained = IDK, True
                res.flags.append("model_error")
        return res

    @contextmanager
    def _stage(self, res: Result, name: str):
        """Time a stage into ``res.timings`` and open a span for it."""
        t0 = time.perf_counter()
        try:
            with tracing.span(f"copilot.{name}") as sp:
                yield sp
        finally:
            res.timings[name] = res.timings.get(name, 0.0) + time.perf_counter() - t0

    def _run(self, question: str, res: Result) -> None:
        if self.input_guard is not None:
            with self._stage(res, "input_guard") as sp:
                v = self.input_guard.check(question)
                sp.set_attribute("app.allowed", v.allowed)
            if not v.allowed:
                res.answer, res.blocked = BLOCKED_INPUT, True
                res.flags += [f"input_blocked: {r}" for r in v.reasons]
                return
        if self.cache is not None and (hit := self.cache.get(question, scope=self.scope)):
            c = hit.value
            res.answer, res.abstained, res.cited, res.mode = (
                c["answer"],
                c["abstained"],
                list(c["cited"]),
                c["mode"],
            )
            res.sources = c["sources"]
            res.cached = True
            res.flags.append("cache_hit")
            return
        with self._stage(res, "retrieve") as sp:
            ret = self.retriever.retrieve(question)
            sp.set_attribute("app.sources", len(ret.sources))
            sp.set_attribute("app.retrieval_cached", ret.cached)
        sources = ret.sources
        if self.trusted_prefixes is not None:
            with self._stage(res, "quarantine"):
                kept, quarantined = scrub_sources(sources, trusted_prefixes=self.trusted_prefixes)
            if quarantined:
                res.quarantined = len(quarantined)
                res.flags += [
                    f"quarantined: {s.doc} ({', '.join(why[:2])})" for s, why in quarantined
                ]
                sources = [
                    Source(n, s.id, s.doc, s.heading, s.text, s.score, s.cosine, s.week, s.day)
                    for n, s in enumerate(kept, 1)
                ]
        res.sources = res.retrieved = sources
        if self.gate is not None:
            with self._stage(res, "gate") as sp:
                res.gate = self.gate.decide(question, Retrieval(question, sources, ret.weeks))
                sp.set_attribute("app.gate_score", float(res.gate.score))
                sp.set_attribute("app.gate_allowed", res.gate.allowed)
            if not res.gate.allowed:
                res.answer, res.abstained, res.mode = IDK, True, "gate"
                res.sources = []  # nothing was good enough to cite: do not show passages that were not used
                res.flags.append(res.gate.reason)
                self._remember(question, res)
                return
        if not sources:
            res.answer, res.abstained, res.mode = IDK, True, "no-sources"
            return
        with self._stage(res, "answer") as sp:
            draft: Draft = self.answerer.answer(question, sources, ret.weeks)
            sp.set_attribute("app.mode", draft.mode)
            sp.set_attribute("app.prompt_tokens", draft.prompt_tokens)
            sp.set_attribute("app.completion_tokens", draft.completion_tokens)
        res.answer, res.abstained, res.mode = draft.text, draft.abstained, draft.mode
        res.prompt_tokens, res.completion_tokens = draft.prompt_tokens, draft.completion_tokens
        res.flags += [f"answer: {n}" for n in draft.notes]
        res.cited = [n for n in cited_numbers(res.answer) if 1 <= n <= len(sources)]
        if self.output_guard is not None:
            with self._stage(res, "output_guard"):
                out = self.output_guard.check(res.answer)
            if out.violations:
                res.flags += [f"output: {v}" for v in out.violations]
            if out.blocked:
                res.answer, res.abstained, res.cited, res.mode = (
                    out.text,
                    True,
                    [],
                    "blocked-output",
                )
                res.blocked = True
            else:
                res.answer = out.text
                res.cited = [n for n in cited_numbers(res.answer) if 1 <= n <= len(sources)]
        self._remember(question, res)

    def _remember(self, question: str, res: Result) -> None:
        if self.cache is not None and not res.blocked and not res.error:
            self.cache.put(
                question,
                {
                    "answer": res.answer,
                    "abstained": res.abstained,
                    "cited": res.cited,
                    "mode": res.mode,
                    "sources": res.sources,
                },
                scope=self.scope,
            )


def idf_from_index(index) -> dict[str, float]:
    """Inverse document frequency of the stems in the indexed chunks: the lexical weights the extractive answerer scores sentences with."""
    import collections
    import math

    from .answer import tokens

    df: collections.Counter = collections.Counter()
    for c in index.chunks:
        df.update(set(tokens(c["text"])))
    n = max(1, len(index.chunks))
    return {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
