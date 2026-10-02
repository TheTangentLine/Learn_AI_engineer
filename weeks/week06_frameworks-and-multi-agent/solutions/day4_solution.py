"""Week 6 Day 4 - Solution: orchestrator-workers with parallel research subagents and verified, merged citations.

Two ways to run "one question needing several investigations":

  1. SINGLE AGENT   one research agent works through all sub-questions in ONE context (history grows, steps serialise)
  2. ORCHESTRATED   a planner splits the question; N worker agents (Week 5's research agent) run IN PARALLEL, each in
                    its OWN small context; a deterministic merge combines their reports

The merge is code, not a model: worker citations [1], [2] all mean different pages in different workers, so they are
RENUMBERED into one global source list and the merged report is machine-verified like any other. A worker that fails
is reported as unavailable; the others still count.

Also: ``agentic_orchestrator`` lets a MODEL decide how many workers to spawn through a ``research_subquestion`` tool;
parallel tool calls then run concurrently for free (ToolRegistry.execute_all).

  uv run python weeks/week06_frameworks-and-multi-agent/solutions/day4_solution.py --offline
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "weeks/week05_tool-use-and-agents/solutions/weekly"))

from research_agent.agent import ResearchResult, research  # noqa: E402
from research_agent.report import Verification, render_final, verify_report  # noqa: E402
from research_agent.tools import Source  # noqa: E402
from research_agent.web import LocalWeb  # noqa: E402

from common import llm  # noqa: E402
from common.agent import AgentRun, run_agent  # noqa: E402
from common.tools import ToolFailure, ToolRegistry, tool  # noqa: E402

MAX_SUBQUESTIONS = 4
CITE = re.compile(r"\[(\d+)\]")
INVALID_BASE = (
    9000  # a citation that pointed at nothing is moved here so the verifier still flags it
)

# ----------------------------------------------------------------------------- 1. planning


def plan(question: str, *, provider: str | None = None) -> list[str]:
    """Split a question into independent sub-questions (at most MAX_SUBQUESTIONS) with a structured model call."""
    from pydantic import BaseModel, Field

    class Plan(BaseModel):
        subquestions: list[str] = Field(
            description=f"1 to {MAX_SUBQUESTIONS} independent, self-contained questions"
        )

    system = (
        "Split the user's question into independent, self-contained sub-questions that can each be researched "
        "separately. Do not add questions the user did not ask. If the question is already a single question, return it."
    )
    p, _ = llm.structured(question, Plan, system=system, provider=provider, max_tokens=300)
    out = [q.strip() for q in p.subquestions if q.strip()]
    return list(dict.fromkeys(out))[:MAX_SUBQUESTIONS] or [question]


# ----------------------------------------------------------------------------- 2. workers in parallel


@dataclass
class Part:
    subquestion: str
    result: ResearchResult | None = None
    error: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.result is not None and self.result.run.ok


def run_workers(
    subquestions: list[str],
    web: LocalWeb,
    *,
    provider: str | None = None,
    max_steps: int = 10,
    max_cost_usd: float | None = None,
    max_workers: int = 4,
    sequential: bool = False,
) -> list[Part]:
    """One research agent per sub-question. A worker's failure becomes a Part with an error, never an exception."""

    def work(q: str) -> Part:
        t0 = time.perf_counter()
        try:
            r = research(
                q,
                web,
                provider=provider,
                max_steps=max_steps,
                max_cost_usd=max_cost_usd,
                use_notes=False,
            )
            return Part(
                q,
                r,
                "" if r.run.ok else f"worker stopped: {r.run.status} {r.run.error}".strip(),
                time.perf_counter() - t0,
            )
        except Exception as exc:
            return Part(q, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0)

    if sequential or len(subquestions) == 1:
        return [work(q) for q in subquestions]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(subquestions))) as pool:
        return list(pool.map(work, subquestions))  # results in the SAME order as the sub-questions


# ----------------------------------------------------------------------------- 3. the merge (code, not a model)


@dataclass
class Merged:
    body: str
    report: str
    sources: list[Source]
    verification: Verification
    parts: list[Part]
    failed: list[str] = field(default_factory=list)


def merge_reports(parts: list[Part]) -> Merged:
    """Renumber every worker's [n] into ONE global source list (deduplicated by URL), then verify the whole report."""
    global_sources: list[Source] = []
    by_url: dict[str, int] = {}
    sections: list[str] = []
    verifiable: list[
        str
    ] = []  # sections that carry claims; a 'could not be researched' note is not a claim
    failed: list[str] = []
    for part in parts:
        if not part.ok:
            failed.append(part.subquestion)
            sections.append(
                f"## {part.subquestion}\nThis sub-question could not be researched ({part.error or 'no answer'})."
            )
            continue
        remap: dict[int, int] = {}
        for src in part.result.sources:
            if src.url not in by_url:
                by_url[src.url] = len(global_sources) + 1
                global_sources.append(Source(by_url[src.url], src.url, src.title, src.text))
            remap[src.n] = by_url[src.url]
        body = "\n".join(
            line for line in part.result.body.splitlines() if not line.startswith("# ")
        )
        body = CITE.sub(
            lambda m, remap=remap: (
                f"[{remap.get(int(m.group(1)), INVALID_BASE + int(m.group(1)))}]"
            ),
            body,
        )
        sections.append(f"## {part.subquestion}\n{body.strip()}")
        verifiable.append(sections[-1])
    merged_body = "# Research report\n\n" + "\n\n".join(sections)
    return Merged(
        merged_body,
        render_final(merged_body, global_sources),
        global_sources,
        verify_report("\n\n".join(verifiable), global_sources),
        parts,
        failed,
    )


def orchestrate(
    question: str,
    web: LocalWeb,
    *,
    provider: str | None = None,
    subquestions: list[str] | None = None,
    **kw,
) -> Merged:
    return merge_reports(
        run_workers(subquestions or plan(question, provider=provider), web, provider=provider, **kw)
    )


# ----------------------------------------------------------------------------- 4. an agentic orchestrator (the model chooses the fan-out)

ORCHESTRATOR_SYSTEM = (
    "You coordinate research. For a question with several parts, call research_subquestion ONCE PER PART, all in the same "
    "turn so they run in parallel. Each call returns a short, cited report. When you have them all, reply with one "
    "sentence per part, keeping the [n] citations exactly as given. Never add facts the reports do not contain."
)


def make_delegate_tool(
    web: LocalWeb, *, provider: str | None, log: list[Part], max_steps: int = 10
):
    @tool(timeout_s=300)
    def research_subquestion(question: str) -> str:
        """Research ONE self-contained question with a dedicated research agent and return its short, cited report.

        Args:
            question: A complete question that makes sense without any other context.
        """
        part = run_workers([question], web, provider=provider, max_steps=max_steps)[0]
        log.append(part)
        if not part.ok:
            raise ToolFailure(f"the research agent failed: {part.error}")
        body = "\n".join(
            line for line in part.result.body.splitlines() if not line.startswith("# ")
        )
        return (
            body + "\n(Sources: " + "; ".join(f"[{s.n}] {s.url}" for s in part.result.sources) + ")"
        )

    return research_subquestion


def agentic_orchestrator(
    question: str, web: LocalWeb, *, provider: str | None = None, max_steps: int = 6
) -> tuple[AgentRun, list[Part]]:
    log: list[Part] = []
    registry = ToolRegistry([make_delegate_tool(web, provider=provider, log=log)])
    run = run_agent(
        question, registry, system=ORCHESTRATOR_SYSTEM, provider=provider, max_steps=max_steps
    )
    return run, log


# ----------------------------------------------------------------------------- measuring context isolation


@dataclass
class Footprint:
    peak_context: int  # the largest single prompt any one agent saw (estimated tokens)
    total_input: int  # all prompts, all agents
    seconds: float


def footprint(parts: list[Part], seconds: float) -> Footprint:
    runs = [p.result.run for p in parts if p.result]
    return Footprint(
        max((s.usage.input_tokens for r in runs for s in r.steps), default=0),
        sum(r.input_tokens for r in runs),
        seconds,
    )


# ----------------------------------------------------------------------------- demo

QUESTION = (
    "Summarise three findings: what success rates the badly designed and redesigned tools reached in the Week 5 "
    "tool-design experiment, how many baseline misses corrective RAG rescued in Week 4, and what a subprocess sandbox "
    "does not stop according to Week 5 Day 6."
)


def main() -> None:
    from common.corpus import load_course_docs

    print(
        "Run with the scripted model via the tests; a real-model run follows when --live is given."
    )
    if "--live" not in sys.argv:
        return
    from common.local_server import LocalOpenAIServer

    with LocalOpenAIServer() as srv, LocalWeb(load_course_docs()) as web:
        import os

        os.environ.update(LLM_PROVIDER="ollama", LLM_MODEL="local-qwen", OLLAMA_BASE_URL=srv.url)
        llm._ollama.cache_clear()
        t0 = time.perf_counter()
        m = orchestrate(QUESTION, web)
        dt = time.perf_counter() - t0
        print(
            json.dumps(
                {
                    "subquestions": [p.subquestion for p in m.parts],
                    "failed": m.failed,
                    "verification": m.verification.summary(),
                    "seconds": round(dt, 1),
                },
                indent=2,
            )
        )
        print(m.report)


if __name__ == "__main__":
    main()
