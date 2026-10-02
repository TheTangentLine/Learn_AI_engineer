"""The research agent: search_web + fetch_page + notes (an MCP server) -> a cited report, machine-verified."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from common.agent import AgentRun, run_agent
from common.context import clear_old_tool_results
from common.mcp_tools import McpBridge, stdio
from common.tools import ToolRegistry

from .fetch import Fetcher
from .report import Verification, render_final, verify_report
from .tools import Source, SourceLog, make_research_tools
from .web import LocalWeb

NOTES_SERVER = Path(__file__).resolve().parents[2] / "day4_notes_server.py"
NOTE_TOOLS = {"create_note", "append_to_note", "read_note", "search_notes", "list_notes"}

SYSTEM = (
    "You are a careful research assistant. Answer the user's question using ONLY pages you have opened with "
    "fetch_page. Work like this: search_web to find candidate pages, open the most relevant with fetch_page "
    "(read on with start= if the answer is further down), optionally save key findings with create_note, then write "
    "the final report. Report format: markdown, a short title line starting with '# ', then 2 to 6 sentences or "
    "bullets. EVERY factual sentence must end with a citation like [1], the Source number shown when you opened the "
    "page. Never cite a page you did not open. Copy numbers exactly as written. If the pages do not answer the "
    "question, say plainly that they do not; do not guess. Do not write a Sources section: it is added for you."
)


@dataclass
class ResearchResult:
    question: str
    body: str  # the model's report
    report: str  # body + the harness-built Sources section
    sources: list[Source]
    verification: Verification
    run: AgentRun

    @property
    def fetched_slugs(self) -> set[str]:
        return {s.url.rsplit("/", 1)[-1] for s in self.sources}


def research(
    question: str,
    web: LocalWeb,
    *,
    provider: str | None = None,
    model: str | None = None,
    max_steps: int = 14,
    max_cost_usd: float | None = None,
    use_notes: bool = True,
    keep_pages: int = 5,
) -> ResearchResult:
    fetcher = Fetcher({web.host})
    log = SourceLog()
    tools = make_research_tools(web.base, fetcher, log)
    with tempfile.TemporaryDirectory(prefix="research-notes-") as notes_dir:
        if use_notes:
            env = {"NOTES_DIR": notes_dir, "PATH": "/usr/bin:/bin"}
            with McpBridge(stdio(sys.executable, str(NOTES_SERVER), env=env)) as mcp:
                # only the note tools the agent needs, however many the server offers
                registry = ToolRegistry(
                    [*ToolRegistry(tools).tools(), *mcp.registry(allow=NOTE_TOOLS).tools()]
                )
                run = _run(question, registry, provider, model, max_steps, max_cost_usd, keep_pages)
        else:
            run = _run(
                question, ToolRegistry(tools), provider, model, max_steps, max_cost_usd, keep_pages
            )
    body = run.answer
    return ResearchResult(
        question,
        body,
        render_final(body, log.sources),
        log.sources,
        verify_report(body, log.sources),
        run,
    )


def _run(question, registry, provider, model, max_steps, max_cost_usd, keep_pages) -> AgentRun:
    return run_agent(
        question, registry, system=SYSTEM, provider=provider, model=model, max_steps=max_steps,
        max_cost_usd=max_cost_usd, context_hook=clear_old_tool_results(keep_last=keep_pages),
    )  # fmt: skip
