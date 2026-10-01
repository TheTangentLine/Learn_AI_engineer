"""Week 5 Day 4 - the other side: OUR agent loop using the notes MCP server over stdio.

Starts day4_notes_server.py as a subprocess, discovers its tools through MCP, and lets a model answer questions
from the notes. Only read-only tools are exposed to the agent (an allowlist), however many the server offers.

  uv run python weeks/week05_tool-use-and-agents/solutions/day4_client_demo.py             # local Qwen-0.5B
  LLM_PROVIDER=anthropic uv run python weeks/week05_tool-use-and-agents/solutions/day4_client_demo.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.agent import run_agent  # noqa: E402
from common.mcp_tools import McpBridge, stdio  # noqa: E402

SERVER = Path(__file__).parent / "day4_notes_server.py"

SAMPLE_NOTES = {
    "meeting-2026-03-14": "# Platform sync\ntags: meeting, platform\nDecided to upgrade kubernetes to 1.31 in April.\nOwner: Dana.\n",
    "reading-list": "# Reading list\ntags: personal\nDesigning Data-Intensive Applications\nThe kubernetes book\n",
    "caching-ideas": "# Caching ideas\ntags: platform, design\nUse a write-through cache. Cache invalidation is hard.\nCache keys include tenant id.\n",
    "travel": "# Trip to Hanoi\ntags: personal\nFlight lands 2026-05-02 at 14:30. Hotel: Old Quarter, 3 nights.\n",
}
# (question, regex the answer must match, must the agent have searched the notes?)
QUESTIONS = [
    ("Who owns the kubernetes upgrade?", r"\bdana\b", True),
    ("When does my flight to Hanoi land?", r"14:30|2:30", True),
    ("What did I write about cache keys?", r"tenant", True),
    # no such note: the right answer is an explicit "I don't know", reached AFTER looking
    (
        "What is my favourite colour?",
        r"(don't|do not|cannot|can't|couldn't|no (?:note|information|mention)|not (?:found|mentioned|available)|unknown)",
        True,
    ),
]


def main() -> None:
    import re

    provider = os.environ.get("LLM_PROVIDER") or "local"
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in SAMPLE_NOTES.items():
            (Path(tmp) / f"{name}.md").write_text(text)
        env = {**os.environ, "NOTES_DIR": tmp}
        with McpBridge(stdio(sys.executable, str(SERVER), env=env)) as mcp:
            print(f"server tools: {[t['name'] for t in mcp.list_tools()]}")
            print(
                f"resources: {mcp.list_resources()} {mcp.list_resource_templates()}  prompts: {mcp.list_prompts()}\n"
            )
            registry = mcp.registry(allow={"search_notes", "read_note", "list_notes"})
            passed = 0
            for question, pattern, must_search in QUESTIONS:
                run = run_agent(question, registry, provider=provider, max_steps=6)
                ok = (
                    run.ok
                    and re.search(pattern, run.answer, re.I) is not None
                    and (not must_search or bool(run.calls))
                )
                passed += ok
                print(run.trace())
                print(f"==> {'PASS' if ok else 'FAIL'} ({run.status}, {len(run.calls)} calls)\n")
            print(f"passed {passed}/{len(QUESTIONS)}")


if __name__ == "__main__":
    main()
