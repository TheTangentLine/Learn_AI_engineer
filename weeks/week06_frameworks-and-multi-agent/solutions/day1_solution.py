"""Week 6 Day 1 - Solution: the Week 5 file agent, ported into three frameworks, compared with the raw loop.

All four implementations get the SAME tools (Week 5 Day 2's sandboxed file tools), the SAME seven tasks and talk to the
SAME OpenAI-compatible Chat Completions endpoint, so the only variable is the framework:

  raw          common/agent.py (Week 5)             the loop you wrote
  langgraph    StateGraph: model node, tools node, conditional edge
  agents_sdk   OpenAI Agents SDK: Agent + Runner
  pydantic_ai  PydanticAI: Agent + typed tools + usage limits

Each ``run_*`` returns a ``PortResult`` (answer, tool calls made, model requests, normalised status), so the
comparison is mechanical: do they solve the tasks, how many lines did it take, and what happens at a step limit?

  uv run python weeks/week06_frameworks-and-multi-agent/solutions/day1_solution.py    # local Qwen via LocalOpenAIServer
"""

from __future__ import annotations

import asyncio
import inspect
import json
import operator
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # must be set before pydantic_ai is imported
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from _w5 import load  # noqa: E402

from common import agent as agent_module  # noqa: E402
from common import chat  # noqa: E402
from common.agent import run_agent  # noqa: E402
from common.chat import ToolCall  # noqa: E402
from common.tools import Tool, ToolRegistry  # noqa: E402

w5 = load("day2_solution")  # Week 5 Day 2: Workspace, build_workspace, TASKS
SYSTEM = (
    "You are a careful assistant that completes tasks about a project folder by using tools. "
    "Paths are relative to the project root (use '.' for the root, e.g. 'src/app.py'). Never invent facts that a tool "
    "could tell you. If the information is not available, say so. When you are done, reply with the final answer only."
)


@dataclass
class PortResult:
    framework: str
    answer: str = ""
    calls: list[tuple[str, dict]] = field(default_factory=list)
    requests: int = 0  # model calls
    status: str = "done"  # done | max_steps | error
    error: str = ""


def _use_endpoint(base_url: str) -> None:
    """Point common.llm's OpenAI-compatible ('ollama') client at base_url (its client is cached per process)."""
    from common import llm

    os.environ["OLLAMA_BASE_URL"] = base_url
    llm._ollama.cache_clear()


def _recorded(registry: ToolRegistry, calls: list):
    """Execute through our registry (validation, errors-as-text, truncation) and record every call."""

    def run(name: str, args: dict) -> str:
        calls.append((name, args))
        return registry.execute(ToolCall(f"c{len(calls)}", name, args)).content

    return run


# ============================================================================ 1. raw loop (Week 5)


def run_raw(task: str, registry: ToolRegistry, *, base_url: str, max_steps: int = 8) -> PortResult:
    _use_endpoint(base_url)
    r = run_agent(
        task, registry, system=SYSTEM, provider="ollama", model="local-qwen", max_steps=max_steps
    )
    return PortResult(
        "raw",
        r.answer,
        [(c.name, c.args) for c in r.calls],
        len(r.steps),
        {"done": "done", "max_steps": "max_steps"}.get(r.status, r.status),
        r.error,
    )


# ============================================================================ 2. LangGraph


class AgentState(TypedDict):
    messages: Annotated[
        list, operator.add
    ]  # a reducer: nodes return NEW messages, the framework appends


def run_langgraph(
    task: str, registry: ToolRegistry, *, base_url: str, max_steps: int = 8
) -> PortResult:
    from langgraph.errors import GraphRecursionError
    from langgraph.graph import END, START, StateGraph

    _use_endpoint(base_url)
    calls: list[tuple[str, dict]] = []
    execute = _recorded(registry, calls)
    n_requests = [0]

    def model(state: AgentState) -> dict:
        n_requests[0] += 1
        t = chat.turn(
            state["messages"],
            registry.specs(),
            system=SYSTEM,
            provider="ollama",
            model="local-qwen",
        )
        return {"messages": [chat.assistant_message(t)]}

    def tools(state: AgentState) -> dict:
        return {
            "messages": [
                chat.tool_message(c, execute(c["name"], c["args"]))
                for c in state["messages"][-1]["tool_calls"]
            ]
        }

    graph = StateGraph(AgentState)
    graph.add_node("model", model)
    graph.add_node("tools", tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model",
        lambda s: "tools" if s["messages"][-1].get("tool_calls") else END,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "model")
    app = graph.compile()
    try:
        out = app.invoke(
            {"messages": [{"role": "user", "content": task}]},
            {"recursion_limit": 2 * max_steps + 1},
        )
    except GraphRecursionError as exc:
        return PortResult("langgraph", "", calls, n_requests[0], "max_steps", str(exc)[:80])
    return PortResult("langgraph", out["messages"][-1]["content"], calls, n_requests[0])


# ============================================================================ 3. OpenAI Agents SDK


def run_agents_sdk(
    task: str, registry: ToolRegistry, *, base_url: str, max_steps: int = 8
) -> PortResult:
    from agents import (
        Agent,
        FunctionTool,
        MaxTurnsExceeded,
        ModelBehaviorError,
        OpenAIChatCompletionsModel,
        Runner,
        set_tracing_disabled,
    )
    from openai import AsyncOpenAI

    set_tracing_disabled(True)  # otherwise traces are uploaded to OpenAI's servers
    calls: list[tuple[str, dict]] = []
    execute = _recorded(registry, calls)

    def adapt(t: Tool) -> FunctionTool:
        async def invoke(ctx, args_json: str) -> str:
            return await asyncio.to_thread(execute, t.name, json.loads(args_json or "{}"))

        return FunctionTool(
            name=t.name,
            description=t.description,
            params_json_schema=t.parameters,
            on_invoke_tool=invoke,
            strict_json_schema=False,
        )

    model = OpenAIChatCompletionsModel(
        model="local-qwen", openai_client=AsyncOpenAI(base_url=base_url, api_key="local")
    )
    agent = Agent(
        name="file-agent",
        instructions=SYSTEM,
        model=model,
        tools=[adapt(t) for t in registry.tools()],
    )
    try:
        result = Runner.run_sync(agent, task, max_turns=max_steps)
    except MaxTurnsExceeded as exc:
        return PortResult("agents_sdk", "", calls, max_steps, "max_steps", str(exc)[:80])
    except ModelBehaviorError as exc:  # e.g. the model called a tool that does not exist: the SDK raises, it does not recover
        return PortResult("agents_sdk", "", calls, len(calls) + 1, "error", str(exc)[:120])
    return PortResult(
        "agents_sdk", str(result.final_output or ""), calls, len(result.raw_responses)
    )


# ============================================================================ 4. PydanticAI


def run_pydantic_ai(
    task: str, registry: ToolRegistry, *, base_url: str, max_steps: int = 8
) -> PortResult:
    from pydantic_ai import Agent, UsageLimits
    from pydantic_ai import Tool as PTool
    from pydantic_ai.exceptions import UsageLimitExceeded
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    calls: list[tuple[str, dict]] = []
    execute = _recorded(registry, calls)

    def adapt(t: Tool) -> PTool:
        def fn(**kwargs: Any) -> str:
            return execute(t.name, kwargs)

        return PTool.from_schema(
            fn, name=t.name, description=t.description, json_schema=t.parameters
        )

    model = OpenAIChatModel(
        "local-qwen", provider=OpenAIProvider(base_url=base_url, api_key="local")
    )
    agent = Agent(model, system_prompt=SYSTEM, tools=[adapt(t) for t in registry.tools()])
    try:
        result = agent.run_sync(task, usage_limits=UsageLimits(request_limit=max_steps))
    except UsageLimitExceeded as exc:
        return PortResult("pydantic_ai", "", calls, max_steps, "max_steps", str(exc)[:80])
    return PortResult("pydantic_ai", str(result.output), calls, result.usage.requests)


# ============================================================================ 5. Claude Agent SDK (built, NOT run)


def claude_agent_options(
    registry: ToolRegistry, *, max_turns: int = 8, max_budget_usd: float = 0.50
):
    """Build the options for the Claude Agent SDK, which drives the Claude Code agent loop in a subprocess.

    Our tools become an in-process MCP server (named 'files'); the SDK exposes them to the model as ``mcp__files__<name>``.
    Running it needs the Claude Code CLI and Anthropic credentials, neither available where this was written, so it is
    only CONSTRUCTED and unit-tested here, never executed against a model.
    """
    from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server
    from claude_agent_sdk import tool as sdk_tool

    sdk_tools = []
    for t in registry.tools():

        @sdk_tool(t.name, t.description, t.parameters)
        async def handler(args: dict, _name: str = t.name) -> dict:
            res = await asyncio.to_thread(registry.execute, ToolCall("c", _name, args))
            return {"content": [{"type": "text", "text": res.content}], "is_error": res.is_error}

        sdk_tools.append(handler)
    return ClaudeAgentOptions(
        system_prompt=SYSTEM,
        mcp_servers={"files": create_sdk_mcp_server("files", tools=sdk_tools)},
        allowed_tools=[
            f"mcp__files__{t.name}" for t in registry.tools()
        ],  # an allowlist: nothing else (no Bash) is permitted
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
    )


async def run_claude_agent(
    task: str, registry: ToolRegistry
) -> str:  # pragma: no cover - needs the CLI + credentials
    from claude_agent_sdk import ResultMessage, query

    answer = ""
    async for message in query(prompt=task, options=claude_agent_options(registry)):
        if isinstance(message, ResultMessage):
            answer = message.result or ""
    return answer


PORTS = {
    "raw": run_raw,
    "langgraph": run_langgraph,
    "agents_sdk": run_agents_sdk,
    "pydantic_ai": run_pydantic_ai,
}

# ============================================================================ comparison


def code_lines(fn) -> int:
    """Non-blank, non-comment, non-docstring source lines of a function (a rough size measure, not a quality one)."""
    lines = [ln.strip() for ln in inspect.getsource(fn).splitlines()]
    body = [ln for ln in lines if ln and not ln.startswith("#")]
    return len(body)


def run_task(framework: str, task, *, base_url: str, max_steps: int = 8):
    with tempfile.TemporaryDirectory() as tmp:
        ws = w5.build_workspace(Path(tmp))
        res = PORTS[framework](
            task.prompt, ws.registry([w5.calculate]), base_url=base_url, max_steps=max_steps
        )
        # grade on FINAL STATE with the Week 5 checker (it expects an AgentRun-like object)
        shim = w5.AgentRun(
            task.prompt, status="done" if res.status == "done" else res.status, answer=res.answer
        )
        return res, bool(res.status == "done" and task.check(shim, ws))


def main() -> None:
    from common.local_server import LocalOpenAIServer

    with LocalOpenAIServer() as srv:
        print(f"model server: {srv.url}\n")
        table = {}
        for fw in PORTS:
            ok = 0
            calls = 0
            for task in w5.TASKS:
                res, passed = run_task(fw, task, base_url=srv.url)
                ok += passed
                calls += len(res.calls)
                print(
                    f"{fw:12s} {task.id:14s} {'PASS' if passed else 'FAIL'} status={res.status:9s} calls={len(res.calls)} requests={res.requests}"
                )
            table[fw] = (ok, calls)
        print("\nframework     passed  tool calls  core lines")
        for fw, (ok, calls) in table.items():
            size = code_lines(PORTS[fw]) + (
                code_lines(agent_module._run_agent) if fw == "raw" else 0
            )  # raw: count the loop it calls
            print(f"{fw:12s} {ok:4d}/{len(w5.TASKS)} {calls:10d} {size:10d}")


if __name__ == "__main__":
    main()
