"""A framework-free agent loop with the guardrails production needs.

    from common.agent import run_agent
    run = run_agent("Which file defines parse_config?", registry, max_steps=8, max_cost_usd=0.10)
    run.status      # "done" | "max_steps" | "budget" | "stuck" | "truncated" | "error"
    run.answer      # the model's final text ("" unless it actually finished)
    print(run.trace())

The loop is ReAct without the special prompt format: the model's text is its reasoning, its tool calls are
its actions, tool results are the observations. What makes it an *agent* is only that the model decides the
next step; what makes it *safe to run* is everything below (each point has a test):

  * a step budget; the LAST step hides the tools, so the model must answer with what it has
  * a cost budget and a token budget, checked after every step
  * repeated identical calls are blocked (the result is replaced by a message that says so); two fully
    blocked steps in a row end the run as "stuck" instead of burning the budget
  * every tool call is answered, errors included, in order, and parallel calls run concurrently
  * a ``context_hook`` can rewrite the message list before each model call (Day 5: compaction)
  * a full trace: what the model said, what it called, what came back, tokens, cost, latency
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import chat
from .chat import ToolCall
from .llm import Usage
from .tools import ToolRegistry, ToolResult

DEFAULT_SYSTEM = (
    "You are a careful assistant that completes tasks by using tools. Think about what you still need, "
    "call tools to get it, and use the results. Never invent facts that a tool could tell you. "
    "If the information is not available, say so plainly. When you have enough, reply with the final answer only."
)

LAST_STEP_NOTE = (
    "\n\nYou have NO tool calls left. Answer now with what you have found so far. "
    "If something is still missing, say exactly what is missing."
)


@dataclass
class Step:
    index: int
    text: str
    calls: list[ToolCall]
    results: list[ToolResult]
    usage: Usage
    cost_usd: float
    latency_s: float
    blocked: int = 0  # how many of this step's calls were refused as repeats


@dataclass
class AgentRun:
    task: str
    status: str = "running"
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    messages: list[dict] = field(
        default_factory=list
    )  # the WORKING history (a context hook may trim it)
    transcript: list[dict] = field(
        default_factory=list
    )  # every message ever produced, never trimmed
    error: str = ""
    seconds: float = 0.0

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    @property
    def input_tokens(self) -> int:
        return sum(
            s.usage.input_tokens + s.usage.cache_read_tokens + s.usage.cache_write_tokens
            for s in self.steps
        )

    @property
    def output_tokens(self) -> int:
        return sum(s.usage.output_tokens for s in self.steps)

    @property
    def calls(self) -> list[ToolCall]:
        return [c for s in self.steps for c in s.calls]

    @property
    def results(self) -> list[ToolResult]:
        return [r for s in self.steps for r in s.results]

    @property
    def tool_names(self) -> list[str]:
        return [c.name for c in self.calls]

    @property
    def errors(self) -> int:
        return sum(r.is_error for r in self.results)

    @property
    def ok(self) -> bool:
        return self.status == "done"

    def trace(self, width: int = 140) -> str:
        def clip(s: str) -> str:
            s = " ".join(s.split())
            return s if len(s) <= width else s[: width - 1] + "…"

        lines = [f"TASK: {clip(self.task)}"]
        for s in self.steps:
            lines.append(
                f"[{s.index}] {s.usage.input_tokens + s.usage.cache_read_tokens} in / "
                f"{s.usage.output_tokens} out tokens, {s.latency_s:.1f}s"
            )
            if s.text:
                lines.append(f"    say:  {clip(s.text)}")
            for c, r in zip(s.calls, s.results, strict=True):
                lines.append(f"    call: {c.name}({clip(json.dumps(c.args, ensure_ascii=False))})")
                lines.append(f"    {'ERR ' if r.is_error else '-> '}{clip(r.content)}")
        lines.append(f"STATUS: {self.status}" + (f" ({self.error})" if self.error else ""))
        if self.answer:
            lines.append(f"ANSWER: {clip(self.answer)}")
        return "\n".join(lines)


def _key(call: ToolCall) -> str:
    return call.name + json.dumps(call.args, sort_keys=True, default=str)


def run_agent(
    task: str,
    registry: ToolRegistry,
    *,
    system: str = DEFAULT_SYSTEM,
    provider: str | None = None,
    model: str | None = None,
    messages: list[dict] | None = None,
    max_steps: int = 10,
    max_cost_usd: float | None = None,
    max_total_tokens: int | None = None,
    repeat_limit: int = 2,
    turn_max_tokens: int = 2048,
    context_hook: Callable[[list[dict], AgentRun], list[dict]] | None = None,
    on_step: Callable[[Step], None] | None = None,
    **turn_kwargs: Any,
) -> AgentRun:
    """Run until the model answers or a limit trips. Never raises for model/tool/provider failures."""
    run = AgentRun(task)
    run.messages = list(messages or []) + [{"role": "user", "content": task}]
    run.transcript = list(run.messages)
    seen: dict[str, tuple[int, str]] = {}  # call key -> (times executed, first result text)
    blocked_in_a_row = 0
    t_start = time.perf_counter()

    for index in range(1, max_steps + 1):
        last = index == max_steps
        if context_hook:
            run.messages = context_hook(run.messages, run)
        try:
            t = chat.turn(
                run.messages,
                registry.specs(),
                system=system + (LAST_STEP_NOTE if last else ""),
                provider=provider,
                model=model,
                max_tokens=turn_max_tokens,
                tool_choice="none" if last else None,
                **turn_kwargs,
            )
        except Exception as exc:  # provider outage, auth, rate limit after retries ...
            run.status, run.error = "error", f"{type(exc).__name__}: {exc}"
            break

        step = Step(index, t.text, list(t.tool_calls), [], t.usage, t.cost_usd, t.latency_s)
        run.steps.append(step)

        if not t.wants_tools or last:
            # on the last step the tools are hidden; a model that STILL asks for one has not finished
            run.answer = t.text
            if t.wants_tools:  # never executed, so they must not appear as calls that were made
                run.error = (
                    f"model still asked for {len(t.tool_calls)} tool call(s) on the last step"
                )
                step.calls = []
            run.status = (
                "truncated"
                if t.stop_reason == "max_tokens" and not t.wants_tools
                else ("done" if not t.wants_tools else "max_steps")
            )
            if on_step:
                on_step(step)
            break

        # keep the model's request in the history, then answer EVERY call, in order
        run.messages.append(chat.assistant_message(t))
        run.transcript.append(run.messages[-1])
        results: list[ToolResult | None] = [None] * len(t.tool_calls)
        to_run: list[tuple[int, ToolCall]] = []
        for i, call in enumerate(t.tool_calls):
            n, first = seen.get(_key(call), (0, ""))
            if n >= repeat_limit:
                step.blocked += 1
                results[i] = ToolResult(
                    call.id,
                    call.name,
                    f"Blocked: you already called {call.name} with exactly these arguments {n} times"
                    + (f" and got: {first[:200]!r}" if first else " (in this same step)")
                    + ". Do not repeat it. Use that result, try different arguments, or give your final answer.",
                    True,
                )
            else:
                to_run.append((i, call))
                seen[_key(call)] = (
                    n + 1,
                    first,
                )  # count at SCHEDULE time: duplicates inside one step count
        for (i, call), res in zip(
            to_run, registry.execute_all([c for _, c in to_run]), strict=True
        ):
            results[i] = res
            k = _key(call)
            n, first = seen[k]
            seen[k] = (n, first or res.content)
        step.results = [r for r in results if r is not None]
        for call, res in zip(step.calls, step.results, strict=True):
            run.messages.append(chat.tool_message(call, res.content, res.is_error))
            run.transcript.append(run.messages[-1])
        if on_step:
            on_step(step)

        blocked_in_a_row = blocked_in_a_row + 1 if step.blocked == len(step.calls) else 0
        if blocked_in_a_row >= 2:
            run.status = "stuck"
            break
        if max_cost_usd is not None and run.cost_usd > max_cost_usd:
            run.status, run.error = "budget", f"cost ${run.cost_usd:.4f} > ${max_cost_usd:.4f}"
            break
        if max_total_tokens is not None and run.input_tokens + run.output_tokens > max_total_tokens:
            run.status = "budget"
            run.error = f"tokens {run.input_tokens + run.output_tokens} > {max_total_tokens}"
            break
    else:  # pragma: no cover - the last step always breaks above; kept so the loop cannot fall through silently
        run.status = "max_steps"

    run.seconds = time.perf_counter() - t_start
    return run
