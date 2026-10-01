"""Tools for agents: schema from a Python function, validated execution, errors the MODEL can act on.

    from common.tools import tool, ToolRegistry

    @tool
    def multiply(a: float, b: float) -> float:
        '''Multiply two numbers.

        Args:
            a: First factor.
            b: Second factor.
        '''
        return a * b

    registry = ToolRegistry([multiply])
    registry.specs()                         # -> [{"name", "description", "parameters"}] for chat.turn(...)
    registry.execute(call)                   # -> ToolResult(content, is_error, seconds), never raises
    registry.execute_all(calls)              # parallel, results in the SAME order as the calls

Design rules (each has a test): a tool never crashes the agent loop; errors are returned as text that says
what was wrong AND what is allowed, so the model can retry; huge outputs are truncated with a visible note;
slow tools time out; unknown tools and invalid arguments are errors, not exceptions.
"""

from __future__ import annotations

import inspect
import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import ConfigDict, ValidationError, create_model

from .chat import ToolCall

MAX_RESULT_CHARS = 8000


class ToolFailure(Exception):
    """Raise from a tool to report a failure whose message is shown to the model VERBATIM
    (other exceptions are shown as 'Tool x failed: ExcType: message')."""


def _param_docs(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (description, {param: text})."""
    if not doc:
        return "", {}
    doc = inspect.cleandoc(doc)
    head, _, args = doc.partition("\nArgs:")
    params: dict[str, str] = {}
    for m in re.finditer(r"^\s+(\w+)(?:\s*\([^)]*\))?:\s*(.+)$", args.split("\nReturns:")[0], re.M):
        params[m.group(1)] = m.group(2).strip()
    return head.strip(), params


def _clean_schema(node: Any) -> Any:
    """Drop pydantic's per-field/model 'title's and forbid extra properties (stricter for the model)."""
    if isinstance(node, dict):
        node = {k: _clean_schema(v) for k, v in node.items() if k != "title"}
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
        return node
    if isinstance(node, list):
        return [_clean_schema(x) for x in node]
    return node


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any] | None
    model: Any  # pydantic model validating the arguments (None for remote tools, validated by their server)
    parameters: dict[str, Any]
    timeout_s: float = 30.0
    max_chars: int = MAX_RESULT_CHARS
    raw_fn: Callable[[dict[str, Any]], Any] | None = (
        None  # remote tools: takes the raw argument dict
    )

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout_s: float = 30.0,
    max_chars: int = MAX_RESULT_CHARS,
):
    """Decorator: turn a typed Python function into a Tool."""

    def build(f: Callable) -> Tool:
        head, pdocs = _param_docs(f.__doc__)
        hints = get_type_hints(f)
        fields: dict[str, Any] = {}
        for pname, p in inspect.signature(f).parameters.items():
            ann = hints.get(pname, str)
            default = ... if p.default is inspect.Parameter.empty else p.default
            fields[pname] = (ann, default)
        # extra="forbid": a stray/misspelt argument is an ERROR the model can fix, not a silent default
        model = create_model(f"{f.__name__}_args", __config__=ConfigDict(extra="forbid"), **fields)
        schema = _clean_schema(model.model_json_schema())
        for pname, text in pdocs.items():
            if pname in schema.get("properties", {}):
                schema["properties"][pname]["description"] = text
        return Tool(
            name or f.__name__,
            description or head or f.__name__,
            f,
            model,
            schema,
            timeout_s,
            max_chars,
        )

    return build(fn) if fn else build


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False
    seconds: float = 0.0


def _format_validation(tool: Tool, exc: ValidationError) -> str:
    problems = "; ".join(
        f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in exc.errors()
    )
    required = tool.parameters.get("required", [])
    props = ", ".join(
        f"{k} ({v.get('type', 'any')})" for k, v in tool.parameters.get("properties", {}).items()
    )
    return f"Invalid arguments for {tool.name}: {problems}. Expected: {props or 'no arguments'}; required: {required}."


def _stringify(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    if len(text) > limit:
        text = (
            text[:limit] + f"\n[truncated: {len(text) - limit} more characters; narrow the request]"
        )
    return text


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"duplicate tool name {t.name!r}")
        self._tools[t.name] = t

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def execute(self, call: ToolCall) -> ToolResult:
        t0 = time.perf_counter()
        t = self._tools.get(call.name)
        if t is None:
            return ToolResult(
                call.id,
                call.name,
                f"Unknown tool '{call.name}'. Available tools: {', '.join(self._tools) or 'none'}.",
                True,
                time.perf_counter() - t0,
            )
        if t.raw_fn is not None:  # a remote tool (e.g. MCP): its server validates, we only forward
            raw_fn = t.raw_fn
            run_tool: Callable[[], Any] = lambda: raw_fn(dict(call.args))  # noqa: E731
        else:
            try:
                args = t.model.model_validate(call.args)
            except ValidationError as exc:
                return ToolResult(
                    call.id, call.name, _format_validation(t, exc), True, time.perf_counter() - t0
                )
            # pass the VALIDATED values (nested models stay models, enums stay enums); model_dump() would flatten them
            kwargs = {name: getattr(args, name) for name in type(args).model_fields}
            fn = t.fn
            run_tool = lambda: fn(**kwargs)  # noqa: E731
        # a worker thread lets us enforce a timeout; a timed-out tool keeps running in the background,
        # so truly untrusted work belongs in a subprocess/container (Day 6)
        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(run_tool)
        try:
            out = fut.result(timeout=t.timeout_s)
            res = ToolResult(
                call.id, call.name, _stringify(out, t.max_chars), False, time.perf_counter() - t0
            )
        except FutureTimeout:
            res = ToolResult(
                call.id,
                call.name,
                f"Tool {t.name} timed out after {t.timeout_s:g}s.",
                True,
                time.perf_counter() - t0,
            )
        except ToolFailure as exc:
            res = ToolResult(call.id, call.name, str(exc), True, time.perf_counter() - t0)
        except Exception as exc:  # the loop must survive any tool failure
            res = ToolResult(
                call.id,
                call.name,
                f"Tool {t.name} failed: {type(exc).__name__}: {exc}",
                True,
                time.perf_counter() - t0,
            )
        finally:
            ex.shutdown(wait=False)
        return res

    def execute_all(
        self, calls: list[ToolCall], parallel: bool = True, max_workers: int = 8
    ) -> list[ToolResult]:
        """Run every call; results come back in the SAME order as `calls` (providers need all of them)."""
        if not parallel or len(calls) <= 1:
            return [self.execute(c) for c in calls]
        with ThreadPoolExecutor(max_workers=min(max_workers, len(calls))) as pool:
            return list(pool.map(self.execute, calls))
