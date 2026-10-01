"""Use any MCP server's tools from the (synchronous) agent loop in ``common/agent.py``.

    from common.mcp_tools import McpBridge, stdio

    with McpBridge(stdio("python", "my_server.py")) as mcp:
        registry = mcp.registry(allow={"search_notes", "read_note"})   # expose ONLY what the agent needs
        run = run_agent("What did I write about caching?", registry)
        text = mcp.read_resource("notes://index")

Why a bridge: the MCP SDK is async and keeps one connection open for the whole session, while our tool
registry is called from worker threads. The bridge owns a background thread with an event loop that holds the
connection, and exposes plain blocking methods. ``server`` can be a ``StdioServerParameters`` (a subprocess),
a URL string (streamable HTTP), or an in-process ``MCPServer`` (fast tests, no subprocess).

Security note: an MCP server is *code you are choosing to trust*, and its tool descriptions go straight into
your prompt. ``registry(allow=...)`` is an allowlist for exactly that reason (Week 8 returns to this).
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterable
from typing import Any

from mcp import Client, StdioServerParameters

from .tools import Tool, ToolFailure, ToolRegistry


def stdio(
    command: str, *args: str, env: dict[str, str] | None = None, cwd: str | None = None
) -> StdioServerParameters:
    return StdioServerParameters(command=command, args=list(args), env=env, cwd=cwd)


class McpError(RuntimeError):
    """The server could not be started or the connection failed (not a tool-level error)."""


def _text(result: Any) -> str:
    """Flatten a CallToolResult/ReadResourceResult into text the model can read."""
    parts = []
    for block in getattr(result, "content", None) or []:
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
        else:
            parts.append(f"[{getattr(block, 'type', 'content')} content omitted]")
    if not parts and getattr(result, "structured_content", None) is not None:
        parts.append(json.dumps(result.structured_content, ensure_ascii=False))
    return "\n".join(parts)


class McpBridge:
    def __init__(
        self, server: Any, *, startup_timeout_s: float = 30.0, call_timeout_s: float = 60.0
    ):
        self.server = server
        self.startup_timeout_s = startup_timeout_s
        self.call_timeout_s = call_timeout_s
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: Client | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._tools_cache: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> McpBridge:
        if self._thread is not None:
            raise McpError("bridge already started")
        self._thread = threading.Thread(target=self._run, name="mcp-bridge", daemon=True)
        self._thread.start()
        if not self._ready.wait(self.startup_timeout_s):
            raise McpError(f"MCP server did not start within {self.startup_timeout_s:g}s")
        if self._error is not None:
            raise McpError(
                f"MCP server failed to start: {type(self._error).__name__}: {self._error}"
            ) from self._error
        return self

    def _run(self) -> None:
        async def main() -> None:
            self._stop = asyncio.Event()
            self._loop = asyncio.get_running_loop()
            try:
                async with Client(self.server) as client:
                    self._client = client
                    self._ready.set()
                    await self._stop.wait()
            except BaseException as exc:  # startup failure, or the connection dying later
                self._error = exc
                self._ready.set()

        asyncio.run(main())

    def close(self) -> None:
        if self._loop is not None and self._stop is not None and self._thread is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(timeout=10)
        self._thread = None
        self._client = None

    def __enter__(self) -> McpBridge:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _call(self, coro_factory) -> Any:
        if self._client is None or self._loop is None:
            raise McpError(
                "bridge is not running (use `with McpBridge(...) as mcp:` or call start())"
            )
        fut = asyncio.run_coroutine_threadsafe(coro_factory(self._client), self._loop)
        try:
            return fut.result(timeout=self.call_timeout_s)
        except TimeoutError:
            fut.cancel()
            raise McpError(f"MCP call timed out after {self.call_timeout_s:g}s") from None

    # ------------------------------------------------------------------ MCP operations

    def list_tools(self) -> list[dict[str, Any]]:
        """Tool specs in our format: {name, description, parameters}."""
        if self._tools_cache is None:
            res = self._call(lambda c: c.list_tools())
            self._tools_cache = [
                {
                    "name": t.name,
                    "description": t.description or t.name,
                    "parameters": t.input_schema,
                }
                for t in res.tools
            ]
        return self._tools_cache

    def call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Run a tool. A tool-level error (``is_error``) raises ToolFailure with the server's message."""
        res = self._call(lambda c: c.call_tool(name, args))
        text = _text(res)
        if res.is_error:
            raise ToolFailure(text or f"tool {name} failed")
        return text

    def read_resource(self, uri: str) -> str:
        res = self._call(lambda c: c.read_resource(uri))
        return "\n".join(getattr(c, "text", "") or "[binary content omitted]" for c in res.contents)

    def list_resources(self) -> list[str]:
        return [str(r.uri) for r in self._call(lambda c: c.list_resources()).resources]

    def list_resource_templates(self) -> list[str]:
        return [
            t.uri_template
            for t in self._call(lambda c: c.list_resource_templates()).resource_templates
        ]

    def list_prompts(self) -> list[str]:
        return [p.name for p in self._call(lambda c: c.list_prompts()).prompts]

    def get_prompt(self, name: str, args: dict[str, str] | None = None) -> str:
        res = self._call(lambda c: c.get_prompt(name, args or {}))
        return "\n".join(f"{m.role}: {getattr(m.content, 'text', '')}" for m in res.messages)

    # ------------------------------------------------------------------ into the agent loop

    def registry(
        self,
        *,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] = (),
        prefix: str = "",
        timeout_s: float = 60.0,
    ) -> ToolRegistry:
        """A ToolRegistry whose tools forward to the server. ``allow`` (if given) is an allowlist."""
        allowed = set(allow) if allow is not None else None
        denied = set(deny)
        tools = []
        for spec in self.list_tools():
            name = spec["name"]
            if (allowed is not None and name not in allowed) or name in denied:
                continue
            tools.append(
                Tool(
                    prefix + name,
                    spec["description"],
                    None,
                    None,
                    spec["parameters"],
                    timeout_s=timeout_s,
                    raw_fn=lambda args, _n=name: self.call_tool(_n, args),
                )
            )
        if allowed is not None and (missing := allowed - {s["name"] for s in self.list_tools()}):
            raise McpError(f"allowed tools not offered by the server: {sorted(missing)}")
        return ToolRegistry(tools)
