"""Tracing for the agent stack: OpenTelemetry spans that follow the GenAI semantic conventions.

    from common import tracing

    with tracing.capture() as rec:                  # in-memory, for tests and notebooks
        support.handle("c1", "I was charged twice for INV-3001")
    print(tracing.render_tree(rec.spans))           # invoke_workflow > triage > chat ... > invoke_agent > chat / execute_tool
    tracing.summarize(rec.spans)                    # model calls, tokens, cost, tool calls, errors, per agent

    with tracing.session(tracing.JsonlSpanExporter("out/spans.jsonl")):     # production-style: ship spans somewhere
        ...

What it records (names are the OpenTelemetry GenAI conventions, which are still "development" status: pin your
backend's version and expect renames; this repo's tests compare the strings with the installed semconv package):
  * ``chat {model}``              one model call: provider, model, max tokens, token usage, finish reason, cost
  * ``execute_tool {name}``       one tool call: name, call id, ok / error
  * ``invoke_agent {name}``       one agent loop; ``invoke_workflow {name}`` one conversation turn (the app's root span)

Design rules (each is tested):
  * NO-OP unless you turn it on: without a tracer provider the API does nothing, and ``span()`` never raises.
  * PRIVACY BY DEFAULT: prompts, replies, tool arguments and tool results are NOT recorded unless
    ``capture_content=True``, and even then they pass through your ``redact`` function and a length cap. Span
    exception messages and tool error text are also withheld unless content capture is on (they quote user data).
  * Context survives threads: tool calls run in worker threads (Week 5), so ``common.tools`` copies the caller's
    context into them; otherwise tool spans would be orphans with no parent.
  * Cost and agent status go under the ``app.*`` namespace, because the conventions define no cost attribute.
"""

from __future__ import annotations

import functools
import hashlib
import hmac
import json
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import chat, llm, tools

try:  # the API is tiny and may be missing; everything below degrades to a no-op
    from opentelemetry import trace as _trace
    from opentelemetry.trace import SpanKind, Status, StatusCode

    AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where opentelemetry is absent
    AVAILABLE = False

# ----------------------------------------------------------------------------- attribute names (GenAI conventions)
OPERATION = "gen_ai.operation.name"
PROVIDER = "gen_ai.provider.name"
AGENT_NAME = "gen_ai.agent.name"
WORKFLOW_NAME = "gen_ai.workflow.name"
CONVERSATION_ID = "gen_ai.conversation.id"
REQUEST_MODEL = "gen_ai.request.model"
REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
RESPONSE_MODEL = "gen_ai.response.model"
FINISH_REASONS = "gen_ai.response.finish_reasons"
OUTPUT_TYPE = "gen_ai.output.type"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
CACHE_READ_TOKENS = "gen_ai.usage.cache_read.input_tokens"
CACHE_WRITE_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOOL_TYPE = "gen_ai.tool.type"
TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"
TOOL_RESULT = "gen_ai.tool.call.result"
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"
ERROR_TYPE = "error.type"
# not in the conventions: ours
COST = "app.cost_usd"
STEPS = "app.agent.steps"
AGENT_STATUS = "app.agent.status"
TOOL_CALLS_REQUESTED = "app.tool_calls_requested"
TOOL_ARGS_DIGEST = "app.tool.args_digest"

GENAI_ATTRIBUTES = {
    k: v
    for k, v in globals().items()
    if isinstance(v, str) and v.startswith("gen_ai.") and k.isupper()
}


# ----------------------------------------------------------------------------- configuration
@dataclass
class _Config:
    capture_content: bool = False
    redact: Callable[[str], str] = lambda s: s  # noqa: E731
    max_content_chars: int = 8000
    provider: Any = None  # a TracerProvider; None = the global one


_cfg = _Config()
_DIGEST_KEY = os.urandom(
    16
)  # per process: digests compare within one process's traces, and cannot be dictionary-attacked
_lock = threading.Lock()
_installed = False


class _NoSpan:
    """Stands in for a span when OpenTelemetry is missing."""

    def set_attribute(self, *a, **k):  # noqa: D102
        pass

    def set_attributes(self, *a, **k):  # noqa: D102
        pass

    def set_status(self, *a, **k):  # noqa: D102
        pass

    def add_event(self, *a, **k):  # noqa: D102
        pass

    def is_recording(self) -> bool:  # noqa: D102
        return False


def _tracer():
    provider = _cfg.provider or _trace.get_tracer_provider()
    return provider.get_tracer("ai-refreshment")


def _clean(attrs: dict[str, Any] | None) -> dict[str, Any]:
    """OpenTelemetry attributes cannot be None (the SDK warns and drops them)."""
    return {k: v for k, v in (attrs or {}).items() if v is not None}


@contextmanager
def span(
    name: str, attributes: dict[str, Any] | None = None, *, kind: str = "internal"
) -> Iterator[Any]:
    """Open a span that is the current one inside the block. Exceptions mark it ERROR and propagate.

    The error status carries only the exception CLASS: messages quote user data, so they are recorded only
    when content capture is on."""
    if not AVAILABLE:
        yield _NoSpan()
        return
    kinds = {"internal": SpanKind.INTERNAL, "client": SpanKind.CLIENT, "server": SpanKind.SERVER}
    with _tracer().start_as_current_span(
        name,
        kind=kinds[kind],
        attributes=_clean(attributes),
        record_exception=False,
        set_status_on_exception=False,
    ) as sp:
        try:
            yield sp
        except Exception as exc:
            sp.set_attribute(ERROR_TYPE, type(exc).__qualname__)
            sp.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            if (
                _cfg.capture_content
            ):  # not record_exception(): it would copy the message past the redactor
                sp.add_event(
                    "exception",
                    {
                        "exception.type": type(exc).__qualname__,
                        "exception.message": _content(str(exc)),
                    },
                )
            raise


def current_trace_id() -> str | None:
    """The trace id of the span we are inside (32 hex characters), or None when tracing is off. Store it next to a user
    action (feedback, a complaint, an escalation) and a person can jump from the action to the whole trace."""
    if not AVAILABLE:
        return None
    ctx = _trace.get_current_span().get_span_context()
    return f"{ctx.trace_id:032x}" if ctx.is_valid else None


def set_error(sp: Any, error_type: str, description: str = "") -> None:
    sp.set_attribute(ERROR_TYPE, error_type)
    sp.set_status(Status(StatusCode.ERROR, description) if AVAILABLE else None)


def args_digest(args: Any) -> str:
    """A keyed hash of the tool arguments (canonical JSON). It lets a trace say "the same call was made twice" without
    recording the arguments (an invoice id or a name is low-entropy: a plain hash could be reversed by trying them all)."""
    blob = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False).encode()
    return hmac.new(_DIGEST_KEY, blob, hashlib.sha256).hexdigest()[:12]


def _content(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    text = _cfg.redact(text)
    cap = _cfg.max_content_chars
    return text if len(text) <= cap else text[:cap] + f"...[truncated {len(text) - cap} chars]"


def _record_usage(sp: Any, model: str, usage: llm.Usage, cost: float, finish: str | None) -> None:
    """Token counts. The conventions define ``input_tokens`` as ALL input tokens (cached ones included), while
    this repo's ``Usage.input_tokens`` excludes cache reads/writes: add them back, and report the cached part too."""
    sp.set_attribute(RESPONSE_MODEL, model)
    sp.set_attribute(
        INPUT_TOKENS, usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
    )
    sp.set_attribute(OUTPUT_TOKENS, usage.output_tokens)
    if usage.cache_read_tokens:
        sp.set_attribute(CACHE_READ_TOKENS, usage.cache_read_tokens)
    if usage.cache_write_tokens:
        sp.set_attribute(CACHE_WRITE_TOKENS, usage.cache_write_tokens)
    sp.set_attribute(COST, float(cost))
    if finish:
        sp.set_attribute(FINISH_REASONS, [finish])


# ----------------------------------------------------------------------------- wrappers installed by instrument()
def _wrap_turn(orig):
    @functools.wraps(orig)
    def turn(
        messages, tools=None, *, system=None, provider=None, model=None, max_tokens=4096, **kw
    ):
        prov, mod = llm.resolve(provider, model)
        attrs = {
            OPERATION: "chat",
            PROVIDER: prov,
            REQUEST_MODEL: mod,
            REQUEST_MAX_TOKENS: max_tokens,
        }
        with span(f"chat {mod}", attrs, kind="client") as sp:
            if _cfg.capture_content:
                sp.set_attribute(INPUT_MESSAGES, _content({"system": system, "messages": messages}))
            t = orig(
                messages,
                tools,
                system=system,
                provider=provider,
                model=model,
                max_tokens=max_tokens,
                **kw,
            )
            _record_usage(sp, t.model, t.usage, t.cost_usd, t.stop_reason)
            sp.set_attribute(TOOL_CALLS_REQUESTED, len(t.tool_calls))
            if _cfg.capture_content:
                sp.set_attribute(
                    OUTPUT_MESSAGES,
                    _content({"text": t.text, "tool_calls": [c.as_dict() for c in t.tool_calls]}),
                )
            return t

    return turn


def _wrap_complete(orig, *, structured: bool):
    @functools.wraps(orig)
    def call(messages, *args, system=None, provider=None, model=None, **kw):
        prov, mod = llm.resolve(provider, model)
        attrs = {OPERATION: "chat", PROVIDER: prov, REQUEST_MODEL: mod}
        if structured:
            attrs[OUTPUT_TYPE] = "json"
        with span(f"chat {mod}", attrs, kind="client") as sp:
            if _cfg.capture_content:
                sp.set_attribute(INPUT_MESSAGES, _content({"system": system, "messages": messages}))
            out = orig(messages, *args, system=system, provider=provider, model=model, **kw)
            resp = out[1] if structured else out
            _record_usage(sp, resp.model, resp.usage, resp.cost_usd, resp.stop_reason)
            if _cfg.capture_content:
                sp.set_attribute(OUTPUT_MESSAGES, _content(resp.text))
            return out

    return call


def _wrap_execute(orig):
    @functools.wraps(orig)
    def execute(self, call):
        attrs = {
            OPERATION: "execute_tool",
            TOOL_NAME: call.name,
            TOOL_CALL_ID: call.id,
            TOOL_TYPE: "function",
            TOOL_ARGS_DIGEST: args_digest(call.args),
        }
        with span(f"execute_tool {call.name}", attrs) as sp:
            if _cfg.capture_content:
                sp.set_attribute(TOOL_ARGUMENTS, _content(call.args))
            res = orig(self, call)
            if res.is_error:  # the loop survives tool errors (they are returned as text), so the span must show them
                set_error(
                    sp,
                    "tool_error",
                    _content(res.content) if _cfg.capture_content else "tool returned an error",
                )
            if _cfg.capture_content:
                sp.set_attribute(TOOL_RESULT, _content(res.content))
            return res

    return execute


@contextmanager
def instrument(
    *,
    capture_content: bool = False,
    redact: Callable[[str], str] | None = None,
    max_content_chars: int = 8000,
    tracer_provider: Any = None,
) -> Iterator[None]:
    """Patch ``chat.turn``, ``llm.complete``, ``llm.structured`` and ``ToolRegistry.execute`` to emit spans.

    Install it INSIDE any ``fake_llm`` block (it wraps whatever is there when it starts). It refuses to nest:
    two layers would emit every span twice."""
    global _installed, _cfg
    with _lock:
        if _installed:
            raise RuntimeError("tracing.instrument() is already active")
        _installed = True
    previous = _cfg
    _cfg = _Config(capture_content, redact or (lambda s: s), max_content_chars, tracer_provider)
    patches = [
        (chat, "turn", _wrap_turn(chat.turn)),
        (llm, "complete", _wrap_complete(llm.complete, structured=False)),
        (llm, "structured", _wrap_complete(llm.structured, structured=True)),
        (tools.ToolRegistry, "execute", _wrap_execute(tools.ToolRegistry.execute)),
    ]
    originals = [(obj, name, getattr(obj, name)) for obj, name, _ in patches]
    for obj, name, new in patches:
        setattr(obj, name, new)
    try:
        yield
    finally:
        for (obj, name, orig), (_, _, new) in zip(originals, patches, strict=True):
            if (
                getattr(obj, name) is new
            ):  # someone patched on top of us (a fake that exited first): do not clobber it
                setattr(obj, name, orig)
        _cfg = previous
        with _lock:
            _installed = False


# ----------------------------------------------------------------------------- finished spans as plain data
@dataclass
class SpanRecord:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start_ns: int
    end_ns: int
    status: str  # unset | ok | error
    status_description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6

    def get(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)

    @property
    def is_error(self) -> bool:
        return self.status == "error"


def to_record(s: Any) -> SpanRecord:
    """An SDK ``ReadableSpan`` -> ``SpanRecord`` (JSON-friendly, independent of the SDK)."""
    ctx, parent = s.get_span_context(), s.parent
    attrs = {k: list(v) if isinstance(v, tuple) else v for k, v in (s.attributes or {}).items()}
    return SpanRecord(
        f"{ctx.trace_id:032x}",
        f"{ctx.span_id:016x}",
        f"{parent.span_id:016x}" if parent else None,
        s.name,
        s.start_time,
        s.end_time,
        s.status.status_code.name.lower(),
        s.status.description or "",
        attrs,
        [{"name": e.name, "attributes": dict(e.attributes or {})} for e in s.events],
    )


# ----------------------------------------------------------------------------- exporters
def JsonlSpanExporter(path: str | Path):  # noqa: N802 - a class, built lazily so the SDK import stays optional
    """Append every finished span to a JSON-lines file (one ``SpanRecord`` per line): cheap, greppable, replayable."""
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class _Jsonl(SpanExporter):
        def __init__(self, p):
            self.path = Path(p)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._lock = threading.Lock()

        def export(self, spans):
            lines = "".join(json.dumps(asdict(to_record(s)), default=str) + "\n" for s in spans)
            with self._lock, self.path.open("a") as f:
                f.write(lines)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    return _Jsonl(path)


def load_spans(path: str | Path) -> list[SpanRecord]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [SpanRecord(**r) for r in rows]


def TailSamplingExporter(  # noqa: N802
    inner: Any,
    *,
    keep_fraction: float = 0.1,
    slow_ms: float | None = None,
    keep_errors: bool = True,
    max_buffered_traces: int = 1000,
):
    """Decide per TRACE, after it finished: keep every trace that had an error or was slower than ``slow_ms``, and a
    ``keep_fraction`` of the rest. (Head sampling decides at the first span, before anyone knows the trace is
    interesting, so it throws away exactly the failures you want to read.) Spans are buffered per trace until the
    root span ends. The random part is a function of the trace id, so every service in a system keeps the same traces.
    Production systems usually do this in the collector; this is the same logic, in-process, for one service."""
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class _Tail(SpanExporter):
        def __init__(self):
            self.inner, self.buffer = inner, {}
            self.kept = self.dropped = self.evicted = 0
            self._lock = threading.Lock()

        def _keep(self, spans) -> bool:
            if keep_errors and any(s.status.status_code.name == "ERROR" for s in spans):
                return True
            root = next((s for s in spans if s.parent is None), None)
            if (
                slow_ms is not None
                and root is not None
                and (root.end_time - root.start_time) / 1e6 >= slow_ms
            ):
                return True
            return (spans[0].get_span_context().trace_id % 10_000) < keep_fraction * 10_000

        def export(self, spans):
            with self._lock:
                for s in spans:
                    tid = s.get_span_context().trace_id
                    if tid not in self.buffer and len(self.buffer) >= max_buffered_traces:
                        del self.buffer[
                            next(iter(self.buffer))
                        ]  # bounded memory: evict the oldest unfinished trace
                        self.evicted += 1
                    self.buffer.setdefault(tid, []).append(s)
                    if s.parent is None:  # the root ended: the trace is complete
                        done = self.buffer.pop(tid)
                        if self._keep(done):
                            self.kept += 1
                            self.inner.export(done)
                        else:
                            self.dropped += 1
            return SpanExportResult.SUCCESS

        def shutdown(self):
            self.buffer.clear()  # unfinished traces are not exported
            self.inner.shutdown()

    return _Tail()


def otlp_exporter(
    endpoint: str = "localhost:4317",
    *,
    insecure: bool = True,
    headers: dict[str, str] | None = None,
):
    """OTLP over gRPC: what Phoenix, an OpenTelemetry Collector, Jaeger and most vendors accept."""
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(endpoint=endpoint, insecure=insecure, headers=headers)


# ----------------------------------------------------------------------------- sessions
@contextmanager
def session(
    *exporters: Any,
    service_name: str = "ai-refreshment",
    sync: bool = True,
    **instrument_kwargs: Any,
) -> Iterator[Any]:
    """Create a tracer provider that feeds ``exporters`` and instrument the model/tool entry points.
    ``sync=True`` exports each span as it ends (tests, debugging); ``sync=False`` batches in the background (production)."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    for e in exporters:
        provider.add_span_processor(SimpleSpanProcessor(e) if sync else BatchSpanProcessor(e))
    try:
        with instrument(tracer_provider=provider, **instrument_kwargs):
            yield provider
    finally:
        provider.force_flush()
        provider.shutdown()


class Recorder:
    def __init__(self, exporter: Any):
        self._exporter = exporter

    @property
    def spans(self) -> list[SpanRecord]:
        return [to_record(s) for s in self._exporter.get_finished_spans()]

    def named(self, prefix: str) -> list[SpanRecord]:
        return [s for s in self.spans if s.name.startswith(prefix)]


@contextmanager
def capture(**instrument_kwargs: Any) -> Iterator[Recorder]:
    """Record spans in memory: ``with capture() as rec: ...; rec.spans``."""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    with session(exporter, **instrument_kwargs):
        yield Recorder(exporter)


# ----------------------------------------------------------------------------- reading traces
def children(spans: Sequence[SpanRecord]) -> dict[str | None, list[SpanRecord]]:
    by_parent: dict[str | None, list[SpanRecord]] = {}
    for s in sorted(spans, key=lambda s: (s.start_ns, s.end_ns)):
        by_parent.setdefault(s.parent_id, []).append(s)
    return by_parent


def roots(spans: Sequence[SpanRecord]) -> list[SpanRecord]:
    """Spans with no parent IN THIS SET. More than one root in a single trace id means a broken context."""
    ids = {s.span_id for s in spans}
    return [s for s in spans if s.parent_id is None or s.parent_id not in ids]


def render_tree(spans: Sequence[SpanRecord]) -> str:
    kids = children(spans)
    ids = {s.span_id for s in spans}
    lines: list[str] = []

    def label(s: SpanRecord) -> str:
        bits = [f"{s.duration_ms:.0f}ms"]
        if INPUT_TOKENS in s.attributes:
            bits.append(f"{s.get(INPUT_TOKENS)} in/{s.get(OUTPUT_TOKENS, 0)} out tok")
        if s.get(COST):
            bits.append(f"${s.get(COST):.4f}")
        if s.get(AGENT_STATUS):
            bits.append(f"status={s.get(AGENT_STATUS)}")
        if s.is_error:
            bits.append(f"ERROR {s.get(ERROR_TYPE, '')}".strip())
        return f"{s.name}  [{', '.join(bits)}]"

    def walk(s: SpanRecord, prefix: str, last: bool, top: bool) -> None:
        lines.append(label(s) if top else f"{prefix}{'└─ ' if last else '├─ '}{label(s)}")
        pad = "" if top else prefix + ("   " if last else "│  ")
        sub = kids.get(s.span_id, [])
        for i, c in enumerate(sub):
            walk(c, pad, i == len(sub) - 1, False)

    for r in sorted(
        (s for s in spans if s.parent_id is None or s.parent_id not in ids),
        key=lambda s: s.start_ns,
    ):
        walk(r, "", True, True)
    return "\n".join(lines)


def summarize(spans: Sequence[SpanRecord]) -> dict[str, Any]:
    """Totals and a per-agent breakdown; each model/tool span is attributed to its nearest ``invoke_agent`` ancestor
    (model calls outside any agent, like the triage classifier, are attributed to ``(no agent)``)."""
    by_id = {s.span_id: s for s in spans}

    def agent_of(s: SpanRecord) -> str:
        p = by_id.get(s.parent_id) if s.parent_id else None
        while p is not None:
            if p.get(OPERATION) == "invoke_agent":
                return p.get(AGENT_NAME, p.name)
            p = by_id.get(p.parent_id) if p.parent_id else None
        return "(no agent)"

    out: dict[str, Any] = {
        "spans": len(spans),
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "tool_calls": 0,
        "tool_errors": 0,
        "errors": sum(1 for s in spans if s.is_error),
        "by_agent": {},
        "by_tool": {},
    }
    for s in spans:
        op = s.get(OPERATION)
        if op == "chat":
            a = out["by_agent"].setdefault(
                agent_of(s),
                {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            )
            out["model_calls"] += 1
            a["model_calls"] += 1
            for key, attr in (("input_tokens", INPUT_TOKENS), ("output_tokens", OUTPUT_TOKENS)):
                out[key] += s.get(attr, 0)
                a[key] += s.get(attr, 0)
            out["cost_usd"] += s.get(COST, 0.0)
            a["cost_usd"] += s.get(COST, 0.0)
        elif op == "execute_tool":
            out["tool_calls"] += 1
            out["tool_errors"] += int(s.is_error)
            t = out["by_tool"].setdefault(
                s.get(TOOL_NAME, "?"), {"calls": 0, "errors": 0, "ms": 0.0}
            )
            t["calls"] += 1
            t["errors"] += int(s.is_error)
            t["ms"] += s.duration_ms
    roots_ = roots(spans)
    out["duration_ms"] = max((r.duration_ms for r in roots_), default=0.0)
    return out


def tool_sequence(spans: Sequence[SpanRecord]) -> list[str]:
    """Tool names in the order they STARTED: a trace-based assertion ("lookup before refund")."""
    return [
        s.get(TOOL_NAME)
        for s in sorted(spans, key=lambda s: s.start_ns)
        if s.get(OPERATION) == "execute_tool"
    ]


def slowest(spans: Sequence[SpanRecord], n: int = 3, *, leaf_only: bool = True) -> list[SpanRecord]:
    """Where the time went: the slowest spans (leaves by default, since a parent always contains its children)."""
    parents = {s.parent_id for s in spans}
    pool = [s for s in spans if not leaf_only or s.span_id not in parents]
    return sorted(pool, key=lambda s: s.duration_ms, reverse=True)[:n]


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0, interval: float = 0.02) -> bool:
    """Poll for an asynchronous export (batch processors, network collectors) instead of sleeping a fixed time."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
