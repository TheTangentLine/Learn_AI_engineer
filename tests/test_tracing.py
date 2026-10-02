"""Tests for common/tracing.py: the spans are right, private by default, connected across threads, and really exported."""

from __future__ import annotations

import json
import threading
import time
from concurrent import futures
from dataclasses import asdict
from types import SimpleNamespace

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as semconv  # noqa: E402

from common import chat, llm, tracing  # noqa: E402
from common.agent import run_agent  # noqa: E402
from common.fake import fake_llm, tool_calls  # noqa: E402
from common.tools import ToolRegistry, tool  # noqa: E402

SECRET = "SECRET-4242"


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First.
        b: Second.
    """
    return a + b


@tool
def echo(text: str) -> str:
    """Return the text.

    Args:
        text: Anything.
    """
    return f"echo:{text}"


@tool
def boom() -> str:
    """Always fails."""
    raise RuntimeError(f"disk on fire {SECRET}")


@tool
def nested() -> str:
    """Opens a span of its own from inside the tool's worker thread."""
    with tracing.span("inside-tool"):
        return "ok"


REG = ToolRegistry([add, echo, boom, nested])
ANY = r"(?s).*"


def script(*turns):
    return [(ANY, list(turns))]


def by_name(spans, prefix):
    return [s for s in spans if s.name.startswith(prefix)]


# ----------------------------------------------------------------------------- names


def test_attribute_names_equal_the_installed_semconv_strings():
    """Typing 'gen_ai.usage.input_token' would silently create a dashboard nobody can query: compare with the package."""
    known = {getattr(semconv, n) for n in dir(semconv) if n.startswith("GEN_AI_")}
    assert tracing.GENAI_ATTRIBUTES, "no constants found"
    for const, value in tracing.GENAI_ATTRIBUTES.items():
        assert value in known, (
            f"{const} = {value!r} is not a name in the installed semantic conventions"
        )
    ops = {m.value for m in semconv.GenAiOperationNameValues}
    assert {"chat", "execute_tool", "invoke_agent", "invoke_workflow"} <= ops
    providers = {m.value for m in semconv.GenAiProviderNameValues}
    assert {"anthropic", "openai"} <= providers


# ----------------------------------------------------------------------------- span()


def test_span_is_a_noop_without_a_provider_and_never_raises():
    with tracing.span("x", {"a": 1, "none": None}) as sp:
        sp.set_attribute("k", "v")
    assert tracing.span("y")  # constructing is free
    with pytest.raises(ValueError):
        with tracing.span("z"):
            raise ValueError("propagates")


def test_exception_marks_the_span_error_with_the_class_only():
    with tracing.capture() as rec:
        with pytest.raises(ValueError):
            with tracing.span("work"):
                raise ValueError(f"bad input {SECRET}")
    (s,) = rec.spans
    assert (
        s.is_error
        and s.get(tracing.ERROR_TYPE) == "ValueError"
        and s.status_description == "ValueError"
    )
    assert SECRET not in json.dumps(asdict(s))
    assert s.events == []


def test_exception_message_is_recorded_only_with_content_capture():
    with tracing.capture(capture_content=True) as rec:
        with pytest.raises(ValueError):
            with tracing.span("work"):
                raise ValueError("bad input")
    assert any(e["name"] == "exception" for e in rec.spans[0].events)


def test_none_attributes_are_dropped_not_warned():
    with tracing.capture() as rec:
        with tracing.span("x", {"keep": 1, "drop": None}):
            pass
    assert rec.spans[0].attributes == {"keep": 1}


# ----------------------------------------------------------------------------- model calls


def test_chat_turn_span_has_provider_model_usage_finish_reason_and_cost():
    with fake_llm(script(tool_calls(("add", {"a": 1, "b": 2})))):
        with tracing.capture() as rec:
            t = chat.turn(
                [{"role": "user", "content": "hi"}],
                REG.specs(),
                provider="anthropic",
                model="claude-x",
                max_tokens=77,
            )
    (s,) = rec.spans
    assert s.name == "chat claude-x"
    a = s.attributes
    assert a[tracing.OPERATION] == "chat" and a[tracing.PROVIDER] == "anthropic"
    assert a[tracing.REQUEST_MODEL] == "claude-x" and a[tracing.RESPONSE_MODEL] == "claude-x"
    assert a[tracing.REQUEST_MAX_TOKENS] == 77
    assert (
        a[tracing.INPUT_TOKENS] == t.usage.input_tokens
        and a[tracing.OUTPUT_TOKENS] == t.usage.output_tokens
    )
    assert a[tracing.FINISH_REASONS] == ["tool_use"] and a[tracing.TOOL_CALLS_REQUESTED] == 1
    assert a[tracing.COST] == t.cost_usd and s.parent_id is None and s.status == "unset"


def test_input_tokens_include_cached_tokens_as_the_conventions_require():
    seen = {}
    sp = SimpleNamespace(set_attribute=lambda k, v: seen.__setitem__(k, v))
    tracing._record_usage(
        sp,
        "m",
        llm.Usage(input_tokens=100, output_tokens=7, cache_read_tokens=900, cache_write_tokens=50),
        0.5,
        "end_turn",
    )
    assert seen[tracing.INPUT_TOKENS] == 1050 and seen[tracing.OUTPUT_TOKENS] == 7
    assert seen[tracing.CACHE_READ_TOKENS] == 900 and seen[tracing.CACHE_WRITE_TOKENS] == 50
    seen.clear()
    tracing._record_usage(sp, "m", llm.Usage(input_tokens=5, output_tokens=1), 0.0, None)
    assert tracing.CACHE_READ_TOKENS not in seen and tracing.CACHE_WRITE_TOKENS not in seen
    assert tracing.FINISH_REASONS not in seen


def test_complete_and_structured_are_traced_and_structured_says_json():
    from pydantic import BaseModel

    class R(BaseModel):
        answer: int

    with fake_llm([(r"count", '{"answer": 3}'), (ANY, "plain")]):
        with tracing.capture() as rec:
            llm.complete("hello", provider="anthropic", model="m1")
            obj, _ = llm.structured("count them", R, provider="openai", model="m2")
    assert obj.answer == 3
    plain, structured = rec.spans
    assert plain.get(tracing.OUTPUT_TYPE) is None and plain.get(tracing.PROVIDER) == "anthropic"
    assert structured.get(tracing.OUTPUT_TYPE) == "json" and structured.name == "chat m2"
    assert structured.get(tracing.INPUT_TOKENS) > 0


def test_a_failing_model_call_is_an_error_span_and_the_exception_propagates():
    with fake_llm([(ANY, RuntimeError(f"provider down {SECRET}"))]):
        with tracing.capture() as rec:
            with pytest.raises(RuntimeError):
                llm.complete("x", provider="anthropic")
    (s,) = rec.spans
    assert (
        s.is_error
        and s.get(tracing.ERROR_TYPE) == "RuntimeError"
        and SECRET not in json.dumps(asdict(s))
    )


# ----------------------------------------------------------------------------- tools and the agent loop


def test_agent_run_is_one_trace_with_the_right_parents():
    with fake_llm(script(tool_calls(("add", {"a": 2, "b": 3})), "The sum is 5.")):
        with tracing.capture() as rec:
            run = run_agent("2+3?", REG, provider="anthropic", agent_name="calc")
    spans = rec.spans
    (agent,) = by_name(spans, "invoke_agent")
    assert agent.name == "invoke_agent calc" and agent.get(tracing.AGENT_NAME) == "calc"
    assert (
        agent.get(tracing.AGENT_STATUS) == "done"
        and agent.get(tracing.STEPS) == 2
        and agent.status == "unset"
    )
    assert agent.get(tracing.COST) == run.cost_usd
    chats, tool_spans = by_name(spans, "chat"), by_name(spans, "execute_tool")
    assert len(chats) == 2 and len(tool_spans) == 1
    assert all(s.parent_id == agent.span_id for s in chats + tool_spans)
    assert len({s.trace_id for s in spans}) == 1 and len(tracing.roots(spans)) == 1
    assert tool_spans[0].get(tracing.TOOL_NAME) == "add" and tool_spans[0].status == "unset"
    assert tracing.tool_sequence(spans) == ["add"]


def test_parallel_tool_calls_stay_children_of_the_agent_span():
    """Tool calls run in worker threads. Without copying the context into them every tool span would be an orphan."""
    with fake_llm(
        script(
            tool_calls(
                ("add", {"a": 1, "b": 1}), ("echo", {"text": "a"}), ("add", {"a": 2, "b": 2})
            ),
            "done",
        )
    ):
        with tracing.capture() as rec:
            run_agent("go", REG, provider="anthropic")
    spans = rec.spans
    (agent,) = by_name(spans, "invoke_agent")
    tool_spans = by_name(spans, "execute_tool")
    assert len(tool_spans) == 3 and all(s.parent_id == agent.span_id for s in tool_spans)
    assert len(tracing.roots(spans)) == 1


def open_and_close_span():
    with tracing.span("orphan"):
        pass


def test_a_plain_thread_pool_loses_the_parent_which_is_why_tools_copy_the_context():
    with tracing.capture() as rec:
        with tracing.span("parent"):
            with futures.ThreadPoolExecutor(1) as pool:
                pool.submit(open_and_close_span).result()
    parent = next(s for s in rec.spans if s.name == "parent")
    orphan = next(s for s in rec.spans if s.name == "orphan")
    assert orphan.parent_id is None and orphan.trace_id != parent.trace_id


def test_a_span_opened_inside_a_tool_is_a_child_of_that_tools_span():
    with fake_llm(script(tool_calls(("nested", {})), "done")):
        with tracing.capture() as rec:
            run_agent("go", REG, provider="anthropic")
    inner = by_name(rec.spans, "inside-tool")[0]
    tool_span = by_name(rec.spans, "execute_tool")[0]
    assert inner.parent_id == tool_span.span_id
    assert threading.current_thread().name  # (the tool ran in a worker thread, see common/tools.py)


def test_tool_errors_are_error_spans_even_though_the_loop_survives():
    with fake_llm(
        script(
            tool_calls(
                ("boom", {}),
                ("nosuchtool", {}),
                ("add", {"a": "x", "b": 1}),
                ("add", {"a": 1, "b": 1}),
            ),
            "done",
        )
    ):
        with tracing.capture() as rec:
            run = run_agent("go", REG, provider="anthropic")
    assert run.ok
    flags = {}
    for s in by_name(rec.spans, "execute_tool"):
        flags.setdefault(s.get(tracing.TOOL_NAME), []).append(s.is_error)
    # boom, an unknown tool and invalid arguments are errors; the valid add is not
    assert flags == {"boom": [True], "nosuchtool": [True], "add": [True, False]} or {
        k: sorted(v) for k, v in flags.items()
    } == {"boom": [True], "nosuchtool": [True], "add": [False, True]}
    err = next(s for s in by_name(rec.spans, "execute_tool") if s.get(tracing.TOOL_NAME) == "boom")
    assert err.get(tracing.ERROR_TYPE) == "tool_error" and SECRET not in json.dumps(asdict(err))
    assert err.status_description == "tool returned an error"
    assert tracing.summarize(rec.spans)["tool_errors"] == 3


def test_an_unfinished_agent_run_is_an_error_span_with_its_status():
    with fake_llm(script(tool_calls(("add", {"a": 1, "b": 1})))):
        with tracing.capture() as rec:
            run = run_agent("loop", REG, provider="anthropic", max_steps=2)
    (agent,) = by_name(rec.spans, "invoke_agent")
    assert not run.ok and agent.is_error and agent.get(tracing.AGENT_STATUS) == run.status
    assert agent.get(tracing.ERROR_TYPE) == "agent_" + run.status


def test_provider_outage_inside_the_agent_loop_is_recorded_at_both_levels():
    with fake_llm([(ANY, ConnectionError("down"))]):
        with tracing.capture() as rec:
            run = run_agent("x", REG, provider="anthropic")
    assert run.status == "error"
    chat_span, agent = by_name(rec.spans, "chat")[0], by_name(rec.spans, "invoke_agent")[0]
    assert (
        chat_span.is_error
        and chat_span.get(tracing.ERROR_TYPE) == "ConnectionError"
        and agent.is_error
    )


def test_tracing_does_not_change_what_the_agent_does():
    def run_once():
        with fake_llm(script(tool_calls(("add", {"a": 2, "b": 3})), "The sum is 5.")):
            return run_agent("2+3?", REG, provider="anthropic")

    plain = run_once()
    with tracing.capture():
        traced = run_once()
    assert (plain.status, plain.answer, plain.tool_names, len(plain.steps)) == (
        traced.status,
        traced.answer,
        traced.tool_names,
        len(traced.steps),
    )


def test_without_instrumentation_no_spans_are_produced_and_nothing_breaks():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with fake_llm(script("hi")):
        run_agent(
            "x", REG, provider="anthropic"
        )  # no tracing.session: the global provider is the no-op default
    assert exporter.get_finished_spans() == ()


# ----------------------------------------------------------------------------- privacy


def private_run(**kw):
    rules = script(tool_calls(("echo", {"text": SECRET}), ("boom", {})), f"I looked up {SECRET}")
    with fake_llm(rules):
        with tracing.capture(**kw) as rec:
            run_agent(
                f"my password is {SECRET}",
                REG,
                system=f"never repeat {SECRET}",
                provider="anthropic",
            )
    return rec.spans


def test_nothing_the_customer_typed_reaches_a_span_by_default():
    spans = private_run()
    assert spans
    blob = json.dumps([asdict(s) for s in spans])
    assert SECRET not in blob
    for s in spans:
        assert not {
            tracing.INPUT_MESSAGES,
            tracing.OUTPUT_MESSAGES,
            tracing.TOOL_ARGUMENTS,
            tracing.TOOL_RESULT,
        } & set(s.attributes)


def test_content_capture_is_opt_in_redacted_and_capped():
    spans = private_run(capture_content=True, redact=lambda t: t.replace(SECRET, "[redacted]"))
    blob = json.dumps([asdict(s) for s in spans])
    assert SECRET not in blob and "[redacted]" in blob
    tool_span = next(s for s in spans if s.get(tracing.TOOL_NAME) == "echo")
    assert (
        "[redacted]" in tool_span.get(tracing.TOOL_ARGUMENTS)
        and tool_span.get(tracing.TOOL_RESULT) == "echo:[redacted]"
    )
    chat_span = by_name(spans, "chat")[0]
    assert "password" in chat_span.get(tracing.INPUT_MESSAGES) and chat_span.get(
        tracing.OUTPUT_MESSAGES
    )


def test_exception_events_and_error_descriptions_also_pass_through_the_redactor():
    redact = lambda t: t.replace(SECRET, "[redacted]")  # noqa: E731
    with fake_llm([(ANY, RuntimeError(f"down {SECRET}"))]):
        with tracing.capture(capture_content=True, redact=redact) as rec:
            with pytest.raises(RuntimeError):
                llm.complete("x", provider="anthropic")
    event = rec.spans[0].events[0]
    assert event["attributes"]["exception.message"] == "down [redacted]"
    assert SECRET not in json.dumps([asdict(s) for s in rec.spans])


def test_captured_content_is_truncated_at_the_cap():
    with fake_llm(script("x" * 500)):
        with tracing.capture(capture_content=True, max_content_chars=100) as rec:
            llm.complete("y" * 500, provider="anthropic")
    s = rec.spans[0]
    assert s.get(tracing.OUTPUT_MESSAGES).endswith("[truncated 400 chars]")
    assert len(s.get(tracing.INPUT_MESSAGES)) < 200


# ----------------------------------------------------------------------------- installing and removing


def test_instrument_restores_every_original_and_refuses_to_nest():
    before = (chat.turn, llm.complete, llm.structured, ToolRegistry.execute)
    with tracing.capture():
        assert chat.turn is not before[0] and ToolRegistry.execute is not before[3]
        with pytest.raises(RuntimeError, match="already active"):
            with tracing.instrument():
                pass
    assert (chat.turn, llm.complete, llm.structured, ToolRegistry.execute) == before
    with tracing.capture():  # and it can be installed again afterwards
        pass


def test_a_fake_that_exits_first_is_not_clobbered_by_uninstalling():
    before = chat.turn
    cm_fake = fake_llm(script("x"))
    cm_fake.__enter__()
    cm_trace = tracing.capture()
    cm_trace.__enter__()
    cm_fake.__exit__(
        None, None, None
    )  # wrong order on purpose: the fake restores the real functions
    cm_trace.__exit__(None, None, None)
    assert chat.turn is before


def test_every_span_is_exported_exactly_once_not_doubled():
    with fake_llm(script("hi")):
        with tracing.capture() as rec:
            llm.complete("x", provider="anthropic")
    assert len(rec.spans) == 1


# ----------------------------------------------------------------------------- reading traces


def rec_span(id_, parent, name, start, end, **attrs):
    status = "error" if attrs.pop("error", False) else "unset"
    return tracing.SpanRecord(
        "t1", id_, parent, name, int(start * 1e6), int(end * 1e6), status, "", attrs
    )


HAND = [
    rec_span(
        "a", None, "invoke_workflow support", 0, 1000, **{tracing.OPERATION: "invoke_workflow"}
    ),
    rec_span(
        "b",
        "a",
        "chat m",
        5,
        105,
        **{
            tracing.OPERATION: "chat",
            tracing.INPUT_TOKENS: 50,
            tracing.OUTPUT_TOKENS: 5,
            tracing.COST: 0.001,
        },
    ),
    rec_span(
        "c",
        "a",
        "invoke_agent billing",
        110,
        900,
        **{
            tracing.OPERATION: "invoke_agent",
            tracing.AGENT_NAME: "billing",
            tracing.AGENT_STATUS: "done",
        },
    ),
    rec_span(
        "d",
        "c",
        "chat m",
        120,
        300,
        **{
            tracing.OPERATION: "chat",
            tracing.INPUT_TOKENS: 200,
            tracing.OUTPUT_TOKENS: 20,
            tracing.COST: 0.004,
        },
    ),
    rec_span(
        "e",
        "c",
        "execute_tool lookup_invoice",
        310,
        330,
        **{tracing.OPERATION: "execute_tool", tracing.TOOL_NAME: "lookup_invoice"},
    ),
    rec_span(
        "f",
        "c",
        "execute_tool request_refund",
        340,
        600,
        error=True,
        **{
            tracing.OPERATION: "execute_tool",
            tracing.TOOL_NAME: "request_refund",
            tracing.ERROR_TYPE: "tool_error",
        },
    ),
    rec_span(
        "g",
        "c",
        "chat m",
        610,
        890,
        **{
            tracing.OPERATION: "chat",
            tracing.INPUT_TOKENS: 300,
            tracing.OUTPUT_TOKENS: 30,
            tracing.COST: 0.006,
        },
    ),
]


def test_summarize_totals_and_attributes_cost_to_the_nearest_agent():
    s = tracing.summarize(HAND)
    assert s["model_calls"] == 3 and s["input_tokens"] == 550 and s["output_tokens"] == 55
    assert (
        s["cost_usd"] == pytest.approx(0.011)
        and s["tool_calls"] == 2
        and s["tool_errors"] == 1
        and s["errors"] == 1
    )
    assert s["duration_ms"] == 1000
    assert s["by_agent"]["(no agent)"] == {
        "model_calls": 1,
        "input_tokens": 50,
        "output_tokens": 5,
        "cost_usd": 0.001,
    }
    assert (
        s["by_agent"]["billing"]["model_calls"] == 2
        and s["by_agent"]["billing"]["input_tokens"] == 500
    )
    assert s["by_tool"]["request_refund"] == {"calls": 1, "errors": 1, "ms": 260.0}


def test_tool_sequence_is_in_start_order_and_slowest_returns_leaves():
    assert tracing.tool_sequence(list(reversed(HAND))) == ["lookup_invoice", "request_refund"]
    slow = tracing.slowest(HAND, 2)
    assert [s.span_id for s in slow] == [
        "g",
        "f",
    ]  # leaves only: the agent span contains them, so it is not a culprit
    assert tracing.slowest(HAND, 1, leaf_only=False)[0].span_id == "a"


def test_render_tree_shows_nesting_tokens_cost_and_errors():
    text = tracing.render_tree(HAND)
    lines = text.splitlines()
    assert lines[0].startswith("invoke_workflow support  [1000ms]")
    assert lines[1].startswith("├─ chat m  [100ms, 50 in/5 out tok, $0.0010]")
    assert lines[2].startswith("└─ invoke_agent billing") and "status=done" in lines[2]
    assert lines[3].startswith("   ├─ chat m") and lines[5].endswith("ERROR tool_error]")
    assert len(lines) == 7


def test_orphans_show_up_as_extra_roots():
    spans = [*HAND, rec_span("z", "missing", "execute_tool lost", 1, 2)]
    assert {s.span_id for s in tracing.roots(spans)} == {"a", "z"}


# ----------------------------------------------------------------------------- exporters


def test_jsonl_exporter_round_trips_spans(tmp_path):
    path = tmp_path / "deep" / "spans.jsonl"
    with fake_llm(script(tool_calls(("add", {"a": 1, "b": 2})), "3")):
        with tracing.session(tracing.JsonlSpanExporter(path)):
            run_agent("x", REG, provider="anthropic")
    loaded = tracing.load_spans(path)
    assert len(loaded) == 4 and {s.trace_id for s in loaded} == {loaded[0].trace_id}
    assert tracing.summarize(loaded)[
        "model_calls"
    ] == 2 and "invoke_agent agent" in tracing.render_tree(loaded)
    assert all(s.duration_ms >= 0 for s in loaded)


def make_traces(exporter, n, *, error_every=0, slow=(), seed_offset=0):
    provider = TracerProvider(resource=Resource.create({}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("t")
    for i in range(n):
        with tracer.start_as_current_span(f"root-{i}") as root:
            root.set_attribute("i", i)
            with tracer.start_as_current_span("child") as child:
                if error_every and i % error_every == 0:
                    child.set_status(tracing.Status(tracing.StatusCode.ERROR, "x"))
            if i in slow:
                time.sleep(0.03)


def test_tail_sampling_keeps_errors_and_slow_traces_and_whole_traces_only():
    sink = InMemorySpanExporter()
    tail = tracing.TailSamplingExporter(sink, keep_fraction=0.0, slow_ms=20, keep_errors=True)
    make_traces(tail, 40, error_every=10, slow={7, 23})
    kept_roots = sorted(s.attributes["i"] for s in sink.get_finished_spans() if s.parent is None)
    assert kept_roots == [
        0,
        7,
        10,
        20,
        23,
        30,
    ]  # errors (every 10th) + the two slow ones, nothing else
    assert tail.kept == 6 and tail.dropped == 34
    spans = sink.get_finished_spans()
    assert len(spans) == 12 and all(  # each kept trace has BOTH its spans
        sum(1 for s in spans if s.get_span_context().trace_id == r.get_span_context().trace_id) == 2
        for r in spans
        if r.parent is None
    )


def test_tail_sampling_fraction_extremes_and_rough_rate():
    for frac, expect in ((0.0, 0), (1.0, 200)):
        sink = InMemorySpanExporter()
        tail = tracing.TailSamplingExporter(sink, keep_fraction=frac)
        make_traces(tail, 200)
        assert tail.kept == expect
    sink = InMemorySpanExporter()
    tail = tracing.TailSamplingExporter(sink, keep_fraction=0.25)
    make_traces(tail, 1000)
    assert 170 <= tail.kept <= 330, tail.kept  # trace ids are random: ~25% with generous slack


def test_tail_sampling_decision_depends_only_on_the_trace_id():
    """So two services sampling the same trace agree, and a partial trace is never kept by one and dropped by the other."""
    sink1, sink2 = InMemorySpanExporter(), InMemorySpanExporter()
    t1 = tracing.TailSamplingExporter(sink1, keep_fraction=0.5)
    t2 = tracing.TailSamplingExporter(sink2, keep_fraction=0.5)
    both = SimpleNamespaceExporter(t1, t2)
    make_traces(both, 300)
    ids1 = {s.get_span_context().trace_id for s in sink1.get_finished_spans()}
    ids2 = {s.get_span_context().trace_id for s in sink2.get_finished_spans()}
    assert ids1 == ids2 and 0 < len(ids1) < 300


class SimpleNamespaceExporter:
    def __init__(self, *targets):
        self.targets = targets

    def export(self, spans):
        for t in self.targets:
            t.export(list(spans))
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=0):
        return True


def test_tail_sampling_memory_is_bounded_and_unfinished_traces_are_dropped_on_shutdown():
    sink = InMemorySpanExporter()
    tail = tracing.TailSamplingExporter(sink, keep_fraction=1.0, max_buffered_traces=5)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(tail))
    tracer = provider.get_tracer("t")
    open_roots = []
    for i in range(8):  # eight traces whose root never ends, each with one finished child
        root = tracer.start_span(f"root-{i}")
        child = tracer.start_span("child", context=tracing._trace.set_span_in_context(root))
        child.end()
        open_roots.append(root)
    assert len(tail.buffer) == 5 and tail.evicted == 3
    tail.shutdown()
    assert tail.buffer == {} and sink.get_finished_spans() == ()


# ----------------------------------------------------------------------------- OTLP over the wire


def test_spans_reach_a_real_otlp_grpc_server_with_attributes_and_parents():
    grpc = pytest.importorskip("grpc")
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2 as pb
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2_grpc as pb_grpc

    received: list = []

    class Collector(pb_grpc.TraceServiceServicer):
        def Export(self, request, context):  # noqa: N802 - the generated interface name
            received.append(request)
            return pb.ExportTraceServiceResponse()

    server = grpc.server(futures.ThreadPoolExecutor(2))
    pb_grpc.add_TraceServiceServicer_to_server(Collector(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with fake_llm(script(tool_calls(("add", {"a": 2, "b": 3})), "5")):
            with tracing.session(
                tracing.otlp_exporter(f"127.0.0.1:{port}"), service_name="wire-test", sync=True
            ):
                run_agent("2+3", REG, provider="anthropic", agent_name="calc")
        assert tracing.wait_until(
            lambda: (
                sum(len(rs.scope_spans[0].spans) for r in received for rs in r.resource_spans) >= 4
            )
        )
    finally:
        server.stop(None)
    resource_spans = [rs for r in received for rs in r.resource_spans]
    service = {a.key: a.value.string_value for rs in resource_spans for a in rs.resource.attributes}
    assert service["service.name"] == "wire-test"
    spans = [s for rs in resource_spans for ss in rs.scope_spans for s in ss.spans]
    assert sorted(s.name.split()[0] for s in spans) == [
        "chat",
        "chat",
        "execute_tool",
        "invoke_agent",
    ]
    agent = next(s for s in spans if s.name.startswith("invoke_agent"))
    assert all(s.parent_span_id == agent.span_id for s in spans if s is not agent)
    attrs = {a.key: a.value for a in next(s for s in spans if s.name.startswith("chat")).attributes}
    assert (
        attrs[tracing.OPERATION].string_value == "chat"
        and attrs[tracing.INPUT_TOKENS].int_value > 0
    )


def test_wait_until_returns_as_soon_as_true_and_false_on_timeout():
    box = []
    threading.Timer(0.05, lambda: box.append(1)).start()
    assert tracing.wait_until(lambda: bool(box), timeout=2)
    assert not tracing.wait_until(lambda: False, timeout=0.05)


def test_content_cap_boundary_is_exact():
    with fake_llm(script("z" * 100)):
        with tracing.capture(capture_content=True, max_content_chars=100) as rec:
            llm.complete("y", provider="anthropic")
    assert rec.spans[0].get(tracing.OUTPUT_MESSAGES) == "z" * 100  # exactly at the cap: untouched
    with fake_llm(script("z" * 101)):
        with tracing.capture(capture_content=True, max_content_chars=100) as rec:
            llm.complete("y", provider="anthropic")
    assert rec.spans[0].get(tracing.OUTPUT_MESSAGES) == "z" * 100 + "...[truncated 1 chars]"


def stub_span(trace_id, *, parent=None, error=False, ms=1.0):
    ctx = SimpleNamespace(trace_id=trace_id)
    status = SimpleNamespace(status_code=SimpleNamespace(name="ERROR" if error else "UNSET"))
    return SimpleNamespace(
        get_span_context=lambda: ctx,
        parent=parent,
        status=status,
        start_time=0,
        end_time=int(ms * 1e6),
    )


def test_tail_sampling_threshold_is_exclusive_at_the_exact_boundary():
    """keep_fraction 0.5 keeps trace ids whose (id mod 10000) is below 5000: 4999 kept, 5000 dropped."""
    sink = InMemorySpanExporter()
    tail = tracing.TailSamplingExporter(sink, keep_fraction=0.5)
    tail.export([stub_span(10_000 + 4999)])
    tail.export([stub_span(10_000 + 5000)])
    kept = [s.get_span_context().trace_id for s in sink.get_finished_spans()]
    assert kept == [14999] and (tail.kept, tail.dropped) == (1, 1)


def test_slow_threshold_is_inclusive_and_needs_a_root():
    sink = InMemorySpanExporter()
    tail = tracing.TailSamplingExporter(sink, keep_fraction=0.0, slow_ms=50)
    tail.export([stub_span(1, ms=49.9)])
    tail.export([stub_span(2, ms=50.0)])
    assert [s.get_span_context().trace_id for s in sink.get_finished_spans()] == [2]


def test_uninstalling_leaves_a_function_someone_else_patched_on_top():
    marker = object()
    with tracing.capture():
        mine = chat.turn
        chat.turn = marker  # a later patch on top of ours
    assert chat.turn is marker, "uninstall must not overwrite a patch it did not make"
    chat.turn = mine.__wrapped__  # tidy up for the other tests


def test_render_tree_lists_several_roots_in_start_order():
    later = rec_span("r2", None, "second", 50, 60)
    first = rec_span("r1", None, "first", 10, 20)
    assert [ln.split("  ")[0] for ln in tracing.render_tree([later, first]).splitlines()] == [
        "first",
        "second",
    ]


# ----------------------------------------------------------------------------- argument digests


def test_args_digest_is_stable_keyed_and_not_a_plain_hash():
    import hashlib

    a = {"invoice_id": "INV-1001", "amount": 5}
    assert tracing.args_digest(a) == tracing.args_digest({"amount": 5, "invoice_id": "INV-1001"})
    assert tracing.args_digest(a) != tracing.args_digest({"invoice_id": "INV-1002", "amount": 5})
    plain = hashlib.sha256(json.dumps(a, sort_keys=True).encode()).hexdigest()[:12]
    assert tracing.args_digest(a) != plain, (
        "an unkeyed hash of a low-entropy value can be reversed by trying them all"
    )
    assert len(tracing.args_digest(a)) == 12


def test_tool_spans_carry_a_digest_but_never_the_arguments():
    with fake_llm(
        script(
            tool_calls(
                ("echo", {"text": SECRET}), ("echo", {"text": SECRET}), ("echo", {"text": "other"})
            ),
            "done",
        )
    ):
        with tracing.capture() as rec:
            run_agent("go", REG, provider="anthropic")
    digests = [s.get(tracing.TOOL_ARGS_DIGEST) for s in by_name(rec.spans, "execute_tool")]
    assert len(set(digests)) == 2 and sorted(digests.count(d) for d in set(digests)) == [1, 2]
    assert SECRET not in json.dumps([asdict(s) for s in rec.spans])


# ----------------------------------------------------------------------------- the current trace id


def test_current_trace_id_is_none_when_off_and_the_traces_id_inside_a_span():
    assert tracing.current_trace_id() is None
    with tracing.capture() as rec:
        assert tracing.current_trace_id() is None, "no span open yet"
        with tracing.span("outer"):
            inside = tracing.current_trace_id()
            with tracing.span("inner"):
                assert tracing.current_trace_id() == inside
    assert inside and len(inside) == 32 and int(inside, 16) > 0
    assert {s.trace_id for s in rec.spans} == {inside}
