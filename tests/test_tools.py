"""Tests for common/tools.py: schemas from signatures, validation messages, failure containment, parallelism."""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Literal

import pytest
from pydantic import BaseModel

from common.chat import ToolCall
from common.tools import ToolRegistry, tool


@tool
def multiply(a: float, b: float = 2.0) -> float:
    """Multiply two numbers.

    Args:
        a: First factor.
        b: Second factor (default 2).

    Returns:
        The product.
    """
    return a * b


class Unit(StrEnum):
    c = "celsius"
    f = "fahrenheit"


@tool(name="convert_temp", description="Convert a temperature.")
def convert(
    value: float, to: Unit, rounding: Literal["none", "int"] = "none", tags: list[str] | None = None
) -> str:
    return f"{value}->{to.value}/{rounding}/{tags}"


def call(name, args, cid="c1"):
    return ToolCall(cid, name, args)


def test_schema_comes_from_the_signature_and_docstring():
    spec = multiply.spec()
    assert spec["name"] == "multiply" and spec["description"] == "Multiply two numbers."
    p = spec["parameters"]
    assert p["type"] == "object" and p["required"] == ["a"] and p["additionalProperties"] is False
    assert p["properties"]["a"] == {"type": "number", "description": "First factor."}
    assert (
        p["properties"]["b"]["description"] == "Second factor (default 2)."
        and p["properties"]["b"]["default"] == 2.0
    )
    assert "title" not in str(p), "titles are noise for the model"


def test_enums_literals_and_optional_lists_are_expressed():
    p = convert.spec()["parameters"]
    assert convert.name == "convert_temp" and convert.description == "Convert a temperature."
    assert set(p["required"]) == {"value", "to"}
    blob = str(p)
    assert (
        "celsius" in blob
        and "fahrenheit" in blob
        and "'none'" in blob
        and "'int'" in blob
        and "array" in blob
    )


def test_valid_call_runs_and_coerces_types():
    reg = ToolRegistry([multiply, convert])
    r = reg.execute(call("multiply", {"a": "3", "b": 4}))  # "3" is coerced to 3.0 by validation
    assert r.content == "12.0" and not r.is_error and r.call_id == "c1" and r.name == "multiply"
    assert reg.execute(call("multiply", {"a": 5})).content == "10.0", "defaults apply"
    assert (
        reg.execute(call("convert_temp", {"value": 1, "to": "celsius"})).content
        == "1.0->celsius/none/None"
    )


@pytest.mark.parametrize(
    "args,expect",
    [
        ({}, "a: Field required"),
        ({"a": "banana"}, "a: Input should be a valid number"),
        ({"a": 1, "extra": 5}, "extra: Extra inputs are not permitted"),
        ({"a": 1, "q": 5}, "Expected: a (number), b (number)"),
    ],
)
def test_invalid_arguments_return_a_helpful_error_not_an_exception(args, expect):
    r = ToolRegistry([multiply]).execute(call("multiply", args))
    assert r.is_error and r.content.startswith("Invalid arguments for multiply")
    assert expect in r.content
    assert "required: ['a']" in r.content, "the message tells the model what is required"


def test_misspelt_arguments_are_rejected_not_silently_defaulted():
    """A silent drop would run multiply(a=1) with b=2 and return a plausible-looking wrong answer."""
    reg = ToolRegistry([multiply, convert])
    assert reg.execute(call("multiply", {"a": 1, "extra": 5})).is_error
    bad = reg.execute(call("convert_temp", {"value": 1, "to": "kelvin"}))
    assert bad.is_error and "to:" in bad.content


def test_unknown_tool_lists_the_available_ones():
    r = ToolRegistry([multiply, convert]).execute(call("divide", {}))
    assert (
        r.is_error
        and "Unknown tool 'divide'" in r.content
        and "multiply, convert_temp" in r.content
    )


def test_a_crashing_tool_never_crashes_the_loop():
    @tool
    def boom(x: int) -> str:
        """Explode."""
        raise RuntimeError("kaboom")

    r = ToolRegistry([boom]).execute(call("boom", {"x": 1}))
    assert (
        r.is_error and "failed: RuntimeError: kaboom" in r.content and "Traceback" not in r.content
    )


def test_huge_results_are_truncated_with_a_visible_note():
    @tool(max_chars=50)
    def big() -> str:
        """Big output."""
        return "x" * 500

    r = ToolRegistry([big]).execute(call("big", {}))
    assert (
        len(r.content) < 200 and "[truncated: 450 more characters" in r.content and not r.is_error
    )


def test_non_string_results_are_json_encoded():
    @tool
    def rows() -> list:
        """Rows."""
        return [{"a": 1}, {"b": "é"}]

    assert ToolRegistry([rows]).execute(call("rows", {})).content == '[{"a": 1}, {"b": "é"}]'


def test_slow_tools_time_out():
    @tool(timeout_s=0.1)
    def slow() -> str:
        """Slow."""
        time.sleep(1)
        return "late"

    t0 = time.perf_counter()
    r = ToolRegistry([slow]).execute(call("slow", {}))
    assert r.is_error and "timed out after 0.1s" in r.content and time.perf_counter() - t0 < 0.8


def test_parallel_execution_is_faster_and_preserves_order():
    @tool
    def nap(n: int) -> int:
        """Sleep a little and echo."""
        time.sleep(0.2)
        return n

    reg = ToolRegistry([nap])
    calls = [call("nap", {"n": i}, f"c{i}") for i in range(4)]
    t0 = time.perf_counter()
    par = reg.execute_all(calls)
    t_par = time.perf_counter() - t0
    t0 = time.perf_counter()
    seq = reg.execute_all(calls, parallel=False)
    t_seq = time.perf_counter() - t0
    assert [r.content for r in par] == ["0", "1", "2", "3"] and [r.call_id for r in par] == [
        "c0",
        "c1",
        "c2",
        "c3",
    ]
    assert [r.content for r in seq] == ["0", "1", "2", "3"]
    assert t_par < t_seq / 2, (t_par, t_seq)


def test_one_failure_in_a_parallel_batch_does_not_poison_the_others():
    @tool
    def maybe(n: int) -> int:
        """Fail on odd numbers."""
        if n % 2:
            raise ValueError("odd")
        return n

    out = ToolRegistry([maybe]).execute_all([call("maybe", {"n": i}, f"c{i}") for i in range(4)])
    assert [r.is_error for r in out] == [False, True, False, True] and out[2].content == "2"


def test_duplicate_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry([multiply, multiply])


class Point(BaseModel):
    x: float
    y: float


@tool
def move(p: Point) -> str:
    """Move to a point."""
    return f"{p.x},{p.y}"


def test_nested_models_are_also_closed_to_extra_properties():
    schema = move.spec()["parameters"]
    assert schema["$defs"]["Point"]["additionalProperties"] is False
    assert ToolRegistry([move]).execute(call("move", {"p": {"x": 1, "y": 2}})).content == "1.0,2.0"


# ------------------------------------------------------------------ remote (raw) tools and ToolFailure


def _remote(fn, **kw):
    from common.tools import Tool

    return Tool(
        "remote",
        "A remote tool.",
        None,
        None,
        {"type": "object", "properties": {}},
        raw_fn=fn,
        **kw,
    )


def test_remote_tool_receives_the_raw_argument_dict_without_local_validation():
    seen = []
    reg = ToolRegistry([_remote(lambda args: seen.append(args) or "ok")])
    r = reg.execute(ToolCall("1", "remote", {"anything": [1, 2], "goes": None}))
    assert (r.content, r.is_error) == ("ok", False) and seen == [{"anything": [1, 2], "goes": None}]
    args = {"a": 1}
    reg.execute(ToolCall("2", "remote", args))
    assert args == {"a": 1}, "the caller's dict is not handed out for mutation"


def test_tool_failure_message_is_shown_verbatim_other_exceptions_are_prefixed():
    from common.tools import ToolFailure

    def fail(args):
        raise ToolFailure("Note 'x' not found. Existing notes: a, b.")

    def crash(args):
        raise KeyError("k")

    reg = ToolRegistry([_remote(fail)])
    r = reg.execute(ToolCall("1", "remote", {}))
    assert r.is_error and r.content == "Note 'x' not found. Existing notes: a, b."
    r = ToolRegistry([_remote(crash)]).execute(ToolCall("1", "remote", {}))
    assert r.is_error and r.content.startswith("Tool remote failed: KeyError")

    @tool
    def local_fail() -> str:
        """Local tool that raises ToolFailure."""
        raise ToolFailure("plain message")

    assert (
        ToolRegistry([local_fail]).execute(ToolCall("1", "local_fail", {})).content
        == "plain message"
    )


def test_remote_tools_get_truncation_and_timeouts_too():
    big = ToolRegistry([_remote(lambda a: "x" * 100, max_chars=10)])
    assert "[truncated: 90 more characters" in big.execute(ToolCall("1", "remote", {})).content
    import time

    slow = ToolRegistry([_remote(lambda a: time.sleep(2), timeout_s=0.05)])
    r = slow.execute(ToolCall("1", "remote", {}))
    assert r.is_error and "timed out after 0.05s" in r.content
