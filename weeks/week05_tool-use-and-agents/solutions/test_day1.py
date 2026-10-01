"""Tests for Week 5 Day 1: the tools themselves, then the loop (scripted model, then the real SDKs)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))

import day1_solution as d1  # noqa: E402
from fake_llm_server import FakeLLMServer  # noqa: E402

from common import chat, llm  # noqa: E402
from common.chat import ToolCall  # noqa: E402
from common.fake import fake_llm, tool_calls  # noqa: E402
from common.tools import ToolResult  # noqa: E402

REG = d1.default_registry()


def call(name: str, **args):
    return REG.execute(ToolCall("t", name, args))


# ----------------------------------------------------------------------------- calculator


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("1250 * 0.08 + 15", "115"),
        ("0.1 + 0.2", "0.3"),
        ("10 / 4", "2.5"),
        ("7 // 2", "3"),
        ("-7 % 3", "2"),
        ("2 ** 10", "1024"),
        ("sqrt(144) / 3", "4"),
        ("round(3.14159, 2)", "3.14"),
        ("max(1, 5, 3) - min(4, 2)", "3"),
        ("-(2 + 3) * 2", "-10"),
        ("pi", "3.14159265358979"),
    ],
)
def test_calculate_exact_results(expr, expected):
    r = call("calculate", expression=expr)
    assert (r.content, r.is_error) == (expected, False)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__",
        "[x for x in range(3)]",
        "lambda: 1",
        "a + 1",
        "'abc' * 3",
        "True + 1",
        "sqrt(x=4)",
        "(1).real",
    ],
)
def test_calculate_rejects_everything_that_is_not_arithmetic(expr):
    r = call("calculate", expression=expr)
    assert r.is_error and ("unsupported syntax" in r.content or "cannot parse" in r.content)


def test_calculate_limits_and_error_messages_are_actionable():
    assert "too large" in call("calculate", expression="9 ** 9 ** 9").content
    assert "too large" in call("calculate", expression="2 ** -100").content
    assert "too long" in call("calculate", expression="1+" * 150 + "1").content
    r = call("calculate", expression="1 / 0")
    assert r.is_error and "ZeroDivisionError" in r.content
    assert "cannot parse" in call("calculate", expression="2 +* 3").content
    assert "Invalid arguments" in call("calculate", expr="1+1").content, (
        "wrong argument name is an error"
    )


# ----------------------------------------------------------------------------- units


@pytest.mark.parametrize(
    "value,src,dst,expected",
    [
        (5, "km", "mi", "3.106856 mi"),
        (42.195, "kilometers", "MI", "26.218757 mi"),
        (10, "lb", "kg", "4.535924 kg"),
        (1, "gal", "l", "3.785412 l"),
        (98.6, "f", "c", "37 c"),
        (0, "c", "k", "273.15 k"),
        (100, "°C", "fahrenheit", "212 f"),
        (12, "in", "ft", "1 ft"),
        (3, "feet", "inches", "36 in"),
    ],
)
def test_convert_units(value, src, dst, expected):
    r = call("convert_units", value=value, from_unit=src, to_unit=dst)
    assert (r.content, r.is_error) == (expected, False)


def test_convert_units_round_trips_within_tolerance():
    for src, dst in [("km", "mi"), ("lb", "oz"), ("cup", "ml"), ("f", "c")]:
        out = float(
            call("convert_units", value=123.456, from_unit=src, to_unit=dst).content.split()[0]
        )
        back = float(
            call("convert_units", value=out, from_unit=dst, to_unit=src).content.split()[0]
        )
        assert back == pytest.approx(123.456, rel=1e-5)


def test_convert_units_errors_tell_the_model_what_is_valid():
    unknown = call("convert_units", value=1, from_unit="parsec", to_unit="m")
    assert (
        unknown.is_error
        and "'parsec'" in unknown.content
        and "length: mm, cm, m" in unknown.content
    )
    mismatch = call("convert_units", value=1, from_unit="kg", to_unit="m")
    assert (
        mismatch.is_error and "mass (kg)" in mismatch.content and "length (m)" in mismatch.content
    )
    below = call("convert_units", value=-300, from_unit="c", to_unit="f")
    assert below.is_error and "absolute zero" in below.content
    assert (
        "Invalid arguments"
        in call("convert_units", value="abc", from_unit="m", to_unit="km").content
    )


# ----------------------------------------------------------------------------- dates


def test_date_tools_including_leap_years_and_negative_offsets():
    assert call("days_between", start="2026-01-01", end="2026-03-01").content == "59"
    assert call("days_between", start="2024-02-28", end="2024-03-01").content == "2", (
        "2024 is a leap year"
    )
    assert call("days_between", start="2026-03-01", end="2026-01-01").content == "-59"
    assert call("add_days", start="2026-02-10", days=45).content == "2026-03-27"
    assert call("add_days", start="2026-03-01", days=-1).content == "2026-02-28"
    assert call("weekday_of", day="2026-07-04").content == "Saturday"


def test_date_tool_errors(monkeypatch):
    bad = call("days_between", start="March 3", end="2026-01-01")
    assert bad.is_error and "ISO date" in bad.content and "'March 3'" in bad.content
    assert call("days_between", start="2026-02-30", end="2026-03-01").is_error
    assert call("add_days", start="2026-01-01", days=10**9).is_error
    monkeypatch.setenv("COURSE_TODAY", "2031-05-06")
    assert call("current_date").content == "2031-05-06"
    monkeypatch.delenv("COURSE_TODAY")
    assert len(call("current_date").content) == 10


# ----------------------------------------------------------------------------- schemas


def test_every_tool_has_a_strict_schema_with_descriptions():
    specs = {s["name"]: s for s in REG.specs()}
    assert set(specs) == {
        "calculate",
        "convert_units",
        "current_date",
        "days_between",
        "add_days",
        "weekday_of",
    }
    for s in specs.values():
        p = s["parameters"]
        assert p["type"] == "object" and p["additionalProperties"] is False
        assert s["description"]
        assert set(p.get("required", [])) == set(p["properties"])
        assert all(v.get("description") for v in p["properties"].values()), s["name"]
    assert specs["convert_units"]["parameters"]["properties"]["value"]["type"] == "number"
    assert specs["add_days"]["parameters"]["properties"]["days"]["type"] == "integer"
    assert specs["current_date"]["parameters"]["properties"] == {}


# ----------------------------------------------------------------------------- the loop (scripted)


def test_loop_runs_parallel_calls_in_one_turn_and_returns_results_in_order():
    script = [
        (r"tool results are in", "done"),
        (
            r"(?s).*",
            tool_calls(
                ("convert_units", {"value": 5, "from_unit": "km", "to_unit": "mi"}),
                ("convert_units", {"value": 10, "from_unit": "lb", "to_unit": "kg"}),
            ),
        ),
    ]
    # the loop sends results as tool messages; FakeLLM sees "convert_units(...)" call text in the prompt
    with fake_llm([(r"3\.106856", "5 km = 3.1 mi; 10 lb = 4.5 kg."), *script]) as f:
        run = d1.run_tool_loop("Convert 5 km to miles and 10 lb to kg.", REG, provider="anthropic")
    assert run.finished and run.turns == 2 and len(run.calls) == 2
    assert [r.content for r in run.results] == ["3.106856 mi", "4.535924 kg"]
    assert run.answer == "5 km = 3.1 mi; 10 lb = 4.5 kg."
    roles = [m["role"] for m in run.messages]
    assert roles == ["user", "assistant", "tool", "tool"], "one assistant turn, then EVERY result"
    assert [m["tool_call_id"] for m in run.messages[2:]] == [c.id for c in run.calls]
    assert len(f.calls) == 2


def test_loop_feeds_errors_back_so_the_model_can_retry():
    bad = tool_calls(("convert_units", {"value": 1, "from_unit": "parsec", "to_unit": "m"}))
    good = tool_calls(("convert_units", {"value": 1, "from_unit": "km", "to_unit": "m"}))
    with fake_llm(
        [
            (r"1000 m", "1 km is 1000 m."),
            (r"Valid units", good),
            (r"(?s).*", bad),
        ]
    ):
        run = d1.run_tool_loop("1 parsec in m?", REG, provider="anthropic")
    assert run.finished and run.turns == 3
    assert [r.is_error for r in run.results] == [True, False]
    assert run.messages[2]["is_error"] is True and "Valid units" in run.messages[2]["content"]


def test_loop_stops_at_max_turns_and_says_it_did_not_finish():
    with fake_llm(default=tool_calls(("current_date", {}))):
        run = d1.run_tool_loop("loop forever", REG, provider="anthropic", max_turns=3)
    assert not run.finished and run.turns == 3 and run.answer == "" and len(run.results) == 3


def test_loop_survives_unknown_tools_and_bad_arguments():
    with fake_llm(
        [
            (r"Unknown tool", "I cannot do that."),
            (r"(?s).*", tool_calls(("say_hello_in_french", {"language": "fr"}))),
        ]
    ):
        run = d1.run_tool_loop("hello", REG, provider="anthropic")
    assert run.finished and run.results[0].is_error and "Available tools" in run.results[0].content
    with fake_llm(
        [(r"Invalid arguments", "ok"), (r"(?s).*", tool_calls(("calculate", {"expression": 5})))]
    ):
        run = d1.run_tool_loop("x", REG, provider="anthropic")
    assert run.results[0].is_error and run.finished


def test_forced_tool_choice_applies_only_to_the_first_turn():
    seen: list = []

    def spy(messages, tools=None, **kw):
        seen.append(kw.get("tool_choice"))
        return real(messages, tools, **kw)

    with fake_llm(
        [(r"\b4\b", "It is 4."), (r"(?s).*", tool_calls(("calculate", {"expression": "2+2"})))]
    ):
        real = chat.turn  # fake_llm already patched chat.turn; wrap that (it is undone on exit)
        chat.turn = spy
        run = d1.run_tool_loop("2+2?", REG, provider="anthropic", tool_choice="calculate")
    assert seen == ["calculate", None] and run.finished


def test_text_only_answer_makes_no_tool_calls():
    with fake_llm(default="Bonjour"):
        run = d1.run_tool_loop("Say hello in French.", REG, provider="anthropic")
    assert run.finished and run.calls == [] and run.turns == 1 and run.answer == "Bonjour"


# ----------------------------------------------------------------------------- grading


def test_grade_checks_tools_and_answer_independently():
    case = ("q", {"calculate"}, "115")
    ok = d1.Run("q", answer="It is 115.", calls=[ToolCall("1", "calculate", {})])
    assert d1.grade(case, ok) == {"right_tools": True, "right_answer": True, "errors": 0}
    guessed = d1.Run("q", answer="It is 115.")
    assert (
        d1.grade(case, guessed)["right_tools"] is False and d1.grade(case, guessed)["right_answer"]
    )
    wrong_tool = d1.Run("q", answer="x", calls=[ToolCall("1", "weekday_of", {})])
    assert d1.grade(case, wrong_tool)["right_tools"] is False
    no_tool_case = ("hi", set(), "bonjour")
    assert d1.grade(no_tool_case, d1.Run("hi", answer="Bonjour!"))["right_tools"] is True
    assert (
        d1.grade(no_tool_case, d1.Run("hi", calls=[ToolCall("1", "calculate", {})]))["right_tools"]
        is False
    )
    failed = d1.Run(
        "q",
        answer="115",
        calls=[ToolCall("1", "calculate", {})],
        results=[ToolResult("1", "calculate", "e", True)],
    )
    assert d1.grade(case, failed)["errors"] == 1


def test_case_set_covers_each_tool_and_one_no_tool_case():
    covered = set().union(*(c[1] for c in d1.CASES))
    assert {"calculate", "convert_units", "days_between", "add_days", "weekday_of"} <= covered
    assert sum(1 for c in d1.CASES if not c[1]) == 1


# ----------------------------------------------------------------------------- real SDK round trip


@pytest.fixture()
def server(monkeypatch):
    with FakeLLMServer() as srv:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", srv.url)
        monkeypatch.setenv("OPENAI_BASE_URL", srv.url + "/v1")
        monkeypatch.setenv("OLLAMA_BASE_URL", srv.url + "/v1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "t")
        monkeypatch.setenv("OPENAI_API_KEY", "t")
        monkeypatch.setenv("LLM_MAX_RETRIES", "0")
        for fn in (llm._anthropic, llm._openai, llm._ollama):
            fn.cache_clear()
        yield srv
        for fn in (llm._anthropic, llm._openai, llm._ollama):
            fn.cache_clear()


@pytest.mark.parametrize("provider", ["anthropic", "openai", "ollama"])
def test_full_loop_through_each_providers_wire_format(server, provider):
    server.script = [
        {
            "text": "",
            "tool_calls": [
                {"name": "convert_units", "args": {"value": 5, "from_unit": "km", "to_unit": "mi"}},
                {"name": "calculate", "args": {"expression": "2 * 21"}},
            ],
        },
        {"text": "3.1 miles and 42."},
    ]
    run = d1.run_tool_loop("two things", REG, provider=provider)
    assert run.finished and run.answer == "3.1 miles and 42."
    assert [r.content for r in run.results] == ["3.106856 mi", "42"]
    sent = str(server.requests[-1]["body"])
    assert "3.106856 mi" in sent and "42" in sent
    first_tools = server.requests[0]["body"]["tools"]
    assert len(first_tools) == 6


def test_grade_accepts_extra_helper_tools_but_not_missing_ones():
    case = ("q", {"add_days"}, "x")
    run = d1.Run(
        "q", answer="x", calls=[ToolCall("1", "current_date", {}), ToolCall("2", "add_days", {})]
    )
    assert d1.grade(case, run)["right_tools"] is True, "calling current_date first is fine"


def test_add_days_limit_is_enforced_by_our_check_not_by_python_overflow():
    r = call("add_days", start="2026-01-01", days=100_001)
    assert r.is_error and "between -100000 and 100000" in r.content
    assert call("add_days", start="2026-01-01", days=100_000).content == "2299-10-17"
