"""Week 5 Day 1 - Solution: calculator + unit converter + date tools, with parallel calls.

What is here:
  * Four tool families written as plain typed Python functions (``@tool`` builds the JSON Schema):
      calculate        safe arithmetic (an AST walk, never eval())
      convert_units    length / mass / volume / temperature, with errors that list the valid units
      days_between, add_days, weekday_of, current_date
  * ``run_tool_loop``: the minimal "call the model, run its tools, send results back" loop. It handles
    PARALLEL calls (all results go back together, in order). Day 2 grows this into a real agent.
  * A small question set that is graded by *what the model did* (which tools it called) and *what it said*.

  uv run python weeks/week05_tool-use-and-agents/solutions/day1_solution.py            # local Qwen-0.5B
  LLM_PROVIDER=anthropic uv run python weeks/week05_tool-use-and-agents/solutions/day1_solution.py
"""

from __future__ import annotations

import ast
import math
import operator
import os
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common import chat  # noqa: E402
from common.tools import ToolRegistry, ToolResult, tool  # noqa: E402

# ----------------------------------------------------------------------------- calculator

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log10": math.log10,
}
_NAMES = {"pi": math.pi, "e": math.e}
MAX_EXPR_CHARS = 200
MAX_EXPONENT = 64  # 9**9**9 must not hang the agent


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent {right:g} is too large (limit {MAX_EXPONENT})")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"unsupported syntax: {ast.dump(node)[:60]}")


def _num(x: float) -> str:
    """12.0 -> '12', 0.1+0.2 -> '0.3' (15 significant digits hides float noise), 1e21 stays readable."""
    if isinstance(x, int):
        return str(x)
    if x != x or x in (math.inf, -math.inf):
        return str(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.15g}"


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly and return the result.

    Supports + - * / // % **, parentheses, pi, e, and sqrt/abs/round/min/max/log10. Use this for ANY
    arithmetic instead of computing in your head.

    Args:
        expression: Arithmetic only, e.g. "(1250 * 0.08) + 15" or "sqrt(144) / 3".
    """
    if len(expression) > MAX_EXPR_CHARS:
        raise ValueError(f"expression too long ({len(expression)} > {MAX_EXPR_CHARS} characters)")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"cannot parse {expression!r}: {exc.msg}") from None
    return _num(_eval(tree.body))


# ----------------------------------------------------------------------------- units

# factor to a base unit per category
_UNITS: dict[str, dict[str, float]] = {
    "length": {
        "mm": 0.001,
        "cm": 0.01,
        "m": 1.0,
        "km": 1000.0,
        "in": 0.0254,
        "ft": 0.3048,
        "yd": 0.9144,
        "mi": 1609.344,
    },
    "mass": {"mg": 1e-6, "g": 1e-3, "kg": 1.0, "oz": 0.028349523125, "lb": 0.45359237},
    "volume": {
        "ml": 1e-3,
        "l": 1.0,
        "tsp": 0.00492892159375,
        "cup": 0.2365882365,
        "gal": 3.785411784,
    },
}
_TEMPS = ("c", "f", "k")
_ALIASES = {
    "meter": "m", "meters": "m", "metre": "m", "metres": "m", "kilometer": "km", "kilometers": "km",
    "mile": "mi", "miles": "mi", "foot": "ft", "feet": "ft", "inch": "in", "inches": "in",
    "yard": "yd", "yards": "yd", "gram": "g", "grams": "g", "kilogram": "kg", "kilograms": "kg",
    "pound": "lb", "pounds": "lb", "lbs": "lb", "ounce": "oz", "ounces": "oz", "liter": "l",
    "liters": "l", "litre": "l", "litres": "l", "gallon": "gal", "gallons": "gal", "celsius": "c",
    "fahrenheit": "f", "kelvin": "k", "°c": "c", "°f": "f",
}  # fmt: skip


def _canon(unit: str) -> str:
    u = unit.strip().lower()
    return _ALIASES.get(u, u)


def _category(unit: str) -> str | None:
    if unit in _TEMPS:
        return "temperature"
    return next((cat for cat, table in _UNITS.items() if unit in table), None)


def _all_units() -> str:
    parts = [f"{cat}: {', '.join(table)}" for cat, table in _UNITS.items()]
    return "; ".join([*parts, "temperature: c, f, k"])


def _to_kelvin(v: float, u: str) -> float:
    return {"c": v + 273.15, "f": (v - 32) * 5 / 9 + 273.15, "k": v}[u]


def _from_kelvin(k: float, u: str) -> float:
    return {"c": k - 273.15, "f": (k - 273.15) * 9 / 5 + 32, "k": k}[u]


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between units of the same kind (length, mass, volume or temperature).

    Args:
        value: The number to convert.
        from_unit: Unit symbol such as km, mi, ft, kg, lb, l, gal, c, f or k.
        to_unit: Unit symbol to convert into; must measure the same kind of thing as from_unit.
    """
    src, dst = _canon(from_unit), _canon(to_unit)
    for given, canon in ((from_unit, src), (to_unit, dst)):
        if _category(canon) is None:
            raise ValueError(f"unknown unit {given!r}. Valid units: {_all_units()}.")
    cat_s, cat_d = _category(src), _category(dst)
    if cat_s != cat_d:
        raise ValueError(f"cannot convert {cat_s} ({src}) to {cat_d} ({dst}).")
    if cat_s == "temperature":
        k = _to_kelvin(value, src)
        if k < 0:
            raise ValueError(f"{value:g} {src} is below absolute zero.")
        out = _from_kelvin(k, dst)
    else:
        table = _UNITS[cat_s]
        out = value * table[src] / table[dst]
    return f"{_num(round(out, 6))} {dst}"


# ----------------------------------------------------------------------------- dates


def _parse_date(text: str, field_name: str) -> date:
    try:
        return date.fromisoformat(text.strip())
    except ValueError:
        raise ValueError(
            f"{field_name} must be an ISO date like 2026-03-14, got {text!r}"
        ) from None


@tool
def current_date() -> str:
    """Return today's date in ISO format (YYYY-MM-DD). Call this before any 'today', 'tomorrow' or 'next week' reasoning."""
    return os.environ.get("COURSE_TODAY") or date.today().isoformat()


@tool
def days_between(start: str, end: str) -> str:
    """Number of days from start to end (negative when end is before start).

    Args:
        start: Start date, ISO format YYYY-MM-DD.
        end: End date, ISO format YYYY-MM-DD.
    """
    return str((_parse_date(end, "end") - _parse_date(start, "start")).days)


@tool
def add_days(start: str, days: int) -> str:
    """The ISO date that is `days` days after start (use a negative number to go back).

    Args:
        start: Start date, ISO format YYYY-MM-DD.
        days: Whole days to add; may be negative. At most 100000 in either direction.
    """
    if abs(days) > 100_000:
        raise ValueError("days must be between -100000 and 100000")
    return (_parse_date(start, "start") + timedelta(days=days)).isoformat()


@tool
def weekday_of(day: str) -> str:
    """Name of the weekday (Monday..Sunday) of an ISO date.

    Args:
        day: Date in ISO format YYYY-MM-DD.
    """
    return _parse_date(day, "day").strftime("%A")


def default_registry() -> ToolRegistry:
    return ToolRegistry(
        [calculate, convert_units, current_date, days_between, add_days, weekday_of]
    )


# ----------------------------------------------------------------------------- the loop

SYSTEM = (
    "You are a precise assistant with tools for arithmetic, unit conversion and dates. "
    "Use a tool whenever a question needs a calculation, a conversion or a date; never do those in your head. "
    "When several independent lookups are needed, request all of them in the same turn. "
    "After you have the results, answer in one short sentence."
)


@dataclass
class Run:
    question: str
    answer: str = ""
    messages: list[dict] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    calls: list[chat.ToolCall] = field(default_factory=list)
    turns: int = 0
    cost_usd: float = 0.0
    finished: bool = False  # False = we stopped at max_turns while the model still wanted tools

    @property
    def tool_names(self) -> list[str]:
        return [c.name for c in self.calls]


def run_tool_loop(
    question: str,
    registry: ToolRegistry,
    *,
    provider: str | None = None,
    model: str | None = None,
    system: str = SYSTEM,
    max_turns: int = 6,
    tool_choice: str | None = None,
) -> Run:
    """Ask, run the requested tools (in parallel), send ALL results back, repeat until the model answers."""
    run = Run(question, messages=[{"role": "user", "content": question}])
    for _ in range(max_turns):
        t = chat.turn(
            run.messages,
            registry.specs(),
            system=system,
            provider=provider,
            model=model,
            # a forced tool_choice only applies to the first turn; forcing it forever would never let it answer
            tool_choice=tool_choice if run.turns == 0 else None,
        )
        run.turns += 1
        run.cost_usd += t.cost_usd
        if not t.wants_tools:
            run.answer, run.finished = t.text, True
            return run
        run.messages.append(chat.assistant_message(t))
        run.calls += t.tool_calls
        results = registry.execute_all(t.tool_calls)  # same order as the calls
        run.results += results
        for call, res in zip(t.tool_calls, results, strict=True):
            run.messages.append(chat.tool_message(call, res.content, res.is_error))
    return run


# ----------------------------------------------------------------------------- evaluation

CASES = [
    # (question, tools that SHOULD be used (set), substring the answer should contain)
    ("What is 1250 * 0.08 + 15?", {"calculate"}, "115"),
    ("How many miles is 42.195 kilometers?", {"convert_units"}, "26.2"),
    ("Convert 98.6 fahrenheit to celsius.", {"convert_units"}, "37"),
    ("How many days are there between 2026-01-01 and 2026-03-01?", {"days_between"}, "59"),
    ("What date is 45 days after 2026-02-10?", {"add_days"}, "2026-03-27"),
    ("What day of the week is 2026-07-04?", {"weekday_of"}, "Saturday"),
    ("Convert 5 km to miles and 10 lb to kg.", {"convert_units"}, "3.1"),
    ("Say hello in French.", set(), "bonjour"),  # must NOT call a tool
]


def grade(case: tuple[str, set[str], str], run: Run) -> dict:
    _, expected_tools, needle = case
    used = set(run.tool_names)
    return {
        "right_tools": (not used) if not expected_tools else expected_tools <= used,
        "right_answer": needle.lower() in run.answer.lower(),
        "errors": sum(r.is_error for r in run.results),
    }


def main() -> None:
    provider = os.environ.get("LLM_PROVIDER") or "local"
    os.environ.setdefault("COURSE_TODAY", "2026-10-01")
    registry = default_registry()
    print(f"provider={provider}  tools={registry.names()}\n")
    rows = []
    for case in CASES:
        run = run_tool_loop(case[0], registry, provider=provider)
        g = grade(case, run)
        rows.append(g)
        print(f"Q: {case[0]}")
        print(f"   calls={[(c.name, c.args) for c in run.calls]}")
        print(f"   results={[r.content for r in run.results]}")
        print(f"   answer={run.answer!r}  {g}\n")
    n = len(rows)
    print(
        f"right tools: {sum(r['right_tools'] for r in rows)}/{n}   "
        f"right answer: {sum(r['right_answer'] for r in rows)}/{n}   "
        f"tool errors: {sum(r['errors'] for r in rows)}"
    )


if __name__ == "__main__":
    main()
