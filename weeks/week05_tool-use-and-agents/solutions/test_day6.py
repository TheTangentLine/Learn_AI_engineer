"""Tests for Week 5 Day 6: the dataset and answer key, pandas solutions executed in the REAL sandbox, the
checkers, error recovery through the agent loop, and the escape matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))

import day6_solution as d6  # noqa: E402
from day3_solution import lint_spec  # noqa: E402

from common.chat import ToolCall  # noqa: E402
from common.fake import fake_llm, tool_calls  # noqa: E402
from common.sandbox import DockerSandbox, SubprocessSandbox  # noqa: E402
from common.tools import ToolRegistry  # noqa: E402

SB = SubprocessSandbox(timeout_s=20)

# ----------------------------------------------------------------------------- data and key


def test_dataset_is_deterministic_messy_and_the_answer_key_is_stable():
    assert d6.build_rows() == d6.ROWS and len(d6.ROWS) == 240
    assert sum(r["units"] == "" for r in d6.ROWS) == 6
    assert d6.CSV_TEXT.splitlines()[0] == "order_id,date,region,product,units,unit_price"
    assert d6.answer_key() == d6.KEY
    assert d6.build_rows(seed=8) != d6.ROWS
    assert d6.KEY["missing_units"] == 6 and set(d6.KEY) >= {
        "total_revenue",
        "top_region",
        "top_month",
    }


def test_answer_key_ignores_missing_units_and_the_answers_are_not_ties():
    rows = [
        {
            "order_id": 1,
            "date": "2026-01-05",
            "region": "north",
            "product": "A",
            "units": 2,
            "unit_price": 10.0,
        },
        {
            "order_id": 2,
            "date": "2026-01-06",
            "region": "south",
            "product": "B",
            "units": "",
            "unit_price": 1000.0,
        },
        {
            "order_id": 3,
            "date": "2026-02-01",
            "region": "south",
            "product": "B",
            "units": 1,
            "unit_price": 5.0,
        },
    ]
    k = d6.answer_key(rows)
    assert (
        k["total_revenue"] == 25
        and k["top_region"] == "north"
        and k["missing_units"] == 1
        and k["top_month"] == "2026-01"
    )
    assert k["avg_units_b"] == 1 and k["top_product_units"] == "A"
    by = {}
    for r in d6._valid(d6.ROWS):
        by[r["region"]] = by.get(r["region"], 0) + r["units"] * r["unit_price"]
    top_two = sorted(by.values())[-2:]
    assert top_two[1] - top_two[0] > 50, "the top region must be unambiguous"


# ----------------------------------------------------------------------------- pandas solutions run in the real sandbox

PANDAS = {
    "total-revenue": "import pandas as pd\ndf = pd.read_csv('sales.csv').dropna(subset=['units'])\nprint(round((df.units * df.unit_price).sum(), 2))",
    "top-region": "import pandas as pd\ndf = pd.read_csv('sales.csv').dropna(subset=['units'])\ndf['rev'] = df.units * df.unit_price\nprint(df.groupby('region').rev.sum().idxmax())",
    "avg-units-b": "import pandas as pd\ndf = pd.read_csv('sales.csv')\nprint(round(df[df['product'] == 'B'].units.mean(), 4))",
    "missing": "import pandas as pd\ndf = pd.read_csv('sales.csv')\nprint(int(df.units.isna().sum()))",
    "top-month": "import pandas as pd\ndf = pd.read_csv('sales.csv').dropna(subset=['units'])\ndf['rev'] = df.units * df.unit_price\nprint(df.groupby(df['date'].str[:7]).rev.sum().idxmax())",
    "north-march": "import pandas as pd\ndf = pd.read_csv('sales.csv').dropna(subset=['units'])\nm = df[(df.region == 'north') & (df.date.str[:7] == '2026-03')]\nprint(round((m.units * m.unit_price).sum(), 2))",
    "top-product": "import pandas as pd\ndf = pd.read_csv('sales.csv')\nprint(df.groupby('product').units.sum().idxmax())",
}
ANSWERS = {
    "total-revenue": lambda out: f"The total revenue is {out}.",
    "top-region": lambda out: f"The {out} region.",
    "avg-units-b": lambda out: f"On average {out} units per order.",
    "missing": lambda out: f"{out} rows have missing units.",
    "top-month": lambda out: f"The best month is {out}.",
    "north-march": lambda out: f"North earned {out} in March.",
    "top-product": lambda out: f"Product {out} sold the most units.",
}


@pytest.mark.parametrize("task", d6.TASKS, ids=lambda t: t.id)
def test_pandas_solution_run_in_the_sandbox_agrees_with_the_plain_python_key(task):
    code = PANDAS[task.id]
    printed = SB.run(code, files={"sales.csv": d6.CSV_TEXT})
    assert printed.ok, printed.stderr
    out = printed.stdout.strip()
    rules = [
        (
            r"run_python\(",
            lambda p, call: ANSWERS[task.id](call.messages[-1]["content"].strip()),
        ),  # after the tool ran
        (r"(?s).*", tool_calls(("run_python", {"code": code}))),
    ]
    with fake_llm(rules):
        run, passed, log = d6.run_task(task, provider="anthropic", sandbox=SB)
    assert passed, (run.trace(), out)
    assert run.tool_names == ["run_python"] and len(log) == 1 and log[0].ok


@pytest.mark.parametrize("task", d6.TASKS, ids=lambda t: t.id)
def test_each_checker_rejects_a_confident_wrong_answer(task):
    wrong = {
        "total-revenue": "The total revenue is 12345.67.",
        "top-region": "The region with the highest revenue is "
        + next(r for r in d6.REGIONS if r != d6.KEY["top_region"])
        + ".",
        "avg-units-b": "About 3.2 units.",
        "missing": "There are 9 rows with missing units.",
        "top-month": "The best month is "
        + ("2026-01" if d6.KEY["top_month"] != "2026-01" else "2026-02")
        + ".",
        "north-march": "About 999.",
        "top-product": "Product "
        + next(p for p in d6.PRODUCTS if p != d6.KEY["top_product_units"])
        + " sold the most.",
    }
    assert not task.check(wrong[task.id])
    assert not task.check("I could not determine that.") and not task.check("")


def test_checkers_reject_answers_that_hedge_by_naming_several_candidates():
    top = d6.KEY["top_region"]
    other = next(r for r in d6.REGIONS if r != top)
    region = d6.TASKS[1]
    assert region.check(f"{top} is the top region.") and not region.check(
        f"Either {top} or {other}."
    )
    month = d6.TASKS[4]
    assert month.check(f"{d6.KEY['top_month']}") and not month.check(
        f"{d6.KEY['top_month']} or 2026-01 or 2026-02"
    )


def test_number_helpers():
    assert d6.numbers_in("Total: 1,234.50 and -3 and 7.") == [1234.5, -3.0, 7.0]
    assert d6.numbers_in("no digits") == [] and d6.numbers_in("") == []
    assert d6.close_to("revenue 1000.4", 1000) and not d6.close_to("revenue 1010", 1000)
    assert d6.close_to("1,000", 1000) and not d6.close_to("nothing", 1000)
    assert not d6.close_to("10", 0.0 + 1000) and d6.close_to("5 and 1000", 1000), (
        "any number may match"
    )


# ----------------------------------------------------------------------------- the tool and the loop


def test_run_python_tool_schema_is_documented_and_describes_the_sandbox_contract():
    tool_ = d6.make_run_python(SB)
    spec = tool_.spec()
    assert (
        lint_spec(spec) == []
        and "no network" in spec["description"].lower()
        and "do NOT persist" in spec["description"]
    )
    assert lint_spec(d6.describe_data.spec()) == []
    assert (
        "240 rows"
        in ToolRegistry([d6.describe_data]).execute(ToolCall("1", "describe_data", {})).content
    )


def test_each_call_gets_the_data_fresh_and_errors_return_a_short_traceback_tail():
    log = []
    reg = ToolRegistry([d6.make_run_python(SB, log=log)])
    first = reg.execute(
        ToolCall(
            "1",
            "run_python",
            {
                "code": "import pandas as pd\ndf = pd.read_csv('sales.csv')\ndf['x'] = 1\ndf.to_csv('out.csv')\nprint(df.shape)"
            },
        )
    )
    assert first.content.startswith("(240, 7)") and "out.csv" in first.content
    second = reg.execute(ToolCall("2", "run_python", {"code": "print(df.shape)"}))
    assert "NameError" in second.content and "name 'df' is not defined" in second.content, (
        "variables do not persist"
    )
    assert len(second.content.splitlines()) < 12 and second.content.startswith("stderr:")
    third = reg.execute(
        ToolCall("3", "run_python", {"code": "import os; print(os.path.exists('out.csv'))"})
    )
    assert third.content.strip() == "False"
    assert len(log) == 3


def test_the_agent_recovers_from_a_code_error_using_the_traceback():
    bad = "import pandas as pd\ndf = pd.read_csv('sales.csv')\nprint(df['units_sold'].sum())"
    good = PANDAS["missing"]
    script = [
        (
            r"KeyError: 'units_sold'",
            [tool_calls(("run_python", {"code": good}))],
        ),  # after seeing the error
        (
            r"(?s).*",
            [
                tool_calls(("run_python", {"code": bad})),
                tool_calls(("run_python", {"code": good})),
                "6 rows have missing units.",
            ],
        ),
    ]
    with fake_llm(
        [
            (
                r"(?s).*",
                [
                    tool_calls(("run_python", {"code": bad})),
                    tool_calls(("run_python", {"code": good})),
                    "6 rows have missing units.",
                ],
            )
        ]
    ):
        run, passed, log = d6.run_task(d6.TASKS[3], provider="anthropic", sandbox=SB)
    assert passed and [r.ok for r in log] == [False, True]
    assert "KeyError: 'units_sold'" in run.results[0].content and run.errors == 0, (
        "a failing script is a normal result, not a tool error"
    )
    del script


def test_the_agent_survives_hostile_code_and_still_answers():
    boom = tool_calls(("run_python", {"code": "while True: pass"}))
    sandbox = SubprocessSandbox(timeout_s=1)
    with fake_llm([(r"(?s).*", [boom, "I could not compute it."])]):
        run, passed, log = d6.run_task(d6.TASKS[0], provider="anthropic", sandbox=sandbox)
    assert run.ok and not passed and not log[0].ok
    assert "[stopped:" in run.results[0].content


def test_make_sandbox_chooses_by_argument_and_environment(monkeypatch):
    assert isinstance(d6.make_sandbox("docker"), DockerSandbox) and isinstance(
        d6.make_sandbox("subprocess"), SubprocessSandbox
    )
    monkeypatch.setenv("SANDBOX", "docker")
    assert isinstance(d6.make_sandbox(), DockerSandbox)
    monkeypatch.delenv("SANDBOX")
    assert isinstance(d6.make_sandbox(), SubprocessSandbox)


# ----------------------------------------------------------------------------- the escape matrix


def test_escape_matrix_on_the_host_blocks_resource_abuse_but_allows_the_documented_gaps():
    rows = {name: host for name, host, _ in d6.escape_matrix(docker=False)}
    blocked = [
        "infinite loop",
        "output flood (endless print)",
        "write a 50 MB file",
        "read your environment variables",
    ]
    allowed = [
        "read your secrets (a canary file outside)",
        "open a network connection",
        "write a file outside its directory",
    ]
    for name in blocked:
        assert rows[name] == "BLOCKED", name
    for name in allowed:
        assert rows[name] == "**ALLOWED**", f"{name}: a subprocess sandbox is not a boundary"
    assert (
        rows["fork bomb"] == "(not run on the host)"
        and rows["allocate 600 MB"] == "(not run on the host)"
    )


@pytest.mark.skipif(not DockerSandbox.available(), reason="Docker daemon not running")
def test_escape_matrix_in_docker_blocks_everything():
    rows = d6.escape_matrix(docker=True)
    assert len(rows) == len(d6.ATTEMPTS) and all(dock == "BLOCKED" for _, _, dock in rows), rows


def test_every_attempt_is_classified_and_the_dangerous_ones_never_run_on_the_host():
    assert [a.name for a in d6.ATTEMPTS if not a.host_safe] == ["fork bomb", "allocate 600 MB"]
    assert len({a.name for a in d6.ATTEMPTS}) == len(d6.ATTEMPTS) == 9
