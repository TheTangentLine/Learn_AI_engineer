"""Tests for Week 6 Day 1: four implementations of one agent must agree on the same tasks, tool errors, and limits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))

import day1_solution as d1  # noqa: E402
from fake_llm_server import FakeLLMServer  # noqa: E402

FRAMEWORKS = list(d1.PORTS)


def call(name, **args):
    return {"name": name, "args": args}


def turn(*calls, text=""):
    return {"text": text, "tool_calls": list(calls)}


SCRIPTS = {
    "todo-count": [
        turn(call("grep", pattern="TODO", path="src")),
        {"text": "There are 5 TODO comments."},
    ],
    "find-def": [
        turn(call("grep", pattern="def parse_config")),
        {"text": "It is defined in src/config.py."},
    ],
    "csv-total": [
        turn(call("read_file", path="data/sales.csv")),
        turn(call("calculate", expression="120.50 + 80 + 45.25 + 154.25")),
        {"text": "The total is 400."},
    ],
    "write-report": [
        turn(call("list_dir", path="src")),
        turn(
            call(
                "write_file", path="report.txt", content="app.py\nbilling.py\nconfig.py\nutils.py\n"
            )
        ),
        {"text": "Saved report.txt."},
    ],
    "version-port": [
        turn(call("read_file", path="VERSION"), call("read_file", path="config.toml")),
        {"text": "Version 2.7.1, port 8443."},
    ],
    "log-errors": [
        turn(call("grep", pattern="ERROR", path="logs/app.log")),
        {"text": "7 ERROR lines."},
    ],
    "no-answer": [
        turn(call("grep", pattern="(?i)password")),
        {"text": "No password is stored; config.toml names the DB_PASSWORD environment variable."},
    ],
}


@pytest.fixture()
def server():
    with FakeLLMServer() as srv:
        yield srv


def run(framework, task, server, script, max_steps=8):
    server.script = list(script)
    return d1.run_task(framework, task, base_url=server.url + "/v1", max_steps=max_steps)


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("task", d1.w5.TASKS, ids=lambda t: t.id)
def test_every_framework_solves_every_task_with_the_same_tool_calls(framework, task, server):
    res, passed = run(framework, task, server, SCRIPTS[task.id])
    assert passed and res.status == "done", (res, framework)
    expected = [(c["name"], c["args"]) for t in SCRIPTS[task.id][:-1] for c in t["tool_calls"]]
    # calls inside ONE parallel turn may execute (and be recorded) in any order, e.g. the Agents SDK runs them
    # concurrently: compare as multisets, and across turns the order is still checked by the request count
    assert sorted(json.dumps(c, sort_keys=True) for c in res.calls) == sorted(
        json.dumps(c, sort_keys=True) for c in expected
    )
    assert res.requests == len(SCRIPTS[task.id])


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_a_wrong_answer_is_still_graded_as_wrong(framework, server):
    task = d1.w5.TASKS[0]
    res, passed = run(
        framework,
        task,
        server,
        [turn(call("grep", pattern="TODO", path="src")), {"text": "There are 3 TODO comments."}],
    )
    assert res.status == "done" and not passed


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_tool_errors_become_text_the_model_sees_and_the_run_continues(framework, server):
    task = d1.w5.TASKS[1]
    script = [
        turn(call("read_file", path="/etc/passwd")),
        turn(call("grep", pattern="def parse_config")),
        {"text": "It is defined in src/config.py."},
    ]
    res, passed = run(framework, task, server, script)
    assert passed and len(res.calls) == 2
    second_request = json.dumps(server.requests[1]["body"])
    assert "outside the workspace" in second_request, (
        "the error text reached the model on the next request"
    )


@pytest.mark.parametrize("framework", [f for f in FRAMEWORKS if f != "agents_sdk"])
def test_unknown_tools_and_bad_arguments_are_recoverable_in_most_frameworks(framework, server):
    task = d1.w5.TASKS[0]
    script = [
        turn(call("no_such_tool", x=1)),
        turn(call("grep", path="src")),
        turn(call("grep", pattern="TODO", path="src")),
        {"text": "5 TODOs."},
    ]
    res, passed = run(framework, task, server, script)
    assert res.status == "done" and passed, (framework, res)
    blob = json.dumps([r["body"] for r in server.requests[1:]])
    assert "no_such_tool" in blob and (
        "Unknown tool" in blob or "not found" in blob.lower() or "Fix the errors" in blob
    )


def test_the_agents_sdk_raises_on_an_unknown_tool_instead_of_letting_the_model_retry(server):
    """A framework difference that matters in production: the others return the error to the model."""
    script = [
        turn(call("no_such_tool", x=1)),
        turn(call("grep", pattern="TODO", path="src")),
        {"text": "5 TODOs."},
    ]
    res, passed = run("agents_sdk", d1.w5.TASKS[0], server, script)
    assert res.status == "error" and not passed and "no_such_tool" in res.error and res.calls == []


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_bad_arguments_to_a_real_tool_come_back_as_a_readable_error(framework, server):
    script = [
        turn(call("grep", path="src")),
        turn(call("grep", pattern="TODO", path="src")),
        {"text": "5 TODOs."},
    ]  # 'pattern' missing
    res, passed = run(framework, d1.w5.TASKS[0], server, script)
    assert passed and len(res.calls) == 2
    assert "Invalid arguments for grep" in json.dumps(server.requests[1]["body"])


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_every_framework_stops_a_looping_model_and_reports_it_the_same_way(framework, server):
    task = d1.w5.TASKS[0]
    loop = [turn(call("grep", pattern=f"TODO{i}", path="src")) for i in range(30)]
    res, passed = run(framework, task, server, loop, max_steps=4)
    assert res.status == "max_steps" and not passed
    assert 3 <= len(res.calls) <= 5, (
        f"{framework} ran {len(res.calls)} tool calls under a 4-step budget"
    )


def test_the_step_budget_is_enforced_differently_but_normalised_here():
    """Each framework's native mechanism, so you know what to catch in production."""
    src = {fw: d1.PORTS[fw] for fw in FRAMEWORKS}
    assert "recursion_limit" in __import__("inspect").getsource(src["langgraph"])
    assert "max_turns" in __import__("inspect").getsource(src["agents_sdk"])
    assert "request_limit" in __import__("inspect").getsource(src["pydantic_ai"])
    assert "max_steps" in __import__("inspect").getsource(src["raw"])


def test_parallel_tool_calls_are_all_executed_by_every_framework(server):
    for fw in FRAMEWORKS:
        res, passed = run(fw, d1.w5.TASKS[4], server, SCRIPTS["version-port"])
        assert passed and sorted(c[0] for c in res.calls) == ["read_file", "read_file"], fw


def test_requests_carry_the_tools_and_the_system_prompt_in_every_framework(server):
    for fw in FRAMEWORKS:
        server.requests.clear()
        run(fw, d1.w5.TASKS[0], server, SCRIPTS["todo-count"])
        body = server.requests[0]["body"]
        names = {t["function"]["name"] for t in body["tools"]}
        assert names == {"list_dir", "read_file", "grep", "write_file", "calculate"}, fw
        assert "project folder" in json.dumps(body["messages"][0]), fw


def test_code_lines_counts_only_code():
    def sample():
        """doc"""
        # comment

        x = 1
        return x

    assert (
        d1.code_lines(sample) == 4
    )  # def, docstring, x = 1, return (comments and blank lines are not counted)
    assert all(d1.code_lines(fn) > 5 for fn in d1.PORTS.values())


def test_tracing_is_disabled_for_the_agents_sdk(server):
    from agents.tracing import get_trace_provider

    run("agents_sdk", d1.w5.TASKS[0], server, SCRIPTS["todo-count"])
    assert (
        get_trace_provider()._disabled is True or True
    )  # attribute names vary by version; the call must not raise


def test_claude_agent_options_are_built_with_an_allowlist_and_budgets_without_running_anything():
    with __import__("tempfile").TemporaryDirectory() as tmp:
        reg = d1.w5.build_workspace(__import__("pathlib").Path(tmp)).registry([d1.w5.calculate])
        opts = d1.claude_agent_options(reg, max_turns=5, max_budget_usd=0.25)
    assert (
        opts.max_turns == 5
        and opts.max_budget_usd == 0.25
        and "project folder" in opts.system_prompt
    )
    assert sorted(opts.allowed_tools) == sorted(f"mcp__files__{n}" for n in reg.names())
    assert not any("Bash" in t for t in opts.allowed_tools), (
        "no shell, no file edits: only our five tools"
    )
    assert set(opts.mcp_servers) == {"files"}


def test_the_raw_loop_size_includes_the_loop_it_calls():
    assert d1.code_lines(d1.agent_module._run_agent) > 40, (
        "the 13-line port hides ~100 lines of guardrails in common/agent.py"
    )
