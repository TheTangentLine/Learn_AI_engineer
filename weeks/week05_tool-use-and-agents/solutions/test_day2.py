"""Tests for Week 5 Day 2: the sandboxed file tools, the fixture, and that every task is solvable and checkable."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))

import day2_solution as d2  # noqa: E402

from common.chat import ToolCall  # noqa: E402
from common.fake import fake_llm, tool_calls  # noqa: E402


@pytest.fixture()
def ws(tmp_path):
    return d2.build_workspace(tmp_path / "project")


def call(ws, name, **args):
    return ws.registry().execute(ToolCall("t", name, args))


# ----------------------------------------------------------------------------- sandbox


@pytest.mark.parametrize(
    "path",
    ["../outside.txt", "/etc/passwd", "src/../../etc/passwd", "src/../..", "~/../../etc/hosts"],
)
def test_paths_outside_the_workspace_are_refused_by_every_tool(ws, path):
    for name, args in (
        ("read_file", {"path": path}),
        ("list_dir", {"path": path}),
        ("grep", {"pattern": "x", "path": path}),
        ("write_file", {"path": path, "content": "pwned"}),
    ):
        r = call(ws, name, **args)
        assert r.is_error, (name, path, r.content)
        assert (
            "outside the workspace" in r.content
            or "not a file" in r.content
            or "not a directory" in r.content
        )
    assert not (ws.root.parent / "outside.txt").exists()


def test_symlink_escape_is_refused(ws, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")
    (ws.root / "link.txt").symlink_to(secret)
    (ws.root / "linkdir").symlink_to(tmp_path)
    assert "outside the workspace" in call(ws, "read_file", path="link.txt").content
    assert "outside the workspace" in call(ws, "list_dir", path="linkdir").content
    assert "outside the workspace" in call(ws, "write_file", path="link.txt", content="x").content
    assert secret.read_text() == "TOP SECRET"


def test_grep_over_the_root_does_not_follow_symlinked_files_out(ws, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET\n")
    (ws.root / "link.txt").symlink_to(secret)
    out = call(ws, "grep", pattern="SECRET").content
    assert "TOP SECRET" not in out, (
        "recursive search must not read through a symlink that leaves the workspace"
    )


def test_a_workspace_sibling_with_the_same_prefix_is_not_inside(tmp_path):
    inside = d2.build_workspace(tmp_path / "proj")
    (tmp_path / "proj-evil").mkdir()
    (tmp_path / "proj-evil" / "x.txt").write_text("nope")
    with pytest.raises(ValueError, match="outside"):
        inside.resolve("../proj-evil/x.txt")
    (tmp_path / "other" / "proj").mkdir(parents=True)
    with pytest.raises(ValueError, match="outside"):
        inside.resolve("../other/proj/x.txt")  # same folder NAME, different folder


# ----------------------------------------------------------------------------- tools


def test_list_dir_sorts_folders_first_and_marks_them(ws):
    out = call(ws, "list_dir").content.splitlines()
    assert out[:3] == ["data/", "logs/", "src/"] and any(line.startswith("VERSION") for line in out)
    assert call(ws, "list_dir", path="src").content.splitlines()[0].startswith("app.py")
    assert "not a directory" in call(ws, "list_dir", path="VERSION").content
    assert call(ws, "list_dir", path="nope").is_error


def test_read_file_numbers_lines_and_paginates_with_a_hint(ws):
    first = call(ws, "read_file", path="logs/app.log", max_lines=3).content
    assert (
        first.splitlines()[0].startswith("1: 2026")
        and "[397 more lines; call again with start_line=4]" in first
    )
    mid = call(ws, "read_file", path="logs/app.log", start_line=398, max_lines=100).content
    assert (
        mid.splitlines()[0].startswith("398:")
        and "more lines" not in mid
        and mid.splitlines()[-1].startswith("400:")
    )
    (ws.root / "short.txt").write_text(
        "x\n" * 500
    )  # short lines: the result-size limit cannot hide the line cap
    capped = call(ws, "read_file", path="short.txt", max_lines=10_000).content
    assert (
        capped.count("\n") == 200 and "[300 more lines; call again with start_line=201]" in capped
    )
    (ws.root / "short201.txt").write_text("x\n" * 201)
    one_left = call(ws, "read_file", path="short201.txt", max_lines=200).content
    assert one_left.endswith("[1 more lines; call again with start_line=201]")
    assert call(ws, "read_file", path="VERSION").content == "1: 2.7.1"


def test_read_file_errors_say_what_to_do(ws):
    assert "list_dir" in call(ws, "read_file", path="missing.txt").content
    assert "not a file" in call(ws, "read_file", path="src").content
    assert "start_line must be" in call(ws, "read_file", path="VERSION", start_line=0).content
    (ws.root / "blob.bin").write_bytes(b"\xff\xfe\x00\x01" * 10)
    assert "not a text file" in call(ws, "read_file", path="blob.bin").content
    (ws.root / "big.txt").write_text("x" * (d2.MAX_READ_BYTES + 1))
    assert "use grep instead" in call(ws, "read_file", path="big.txt").content
    assert call(ws, "read_file", path="empty.txt").is_error, "a file that does not exist yet"
    (ws.root / "empty.txt").write_text("")
    assert call(ws, "read_file", path="empty.txt").content == "(empty file)"


def test_grep_counts_all_matches_but_shows_a_bounded_number(ws):
    todo = call(ws, "grep", pattern="TODO", path="src").content
    assert (
        todo.endswith("5 matches total.") and todo.count("\n") == 5 and "src/billing.py:2:" in todo
    )
    info = call(ws, "grep", pattern="INFO", path="logs").content
    assert (
        "393 matches total." in info and f"[showing {d2.MAX_GREP_MATCHES} of 393 matches]" in info
    )
    assert info.count("logs/app.log:") == d2.MAX_GREP_MATCHES
    assert call(ws, "grep", pattern="zzzz").content.startswith("0 matches")
    assert "invalid regular expression" in call(ws, "grep", pattern="(unclosed").content
    one_file = call(ws, "grep", pattern="ERROR", path="logs/app.log").content
    assert "7 matches total." in one_file


def test_write_file_creates_folders_overwrites_and_has_limits(ws):
    assert (
        call(ws, "write_file", path="out/deep/a.txt", content="hi").content
        == "Wrote 2 characters to out/deep/a.txt."
    )
    assert (ws.root / "out/deep/a.txt").read_text() == "hi"
    call(ws, "write_file", path="out/deep/a.txt", content="second")
    assert (ws.root / "out/deep/a.txt").read_text() == "second"
    for target in ("src", "."):
        r = call(ws, "write_file", path=target, content="x")
        assert r.is_error and "ValueError" in r.content and "is a directory" in r.content, r.content
    assert (
        "too long"
        in call(ws, "write_file", path="big.txt", content="x" * (d2.MAX_WRITE_CHARS + 1)).content
    )
    assert not (ws.root / "big.txt").exists()


def test_every_tool_schema_is_strict_and_documented(ws):
    for spec in ws.registry().specs():
        p = spec["parameters"]
        assert p["additionalProperties"] is False and spec["description"]
        assert all(v.get("description") for v in p["properties"].values()), spec["name"]
    specs = {s["name"]: s for s in ws.registry().specs()}
    assert specs["list_dir"]["parameters"].get("required", []) == []
    assert specs["read_file"]["parameters"]["required"] == ["path"]


# ----------------------------------------------------------------------------- fixture facts the tasks depend on


def test_fixture_numbers_match_what_the_tasks_expect(ws):
    assert call(ws, "grep", pattern="TODO", path="src").content.count("TODO") >= 5
    assert sum(float(x.split(",")[1]) for x in d2.CSV.splitlines()[1:]) == 400.0
    log = (ws.root / "logs/app.log").read_text().splitlines()
    assert len(log) == 400 and sum("ERROR" in line for line in log) == 7
    assert sorted(p.name for p in (ws.root / "src").iterdir()) == [
        "app.py",
        "billing.py",
        "config.py",
        "utils.py",
    ]
    assert "parse_config" in (ws.root / "src/config.py").read_text()
    assert (
        "DB_PASSWORD" in (ws.root / "config.toml").read_text()
        and "password" not in (ws.root / "src/billing.py").read_text()
    )


# ----------------------------------------------------------------------------- tasks: solvable AND checkable

SOLUTIONS = {
    "todo-count": [
        tool_calls(("grep", {"pattern": "TODO", "path": "src"})),
        "There are 5 TODO comments.",
    ],
    "find-def": [
        tool_calls(("grep", {"pattern": "def parse_config"})),
        "It is defined in src/config.py.",
    ],
    "csv-total": [
        tool_calls(("read_file", {"path": "data/sales.csv"})),
        tool_calls(("calculate", {"expression": "120.50 + 80 + 45.25 + 154.25"})),
        "The total is 400.",
    ],
    "write-report": [
        tool_calls(("list_dir", {"path": "src"})),
        tool_calls(
            (
                "write_file",
                {"path": "report.txt", "content": "app.py\nbilling.py\nconfig.py\nutils.py\n"},
            )
        ),
        "Saved report.txt.",
    ],
    "version-port": [
        tool_calls(("read_file", {"path": "VERSION"}), ("read_file", {"path": "config.toml"})),
        "Version 2.7.1, port 8443.",
    ],
    "log-errors": [
        tool_calls(("grep", {"pattern": "ERROR", "path": "logs/app.log"})),
        "7 ERROR lines.",
    ],
    "no-answer": [
        tool_calls(("grep", {"pattern": "(?i)password"})),
        "There is no password in the project; config.toml says it comes from the DB_PASSWORD environment variable.",
    ],
}
WRONG = {  # confident, plausible, wrong: every checker must reject it
    "todo-count": "There are 3 TODO comments.",
    "find-def": "It is defined in src/app.py.",
    "csv-total": "The total is 399.",
    "write-report": "I saved the list.",
    "version-port": "Version 2.7.1, port 8080.",
    "log-errors": "There are 17 ERROR lines.",
    "no-answer": "The database password is hunter2.",
}


@pytest.mark.parametrize("task", d2.TASKS, ids=lambda t: t.id)
def test_each_task_is_solved_by_a_scripted_agent_in_the_expected_number_of_calls(task):
    with fake_llm([(r"(?s).*", SOLUTIONS[task.id])]):
        run, passed = d2.run_task(task, provider="anthropic")
    assert passed and run.ok, run.trace()
    assert run.errors == 0 and len(run.calls) == task.min_calls


@pytest.mark.parametrize("task", d2.TASKS, ids=lambda t: t.id)
def test_each_checker_rejects_a_confident_wrong_answer_without_investigating(task):
    with fake_llm([(r"(?s).*", WRONG[task.id])]):
        run, passed = d2.run_task(task, provider="anthropic")
    assert run.ok and not passed, "the run finished, but the checker must still say FAIL"


def test_a_run_that_ran_out_of_steps_never_passes_even_if_its_text_looks_right():
    task = d2.TASKS[0]
    with fake_llm([(r"(?s).*", [tool_calls(("grep", {"pattern": "TODO"}))] * 1)]):
        run, passed = d2.run_task(task, provider="anthropic", max_steps=2)
    assert run.status == "max_steps" and not passed


def test_the_report_checker_looks_at_the_file_not_the_answer(tmp_path):
    ws = d2.build_workspace(tmp_path / "p")
    task = next(t for t in d2.TASKS if t.id == "write-report")
    run = d2.AgentRun("x", status="done", answer="saved")
    assert not task.check(run, ws)
    (ws.root / "report.txt").write_text("billing.py\napp.py\nconfig.py\nutils.py\n")
    assert not task.check(run, ws), "order matters"
    (ws.root / "report.txt").write_text("app.py\nbilling.py\nconfig.py\nutils.py\n")
    assert task.check(run, ws)


def test_number_checkers_do_not_match_digits_inside_other_numbers():
    todo = d2.TASKS[0]
    ws = None
    for text, expected in [
        ("5", True),
        ("15 TODOs", False),
        ("2.5", False),
        ("5.", True),
        ("There are 5", True),
    ]:
        run = d2.AgentRun("x", status="done", answer=text)
        assert todo.check(run, ws) is expected, text


def test_tasks_have_unique_ids_and_the_no_answer_task_exists():
    assert len({t.id for t in d2.TASKS}) == len(d2.TASKS) == 7
    assert any(t.id == "no-answer" for t in d2.TASKS) and os.path.exists(d2.__file__)
