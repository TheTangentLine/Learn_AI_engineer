"""Week 5 Day 2 - Solution: a raw-loop agent that solves multi-step file tasks (no framework).

The loop itself lives in ``common/agent.py`` (read it: ~150 lines). This file adds:
  * ``Workspace``: a sandboxed project directory + four file tools (list_dir, read_file, grep, write_file).
    Every path is resolved and must stay inside the workspace (``..``, absolute paths and symlinks are refused).
  * ``build_workspace``: a deterministic fixture project (code with TODOs, a CSV, a config, a 400-line log).
  * ``TASKS``: seven tasks, each with a checker that looks at the FINAL STATE (answer text and files written),
    including one with no answer in the data, where the right behaviour is to say so.

  uv run python weeks/week05_tool-use-and-agents/solutions/day2_solution.py             # local Qwen-0.5B
  LLM_PROVIDER=anthropic uv run python weeks/week05_tool-use-and-agents/solutions/day2_solution.py
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from day1_solution import calculate  # noqa: E402

from common.agent import AgentRun, run_agent  # noqa: E402
from common.tools import ToolRegistry, tool  # noqa: E402

MAX_READ_BYTES = 1_000_000
MAX_WRITE_CHARS = 100_000
MAX_GREP_MATCHES = 50

# ----------------------------------------------------------------------------- sandboxed file tools


class Workspace:
    """A directory the agent may touch, and nothing outside it."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def resolve(self, path: str) -> Path:
        p = (self.root / path).resolve()  # resolves ".." AND symlinks before the containment check
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"path {path!r} is outside the workspace")
        return p

    def rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix() or "."

    def tools(self) -> list:
        ws = self

        @tool
        def list_dir(path: str = ".") -> str:
            """List the files and folders in a directory (folders end with '/').

            Args:
                path: Directory relative to the project root; "." is the root.
            """
            d = ws.resolve(path)
            if not d.is_dir():
                raise ValueError(f"{path!r} is not a directory")
            names = sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name))
            return (
                "\n".join(
                    x.name + ("/" if x.is_dir() else f"  ({x.stat().st_size} bytes)") for x in names
                )
                or "(empty)"
            )

        @tool
        def read_file(path: str, start_line: int = 1, max_lines: int = 100) -> str:
            """Read a text file. Returns numbered lines; says how many lines remain if the file is longer.

            Args:
                path: File relative to the project root.
                start_line: First line to return (1-based).
                max_lines: How many lines to return (at most 200).
            """
            f = ws.resolve(path)
            if not f.is_file():
                raise ValueError(f"{path!r} is not a file (use list_dir to see what exists)")
            if f.stat().st_size > MAX_READ_BYTES:
                raise ValueError(
                    f"{path!r} is larger than {MAX_READ_BYTES} bytes; use grep instead"
                )
            try:
                lines = f.read_text().splitlines()
            except UnicodeDecodeError:
                raise ValueError(f"{path!r} is not a text file") from None
            if start_line < 1:
                raise ValueError("start_line must be >= 1")
            n = max(1, min(max_lines, 200))
            chunk = lines[start_line - 1 : start_line - 1 + n]
            out = "\n".join(f"{start_line + i}: {line}" for i, line in enumerate(chunk))
            more = len(lines) - (start_line - 1 + len(chunk))
            if more > 0:
                out += (
                    f"\n[{more} more lines; call again with start_line={start_line + len(chunk)}]"
                )
            return out or "(empty file)"

        @tool
        def grep(pattern: str, path: str = ".") -> str:
            """Search files for a regular expression. Returns 'file:line: text' for each match and the total count.

            Args:
                pattern: Python regular expression, e.g. "TODO" or "def parse_\\w+".
                path: A file or directory (searched recursively), relative to the project root.
            """
            try:
                rx = re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regular expression: {exc}") from None
            base = ws.resolve(path)
            files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
            # a symlink INSIDE the workspace can point outside it: resolve each file and keep only the inside ones
            files = [f for f in files if f.resolve() == ws.root or ws.root in f.resolve().parents]
            hits: list[str] = []
            total = 0
            for f in files:
                try:
                    lines = f.read_text().splitlines()
                except (UnicodeDecodeError, OSError):
                    continue  # binary or unreadable: skip silently, it cannot match text
                for i, line in enumerate(lines, 1):
                    if rx.search(line):
                        total += 1
                        if len(hits) < MAX_GREP_MATCHES:
                            hits.append(f"{ws.rel(f)}:{i}: {line.strip()[:200]}")
            if not total:
                return f"0 matches for {pattern!r} in {path!r}."
            tail = f"\n[showing {len(hits)} of {total} matches]" if total > len(hits) else ""
            return "\n".join(hits) + f"\n{total} matches total.{tail}"

        @tool
        def write_file(path: str, content: str) -> str:
            """Create or overwrite a text file inside the project.

            Args:
                path: File relative to the project root (parent folders are created).
                content: The full new contents of the file.
            """
            if len(content) > MAX_WRITE_CHARS:
                raise ValueError(
                    f"content too long ({len(content)} > {MAX_WRITE_CHARS} characters)"
                )
            f = ws.resolve(path)
            if f.is_dir():  # also covers the workspace root itself
                raise ValueError(f"{path!r} is a directory")
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content)
            return f"Wrote {len(content)} characters to {ws.rel(f)}."

        return [list_dir, read_file, grep, write_file]

    def registry(self, extra: list | None = None) -> ToolRegistry:
        return ToolRegistry([*self.tools(), *(extra or [])])


# ----------------------------------------------------------------------------- fixture project

APP = '''"""Entry point."""
import config
import billing


def main():
    cfg = config.parse_config("config.toml")  # TODO: handle a missing file
    billing.charge(cfg["port"], 10)  # TODO: retry on failure


if __name__ == "__main__":
    main()
'''
CONFIG = """def parse_config(path):
    # TODO: validate the port range
    out = {}
    for line in open(path):
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"')
    return out
"""
BILLING = """def charge(customer, cents):
    # TODO: idempotency key
    return {"customer": customer, "cents": cents}


def refund(customer, cents):
    # TODO: partial refunds
    return charge(customer, -cents)
"""
UTILS = """def clamp(x, lo, hi):
    return max(lo, min(hi, x))
"""
CSV = "region,amount\nnorth,120.50\nsouth,80.00\neast,45.25\nwest,154.25\n"
TOML = 'name = "orders-api"\nport = 8443\ndb_password_env = "DB_PASSWORD"\n'
ERROR_LINES = {17, 58, 131, 204, 266, 313, 377}  # 7 errors among 400 lines


def build_workspace(root: Path) -> Workspace:
    root = Path(root)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    for name, text in (
        ("app.py", APP),
        ("config.py", CONFIG),
        ("billing.py", BILLING),
        ("utils.py", UTILS),
    ):
        (root / "src" / name).write_text(text)
    (root / "data" / "sales.csv").write_text(CSV)
    (root / "VERSION").write_text("2.7.1\n")
    (root / "config.toml").write_text(TOML)
    (root / "README.md").write_text(
        "# orders-api\n\nA small demo service. See config.toml for settings.\n"
    )
    log = [
        f"2026-09-30 10:{i % 60:02d}:{i % 60:02d} {'ERROR payment timeout' if i in ERROR_LINES else 'INFO request ok'} id={i}"
        for i in range(1, 401)
    ]
    (root / "logs" / "app.log").write_text("\n".join(log) + "\n")
    return Workspace(root)


# ----------------------------------------------------------------------------- tasks and checkers


@dataclass
class Task:
    id: str
    prompt: str
    check: Callable[[AgentRun, Workspace], bool]
    min_calls: int  # fewest tool calls a correct solution can use (for an efficiency number)


def _says(run: AgentRun, pattern: str) -> bool:
    return re.search(pattern, run.answer, re.I) is not None


def _report_ok(run: AgentRun, ws: Workspace) -> bool:
    f = ws.root / "report.txt"
    return f.is_file() and f.read_text().split() == [
        "app.py",
        "billing.py",
        "config.py",
        "utils.py",
    ]


TASKS = [
    Task(
        "todo-count",
        "How many TODO comments are there in total in the src folder?",
        lambda r, w: _says(r, r"(?<![\d.])5(?!\d|\.\d)"),
        1,
    ),
    Task(
        "find-def",
        "Which file defines the function parse_config?",
        lambda r, w: _says(r, r"src/config\.py|config\.py"),
        1,
    ),
    Task(
        "csv-total",
        "What is the total of the amount column in data/sales.csv?",
        lambda r, w: _says(r, r"\b400(\.0+)?\b"),
        2,
    ),
    Task(
        "write-report",
        "List the .py files in src, one filename per line in alphabetical order, and save that list to report.txt.",
        _report_ok,
        2,
    ),
    Task(
        "version-port",
        "What are the version (VERSION file) and the port (config.toml) of this project?",
        lambda r, w: _says(r, r"2\.7\.1") and _says(r, r"8443"),
        2,
    ),
    Task(
        "log-errors",
        "How many ERROR lines are in logs/app.log?",
        lambda r, w: _says(r, r"(?<![\d.])7(?!\d|\.\d)"),
        1,
    ),
    Task("no-answer", "What is the database password?", lambda r, w: _says(r, r"DB_PASSWORD"), 1),
]


def run_task(task: Task, *, provider: str | None, max_steps: int = 8) -> tuple[AgentRun, bool]:
    with tempfile.TemporaryDirectory() as tmp:
        ws = build_workspace(Path(tmp))
        run = run_agent(
            task.prompt, ws.registry([calculate]), provider=provider, max_steps=max_steps
        )
        return run, bool(run.ok and task.check(run, ws))


def main() -> None:
    provider = os.environ.get("LLM_PROVIDER") or "local"
    print(f"provider={provider}\n")
    rows = []
    for task in TASKS:
        run, passed = run_task(task, provider=provider)
        rows.append((task, run, passed))
        print(run.trace())
        print(
            f"==> {task.id}: {'PASS' if passed else 'FAIL'} ({run.status}, {len(run.calls)} calls, {run.input_tokens} in-tokens)\n"
        )
    ok = sum(p for _, _, p in rows)
    print(f"passed {ok}/{len(rows)};  statuses: {[r.status for _, r, _ in rows]}")
    print(
        f"total tool calls {sum(len(r.calls) for _, r, _ in rows)} (minimum possible {sum(t.min_calls for t, _, _ in rows)}), "
        f"tool errors {sum(r.errors for _, r, _ in rows)}, cost ${sum(r.cost_usd for _, r, _ in rows):.4f}"
    )


if __name__ == "__main__":
    main()
