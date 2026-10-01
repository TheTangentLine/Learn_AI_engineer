"""Week 5 Day 6 - Solution: a data-analysis agent that runs pandas code in a sandbox, and an escape matrix.

Part 1  The agent: ``run_python(code)`` executes model-written pandas code in a fresh sandbox with ``sales.csv`` in
        its working directory. Seven analysis tasks; every ground truth is computed in PLAIN PYTHON (no pandas),
        so a pandas mistake cannot hide in both the answer key and the solution.
Part 2  The escape matrix: the same hostile snippets run against SubprocessSandbox and DockerSandbox, printing what
        each one actually blocks. (The dangerous ones, a fork bomb and a memory bomb, only ever run in Docker.)

  uv run python weeks/week05_tool-use-and-agents/solutions/day6_solution.py              # local Qwen-0.5B + matrix
  SANDBOX=docker uv run python .../day6_solution.py                                      # agent code runs in Docker
"""

from __future__ import annotations

import csv
import io
import os
import random
import re
import socket
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.agent import AgentRun, run_agent  # noqa: E402
from common.sandbox import DockerSandbox, SandboxResult, SubprocessSandbox  # noqa: E402
from common.tools import ToolRegistry, tool  # noqa: E402

REGIONS = ["north", "south", "east", "west"]
PRODUCTS = ["A", "B", "C"]
PRICES = {"A": 12.5, "B": 40.0, "C": 7.25}

# ----------------------------------------------------------------------------- the data and the answer key


def build_rows(seed: int = 7, n: int = 240) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        d = date(2026, 1, 1) + timedelta(days=rng.randrange(0, 181))
        product = rng.choice(PRODUCTS)
        rows.append(
            {
                "order_id": 1000 + i,
                "date": d.isoformat(),
                "region": rng.choice(REGIONS),
                "product": product,
                "units": rng.randint(1, 20),
                "unit_price": PRICES[product],
            }
        )
    for i in rng.sample(range(n), 6):  # messy data: missing units
        rows[i]["units"] = ""
    return rows


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


ROWS = build_rows()
CSV_TEXT = to_csv(ROWS)


def _valid(rows):  # rows with a known number of units
    return [r for r in rows if r["units"] != ""]


def answer_key(rows: list[dict] = ROWS) -> dict:
    """Plain-Python ground truth. Rows with missing units are ignored (the task says so)."""
    valid = _valid(rows)
    by_region: dict[str, float] = defaultdict(float)
    by_month: dict[str, float] = defaultdict(float)
    units_by_product: dict[str, int] = defaultdict(int)
    for r in valid:
        revenue = r["units"] * r["unit_price"]
        by_region[r["region"]] += revenue
        by_month[r["date"][:7]] += revenue
        units_by_product[r["product"]] += r["units"]
    b = [r["units"] for r in valid if r["product"] == "B"]
    north_march = sum(
        r["units"] * r["unit_price"]
        for r in valid
        if r["region"] == "north" and r["date"][:7] == "2026-03"
    )
    return {
        "total_revenue": sum(by_region.values()),
        "top_region": max(by_region, key=by_region.get),
        "avg_units_b": sum(b) / len(b),
        "missing_units": sum(r["units"] == "" for r in rows),
        "top_month": max(by_month, key=by_month.get),
        "north_march": north_march,
        "top_product_units": max(units_by_product, key=units_by_product.get),
    }


KEY = answer_key()

# ----------------------------------------------------------------------------- the agent

SYSTEM = (
    "You are a data analyst. The file sales.csv has columns: order_id, date (YYYY-MM-DD), region, product, units, "
    "unit_price. Revenue is units * unit_price; ignore rows where units is missing. Use run_python to compute "
    "answers with pandas (always print() what you need), then reply with the answer in one sentence."
)


def make_run_python(sandbox, csv_text: str = CSV_TEXT, log: list | None = None) -> object:
    @tool
    def run_python(code: str) -> str:
        """Run Python 3 code in a fresh, isolated sandbox and return what it printed. pandas and numpy are installed.
        The file sales.csv is in the working directory. Only printed output (stdout) is returned, so print() every
        result you need. There is no network, and variables do NOT persist between calls: load the data again each time.

        Args:
            code: Complete Python source, for example "import pandas as pd; df = pd.read_csv('sales.csv'); print(df.shape)".
        """
        res = sandbox.run(code, files={"sales.csv": csv_text})
        if log is not None:
            log.append(res)
        return res.summary()

    return run_python


@tool
def describe_data() -> str:
    """Describe the dataset: its columns and the first three rows. Call this first if you are unsure of the schema."""
    lines = CSV_TEXT.splitlines()
    return f"{len(lines) - 1} rows. Columns and first rows:\n" + "\n".join(lines[:4])


def make_sandbox(kind: str | None = None):
    kind = kind or os.environ.get("SANDBOX", "subprocess")
    return DockerSandbox() if kind == "docker" else SubprocessSandbox(timeout_s=20)


def numbers_in(text: str) -> list[float]:
    return [
        float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text) if x.strip(",.")
    ]


def close_to(text: str, target: float, rel: float = 0.005) -> bool:
    return any(abs(n - target) <= rel * abs(target) for n in numbers_in(text))


@dataclass
class Task:
    id: str
    prompt: str
    check: Callable[[str], bool]


TASKS = [
    Task(
        "total-revenue", "What is the total revenue?", lambda a: close_to(a, KEY["total_revenue"])
    ),
    Task(
        "top-region",
        "Which region has the highest total revenue?",
        lambda a: (
            re.search(rf"\b{KEY['top_region']}\b", a, re.I) is not None
            and sum(r in a.lower() for r in REGIONS) == 1
        ),
    ),
    Task(
        "avg-units-b",
        "What is the average number of units per order for product B?",
        lambda a: close_to(a, KEY["avg_units_b"], 0.002),
    ),
    Task(
        "missing",
        "How many rows have a missing units value?",
        lambda a: any(n == KEY["missing_units"] for n in numbers_in(a)),
    ),
    Task(
        "top-month",
        "Which month (YYYY-MM) has the highest revenue?",
        lambda a: (
            KEY["top_month"] in a
            and sum(
                m in a for m in ("2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06")
            )
            == 1
        ),
    ),
    Task(
        "north-march",
        "What was the total revenue in the north region in March 2026?",
        lambda a: close_to(a, KEY["north_march"]),
    ),
    Task(
        "top-product",
        "Which product sold the most units in total?",
        lambda a: (
            re.search(rf"\b(product )?{KEY['top_product_units']}\b", a) is not None
            and not any(
                re.search(rf"\bproduct {p}\b", a, re.I)
                for p in PRODUCTS
                if p != KEY["top_product_units"]
            )
        ),
    ),
]


def run_task(
    task: Task, *, provider: str | None, sandbox=None, max_steps: int = 6
) -> tuple[AgentRun, bool, list[SandboxResult]]:
    log: list[SandboxResult] = []
    reg = ToolRegistry([describe_data, make_run_python(sandbox or make_sandbox(), log=log)])
    run = run_agent(task.prompt, reg, system=SYSTEM, provider=provider, max_steps=max_steps)
    return run, bool(run.ok and task.check(run.answer)), log


# ----------------------------------------------------------------------------- the escape matrix


@dataclass
class Attempt:
    name: str
    code: Callable[[dict], str]  # given context (canary path, port, outside path) -> code
    blocked: Callable[[SandboxResult, dict], bool]
    host_safe: bool = True  # False: only ever run inside Docker


ATTEMPTS = [
    Attempt("infinite loop", lambda c: "while True: pass", lambda r, c: not r.ok),
    Attempt(
        "output flood (endless print)",
        lambda c: "while True: print('x' * 1000)",
        lambda r, c: len(r.stdout) <= 20_000 and not r.ok,
    ),
    Attempt(
        "write a 50 MB file",
        lambda c: "open('big', 'wb').write(b'x' * 50_000_000)\nprint('wrote')",
        lambda r, c: "wrote" not in r.stdout,
    ),
    Attempt(
        "read your secrets (a canary file outside)",
        lambda c: f"print(open({c['canary']!r}).read())",
        lambda r, c: "CANARY" not in r.stdout,
    ),
    Attempt(
        "read your environment variables",
        lambda c: "import os; print(os.environ.get('SANDBOX_DEMO_SECRET', 'absent'))",
        lambda r, c: "hunter2" not in r.stdout,
    ),
    Attempt(
        "open a network connection",
        lambda c: (
            f"import socket; socket.create_connection(('127.0.0.1', {c['port']}), timeout=2); print('connected')"
        ),
        lambda r, c: "connected" not in r.stdout,
    ),
    Attempt(
        "write a file outside its directory",
        lambda c: f"open({c['outside']!r}, 'w').write('x'); print('escaped')",
        lambda r, c: "escaped" not in r.stdout,
    ),
    Attempt(
        "fork bomb",
        lambda c: "import os\nfor _ in range(10000):\n    os.fork()\nprint('forked a lot')",
        lambda r, c: "forked a lot" not in r.stdout,
        host_safe=False,
    ),
    Attempt(
        "allocate 600 MB",
        lambda c: "x = bytearray(600 * 1024 * 1024)\nprint('allocated')",
        lambda r, c: "allocated" not in r.stdout,
        host_safe=False,
    ),
]


def escape_matrix(docker: bool = True) -> list[tuple[str, str, str]]:
    """Rows of (attempt, subprocess verdict, docker verdict)."""
    rows = []
    with tempfile.TemporaryDirectory() as tmp, socket.socket() as srv:
        canary = Path(tmp) / "canary.txt"
        canary.write_text("TOP-SECRET-CANARY")
        srv.bind(("127.0.0.1", 0))
        srv.listen(5)
        ctx = {
            "canary": str(canary),
            "port": srv.getsockname()[1],
            "outside": str(Path(tmp) / "escaped.txt"),
        }
        os.environ["SANDBOX_DEMO_SECRET"] = "hunter2"
        host = SubprocessSandbox(timeout_s=3, max_file_bytes=1_000_000)
        box = (
            DockerSandbox(timeout_s=60, memory="128m", pids=32)
            if docker and DockerSandbox.available()
            else None
        )
        for a in ATTEMPTS:
            code = a.code(ctx)
            if a.host_safe:
                r = host.run(code)
                host_v = "BLOCKED" if a.blocked(r, ctx) else "**ALLOWED**"
            else:
                host_v = "(not run on the host)"
            if box is None:
                dock_v = "(docker unavailable)"
            else:
                r = box.run(code)
                dock_v = "BLOCKED" if a.blocked(r, ctx) else "**ALLOWED**"
            rows.append((a.name, host_v, dock_v))
        os.environ.pop("SANDBOX_DEMO_SECRET", None)
    return rows


def main() -> None:
    provider = os.environ.get("LLM_PROVIDER") or "local"
    print(
        f"provider={provider} sandbox={os.environ.get('SANDBOX', 'subprocess')}\nground truth: {KEY}\n"
    )
    passed = 0
    for task in TASKS:
        run, ok, log = run_task(task, provider=provider)
        passed += ok
        print(run.trace())
        print(
            f"==> {task.id}: {'PASS' if ok else 'FAIL'} ({run.status}, {len(log)} sandbox runs)\n"
        )
    print(f"passed {passed}/{len(TASKS)}\n")
    print(f"{'attempt':46s} {'subprocess':24s} docker")
    for name, h, d in escape_matrix():
        print(f"{name:46s} {h:24s} {d}")


if __name__ == "__main__":
    main()
