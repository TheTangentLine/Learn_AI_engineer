#!/usr/bin/env python3
"""Tiny mutation tester: apply each (find -> replace) edit to a source file, run the tests, restore.

    python scripts/mutate.py common/sqlsafe.py tests/test_sqlsafe.py \
        "if len(statements) != 1:=>if len(statements) < 1:" "comments=False=>comments=True"

A mutant is KILLED if the tests fail (or hang past the timeout) and SURVIVES if they pass. Surviving
mutants point at behaviour your tests do not check. The file is ALWAYS restored (try/finally).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT = 45


def main() -> int:
    src, tests, *edits = sys.argv[1:]
    path = Path(src)
    original = path.read_text()
    survived = []
    try:
        for edit in edits:
            find, _, repl = edit.partition("=>")
            if find not in original:
                print(f"SKIP   (pattern not found) {find[:60]!r}")
                continue
            path.write_text(original.replace(find, repl, 1))
            try:
                # a fresh bytecode cache per mutant: two same-size edits written within one second would
                # otherwise share a stale .pyc (mtime has 1s resolution) and give a false KILLED/SURVIVED
                with tempfile.TemporaryDirectory() as pycache:
                    env = {**os.environ, "PYTHONPYCACHEPREFIX": pycache}
                    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", tests],
                                       capture_output=True, text=True, timeout=TIMEOUT, env=env)
                status = "KILLED" if r.returncode != 0 else "SURVIVED"
            except subprocess.TimeoutExpired:
                status = "KILLED (timeout)"
            print(f"{status:<17} {find[:48]!r} -> {repl[:30]!r}")
            if status == "SURVIVED":
                survived.append(find)
    finally:
        path.write_text(original)
    print(f"\n{len(edits) - len(survived)} killed/skipped, {len(survived)} survived")
    return 1 if survived else 0


if __name__ == "__main__":
    raise SystemExit(main())
