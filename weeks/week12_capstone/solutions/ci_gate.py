"""The CI gate as a command: run the product on the dev split and the security suite, compare with the stored baseline, exit 1 if the contract no longer holds.

  uv run python weeks/week12_capstone/solutions/ci_gate.py                  # check
  uv run python weeks/week12_capstone/solutions/ci_gate.py --update-baseline # accept the current behaviour as the new baseline (review the diff!)

The thresholds are in ``copilot/evalgate.py``. The latency budget only warns: latency depends on the machine.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from copilot import evalgate as EG
from copilot import evaluate as E
from day4_solution import BASELINE, poisoned_lab_index, poisoning
from lab import Lab


def main(argv: list[str]) -> int:
    lab = Lab()
    tmp = tempfile.mkdtemp(prefix="w12-ci-")
    try:
        table = poisoning(lab, poisoned_lab_index(lab, tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    attacks = table["extractive answerer"]["quarantine + output guard"]
    runs = E.run_golden(lab.copilot(), lab.items, split="dev")
    rep = E.report(runs)
    current = EG.snapshot(runs, rep["retrieved_any"][0], attacks, rep["latency"]["p95"])
    if "--update-baseline" in argv:
        BASELINE.write_text(json.dumps(current, indent=1, sort_keys=True))
        print(
            f"baseline updated: {sum(v['passed'] for v in current['items'].values())}/{len(current['items'])} items pass, {attacks} attacks through"
        )
        return 0
    baseline = json.loads(BASELINE.read_text()) if Path(BASELINE).exists() else None
    d = EG.gate(current, baseline)
    print(d.markdown())
    return 0 if d.passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
