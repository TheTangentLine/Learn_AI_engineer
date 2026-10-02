"""llmops: the production harness for the Week 6 support system (Week 7 weekly challenge).

    python -m llmops run      --variant lean+hide+direct --provider local --out runs/candidate
    python -m llmops gate     --baseline runs/baseline --candidate runs/candidate --summary-file "$GITHUB_STEP_SUMMARY"
    python -m llmops dashboard --runs runs/baseline runs/candidate --out dashboard.html
    python -m llmops monitor  --spans prod-spans.jsonl --reference ref-spans.jsonl

Four parts, one artifact format:
  runner.py     runs the 50-case suite with tracing, trials and a spend cap, and writes a RUN DIRECTORY
  gate.py       compares two run directories: quality (Day 3), cost, latency, dataset, completeness
  dashboard.py  one self-contained HTML page (inline SVG, no scripts, everything escaped)
  monitor.py    reads production traces and raises alerts (defect rate, escalations, route drift, cost spikes)
"""

import sys
from pathlib import Path

_SOLUTIONS = str(
    Path(__file__).resolve().parents[2]
)  # .../solutions: evalcases, evalgate, costlab, traceeval, day1_solution ...
if _SOLUTIONS not in sys.path:
    sys.path.append(
        _SOLUTIONS
    )  # appended, never inserted first: it must not shadow the caller's own modules
_ROOT = str(Path(__file__).resolve().parents[5])
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
