# llmops: the production harness (Week 7 weekly challenge)

```bash
cd weeks/week07_evals-observability-llmops/solutions/weekly
python -m llmops run --provider scripted --variant base --out /tmp/runs/baseline                      # offline, deterministic
python -m llmops run --provider scripted --variant base --fault skip_lookup --name broken --out /tmp/runs/broken
python -m llmops gate --baseline /tmp/runs/baseline --candidate /tmp/runs/broken                      # exit code 1
python -m llmops dashboard --runs /tmp/runs/baseline /tmp/runs/broken --out /tmp/runs/dashboard.html
python -m llmops run --provider local --variant lean+hide+direct --out /tmp/runs/candidate            # the real local model (slow the first time)
python -m llmops monitor --spans prod.jsonl --reference reference.jsonl
python -m llmops.legacy ../../../../outputs ../../../../outputs/w7d7_runs                               # import the Day 5 real runs
```

| file | job |
|---|---|
| `llmops/runner.py` | run the 50 cases with tracing, trials and a spend cap; write / load / promote a run directory |
| `llmops/gate.py` | completeness, dataset, price card, quality (Day 3), cost, latency; markdown; exit code |
| `llmops/dashboard.py` | one HTML page: inline SVG, no scripts, no external requests, everything escaped |
| `llmops/monitor.py` | production traces -> alerts (defect rate, escalation rate, route drift, cost spike) |
| `llmops/drill.py` | incident drills: synthetic traffic with a known incident, to test the monitor |
| `llmops/legacy.py` | import the Day 5 saved runs as run directories |
| `test_day7.py` | the tests |
