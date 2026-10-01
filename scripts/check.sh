#!/usr/bin/env bash
# One command to verify the whole repo: lint + every offline test.
#   ./scripts/check.sh          # fast; tests that need Postgres/Docker are skipped automatically
#
# Each test directory runs in its OWN pytest process: different weeks reuse module names
# (day3_solution.py, test_day4.py, ...), which would collide inside a single process.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
status=0

echo "== ruff"
$PY -m ruff check . || status=1

echo "== pytest (one process per test directory)"
total_pass=0; total_skip=0; failed_dirs=()
for dir in tests $(find weeks -name 'test_*.py' -exec dirname {} \; | sort -u); do
  out=$($PY -m pytest -q -p no:cacheprovider "$dir" 2>&1); rc=$?
  line=$(echo "$out" | tail -1)
  printf '%-75s %s\n' "$dir" "$line"
  p=$(echo "$line" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo 0)
  s=$(echo "$line" | grep -oE '[0-9]+ skipped' | grep -oE '[0-9]+' || echo 0)
  total_pass=$((total_pass + ${p:-0})); total_skip=$((total_skip + ${s:-0}))
  if [ $rc -ne 0 ]; then failed_dirs+=("$dir"); status=1; fi
done
echo "----"
echo "passed: $total_pass, skipped: $total_skip, failing directories: ${#failed_dirs[@]}"
for d in "${failed_dirs[@]:-}"; do [ -n "$d" ] && echo "  FAILED: $d"; done
exit $status
