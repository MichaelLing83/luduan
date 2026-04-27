#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! cargo llvm-cov --version >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: cargo-llvm-cov is required to run coverage.

Install it with:
  cargo install cargo-llvm-cov

Then rerun:
  ./test.sh
EOF
  exit 1
fi

coverage_json="$(mktemp -t luduan-coverage.XXXXXX.json)"
coverage_log="$(mktemp -t luduan-coverage.XXXXXX.log)"
trap 'rm -f "$coverage_json" "$coverage_log"' EXIT

if ! cargo llvm-cov clean --workspace >"$coverage_log" 2>&1; then
  cat "$coverage_log" >&2
  exit 1
fi

if ! cargo llvm-cov \
  --workspace \
  --all-targets \
  --json \
  --summary-only \
  --output-path "$coverage_json" \
  "$@" >"$coverage_log" 2>&1; then
  cat "$coverage_log" >&2
  exit 1
fi

python3 - "$coverage_json" "$coverage_log" <<'PY'
import json
import os
import re
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    coverage = json.load(file)["data"][0]
with open(sys.argv[2], encoding="utf-8") as file:
    test_log = file.read()

root = os.getcwd()
files = coverage["files"]
totals = coverage["totals"]["lines"]
tests_run = sum(int(value) for value in re.findall(r"running (\d+) tests?", test_log))
tests_passed = sum(int(value) for value in re.findall(r"test result: ok\. (\d+) passed", test_log))

print("Test summary")
print()
print(f"Tests run:    {tests_run}")
print(f"Tests passed: {tests_passed}")
print()
print("Coverage summary")
print()
print(f"{'File':<40} {'Lines':>8} {'Line coverage':>14}")
print("-" * 64)
for file in files:
    filename = os.path.relpath(file["filename"], root)
    lines = file["summary"]["lines"]
    print(f"{filename:<40} {lines['count']:>8} {lines['percent']:>13.2f}%")
print("-" * 64)
print(f"{'TOTAL':<40} {totals['count']:>8} {totals['percent']:>13.2f}%")
PY
