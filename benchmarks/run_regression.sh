#!/usr/bin/env bash
# Run the full benchmark regression suite and write the report to the current
# working directory. Call this from the repo root or any directory you want
# the report dropped into.
#
# Usage:
#   ./benchmarks/run_regression.sh [--diff-branch <branch>] [extra args...]
#
# All arguments are forwarded to run_comparison.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$(pwd)/bench_results"
REPORT="$(pwd)/regression-report.txt"

cd "$REPO_ROOT"

python benchmarks/run_comparison.py \
    --baseline-branch main \
    --results-dir "$OUTPUT_DIR" \
    --solvers-dir /tmp/qocogen_bench_solvers \
    "$@"

# run_comparison.py writes the report into results_dir; copy to cwd.
if [ -f "$OUTPUT_DIR/regression-report.txt" ]; then
    cp "$OUTPUT_DIR/regression-report.txt" "$REPORT"
    echo ""
    echo "Report written to $REPORT"
    echo ""
    cat "$REPORT"
fi
