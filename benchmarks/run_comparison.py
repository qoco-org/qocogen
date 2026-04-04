"""
Compare qocogen benchmark results across two git branches.

Checks out each branch, pip-installs it, runs benchmarks in a subprocess
(so import caching cannot pollute results), then generates a regression report.
MPC problem data is bundled in benchmarks/data/ — no external repo needed.

Usage (local):
    python benchmarks/run_comparison.py \\
        [--baseline-branch main] \\
        [--diff-branch <your-branch>] \\
        [--results-dir /tmp/qocogen_bench_results] \\
        [--solvers-dir /tmp/qocogen_bench_solvers]

Usage (CI — BRANCH_NAME env var is set to the PR branch):
    python benchmarks/run_comparison.py
"""

import argparse
import json
import os
import subprocess
import sys


RESULTS_DIR = "/tmp/qocogen_bench_results"
SOLVERS_DIR = "/tmp/qocogen_bench_solvers"


def _git(args, **kwargs):
    return subprocess.run(["git"] + args, check=True, **kwargs)


def _current_branch():
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _current_sha():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _short_sha():
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _checkout(branch_or_sha):
    print(f"\n==> git checkout {branch_or_sha}")
    try:
        _git(["checkout", branch_or_sha])
    except subprocess.CalledProcessError:
        # In CI, fetch from origin first.
        _git(["fetch", "origin", branch_or_sha])
        _git(["checkout", "-B", branch_or_sha, f"origin/{branch_or_sha}"])


def _pip_install():
    print("==> pip install -e . -q")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
        check=True,
    )


def _run_benchmarks_subprocess(output_csv, solvers_dir, mpc_problems=None):
    """Run run_benchmarks.py in a fresh subprocess to avoid import caching."""
    cmd = [
        sys.executable, "-m", "benchmarks.run_benchmarks",
        "--output", output_csv,
        "--solvers-dir", solvers_dir,
    ]
    if mpc_problems:
        cmd += ["--mpc-problems"] + list(mpc_problems)

    print(f"==> {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: run_benchmarks returned exit code {result.returncode}")


def run_comparison(
    baseline_branch="main",
    diff_branch=None,
    results_dir=RESULTS_DIR,
    solvers_dir=SOLVERS_DIR,
    mpc_problems=None,
):
    """
    Run benchmarks on two branches and generate a regression report.

    Parameters
    ----------
    baseline_branch : str
        Branch to use as baseline (default "main").
    diff_branch : str, optional
        Branch to compare against baseline. If None, uses the current HEAD.
    results_dir : str
        Where to write the CSV results and regression report.
    solvers_dir : str
        Where to place generated solvers (re-used across branches).
    mpc_problems : list of str, optional
        Specific MPC problem names to include (default: all bundled problems).
    """
    os.makedirs(results_dir, exist_ok=True)

    # In CI, BRANCH_NAME is the PR branch name set by the workflow.
    ci_branch = os.environ.get("BRANCH_NAME")

    # Determine which ref represents the diff (PR / feature branch).
    if diff_branch is None:
        diff_branch = ci_branch if ci_branch else _current_branch()

    # Save the current state so we can restore it.
    original_sha = _current_sha()
    original_branch = _current_branch()

    baseline_csv = os.path.join(results_dir, "main.csv")
    diff_csv = os.path.join(results_dir, "branch.csv")
    meta_path = os.path.join(results_dir, "meta.json")
    meta = {}

    try:
        # ── Run diff branch benchmarks ─────────────────────────────────────────
        # In CI the repo is already at the diff branch; locally we check it out.
        if not ci_branch:
            _checkout(diff_branch)
        _pip_install()
        meta["diff_branch"] = diff_branch
        meta["diff_sha"] = _short_sha()
        print(f"\n{'='*60}")
        print(f"Running benchmarks for BRANCH: {diff_branch} ({meta['diff_sha']})")
        print(f"{'='*60}\n")
        _run_benchmarks_subprocess(
            output_csv=diff_csv,
            solvers_dir=os.path.join(solvers_dir, "branch"),
            mpc_problems=mpc_problems,
        )

        # ── Run baseline benchmarks ────────────────────────────────────────────
        _checkout(baseline_branch)
        _pip_install()
        meta["baseline_branch"] = baseline_branch
        meta["baseline_sha"] = _short_sha()
        print(f"\n{'='*60}")
        print(f"Running benchmarks for BASELINE: {baseline_branch} ({meta['baseline_sha']})")
        print(f"{'='*60}\n")
        _run_benchmarks_subprocess(
            output_csv=baseline_csv,
            solvers_dir=os.path.join(solvers_dir, "main"),
            mpc_problems=mpc_problems,
        )

    finally:
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        # Always restore the original branch so the repo isn't left on main.
        print(f"\n==> Restoring original branch ({original_branch or original_sha})")
        try:
            _checkout(original_sha)
            _pip_install()
        except Exception as e:
            print(f"WARNING: Could not restore original branch: {e}")

    # ── Generate regression report ─────────────────────────────────────────────
    if not os.path.isfile(baseline_csv) or not os.path.isfile(diff_csv):
        print("\nERROR: One or both result CSVs are missing; cannot generate report.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Generating regression report")
    print(f"{'='*60}\n")
    subprocess.run(
        [
            sys.executable, "-m", "benchmarks.utils.regression_report",
            baseline_csv,
            diff_csv,
        ],
        env={**os.environ,
             "ARTIFACT_URL": os.environ.get("ARTIFACT_URL", ""),
             "PR_NUMBER": os.environ.get("PR_NUMBER", "")},
        check=True,
    )

    report_path = "/tmp/regression-report.txt"
    if os.path.isfile(report_path):
        print(f"\nReport saved to {report_path}")
        # Also copy into results_dir for artifact upload.
        import shutil
        shutil.copy(report_path, os.path.join(results_dir, "regression-report.txt"))


def main():
    parser = argparse.ArgumentParser(
        description="Compare qocogen across two branches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--baseline-branch", default="main",
        help="Baseline branch name (default: main).",
    )
    parser.add_argument(
        "--diff-branch", default=None,
        help="Branch to compare (default: current HEAD, or BRANCH_NAME env var in CI).",
    )
    parser.add_argument(
        "--results-dir", default=RESULTS_DIR,
        help=f"Directory for CSV results and report (default: {RESULTS_DIR}).",
    )
    parser.add_argument(
        "--solvers-dir", default=SOLVERS_DIR,
        help=f"Directory for generated solvers (default: {SOLVERS_DIR}).",
    )
    parser.add_argument(
        "--mpc-problems", nargs="*", default=None,
        help="MPC problem stems to include (default: all bundled problems).",
    )
    args = parser.parse_args()

    run_comparison(
        baseline_branch=args.baseline_branch,
        diff_branch=args.diff_branch,
        results_dir=args.results_dir,
        solvers_dir=args.solvers_dir,
        mpc_problems=args.mpc_problems,
    )


if __name__ == "__main__":
    main()
