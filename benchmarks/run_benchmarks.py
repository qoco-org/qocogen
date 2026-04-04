"""
Run qocogen benchmarks for the currently installed version of qocogen.

For each problem:
  1. Generate the solver (timed).
  2. Build with cmake.
  3. Execute the runtest binary.
  4. Record codegen time, solve time, and solve status.

Usage:
    python -m benchmarks.run_benchmarks \\
        --output /tmp/results/branch.csv \\
        --solvers-dir /tmp/qocogen_bench_solvers \\
        [--mpc-data-dir ../qoco-benchmarks/mpc/data] \\
        [--mpc-problems toyExample_1 dcMotor_1 ...]
"""

import argparse
import os
import sys
import time
import math
import csv
import shutil

import qocogen

from benchmarks.problems import socp_problems
from benchmarks.problems import mpc_problems
from benchmarks.utils.build_and_run import build_solver, run_solver, read_result

# Bundled MPC data directory (benchmarks/data/).
DEFAULT_MPC_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _run_problem(name, n, m, p, P, c, A, b, G, h, l, nsoc, q, solvers_dir):
    """
    Generate, build, and run a single problem.

    Returns a dict with keys: name, codegen_time_s, solved, obj, solve_time_s.
    On any failure after codegen, solved=False, obj/solve_time_s=NaN.
    """
    solver_dir = os.path.join(solvers_dir, name)

    # Remove stale solver directory if present.
    if os.path.exists(solver_dir):
        shutil.rmtree(solver_dir)

    print(f"  [{name}] generating solver...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        qocogen.generate_solver(n, m, p, P, c, A, b, G, h, l, nsoc, q, solvers_dir, name)
    except Exception as e:
        codegen_time = time.perf_counter() - t0
        print(f"FAILED (codegen): {e}")
        return dict(name=name, codegen_time_s=codegen_time, solved=0,
                    obj=math.nan, solve_time_s=math.nan)
    codegen_time = time.perf_counter() - t0
    print(f"done ({codegen_time:.3f}s)", end=" ", flush=True)

    print("building...", end=" ", flush=True)
    if not build_solver(solver_dir):
        print("FAILED (build)")
        return dict(name=name, codegen_time_s=codegen_time, solved=0,
                    obj=math.nan, solve_time_s=math.nan)

    print("running...", end=" ", flush=True)
    if not run_solver(solver_dir):
        print("FAILED (run)")
        return dict(name=name, codegen_time_s=codegen_time, solved=0,
                    obj=math.nan, solve_time_s=math.nan)

    try:
        solved, obj, solve_time_s = read_result(solver_dir)
    except Exception as e:
        print(f"FAILED (result): {e}")
        return dict(name=name, codegen_time_s=codegen_time, solved=0,
                    obj=math.nan, solve_time_s=math.nan)

    status_str = "solved" if solved == 1 else f"status={solved}"
    print(f"{status_str}, obj={obj:.4g}, solve={solve_time_s*1e3:.3f}ms")
    return dict(name=name, codegen_time_s=codegen_time, solved=int(solved),
                obj=obj, solve_time_s=solve_time_s)


def run_benchmarks(output_csv, solvers_dir, mpc_data_dir=None, mpc_problem_names=None):
    """
    Run all benchmarks and write results to output_csv.

    Parameters
    ----------
    output_csv : str
        Path for the output CSV file.
    solvers_dir : str
        Directory where generated solvers will be placed.
    mpc_data_dir : str, optional
        Path to the directory containing .mat MPC problem files.
        If None or not found, MPC problems are skipped.
    mpc_problem_names : list of str, optional
        MPC problem stems to include (without .mat extension).
        Defaults to benchmarks.problems.mpc_problems.DEFAULT_MPC_NAMES.
    """
    os.makedirs(solvers_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    print(f"\nqocogen version: {qocogen.__version__ if hasattr(qocogen, '__version__') else 'unknown'}")
    print(f"Solvers directory: {solvers_dir}")
    print(f"Output: {output_csv}\n")

    # Collect problems.
    problems = socp_problems.get_problems()

    if mpc_data_dir and os.path.isdir(mpc_data_dir):
        print(f"Loading MPC problems from {mpc_data_dir}")
        problems += mpc_problems.get_problems(mpc_data_dir, names=mpc_problem_names)
    else:
        if mpc_data_dir:
            print(f"MPC data directory not found ({mpc_data_dir}), skipping MPC problems.")
        else:
            print("No MPC data directory specified, skipping MPC problems.")

    print(f"\nRunning {len(problems)} benchmark problem(s):\n")

    results = []
    for prob in problems:
        name = prob[0]
        args = prob[1:]  # n, m, p, P, c, A, b, G, h, l, nsoc, q
        result = _run_problem(name, *args, solvers_dir=solvers_dir)
        results.append(result)

    # Write CSV.
    fieldnames = ["name", "codegen_time_s", "solved", "obj", "solve_time_s"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    n_solved = sum(1 for r in results if r["solved"] == 1)
    print(f"\nDone. {n_solved}/{len(results)} problems solved.")
    print(f"Results written to {output_csv}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run qocogen benchmarks.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--solvers-dir",
        default="/tmp/qocogen_bench_solvers",
        help="Directory for generated solvers (default: /tmp/qocogen_bench_solvers).",
    )
    parser.add_argument(
        "--mpc-data-dir",
        default=None,
        help="Path to the qoco-benchmarks MPC data directory.",
    )
    parser.add_argument(
        "--mpc-problems",
        nargs="*",
        default=None,
        help="MPC problem stems to include (default: built-in subset).",
    )
    args = parser.parse_args()

    # Auto-detect MPC data directory if not specified.
    mpc_data_dir = args.mpc_data_dir
    if mpc_data_dir is None and os.path.isdir(DEFAULT_MPC_DATA_DIR):
        mpc_data_dir = os.path.normpath(DEFAULT_MPC_DATA_DIR)
        print(f"Auto-detected MPC data directory: {mpc_data_dir}")

    run_benchmarks(
        output_csv=args.output,
        solvers_dir=args.solvers_dir,
        mpc_data_dir=mpc_data_dir,
        mpc_problem_names=args.mpc_problems,
    )


if __name__ == "__main__":
    main()
