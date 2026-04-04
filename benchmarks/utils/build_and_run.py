"""Helpers for cmake build and execution of generated solvers."""

import os
import struct
import subprocess


def build_solver(solver_dir):
    """
    Configure and build the generated solver with cmake.

    Returns True on success, False on failure.
    """
    build_dir = os.path.join(solver_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    configure = subprocess.run(
        ["cmake", "-DQOCO_CUSTOM_BUILD_TYPE:STR=Release", ".."],
        cwd=build_dir,
        capture_output=True,
    )
    if configure.returncode != 0:
        print(f"  [cmake configure failed]\n{configure.stderr.decode()}")
        return False

    build = subprocess.run(
        ["cmake", "--build", ".", "--config", "Release"],
        cwd=build_dir,
        capture_output=True,
    )
    if build.returncode != 0:
        print(f"  [cmake build failed]\n{build.stderr.decode()}")
        return False

    return True


def run_solver(solver_dir):
    """
    Execute the runtest binary inside solver_dir/build.

    Returns True on success, False on failure.
    """
    if os.name == "nt":
        binary = os.path.join(solver_dir, "build", "Debug", "runtest.exe")
        cwd = os.path.join(solver_dir, "build", "Debug")
    else:
        binary = os.path.join(solver_dir, "build", "runtest")
        cwd = os.path.join(solver_dir, "build")

    ret = subprocess.run([binary], cwd=cwd, capture_output=True)
    return ret.returncode == 0


def read_result(solver_dir):
    """
    Parse result.bin written by runtest.

    Returns (solved: int, obj: float, solve_time_s: float).
    """
    if os.name == "nt":
        result_path = os.path.join(solver_dir, "build", "Debug", "result.bin")
    else:
        result_path = os.path.join(solver_dir, "build", "result.bin")

    with open(result_path, "rb") as f:
        solved = struct.unpack("B", f.read(1))[0]
        obj = struct.unpack("d", f.read(8))[0]
        solve_time_s = struct.unpack("d", f.read(8))[0]
    return solved, obj, solve_time_s
