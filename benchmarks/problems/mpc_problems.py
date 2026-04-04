"""MPC benchmark problems loaded from .mat files in benchmarks/data/."""

import glob
import os
import sys

# Problems excluded from the default run (too slow or numerically problematic).
DEFAULT_EXCLUDED_NAMES = {
    "matFiles_quadcopter_1",
    "matFiles_quadcopter_2",
    "matFiles_quadcopter_3",
    "matFiles_quadcopter_4",
    "matFiles_quadcopter_5",
    "matFiles_quadcopter_6",
    "matFiles_dcMotor_3",
    "matFiles_dcMotor_4",
    "matFiles_springMass_1",
    "matFiles_springMass_4",
    "matFiles_binaryDistillationColumn_1",
    "matFiles_binaryDistillationColumn_2",
}


def _load_construct_mpc():
    """Import construct_mpc_problem from benchmarks/."""
    benchmarks_dir = os.path.dirname(os.path.dirname(__file__))
    if benchmarks_dir not in sys.path:
        sys.path.insert(0, benchmarks_dir)
    from construct_mpc_problem import construct_mpc_problem
    return construct_mpc_problem


def _load_cvxpy_to_qoco():
    from tests.utils.cvxpy_to_qoco import convert
    return convert


def get_problems(data_dir, names=None):
    """
    Load and convert MPC problems from .mat files.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing .mat files.
    names : list of str, optional
        Stem names to load (without .mat extension).
        Defaults to all .mat files found in data_dir.

    Returns
    -------
    list of (name, n, m, p, P, c, A, b, G, h, l, nsoc, q)
        Problems that loaded successfully.
    """
    import scipy.io

    if names is None:
        mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
        names = [
            os.path.splitext(os.path.basename(f))[0]
            for f in mat_files
            if os.path.splitext(os.path.basename(f))[0] not in DEFAULT_EXCLUDED_NAMES
        ]

    try:
        construct_mpc_problem = _load_construct_mpc()
    except ImportError as e:
        print(f"[mpc_problems] Could not import construct_mpc_problem: {e}. Skipping MPC problems.")
        return []

    try:
        convert = _load_cvxpy_to_qoco()
    except ImportError as e:
        print(f"[mpc_problems] Could not import cvxpy_to_qoco: {e}. Skipping MPC problems.")
        return []

    problems = []
    for stem in names:
        mat_path = os.path.join(data_dir, stem + ".mat")
        if not os.path.isfile(mat_path):
            print(f"[mpc_problems] {mat_path} not found, skipping.")
            continue
        try:
            mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
            prob = construct_mpc_problem(mat)
            n, m, p, P, c, A, b, G, h, l, nsoc, q = convert(prob)
            short_name = stem.removeprefix("matFiles_")
            problems.append((short_name, n, m, p, P, c, A, b, G, h, l, nsoc, q))
        except Exception as e:
            print(f"[mpc_problems] Failed to load {stem}: {e}")

    return problems
