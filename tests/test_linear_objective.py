import qocogen
import numpy as np
from scipy import sparse
from tests.utils.run_generated_solver import *


def test_linear_objective():
    # Define problem data
    n = 2
    P = None
    c = np.array([-1, -2])

    p = 0
    A = None
    b = None

    l = 3
    m = 3
    nsoc = 0
    q = None
    G = sparse.csc_matrix([[-1, 0], [0, -1], [1, 1]])
    h = np.array([0, 0, 1])

    # Solve problem.
    qocogen.generate_solver(
        n, m, p, P, c, A, b, G, h, l, nsoc, q, "tests/", "qoco_custom_lin_obj"
    )
    codegen_solved, codegen_obj, average_runtime_ms = run_generated_solver(
        "tests/qoco_custom_lin_obj"
    )

    opt_obj = -2.000
    assert codegen_solved == 1
    assert abs(codegen_obj - opt_obj) <= 1e-4
