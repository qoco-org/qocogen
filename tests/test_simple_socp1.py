import qocogen
import numpy as np
from scipy import sparse
from tests.utils.run_generated_solver import *


def test_simple_socp():
    # Define problem data
    P = sparse.diags([1, 2, 3, 4, 5, 6], 0)
    P = P.tocsc()

    c = np.array([1, 2, 3, 4, 5, 6])
    G = -sparse.identity(6)
    G = G.tocsc()
    h = np.zeros(6)
    A = sparse.csc_matrix([[1, 1, 0, 0, 0, 0], [0, 1, 2, 0, 0, 0]])
    A = A.tocsc()
    b = np.array([1, 2])

    l = 3
    n = 6
    m = 6
    p = 2
    nsoc = 1
    q = np.array([3])

    qocogen.generate_solver(
        n, m, p, P, c, A, b, G, h, l, nsoc, q, "tests/", "qoco_custom_simple_socp1"
    )
    codegen_solved, codegen_obj, average_runtime_ms = run_generated_solver(
        "tests/qoco_custom_simple_socp1"
    )

    opt_obj = 4.042
    assert codegen_solved == 1
    assert abs(codegen_obj - opt_obj) <= 1e-4
