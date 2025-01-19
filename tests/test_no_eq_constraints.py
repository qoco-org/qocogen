import qocogen
import numpy as np
from scipy import sparse
from tests.utils.run_generated_solver import *


def test_linear_objective():

    n = 6
    P = sparse.diags([1, 2, 3, 4, 5, 6], 0)
    P = P.tocsc()

    c = np.array([1, 2, 3, 4, 5, 6])

    p = 0
    A = None
    b = None

    l = 3
    m = 6
    nsoc = 1
    q = np.array([3])
    G = -sparse.identity(m)
    G = G.tocsc()
    h = np.zeros(m)

    qocogen.generate_solver(
        n, m, p, P, c, A, b, G, h, l, nsoc, q, "tests/", "qoco_custom_no_eq"
    )
    codegen_solved, codegen_obj, average_runtime_ms = run_generated_solver(
        "tests/qoco_custom_no_eq"
    )

    opt_obj = -0.758
    assert codegen_solved == 1
    assert abs(codegen_obj - opt_obj) <= 1e-4
