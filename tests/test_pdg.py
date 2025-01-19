import qocogen
import numpy as np
from scipy import sparse
from tests.utils.run_generated_solver import *


def test_pdg():
    N = 30  # Number of timesteps.
    dt = 0.5  # Discretization interval.
    g0 = 9.8  # Gravitational acceleration.
    zi = np.array([100, 50, 50, -9, 5, -9])  # Initial condition.
    zf = np.zeros(6)  # Terminal condition.
    Q = 1.0 * sparse.eye(6)  # State cost matrix.
    R = 5.0 * sparse.eye(3)  # Input cost matrix.
    vmax = 10.0  # Max inf norm on velocity.
    umax = 12.0  # Maximum thrust.
    thmax = np.deg2rad(35.0)  # Maximum Thrust pointing angle.

    # Number of optimization variables.
    n = 9 * N - 2

    # Number of affine equality constraints (rows of A).
    p = 6 * N + 7

    # Number of conic constraints (rows of G).
    m = 13 * N - 7

    # Dimension of non-negative orthant in cone C.
    l = 6 * N

    # Number of second order cones.
    nsoc = 2 * N - 2

    # Dimension of each second order cone.
    q = np.hstack(
        (4 * np.ones(N - 1, dtype=np.int32), 3 * np.ones(N - 1, dtype=np.int32))
    )

    # Parse cost function.
    Qfull = sparse.kron(sparse.eye(N), Q)
    Rfull = sparse.kron(sparse.eye(N - 1), R)
    P = sparse.block_diag((Qfull, Rfull, 0.0 * sparse.eye(1)))
    c = np.zeros(n)

    # Double integrator dynamics.
    Ad = np.block([[np.eye(3), dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
    Bd = np.block([[0.5 * dt**2 * np.eye(3)], [dt * np.eye(3)]])
    g = np.array([-0.5 * g0 * dt**2, 0, 0, -g0 * dt, 0, 0])

    # Parse dynamics constraint.
    Azdyn = np.block(
        [np.kron(np.eye(N - 1), Ad), np.zeros((6 * (N - 1), 6))]
    ) - np.block([np.zeros((6 * (N - 1), 6)), np.eye(6 * (N - 1))])
    Audyn = np.kron(np.eye(N - 1), Bd)
    Axidyn = np.zeros((6 * (N - 1), 1))

    # Parse boundary conditions.
    Azbc = np.block(
        [
            [np.eye(6), np.zeros((6, 6 * (N - 1)))],
            [np.zeros((6, 6 * (N - 1))), np.eye(6)],
        ]
    )
    Aubc = np.zeros((12, 3 * (N - 1)))
    Axibc = np.zeros((12, 1))

    # Parse slack variable.
    Azslack = np.zeros((1, 6 * N))
    Auslack = np.zeros((1, 3 * (N - 1)))
    Axislack = np.array([1.0])

    # Combine dynamics, boundary conditions, and slack equality into equality constraint matrix A, and vector b.
    A = np.block(
        [[Azdyn, Audyn, Axidyn], [Azbc, Aubc, Axibc], [Azslack, Auslack, Axislack]]
    )
    b = np.hstack((np.kron(np.ones(N - 1), -g), zi, zf, umax))

    # Parse velocity constraint.
    Gzvelocity = np.block(
        [
            [np.kron(np.eye(N), np.block([np.zeros((3, 3)), np.eye(3)]))],
            [np.kron(np.eye(N), np.block([np.zeros((3, 3)), -np.eye(3)]))],
        ]
    )
    Guvelocity = np.zeros((6 * N, 3 * (N - 1)))
    Gxivelocity = np.zeros((6 * N, 1))

    # Parse thrust constraint.
    Gzthrust = np.zeros((4 * (N - 1), 6 * N))
    Guthrust = np.kron(np.eye(N - 1), np.block([[np.zeros((1, 3))], [-np.eye(3)]]))
    Gxithrust = np.kron(np.ones(N - 1), np.array([-1, 0, 0, 0]))
    Gxithrust = Gxithrust[:, np.newaxis]

    # Parse pointing constraint.
    Gzpointing = np.zeros((3 * (N - 1), 6 * N))
    block = -np.eye(3)
    block[0, 0] = -np.tan(thmax)
    Gupointing = np.kron(np.eye(N - 1), block)
    Gxipointing = np.zeros((3 * (N - 1), 1))

    # Combine velocity box constraint, thrust ball constraint, and thrust pointing constraint into G and h.
    G = np.block(
        [
            [Gzvelocity, Guvelocity, Gxivelocity],
            [Gzthrust, Guthrust, Gxithrust],
            [Gzpointing, Gupointing, Gxipointing],
        ]
    )
    h = np.hstack(
        (vmax * np.ones((6 * N)), np.zeros(4 * (N - 1)), np.zeros(3 * (N - 1)))
    )

    # Convert to sparse data type.
    P = sparse.triu(P, format="csc")
    A = sparse.csc_matrix(A)
    G = sparse.csc_matrix(G)

    qocogen.generate_solver(
        n, m, p, P, c, A, b, G, h, l, nsoc, q, "tests/", "qoco_custom_pdg"
    )
    codegen_solved, codegen_obj, average_runtime_ms = run_generated_solver(
        "tests/qoco_custom_pdg"
    )

    # Solve problem.
    opt_obj = 61243.596
    assert codegen_solved == 1
    assert abs(codegen_obj - opt_obj) <= 1e-1
