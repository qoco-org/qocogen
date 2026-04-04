"""SOCP benchmark problems ported from the qocogen test suite."""

import numpy as np
from scipy import sparse


def _simple_socp1():
    P = sparse.diags([1, 2, 3, 4, 5, 6], 0).tocsc()
    c = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    G = -sparse.identity(6).tocsc()
    h = np.zeros(6)
    A = sparse.csc_matrix([[1, 1, 0, 0, 0, 0], [0, 1, 2, 0, 0, 0]])
    b = np.array([1, 2], dtype=float)
    return "simple_socp1", 6, 6, 2, P, c, A, b, G, h, 3, 1, np.array([3])


def _simple_socp2():
    P = sparse.diags([1, 2, 3, 4, 5, 6], 0).tocsc()
    c = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    G = -sparse.identity(6).tocsc()
    h = np.zeros(6)
    A = sparse.csc_matrix([[1, 1, 0, 0, 0, 0], [0, 1, 2, 0, 0, 0]])
    b = np.array([1, 2], dtype=float)
    return "simple_socp2", 6, 6, 2, P, c, A, b, G, h, 0, 2, np.array([3, 3])


def _simple_socp3():
    P = sparse.diags([1, 2, 3, 4, 5, 6], 0).tocsc()
    c = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    G = -sparse.identity(6).tocsc()
    h = np.zeros(6)
    A = sparse.csc_matrix([[1, 1, 0, 0, 0, 0], [0, 1, 2, 0, 0, 0]])
    b = np.array([1, 2], dtype=float)
    return "simple_socp3", 6, 6, 2, P, c, A, b, G, h, 3, 1, np.array([3])


def _linear_objective():
    G = sparse.csc_matrix([[-1, 0], [0, -1], [1, 1]])
    h = np.array([0, 0, 1], dtype=float)
    return "linear_objective", 2, 3, 0, None, np.array([-1, -2], dtype=float), None, None, G, h, 3, 0, None


def _pdg():
    """Powered descent guidance problem (N=30 timesteps)."""
    N = 30
    dt = 0.5
    g0 = 9.8
    Q = 1.0 * sparse.eye(6)
    R = 5.0 * sparse.eye(3)
    vmax = 10.0
    umax = 12.0
    thmax = np.deg2rad(35.0)

    n = 9 * N - 2
    p = 6 * N + 7
    m = 13 * N - 7
    l = 6 * N
    nsoc = 2 * N - 2
    q = np.hstack((4 * np.ones(N - 1, dtype=np.int32), 3 * np.ones(N - 1, dtype=np.int32)))

    Qfull = sparse.kron(sparse.eye(N), Q)
    Rfull = sparse.kron(sparse.eye(N - 1), R)
    P = sparse.triu(sparse.block_diag((Qfull, Rfull, 0.0 * sparse.eye(1))), format="csc")
    c = np.zeros(n)

    Ad = np.block([[np.eye(3), dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
    Bd = np.block([[0.5 * dt**2 * np.eye(3)], [dt * np.eye(3)]])
    g = np.array([-0.5 * g0 * dt**2, 0, 0, -g0 * dt, 0, 0])

    Azdyn = np.block([np.kron(np.eye(N - 1), Ad), np.zeros((6 * (N - 1), 6))]) - np.block([np.zeros((6 * (N - 1), 6)), np.eye(6 * (N - 1))])
    Audyn = np.kron(np.eye(N - 1), Bd)
    Axidyn = np.zeros((6 * (N - 1), 1))

    zi = np.array([100, 50, 50, -9, 5, -9])
    zf = np.zeros(6)
    Azbc = np.block([[np.eye(6), np.zeros((6, 6 * (N - 1)))], [np.zeros((6, 6 * (N - 1))), np.eye(6)]])
    Aubc = np.zeros((12, 3 * (N - 1)))
    Axibc = np.zeros((12, 1))

    Azslack = np.zeros((1, 6 * N))
    Auslack = np.zeros((1, 3 * (N - 1)))
    Axislack = np.array([1.0])

    A = sparse.csc_matrix(np.block([[Azdyn, Audyn, Axidyn], [Azbc, Aubc, Axibc], [Azslack, Auslack, Axislack]]))
    b = np.hstack((np.kron(np.ones(N - 1), -g), zi, zf, umax))

    Gzvelocity = np.block([[np.kron(np.eye(N), np.block([np.zeros((3, 3)), np.eye(3)]))], [np.kron(np.eye(N), np.block([np.zeros((3, 3)), -np.eye(3)]))]])
    Guvelocity = np.zeros((6 * N, 3 * (N - 1)))
    Gxivelocity = np.zeros((6 * N, 1))

    Gzthrust = np.zeros((4 * (N - 1), 6 * N))
    Guthrust = np.kron(np.eye(N - 1), np.block([[np.zeros((1, 3))], [-np.eye(3)]]))
    Gxithrust = np.kron(np.ones(N - 1), np.array([-1, 0, 0, 0]))[:, np.newaxis]

    Gzpointing = np.zeros((3 * (N - 1), 6 * N))
    block = -np.eye(3)
    block[0, 0] = -np.tan(thmax)
    Gupointing = np.kron(np.eye(N - 1), block)
    Gxipointing = np.zeros((3 * (N - 1), 1))

    G = sparse.csc_matrix(np.block([[Gzvelocity, Guvelocity, Gxivelocity], [Gzthrust, Guthrust, Gxithrust], [Gzpointing, Gupointing, Gxipointing]]))
    h = np.hstack((vmax * np.ones(6 * N), np.zeros(4 * (N - 1)), np.zeros(3 * (N - 1))))

    return "pdg", n, m, p, P, c, A, b, G, h, l, nsoc, q


def get_problems():
    """Return list of (name, n, m, p, P, c, A, b, G, h, l, nsoc, q) tuples."""
    return [
        _simple_socp1(),
        _simple_socp2(),
        _simple_socp3(),
        _linear_objective(),
        _pdg(),
    ]
