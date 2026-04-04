import cvxpy as cp
from scipy.sparse import csc_matrix, block_diag, block_array
import numpy as np


def construct_mpc_problem(mat):
    data = mat["problem"]
    x0 = data.x0
    xNr = data.xNr
    A = data.A
    B = data.B
    e = data.e
    f = data.f
    C = data.C
    D = data.D
    M = data.M
    N = data.N
    T = data.T
    yr = data.yr
    ur = data.ur
    ymin = data.ymin
    ymax = data.ymax
    umin = data.umin
    umax = data.umax
    dmin = data.dmin
    dmax = data.dmax
    dNmin = data.dNmin
    dNmax = data.dNmax
    Q = data.Q
    R = data.R
    S = data.S
    P = data.P
    ni = data.ni

    nx = A.shape[0]
    nu = B.shape[1] if len(data.B.shape) > 1 else 1
    x = cp.Variable((nx, ni + 1))
    u = cp.Variable((nu, ni))

    has_outputs = len(C) > 0
    if has_outputs:
        ny = C.shape[0]
        y = cp.Variable((ny, ni))
    con = [x[:, 0] == x0]
    A = csc_matrix(A)
    B = csc_matrix(B)
    B = B if B.shape[0] == nx else B.T
    for i in range(ni):
        f = f if len(f) > 0 else np.zeros(nx)
        con += [x[:, i + 1] == A @ x[:, i] + B @ u[:, i] + f]

    if has_outputs:
        C = csc_matrix(C)
        if len(D) == 0:
            D = np.zeros((ny, nu))
        else:
            D = csc_matrix(D)
            D = D if D.shape[0] == ny else D.T
        for i in range(ni):
            e = e if len(e) > 0 else np.zeros(ny)
            con += [y[:, i] == C @ x[:, i] + D @ u[:, i] + e]

    if np.asarray(umin).size > 0:
        for i in range(ni):
            con += [umin <= u[:, i], u[:, i] <= umax]

    if has_outputs and len(ymin) > 0:
        for i in range(ni):

            idx = np.where(np.isfinite(ymin))[0]
            if len(idx) > 0:
                con += [ymin[idx] <= y[idx, i]]

            idx = np.where(np.isfinite(ymax))[0]
            if len(idx) > 0:
                con += [y[idx, i] <= ymax[idx]]

    if has_outputs and len(dmin) > 0:
        M = csc_matrix(M)
        N = csc_matrix(N)
        M = M if M.shape[1] == ny else M.T
        N = N if N.shape[1] == nu else N.T
        for i in range(ni):
            idx = np.where(np.isfinite(dmin))[0]
            if len(idx) > 0:
                con += [dmin[idx] <= (M @ y[:, i] + N @ u[:, i])[idx]]

            idx = np.where(np.isfinite(dmax))[0]
            if len(idx) > 0:
                con += [(M @ y[:, i] + N @ u[:, i])[idx] <= dmax[idx]]

    if np.asarray(dNmin).size > 0:
        T = csc_matrix(T)
        con += [dNmin <= T @ x[:, ni], T @ x[:, ni] <= dNmax]

    Q = csc_matrix(Q)
    R = csc_matrix(R)
    if len(S) == 0:
        QQ = block_diag((Q, R), format="csc")
    else:
        S = csc_matrix(S.reshape(nx, nu))
        QQ = block_array([[Q, S], [S.T, R]], format="csc")

    if has_outputs and (len(yr) == 0):
        yr = np.zeros(y.shape).tolist()

    if np.asarray(ur).size == 0 or (np.asarray(ur).size == 1 and ur == 0.0):
        ur = np.zeros(u.T.shape).tolist()

    obj = 0.0
    for i in range(ni):
        if has_outputs:
            v = cp.hstack((y[:, i] - yr[i], u[:, i] - ur[i]))
        else:
            v = cp.hstack((x[:, i], u[:, i] - ur[i]))
        obj += cp.quad_form(v, QQ)

    if len(P) > 0:
        P = csc_matrix(P)
        if len(xNr) > 0:
            v = x[:, ni] - xNr
            obj += cp.quad_form(v, P)
        else:
            obj += cp.quad_form(x[:, ni], P)

    problem = cp.Problem(cp.Minimize(obj), con)
    return problem
