import qocogen
import pytest
import numpy as np
import cvxpy as cp
from scipy import sparse
from tests.utils.run_generated_solver import *
from tests.utils.cvxpy_to_qoco import convert

@pytest.mark.skip(reason="Will not pass unless we increase iterative refinement iterations.")
def test_lcvx_bad_scaling():
    tspan = 51
    dt = 1
    x0 = np.array([1000.0, 1000.0, 3000.0, 0.0, 0.0, 0.0])
    g = 9.807
    gs = np.deg2rad(1.0)
    tvc_max = np.deg2rad(25.0)
    rho1 = 0.40 * 411000.0
    rho2 = 411000.0
    m_dry = 25600.0
    m_fuel = 10000.0
    Isp = 311.0

    g0 = 9.807
    m0 = m_dry + m_fuel
    T = int(tspan / dt)
    a = 1 / (Isp * g0)
    nx = 6
    nu = 3

    A = np.array(
        [
            [1.0, 0.0, 0.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    B = np.array(
        [
            [0.5 * dt**2, 0.0, 0.0],
            [0.0, 0.5 * dt**2, 0.0],
            [0.0, 0.0, 0.5 * dt**2],
            [dt, 0.0, 0.0],
            [0.0, dt, 0.0],
            [0.0, 0.0, dt],
        ]
    )
    G = np.array([0.0, 0.0, -0.5 * g * dt**2, 0.0, 0.0, -g * dt])
    S = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    c = np.array([0.0, 0.0, -np.tan(0.5 * np.pi - gs), 0.0, 0.0, 0.0])
    xT = np.zeros((nx))

    x = cp.Variable((nx, T + 1))
    z = cp.Variable(T + 1)
    u = cp.Variable((nu, T + 1))
    s = cp.Variable(T + 1)

    # Objective
    obj = -z[T]

    # IC and TC
    con = [x[:, 0] == x0]
    con += [x[:, T] == xT]
    con += [z[0] == np.log(m0)]
    con += [z[T] >= np.log(m_dry)]

    # Dynamics
    for k in range(T):
        con += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k] + G]
        con += [z[k + 1] == z[k] - a * s[k] * dt]

    # State and Input Constraints
    for k in range(T + 1):
        z0 = np.log(m0 - (a * rho2 * k * dt))
        mu1 = rho1 * np.exp(-z0)
        mu2 = rho2 * np.exp(-z0)
        con += [cp.norm(u[:, k]) <= s[k]]
        con += [mu1 * (1.0 - (z[k] - z0)) <= s[k]]
        con += [s[k] <= mu2 * (1.0 - (z[k] - z0))]
        con += [cp.log(m0 - a * rho2 * k * dt) <= z[k]]
        con += [z[k] <= np.log(m0 - a * rho1 * k * dt)]
        con += [cp.norm(S @ x[:, k]) + c @ x[:, k] <= 0]
        # constraints += [u[2, k] >= s[k] * np.cos(tvc_max)]
        con += [u[2, k] >= cp.norm(u[:, k]) * np.cos(tvc_max)]

    prob = cp.Problem(cp.Minimize(obj), con)
    prob.solve(verbose=True, solver=cp.CLARABEL)

    n, m, p, P, c, A, b, G, h, l, nsoc, q = convert(prob)

    qocogen.generate_solver(
        n,
        m,
        p,
        P,
        c,
        A,
        b,
        G,
        h,
        l,
        nsoc,
        q,
        "tests/",
        "qoco_custom_lcvx_bad_scaling",
        generate_ruiz=True,
    )
    codegen_solved, codegen_obj, average_runtime_ms = run_generated_solver(
        "tests/qoco_custom_lcvx_bad_scaling"
    )

    # Solve problem.
    opt_obj = prob.value
    assert codegen_solved == 1
    assert (abs(codegen_obj - opt_obj) / abs(opt_obj)) <= 1e-3
