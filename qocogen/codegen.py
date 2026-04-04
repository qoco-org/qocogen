# Copyright (c) 2024, Govind M. Chari <govindchari1@gmail.com>
# This source code is licensed under the BSD 3-Clause License

import io
import os
import shutil
import qdldl
import numpy as np
from datetime import datetime
from scipy import sparse
from jinja2 import Environment, FileSystemLoader
from qocogen.codegen_utils import write_license, write_Kelem, write_license_file


def _get_env():
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.globals["declare_array"] = lambda name, size, t: (
        f"   {t} {name}[{size}];" if size > 0 else f"   {t}* {name};"
    )
    env.filters["py_percent"] = lambda fmt, val: fmt % val
    return env


def _timestamp():
    return datetime.now().strftime("%m/%d/%Y %H:%M:%S")


def _render(env, template_name, path, **ctx):
    content = env.get_template(template_name).render(**ctx)
    with open(path, "w") as f:
        f.write(content)


def _generate_solver(
    n, m, p, P, c, A, b, G, h, l, nsoc, q, output_dir, name, generate_ruiz
):
    solver_dir = output_dir + "/" + name

    # Need to handle case where 0 is in P.data, A.data, or G.data.
    if P is not None:
        for i in range(P.nnz):
            if P.data[i] == 0:
                P.data[i] += 1e-16
    if A is not None:
        for i in range(A.nnz):
            if A.data[i] == 0:
                A.data[i] += 1e-16
    if G is not None:
        for i in range(G.nnz):
            if G.data[i] == 0:
                G.data[i] += 1e-16
    print("\n")
    if os.path.exists(solver_dir):
        print("Removing existing solver.")
        shutil.rmtree(solver_dir)

    print("Generating qoco_custom.")
    os.mkdir(solver_dir)
    W = sparse.identity(l)
    W = sparse.csc_matrix(W)
    for qi in q:
        Wsoc = np.ones((qi, qi), dtype=np.float64)
        Wsoc = sparse.csc_matrix(Wsoc)
        W = sparse.bmat([[W, None], [None, Wsoc]])
    W = sparse.csc_matrix(W)

    Wnnz = int((W.nnz - m) / 2 + m)
    Wnnz_cnt = 0

    # Maps sparse 1D index (1,...,m^2) of W to its sparse index (1,...,Wnnz). Note that accessing an lower triangular element of W returns -1.
    Wsparse2dense = -np.ones((m * m))
    for j in range(m):
        for i in range(m):
            if W[i, j] != 0.0 and i <= j:
                Wsparse2dense[i + j * m] = Wnnz_cnt
                Wnnz_cnt += 1

    # Get sparsity pattern of the regularized KKT matrix.
    Preg = P + sparse.identity(n) if P is not None else sparse.identity(n)
    A = A if A is not None else None
    G = G if G is not None else None
    At = A.T if A is not None else None
    Gt = G.T if G is not None else None

    K = sparse.bmat(
        [
            [Preg, At, Gt],
            [A, -sparse.identity(p), None],
            [G, None, -W - 1e3 * sparse.identity(m)],
        ]
    )
    solver = qdldl.Solver(K)
    L, D, perm = solver.factors()

    # Ensure that all elements of L.data are nonzero.
    for i in range(L.nnz):
        if L.data[i] == 0:
            L.data[i] += 1e-16

    # Generate Lidx which indicates which elements of L are nonzero.
    N = n + m + p
    Lidx = [False for _ in range(N**2)]
    for i in range(N):
        for j in range(N):
            if L[i, j] != 0.0:
                Lidx[j * N + i] = True

    write_license_file(solver_dir)
    generate_cmakelists(solver_dir)
    generate_workspace(solver_dir, n, m, p, P, c, A, b, G, h, q, L.nnz, Wnnz)
    generate_cone(solver_dir, m, Wnnz, Wsparse2dense)
    generate_kkt(
        solver_dir, n, m, p, P, c, A, b, G, h, perm, Wsparse2dense, generate_ruiz
    )
    generate_utils(
        solver_dir,
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
        Wsparse2dense,
        Wnnz,
        perm,
        generate_ruiz,
    )

    generate_ldl(solver_dir, n, m, p, P, A, G, perm, Lidx, Wsparse2dense)
    generate_solver(solver_dir, m, Wsparse2dense, generate_ruiz)
    generate_runtest(solver_dir)


def generate_cmakelists(solver_dir):
    env = _get_env()
    _render(env, "CMakeLists.txt.j2", solver_dir + "/CMakeLists.txt")


def generate_workspace(solver_dir, n, m, p, P, c, A, b, G, h, q, Lnnz, Wnnz):
    Pnnz = len(P.data) if P is not None else 0
    Annz = len(A.data) if A is not None else 0
    Gnnz = len(G.data) if G is not None else 0
    qmax = max(q) if len(q) > 0 else 0
    env = _get_env()
    _render(
        env,
        "workspace.h.j2",
        solver_dir + "/workspace.h",
        timestamp=_timestamp(),
        n=n, m=m, p=p,
        Pnnz=Pnnz, Annz=Annz, Gnnz=Gnnz,
        q=q, Lnnz=Lnnz, Wnnz=Wnnz, qmax=qmax,
    )


def generate_ldl(solver_dir, n, m, p, P, A, G, perm, Lidx, Wsparse2dense):
    env = _get_env()
    _render(env, "ldl.h.j2", solver_dir + "/ldl.h", timestamp=_timestamp())

    f = open(solver_dir + "/ldl.c", "w")
    write_license(f)
    f.write('#include "ldl.h"\n\n')
    f.write("void ldl(Workspace* work) {\n")
    N = n + m + p

    # Maps sparse 1D index (1,...,N^2) of L to its sparse index (1,...,Lnnz).
    Lsparse2dense = -np.ones(N**2)

    # Number of nonzeros of L added (Used to get sparse index of the current element under consideration).
    Lnnz = 0

    # The factorization will only access strictly lower triangular elements of L.
    for j in range(N):
        # D update.
        f.write("   work->D[%i] = " % j)
        write_Kelem(f, j, j, n, m, p, P, A, G, perm, Wsparse2dense, True, True)
        for k in range(j):
            if Lidx[k * N + j]:
                f.write(" - work->D[%i] * " % k)
                f.write(
                    "work->L[%i] * work->L[%i]"
                    % (Lsparse2dense[k * N + j], Lsparse2dense[k * N + j])
                )
        f.write(";\n")
        if perm[j] < n:
            f.write("   if (work->D[%i] < 0) {\n" % j)
            f.write("       work->D[%i] = work->settings.kkt_dynamic_reg;\n" % j)
            f.write("   }\n")
            f.write("   else {\n")
            f.write("       work->D[%i] += work->settings.kkt_dynamic_reg;\n" % j)
            f.write("   }\n")
        else:
            f.write("   if (work->D[%i] > 0) {\n" % j)
            f.write("       work->D[%i] = -work->settings.kkt_dynamic_reg;\n" % j)
            f.write("   }\n")
            f.write("   else {\n")
            f.write("       work->D[%i] -= work->settings.kkt_dynamic_reg;\n" % j)
            f.write("   }\n")

        # L update.
        for i in range(j + 1, N):
            if Lidx[j * N + i]:
                Lsparse2dense[j * N + i] = Lnnz
                f.write("   work->L[%i] = " % (Lnnz))
                write_Kelem(f, j, i, n, m, p, P, A, G, perm, Wsparse2dense, True, True)
                for k in range(j):
                    if Lidx[k * N + i] and Lidx[k * N + j]:
                        f.write(
                            " - work->L[%i] * work->L[%i] * work->D[%i]"
                            % (Lsparse2dense[k * N + i], Lsparse2dense[k * N + j], k)
                        )
                f.write(";\n")
                f.write("   work->L[%i] /= work->D[%i];\n" % (Lnnz, j))
                Lnnz += 1
    f.write("}\n\n")

    f.write("void tri_solve(Workspace* work) {\n")
    for i in range(N):
        f.write("   work->xyz[%i] = work->xyz[%i]" % (i, i))
        for j in range(i):
            if Lidx[j * N + i]:
                f.write(
                    " - work->L[%i] * work->xyz[%i]" % (Lsparse2dense[j * N + i], j)
                )
        f.write(";\n")

    for i in range(N):
        f.write("   work->xyz[%i] /= work->D[%i];\n" % (i, i))

    for i in range(N - 1, -1, -1):
        f.write("   work->xyzbuff[%i] = work->xyz[%i]" % (i, i))
        for j in range(i + 1, N):
            if Lidx[i * N + j]:
                f.write(
                    " - work->L[%i] * work->xyzbuff[%i]" % (Lsparse2dense[i * N + j], j)
                )
        f.write(";\n")
    f.write("}\n\n")

    f.write("void kkt_solve(Workspace* work) {\n")
    f.write("   // Permute kkt_rhs and store in xyz.\n")
    f.write("   for (int j = 0; j < work->n + work->m + work->p; ++j) {\n")
    f.write("       work->xyz[j] = work->kkt_rhs[work->perm[j]];\n")
    f.write("   }\n")
    f.write(
        "   qoco_custom_copy_arrayf(work->xyz, work->kkt_rhs, work->n + work->p + work->m);\n\n"
    )

    f.write("   // Solve xyzbuff = K \\ xyz.\n")
    f.write("   tri_solve(work);\n")
    for i in range(N):
        f.write("   work->xyz[%i] = work->xyzbuff[%i];\n" % (perm[i], i))

    f.write("   for (int i = 0; i < work->settings.iter_ref_iters; ++i) {\n")
    f.write("       KKT_perm_product(work->xyzbuff, work->xyz, work);\n")
    f.write("       for (int j = 0; j < work->n + work->m + work->p; ++j) {\n")
    f.write("           work->xyz[j] = work->kkt_rhs[j] - work->xyz[j];\n")
    f.write("       }\n")
    f.write(
        "       qoco_custom_copy_arrayf(work->xyzbuff, work->xyzbuff2, work->n + work->m + work->p);\n"
    )
    f.write("       tri_solve(work);\n")
    f.write(
        "       qoco_custom_axpy(work->xyzbuff2, work->xyzbuff, work->xyz, 1.0, work->n + work->m + work->p);\n"
    )
    f.write(
        "       qoco_custom_copy_arrayf(work->xyz, work->xyzbuff, work->n + work->m + work->p);\n"
    )
    f.write("   }\n")
    f.write("   // Permute xyzbuff and store in xyz.\n")
    f.write("   for (int j = 0; j < work->n + work->m + work->p; ++j) {\n")
    f.write("       work->xyz[work->perm[j]] = work->xyzbuff[j];\n")
    f.write("   }\n")
    f.write("}\n")
    f.close()
    return Lsparse2dense


def _gen_compute_WtW(m, Wsparse2dense):
    buf = io.StringIO()
    for i in range(m):
        for j in range(i, m):
            if Wsparse2dense[j * m + i] != -1:
                buf.write("   work->WtW[%i] = " % Wsparse2dense[j * m + i])
                for k in range(m):
                    row1 = k
                    col1 = j
                    row2 = k
                    col2 = i
                    if col1 < row1:
                        row1, col1 = col1, row1
                    if col2 < row2:
                        row2, col2 = col2, row2
                    if (
                        Wsparse2dense[col1 * m + row1] != -1
                        and Wsparse2dense[col2 * m + row2] != -1
                    ):
                        buf.write(
                            " + work->W[%i] * work->W[%i]"
                            % (
                                Wsparse2dense[col1 * m + row1],
                                Wsparse2dense[col2 * m + row2],
                            )
                        )
                buf.write(";\n")
    return buf.getvalue()


def _gen_nt_multiply(m, Wsparse2dense):
    buf = io.StringIO()
    for i in range(m):
        buf.write("   z[%i] = " % i)
        for j in range(m):
            row = i
            col = j
            if col < row:
                row, col = col, row
            if Wsparse2dense[col * m + row] != -1:
                buf.write(" + W[%i] * x[%i]" % (Wsparse2dense[col * m + row], j))
        buf.write(";\n")
    return buf.getvalue()


def generate_cone(solver_dir, m, Wnnz, Wsparse2dense):
    env = _get_env()
    ts = _timestamp()
    _render(env, "cone.h.j2", solver_dir + "/cone.h", timestamp=ts)
    _render(
        env,
        "cone.c.j2",
        solver_dir + "/cone.c",
        timestamp=ts,
        m=m,
        Wnnz=Wnnz,
        compute_WtW_body=_gen_compute_WtW(m, Wsparse2dense),
        nt_multiply_body=_gen_nt_multiply(m, Wsparse2dense),
    )


def _gen_kkt_perm_product(n, m, p, P, A, G, perm, Wsparse2dense):
    N = n + m + p
    buf = io.StringIO()
    for i in range(N):
        buf.write("   y[%i] = " % i)
        for j in range(N):
            if write_Kelem(buf, i, j, n, m, p, P, A, G, perm, Wsparse2dense, False, True):
                buf.write(" * x[%i]" % j)
                buf.write(" + ")
        buf.write("0;\n")
    return buf.getvalue()


def generate_kkt(
    solver_dir, n, m, p, P, c, A, b, G, h, perm, Wsparse2dense, generate_ruiz
):
    env = _get_env()
    ts = _timestamp()
    _render(
        env, "kkt.h.j2", solver_dir + "/kkt.h",
        timestamp=ts, generate_ruiz=generate_ruiz,
    )
    _render(
        env,
        "kkt.c.j2",
        solver_dir + "/kkt.c",
        timestamp=ts,
        generate_ruiz=generate_ruiz,
        kkt_perm_product_body=_gen_kkt_perm_product(
            n, m, p, P, A, G, perm, Wsparse2dense
        ),
    )


def _gen_load_data(n, m, p, P, c, A, b, G, h, l, nsoc, q, perm, Pnnz, Annz, Gnnz, Wnnz):
    buf = io.StringIO()
    buf.write("void load_data(Workspace* work) {\n")
    buf.write("   work->n = %d;\n" % n)
    buf.write("   work->m = %d;\n" % m)
    buf.write("   work->p = %d;\n" % p)

    for i in range(Pnnz):
        buf.write("   work->P[%i] = %.17g;\n" % (i, P.data[i]))
    buf.write("\n")

    for i in range(len(c)):
        buf.write("   work->c[%i] = %.17g;\n" % (i, c[i]))
    buf.write("\n")

    for i in range(Annz):
        buf.write("   work->A[%i] = %.17g;\n" % (i, A.data[i]))
    buf.write("\n")

    for i in range(len(b)):
        buf.write("   work->b[%i] = %.17g;\n" % (i, b[i]))
    buf.write("\n")

    for i in range(Gnnz):
        buf.write("   work->G[%i] = %.17g;\n" % (i, G.data[i]))
    buf.write("\n")

    for i in range(len(h)):
        buf.write("   work->h[%i] = %.17g;\n" % (i, h[i]))
    buf.write("\n")
    buf.write("   work->l = %d;\n" % l)
    buf.write("   work->nsoc = %d;\n" % nsoc)

    for i in range(len(q)):
        buf.write("   work->q[%i] = %d;\n" % (i, q[i]))
    buf.write("\n")

    for i in range(len(perm)):
        buf.write("   work->perm[%i] = %d;\n" % (i, perm[i]))
    buf.write("\n")
    buf.write("   work->Pnnz = %i;\n" % Pnnz)
    buf.write("   work->Wnnz = %i;\n" % Wnnz)
    buf.write("   work->mu = 0.0;\n")
    buf.write("   work->sigma = 0.0;\n")
    buf.write("   work->a = 0.0;\n")
    buf.write("   work->sol.iters = 0;\n")
    buf.write("   work->sol.pres = 0;\n")
    buf.write("   work->sol.dres = 0;\n")
    buf.write("   work->sol.gap = 0;\n")
    buf.write("   work->sol.obj = 0;\n")
    buf.write("   work->sol.status = QOCO_CUSTOM_UNSOLVED;\n")
    buf.write("}\n\n")
    return buf.getvalue()


def _gen_kktrow_inf_norm(n, m, p, P, A, G, Wsparse2dense):
    N = n + m + p
    identity_perm = np.linspace(0, N - 1, N, dtype=np.int32)
    buf = io.StringIO()
    for i in range(N):
        for j in range(N):
            if write_Kelem(
                buf, i, j, n, m, p, P, A, G, identity_perm, Wsparse2dense, False, False
            ):
                buf.write("   norm[%i] = qoco_max(norm[%i], qoco_abs(" % (i, i))
                write_Kelem(
                    buf, i, j, n, m, p, P, A, G, identity_perm, Wsparse2dense, False, True
                )
                buf.write("));\n")
    return buf.getvalue()


def _gen_ruiz_scale_kkt(n, m, p, P, A, G, Wsparse2dense):
    N = n + m + p
    identity_perm = np.linspace(0, N - 1, N, dtype=np.int32)
    buf = io.StringIO()
    for i in range(n):
        for j in range(i, N):
            if write_Kelem(
                buf, i, j, n, m, p, P, A, G, identity_perm, Wsparse2dense, False, False
            ):
                buf.write("   ")
                write_Kelem(
                    buf, i, j, n, m, p, P, A, G, identity_perm, Wsparse2dense, False, True
                )
                buf.write(" *= d[%i] * d[%i];\n" % (i, j))
    return buf.getvalue()


def _gen_pinf_norm(n, m, p, P, A, G, Wsparse2dense):
    N = n + m + p
    identity_perm = np.linspace(0, N - 1, N, dtype=np.int32)
    buf = io.StringIO()
    for i in range(n):
        for j in range(n):
            if write_Kelem(
                buf, i, j, n, m, p, P, A, G, identity_perm, Wsparse2dense, False, False
            ):
                buf.write("   norm[%i] = qoco_max(norm[%i], qoco_abs(" % (i, i))
                write_Kelem(
                    buf, i, j, n, m, p, P, A, G, identity_perm, Wsparse2dense, False, True
                )
                buf.write("));\n")
    return buf.getvalue()


def _gen_matrix_product(row_range, col_range, row_offset, col_offset,
                         n, m, p, P, A, G, Wsparse2dense):
    N = n + m + p
    identity_perm = np.linspace(0, N - 1, N, dtype=np.int32)
    buf = io.StringIO()
    for i in row_range:
        buf.write("   y[%i] = " % i)
        for j in col_range:
            if write_Kelem(
                buf,
                i + row_offset,
                j + col_offset,
                n, m, p, P, A, G,
                identity_perm, Wsparse2dense, False, True,
            ):
                buf.write(" * x[%i]" % j)
                buf.write(" + ")
        buf.write("0;\n")
    return buf.getvalue()


def generate_utils(
    solver_dir,
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
    Wsparse2dense,
    Wnnz,
    perm,
    generate_ruiz,
):
    Pnnz = len(P.data) if P is not None else 0
    Annz = len(A.data) if A is not None else 0
    Gnnz = len(G.data) if G is not None else 0

    env = _get_env()
    ts = _timestamp()
    _render(
        env, "utils.h.j2", solver_dir + "/utils.h",
        timestamp=ts, generate_ruiz=generate_ruiz,
    )

    ctx = dict(
        timestamp=ts,
        n=n, m=m, p=p, l=l, nsoc=nsoc,
        Pnnz=Pnnz, Annz=Annz, Gnnz=Gnnz,
        n_constraints=l + p + nsoc,
        generate_ruiz=generate_ruiz,
        load_data_body=_gen_load_data(
            n, m, p, P, c, A, b, G, h, l, nsoc, q, perm, Pnnz, Annz, Gnnz, Wnnz
        ),
        Px_body=_gen_matrix_product(
            range(n), range(n), 0, 0, n, m, p, P, A, G, Wsparse2dense
        ),
        Ax_body=_gen_matrix_product(
            range(p), range(n), n, 0, n, m, p, P, A, G, Wsparse2dense
        ),
        Gx_body=_gen_matrix_product(
            range(m), range(n), n + p, 0, n, m, p, P, A, G, Wsparse2dense
        ),
        Atx_body=_gen_matrix_product(
            range(n), range(p), 0, n, n, m, p, P, A, G, Wsparse2dense
        ),
        Gtx_body=_gen_matrix_product(
            range(n), range(m), 0, n + p, n, m, p, P, A, G, Wsparse2dense
        ),
    )
    if generate_ruiz:
        ctx["kktrow_norm_body"] = _gen_kktrow_inf_norm(n, m, p, P, A, G, Wsparse2dense)
        ctx["ruiz_scale_body"] = _gen_ruiz_scale_kkt(n, m, p, P, A, G, Wsparse2dense)
        ctx["pinf_norm_body"] = _gen_pinf_norm(n, m, p, P, A, G, Wsparse2dense)
    else:
        ctx["kktrow_norm_body"] = ""
        ctx["ruiz_scale_body"] = ""
        ctx["pinf_norm_body"] = ""

    _render(env, "utils.c.j2", solver_dir + "/utils.c", **ctx)


def generate_solver(solver_dir, m, Wsparse2dense, generate_ruiz):
    env = _get_env()
    ts = _timestamp()
    _render(env, "qoco_custom.h.j2", solver_dir + "/qoco_custom.h", timestamp=ts)

    wtw_buf = io.StringIO()
    for i in range(m):
        wtw_buf.write("   work->WtW[%i] = 1.0;\n" % Wsparse2dense[i * m + i])

    _render(
        env,
        "qoco_custom.c.j2",
        solver_dir + "/qoco_custom.c",
        timestamp=ts,
        generate_ruiz=generate_ruiz,
        wtw_init_body=wtw_buf.getvalue(),
    )


def generate_runtest(solver_dir):
    env = _get_env()
    _render(env, "runtest.c.j2", solver_dir + "/runtest.c", timestamp=_timestamp())
