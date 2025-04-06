# Copyright (c) 2024, Govind M. Chari <govindchari1@gmail.com>
# This source code is licensed under the BSD 3-Clause License

import os
import shutil
import qdldl
import numpy as np
from scipy import sparse
from qocogen.codegen_utils import write_license, write_Kelem, declare_array


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
    f = open(solver_dir + "/CMakeLists.txt", "a")
    f.write("cmake_minimum_required(VERSION 3.18)\n")
    f.write("project(qoco_custom)\n\n")
    f.write("if(DISABLE_PRINTING)\n")
    f.write("   add_compile_definitions(DISABLE_PRINTING)\n")
    f.write("endif()\n\n")
    f.write("if(QOCO_CUSTOM_BUILD_TYPE STREQUAL Debug)\n")
    f.write("   set(QOCO_CUSTOM_BUILD_TYPE Debug)\n")
    f.write(
        '   set(CMAKE_C_FLAGS "-g -Wall")\n'
    )
    f.write('   if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")\n')
    f.write('       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address,undefined")\n')
    f.write("   endif()\n")
    f.write("else()\n")
    f.write("   set(QOCO_CUSTOM_BUILD_TYPE Release)\n")
    f.write('   set(CMAKE_C_FLAGS "-O3 -Wall -march=native")\n')
    f.write("endif()\n\n")
    f.write("# Detect OS.\n")
    f.write('message(STATUS "We are on a ${CMAKE_SYSTEM_NAME} system")\n')
    f.write('if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")\n')
    f.write('   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra")\n')
    f.write("else()\n")
    f.write("   add_compile_definitions(IS_WINDOWS)\n")
    f.write("   if(QOCO_CUSTOM_BUILD_TYPE STREQUAL Release)\n")
    f.write('       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Ox")\n')
    f.write("   endif()\n")
    f.write("endif()\n")
    f.write('message(STATUS "Build Type: " ${QOCO_CUSTOM_BUILD_TYPE})\n')
    f.write('message(STATUS "Build Flags: " ${CMAKE_C_FLAGS})\n\n')

    f.write('set(qoco_custom_sources "${CMAKE_CURRENT_SOURCE_DIR}/qoco_custom.c"\n\t"${CMAKE_CURRENT_SOURCE_DIR}/cone.c"\n\t"${CMAKE_CURRENT_SOURCE_DIR}/utils.c"\n\t"${CMAKE_CURRENT_SOURCE_DIR}/ldl.c"\n\t"${CMAKE_CURRENT_SOURCE_DIR}/kkt.c")\n\n')
    f.write('set(qoco_custom_headers "${CMAKE_CURRENT_SOURCE_DIR}/qoco_custom.h"\n\t"${CMAKE_CURRENT_SOURCE_DIR}/cone.h"\n\t"${CMAKE_CURRENT_SOURCE_DIR}/utils.h"\n\t"${CMAKE_CURRENT_SOURCE_DIR}/ldl.h"\n\t"${CMAKE_CURRENT_SOURCE_DIR}/kkt.h")\n\n')

    f.write("# Build qoco_custom shared and static library.\n")
    f.write("add_library(qoco_custom SHARED)\n")
    f.write(
        "target_sources(qoco_custom PRIVATE ${qoco_custom_sources} ${qoco_custom_headers})\n"
    )
    f.write('if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")\n')
    f.write("   target_link_libraries(qoco_custom m)\n")
    f.write("endif()\n\n")
    f.write("# Build qoco demo.\n")
    f.write("add_executable(runtest runtest.c)\n")
    f.write('if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")\n')
    f.write("   target_link_libraries(runtest qoco_custom)\n")
    f.write("else()\n")
    f.write("   add_library(qoco_custom_static STATIC)\n")
    f.write(
        "   target_sources(qoco_custom_static PRIVATE ${qoco_custom_sources} ${qoco_custom_headers})\n"
    )
    f.write("   target_link_libraries(runtest qoco_custom_static)\n")
    f.write("endif()\n\n")

    f.close()


def generate_workspace(solver_dir, n, m, p, P, c, A, b, G, h, q, Lnnz, Wnnz):
    f = open(solver_dir + "/workspace.h", "a")
    write_license(f)
    f.write("#ifndef QOCO_CUSTOM_WORKSPACE_H\n")
    f.write("#define QOCO_CUSTOM_WORKSPACE_H\n\n")

    f.write("typedef struct {\n")
    f.write("   int max_iters;\n")
    f.write("   int bisect_iters;\n")
    f.write("   int ruiz_iters;\n")
    f.write("   int iter_ref_iters;\n")
    f.write("   double kkt_static_reg;\n")
    f.write("   double kkt_dynamic_reg;\n")
    f.write("   double abstol;\n")
    f.write("   double reltol;\n")
    f.write("   double abstol_inacc;\n")
    f.write("   double reltol_inacc;\n")
    f.write("   unsigned char verbose;\n")
    f.write("} Settings;\n\n")

    f.write("typedef struct {\n")
    declare_array(f, "x", n, "double")
    declare_array(f, "s", m, "double")
    declare_array(f, "y", p, "double")
    declare_array(f, "z", m, "double")
    f.write("   int iters;\n")
    f.write("   double obj;\n")
    f.write("   double pres;\n")
    f.write("   double dres;\n")
    f.write("   double gap;\n")
    f.write("   unsigned char status;\n")
    f.write("} Solution;\n\n")

    Pnnz = len(P.data) if P is not None else 0
    Annz = len(A.data) if A is not None else 0
    Gnnz = len(G.data) if G is not None else 0
    qmax = max(q) if len(q) > 0 else 0

    f.write("typedef struct {\n")
    f.write("   int n;\n")
    f.write("   int m;\n")
    f.write("   int p;\n")
    declare_array(f, "P", Pnnz, "double")
    declare_array(f, "c", n, "double")
    declare_array(f, "A", Annz, "double")
    declare_array(f, "b", p, "double")
    declare_array(f, "G", Gnnz, "double")
    declare_array(f, "h", m, "double")
    f.write("   int l;\n")
    f.write("   int nsoc;\n")
    declare_array(f, "q", len(q), "int")
    f.write("   int perm[%i];\n" % (n + m + p))
    f.write("   int Pnnz;\n")
    f.write("   int Wnnz;\n")
    declare_array(f, "x", n, "double")
    declare_array(f, "s", m, "double")
    declare_array(f, "y", p, "double")
    declare_array(f, "z", m, "double")
    declare_array(f, "L", Lnnz, "double")
    declare_array(f, "D", (n + m + p), "double")
    declare_array(f, "W", Wnnz, "double")
    declare_array(f, "lambda", m, "double")
    declare_array(f, "xbuff", n, "double")
    declare_array(f, "ubuff1", m, "double")
    declare_array(f, "ubuff2", m, "double")
    declare_array(f, "ubuff3", m, "double")
    declare_array(f, "Ds", m, "double")
    declare_array(f, "Winv", Wnnz, "double")
    declare_array(f, "WtW", Wnnz, "double")
    declare_array(f, "kkt_rhs", (n + m + p), "double")
    declare_array(f, "kkt_res", (n + m + p), "double")
    declare_array(f, "xyz", (n + m + p), "double")
    declare_array(f, "Druiz", n, "double")
    declare_array(f, "Eruiz", p, "double")
    declare_array(f, "Fruiz", m, "double")
    declare_array(f, "Dinvruiz", n, "double")
    declare_array(f, "Einvruiz", p, "double")
    declare_array(f, "Finvruiz", m, "double")
    f.write("   double k;\n")
    f.write("   double kinv;\n")
    declare_array(f, "xyzbuff", (n + m + p), "double")
    declare_array(f, "xyzbuff2", (n + m + p), "double")
    declare_array(f, "sbar", qmax, "double")
    declare_array(f, "zbar", qmax, "double")
    f.write("   double mu;\n")
    f.write("   double sigma;\n")
    f.write("   double a;\n\n")
    f.write("   Settings settings;\n")
    f.write("   Solution sol;\n")
    f.write("} Workspace;\n\n")

    f.write("#endif")
    f.close()


def generate_ldl(solver_dir, n, m, p, P, A, G, perm, Lidx, Wsparse2dense):
    f = open(solver_dir + "/ldl.h", "a")
    write_license(f)
    f.write("#ifndef QOCO_CUSTOM_LDL_H\n")
    f.write("#define QOCO_CUSTOM_LDL_H\n\n")
    f.write('#include "kkt.h"\n')
    f.write('#include "utils.h"\n')
    f.write('#include "workspace.h"\n\n')

    f.write("void ldl(Workspace* work);\n\n")
    f.write("// Solves L*D*L'*xyzbuff = kkt_rhs.\n")
    f.write("void tri_solve(Workspace* work);\n\n")
    f.write("// Solves L*D*L'*xyz = kkt_rhs with iterative refinement.\n")
    f.write("void kkt_solve(Workspace* work);\n")
    f.write("#endif")
    f.close()

    f = open(solver_dir + "/ldl.c", "a")
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


def generate_cone(solver_dir, m, Wnnz, Wsparse2dense):
    # Write header.
    f = open(solver_dir + "/cone.h", "a")
    write_license(f)
    f.write("#ifndef QOCO_CUSTOM_CONE_H\n")
    f.write("#define QOCO_CUSTOM_CONE_H\n\n")
    f.write('#include "utils.h"\n\n')

    f.write("void soc_product(double* u, double* v, double* p, int n);\n")
    f.write("void soc_division(double* lam, double* v, double* d, int n);\n")
    f.write("double soc_residual(double* u, int n);\n")
    f.write("double soc_residual2(double* u, int n);\n")
    f.write("double cone_residual(double* u, int l, int nsoc, int* q);\n")
    f.write("void bring2cone(double* u, int l, int nsoc, int* q);\n")
    f.write(
        "void cone_product(double* u, double* v, double* p, int l, int nsoc, int* q);\n"
    )
    f.write(
        "void cone_division(double* lambda, double* v, double* d, int l, int nsoc, int* q);\n"
    )
    f.write("void compute_mu(Workspace* work);\n")
    f.write("void compute_nt_scaling(Workspace* work);\n")
    f.write("void compute_lambda(Workspace* work);\n")
    f.write("void compute_WtW(Workspace* work);\n")
    f.write(
        "// Computes z = W * x, where W has the same sparsity structure as the Nesterov-Todd scaling matrices.\n"
    )
    f.write("void nt_multiply(double* W, double* x, double* z);\n")
    f.write("double linesearch(double* u, double* Du, double f, Workspace* work);\n")
    f.write("double bisection_search(double* u, double* Du, double f, Workspace* work);\n")
    f.write("double exact_linesearch(double* u, double* Du, double f, Workspace* work);\n")
    f.write("void compute_centering(Workspace* work);\n")
    f.write("#endif")
    f.close()

    # Write source.
    f = open(solver_dir + "/cone.c", "a")
    write_license(f)
    f.write('#include "cone.h"\n\n')

    f.write("void soc_product(double* u, double* v, double* p, int n) {\n")
    f.write("   p[0] = qoco_custom_dot(u, v, n);\n")
    f.write("   for (int i = 1; i < n; ++i) {\n")
    f.write("       p[i] = u[0] * v[i] + v[0] * u[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void soc_division(double* lam, double* v, double* d, int n) {\n")
    f.write("   double f = lam[0] * lam[0] - qoco_custom_dot(&lam[1], &lam[1], n - 1);\n")
    f.write("   double finv = safe_div(1.0, f);\n")
    f.write("   double lam0inv = safe_div(1.0, lam[0]);\n")
    f.write("   double lam1dv1 = qoco_custom_dot(&lam[1], &v[1], n - 1);\n")
    f.write("   d[0] = finv * (lam[0] * v[0] - qoco_custom_dot(&lam[1], &v[1], n - 1));\n")
    f.write("   for (int i = 1; i < n; ++i) {\n")
    f.write(
        "       d[i] = finv * (-lam[i] * v[0] + lam0inv * f * v[i] + lam0inv * lam1dv1 * lam[i]);\n"
    )
    f.write("   }\n")
    f.write("}\n")

    f.write("double soc_residual(double* u, int n) {\n")
    f.write("   double res = 0;\n")
    f.write("   for (int i = 1; i < n; ++i) {\n")
    f.write("      res += u[i] * u[i];\n")
    f.write("   }\n")
    f.write("   res = qoco_sqrt(res) - u[0];\n")
    f.write("   return res;\n")
    f.write("}\n\n")
    f.write("double soc_residual2(double* u, int n) {\n")
    f.write("   double res = u[0] * u[0];\n")
    f.write("   for (int i = 1; i < n; ++i) {\n")
    f.write("      res -= u[i] * u[i];\n")
    f.write("   }\n")
    f.write("   return res;\n")
    f.write("}\n\n")

    f.write("double cone_residual(double* u, int l, int nsoc, int* q) {\n")
    f.write("   double res = -1e7;\n")
    f.write("   int idx;\n")
    f.write("   for (idx = 0; idx < l; ++idx) {\n")
    f.write("      res = qoco_max(-u[idx], res);\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < nsoc; ++i) {\n")
    f.write("      res = qoco_max(soc_residual(&u[idx], q[i]), res);\n")
    f.write("      idx += q[i];\n")
    f.write("   }\n")
    f.write("   return res;\n")
    f.write("}\n\n")

    f.write("void bring2cone(double* u, int l, int nsoc, int* q) {\n")
    f.write("   if (cone_residual(u, l, nsoc, q) >= 0) {\n")
    f.write("      double a = 0.0;\n\n")
    f.write("      int idx;\n")
    f.write("      for (idx = 0; idx < l; ++idx) {\n")
    f.write("         a = qoco_max(a, -u[idx]);\n")
    f.write("      }\n")
    f.write("      a = qoco_max(a, 0.0);\n\n")
    f.write("      for (int i = 0; i < nsoc; ++i) {\n")
    f.write("         double soc_res = soc_residual(&u[idx], q[i]);\n")
    f.write("         if (soc_res > 0 && soc_res > a) {\n")
    f.write("            a = soc_res;\n")
    f.write("         }\n")
    f.write("         idx += q[i];\n")
    f.write("      }\n")
    f.write("      for (idx = 0; idx < l; ++idx) {\n")
    f.write("         u[idx] += (1 + a);\n")
    f.write("      }\n")
    f.write("      for (int i = 0; i < nsoc; ++i) {\n")
    f.write("         u[idx] += (1 + a);\n")
    f.write("         idx += q[i];\n")
    f.write("      }\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write(
        "void cone_product(double* u, double* v, double* p, int l, int nsoc, int* q) {\n"
    )
    f.write("   int idx;\n")
    f.write("   for (idx = 0; idx < l; ++idx) {\n")
    f.write("       p[idx] = u[idx] * v[idx];\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < nsoc; ++i) {\n")
    f.write("       soc_product(&u[idx], &v[idx], &p[idx], q[i]);\n")
    f.write("       idx += q[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write(
        "void cone_division(double* lambda, double* v, double* d, int l, int nsoc, int* q) {\n"
    )
    f.write("   int idx;\n")
    f.write("   for (idx = 0; idx < l; ++idx) {\n")
    f.write("       d[idx] = safe_div(v[idx], lambda[idx]);\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < nsoc; ++i) {\n")
    f.write("       soc_division(&lambda[idx], &v[idx], &d[idx], q[i]);\n")
    f.write("       idx += q[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void compute_mu(Workspace* work) {\n")
    if m == 0:
        f.write("   work->mu = 0.0;\n")
    else:
        f.write("   work->mu = (qoco_custom_dot(work->s, work->z, work->m) / work->m);\n")
    f.write("}\n\n")

    f.write("void compute_nt_scaling(Workspace* work) {\n")
    for i in range(Wnnz):
        f.write("   work->W[%i] = 0.0;\n" % i)
        f.write("   work->Winv[%i] = 0.0;\n" % i)
    f.write("   int idx;\n")
    f.write("   for (idx = 0; idx < work->l; ++idx) {\n")
    f.write("       work->W[idx] = qoco_sqrt(safe_div(work->s[idx], work->z[idx]));\n")
    f.write("       work->Winv[idx] = safe_div(1.0, work->W[idx]);\n")
    f.write("   }\n\n")

    f.write("   int nt_idx = idx;\n")
    f.write("   for (int i = 0; i < work->nsoc; ++i) {\n")
    f.write("       // Compute normalized vectors.\n")
    f.write("       double s_scal = soc_residual2(&work->s[idx], work->q[i]);\n")
    f.write("       s_scal = qoco_sqrt(s_scal);\n")
    f.write("       double f = safe_div(1.0, s_scal);\n")
    f.write("       qoco_custom_scale_arrayf(&work->s[idx], work->sbar, f, work->q[i]);\n\n")

    f.write("       double z_scal = soc_residual2(&work->z[idx], work->q[i]);\n")
    f.write("       z_scal = qoco_sqrt(z_scal);\n")
    f.write("       f = safe_div(1.0, z_scal);\n")
    f.write("       qoco_custom_scale_arrayf(&work->z[idx], work->zbar, f, work->q[i]);\n\n")

    f.write(
        "       double gamma = qoco_sqrt(0.5 * (1 + qoco_custom_dot(work->sbar, work->zbar, work->q[i])));\n"
    )
    f.write("       f = safe_div(1.0, (2 * gamma));\n\n")
    f.write("       // Overwrite sbar with wbar.\n")
    f.write("       work->sbar[0] = f * (work->sbar[0] + work->zbar[0]);\n")
    f.write("       for (int j = 1; j < work->q[i]; ++j) {\n")
    f.write("           work->sbar[j] = f * (work->sbar[j] - work->zbar[j]);\n")
    f.write("       }\n\n")
    f.write("       // Overwrite zbar with v.\n")
    f.write("       f = safe_div(1.0, qoco_sqrt(2 * (work->sbar[0] + 1)));\n")
    f.write("       work->zbar[0] = f * (work->sbar[0] + 1.0);\n")
    f.write("       for (int j = 1; j < work->q[i]; ++j) {\n")
    f.write("           work->zbar[j] = f * work->sbar[j];\n")
    f.write("       }\n\n")
    f.write("       // Compute W for second-order cones.\n")
    f.write("       int shift = 0;\n")
    f.write("       f = qoco_sqrt(safe_div(s_scal, z_scal));\n")
    f.write("       double finv = safe_div(1.0, f);\n")
    f.write("       for (int j = 0; j < work->q[i]; ++j) {\n")
    f.write("           for (int k = 0; k <= j; ++k) {\n")
    f.write(
        "               work->W[nt_idx + shift] = 2 * (work->zbar[k] * work->zbar[j]);\n"
    )
    f.write("               if (j != 0 && k == 0) {\n")
    f.write(
        "                   work->Winv[nt_idx + shift] = -work->W[nt_idx + shift];\n"
    )
    f.write("               }\n")
    f.write("               else {\n")
    f.write(
        "                   work->Winv[nt_idx + shift] = work->W[nt_idx + shift];\n"
    )
    f.write("               }\n")
    f.write("               if (j == k && j == 0) {\n")
    f.write("                   work->W[nt_idx + shift] -= 1;\n")
    f.write("                   work->Winv[nt_idx + shift] -= 1;\n")
    f.write("               }\n")
    f.write("               else if (j == k) {\n")
    f.write("                   work->W[nt_idx + shift] += 1;\n")
    f.write("                   work->Winv[nt_idx + shift] += 1;\n")
    f.write("               }\n")
    f.write("               work->W[nt_idx + shift] *= f;\n")
    f.write("               work->Winv[nt_idx + shift] *= finv;\n")
    f.write("               shift += 1;\n")
    f.write("           }\n")
    f.write("       }\n")
    f.write("       idx += work->q[i];\n")
    f.write("       nt_idx += (work->q[i] * work->q[i] + work->q[i]) / 2;\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void compute_WtW(Workspace* work) {\n")
    for i in range(m):
        for j in range(i, m):
            if Wsparse2dense[j * m + i] != -1:
                f.write("   work->WtW[%i] = " % Wsparse2dense[j * m + i])
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
                        f.write(
                            " + work->W[%i] * work->W[%i]"
                            % (
                                Wsparse2dense[col1 * m + row1],
                                Wsparse2dense[col2 * m + row2],
                            )
                        )
                f.write(";\n")

    f.write("}\n\n")

    f.write("void compute_lambda(Workspace* work) {\n")
    f.write("   nt_multiply(work->W, work->z, work->lambda);\n")
    f.write("}\n\n")

    f.write("void nt_multiply(double* W, double* x, double* z) {\n")
    for i in range(m):
        f.write("   z[%i] = " % i)
        for j in range(m):
            row = i
            col = j
            if col < row:
                row, col = col, row
            if Wsparse2dense[col * m + row] != -1:
                f.write(" + W[%i] * x[%i]" % (Wsparse2dense[col * m + row], j))
        f.write(";\n")
    f.write("}\n\n")

    f.write("double linesearch(double* u, double* Du, double f, Workspace* work) {\n")
    f.write("   if (work->nsoc == 0) {\n")
    f.write("       return exact_linesearch(u, Du, f, work);\n")
    f.write("   }\n")
    f.write("   else {\n")
    f.write("       return bisection_search(u, Du, f, work);\n")
    f.write("   }\n")
    f.write("}\n")

    f.write("double bisection_search(double* u, double* Du, double f, Workspace* work) {\n")
    f.write("   double al = 0.0;\n")
    f.write("   double au = 1.0;\n")
    f.write("   double a = 0.0;\n")
    f.write("   for (int i = 0; i < work->settings.bisect_iters; ++i) {\n")
    f.write("       a = 0.5 * (al + au);\n")
    f.write("       qoco_custom_axpy(Du, u, work->ubuff1, safe_div(a, f), work->m);\n")
    f.write(
        "       if (cone_residual(work->ubuff1, work->l, work->nsoc, work->q) >= 0) {\n"
    )
    f.write("           au = a;\n")
    f.write("       }\n")
    f.write("       else {\n")
    f.write("           al = a;\n")
    f.write("       }\n")
    f.write("   }\n")
    f.write("   return al;\n")
    f.write("}\n")

    f.write("double exact_linesearch(double* u, double* Du, double f, Workspace* work) {\n")
    f.write("   double a = 1.0;\n")
    f.write("   double minval = 0;\n")
    f.write("   // Compute a for LP cones.\n")
    f.write("   for (int i = 0; i < work->l; ++i) {\n")
    f.write("       if (Du[i] < minval * u[i])\n")
    f.write("           minval = Du[i] / u[i];\n\n")
    f.write("   }\n")
    f.write("   if (-f < minval)\n")
    f.write("       a = f;\n")
    f.write("   else\n")
    f.write("       a = -f / minval;\n\n")
    f.write("   return a;\n")
    f.write("}\n")


    f.write("void compute_centering(Workspace* work) {\n")
    f.write(
        "   double a = qoco_min(linesearch(work->z, &work->xyz[work->n + work->p], 1.0, work), linesearch(work->s, work->Ds, 1.0, work));\n"
    )
    f.write(
        "   qoco_custom_axpy(&work->xyz[work->n + work->p], work->z, work->ubuff1, a, work->m);\n"
    )
    f.write("   qoco_custom_axpy(work->Ds, work->s, work->ubuff2, a, work->m);\n")
    f.write(
        "   double rho = safe_div(qoco_custom_dot(work->ubuff1, work->ubuff2, work->m), qoco_custom_dot(work->z, work->s, work->m));\n"
    )
    f.write("   double sigma = qoco_min(1.0, rho);\n")
    f.write("   sigma = qoco_max(0.0, sigma);\n")
    f.write("   sigma = sigma * sigma * sigma;\n")
    f.write("   work->sigma = sigma;\n")
    f.write("}\n")
    f.close()


def generate_kkt(
    solver_dir, n, m, p, P, c, A, b, G, h, perm, Wsparse2dense, generate_ruiz
):
    # Write header.
    f = open(solver_dir + "/kkt.h", "a")
    write_license(f)
    f.write("#ifndef QOCO_CUSTOM_KKT_H\n")
    f.write("#define QOCO_CUSTOM_KKT_H\n\n")
    f.write('#include "cone.h"\n')
    f.write('#include "ldl.h"\n')
    f.write('#include "workspace.h"\n\n')
    if generate_ruiz:
        f.write("void ruiz_equilibration(Workspace* work);\n")
        f.write("void unequilibrate_data(Workspace* work);\n\n")
    f.write("// Computes y = K * x where K is the permuted KKT matrix.\n")
    f.write("void KKT_perm_product(double* x, double* y, Workspace* work);\n")
    f.write("void compute_kkt_residual(Workspace* work);\n")
    f.write("void construct_kkt_aff_rhs(Workspace* work);\n")
    f.write("void construct_kkt_comb_rhs(Workspace* work);\n")
    f.write("void predictor_corrector(Workspace* work);\n")
    f.write("#endif")
    f.close()

    # Write source.
    N = n + m + p
    f = open(solver_dir + "/kkt.c", "a")
    write_license(f)
    f.write('#include "kkt.h"\n\n')
    if generate_ruiz:
        f.write("void ruiz_equilibration(Workspace* work) {\n")
        f.write("   for (int i = 0; i < work->n; ++i) {\n")
        f.write("       work->Druiz[i] = 1.0;\n")
        f.write("       work->Dinvruiz[i] = 1.0;\n")
        f.write("   }\n")
        f.write("   for (int i = 0; i < work->p; ++i) {\n")
        f.write("       work->Eruiz[i] = 1.0;\n")
        f.write("       work->Einvruiz[i] = 1.0;\n")
        f.write("   }\n")
        f.write("   for (int i = 0; i < work->m; ++i) {\n")
        f.write("       work->Fruiz[i] = 1.0;\n")
        f.write("       work->Finvruiz[i] = 1.0;\n")
        f.write("   }\n")
        f.write("   work->k = 1.0;\n")
        f.write("   work->kinv = 1.0;\n\n")
        f.write("   for (int iter = 0; iter < work->settings.ruiz_iters; ++iter) {\n")
        f.write("       for (int i = 0; i < work->n + work->m + work->p; ++i) {\n")
        f.write("           work->xyzbuff[i] = 0.0;\n")
        f.write("       }\n\n")
        f.write("       KKTrow_inf_norm(work->xyzbuff, work);\n")
        f.write("       for (int i = 0; i < work->n + work->m + work->p; ++i) {\n")
        f.write(
            "           work->xyzbuff[i] = safe_div(1.0, qoco_sqrt(work->xyzbuff[i]));\n"
        )
        f.write("       }\n\n")

        f.write('       // g = 1 / max(mean(Pinf), norm(c, "inf"));\n')
        f.write("       double g = 0.0;\n")
        f.write("       Pinf_norm(work->xbuff, work);\n")
        f.write("       for (int i = 0; i < work->n; ++i) {\n")
        f.write("           g += work->xbuff[i];\n")
        f.write("       }\n")
        f.write("       g /= work->n;\n")
        f.write("       double cinf = qoco_custom_inf_norm(work->c, work->n);\n")
        f.write("       g = qoco_max(g, cinf);\n")
        f.write("       g = safe_div(1.0, g);\n")
        f.write("       work->k *= g;\n\n")
        f.write(
            "       // Make scalings for all variables in a second-order cone equal.\n"
        )
        f.write("       int idx = work->l;\n")
        f.write("       for (int j = 0; j < work->nsoc; ++j) {\n")
        f.write("           for (int k = idx + 1; k < idx + work->q[j]; ++k) {\n")
        f.write(
            "               work->xyzbuff[work->n + work->p + k] = work->xyzbuff[work->n + work->p + idx];\n"
        )
        f.write("           }\n")
        f.write("       idx += work->q[j];\n")
        f.write("       }\n\n")

        f.write("       // Scale P by g.\n")
        f.write("       qoco_custom_scale_arrayf(work->P, work->P, g, work->Pnnz);\n")

        f.write("       // Scale c.\n")
        f.write("       qoco_custom_scale_arrayf(work->c, work->c, g, work->n);\n")
        f.write("       qoco_custom_ew_product(work->c, work->xyzbuff, work->c, work->n);\n\n")

        f.write("       // Scale P, A, G.\n")
        f.write("       ruiz_scale_KKT(work->xyzbuff, work);\n\n")

        f.write("       // Update scaling matrices.\n")
        f.write(
            "       qoco_custom_ew_product(work->Druiz, work->xyzbuff, work->Druiz, work->n);\n"
        )
        f.write(
            "       qoco_custom_ew_product(work->Eruiz, &work->xyzbuff[work->n], work->Eruiz, work->p);\n"
        )
        f.write(
            "       qoco_custom_ew_product(work->Fruiz, &work->xyzbuff[work->n + work->p], work->Fruiz, work->m);\n"
        )
        f.write("   }\n")
        f.write("   // Scale b.\n")
        f.write("   qoco_custom_ew_product(work->b, work->Eruiz, work->b, work->p);\n\n")

        f.write("   // Scale h.\n")
        f.write("   qoco_custom_ew_product(work->h, work->Fruiz, work->h, work->m);\n\n")

        f.write("   // Compute Dinv, Einv, Finv.\n")
        f.write("   work->kinv = safe_div(1.0, work->k);\n")
        f.write("   for (int i = 0; i < work->n; ++i) {\n")
        f.write("       work->Dinvruiz[i] = safe_div(1.0, work->Druiz[i]);\n")
        f.write("   }\n")
        f.write("   for (int i = 0; i < work->p; ++i) {\n")
        f.write("       work->Einvruiz[i] = safe_div(1.0, work->Eruiz[i]);\n")
        f.write("   }\n")
        f.write("   for (int i = 0; i < work->m; ++i) {\n")
        f.write("       work->Finvruiz[i] = safe_div(1.0, work->Fruiz[i]);\n")
        f.write("   }\n")
        f.write("}\n\n")

        f.write("void unequilibrate_data(Workspace* work) {\n")
        f.write("   qoco_custom_copy_arrayf(work->Dinvruiz, work->xyzbuff, work->n);\n")
        f.write("   qoco_custom_copy_arrayf(work->Einvruiz, &work->xyzbuff[work->n], work->p);\n")
        f.write(
            "   qoco_custom_copy_arrayf(work->Finvruiz, &work->xyzbuff[work->n + work->p], work->m);\n"
        )
        f.write("   ruiz_scale_KKT(work->xyzbuff, work);\n")
        f.write("   qoco_custom_scale_arrayf(work->P, work->P, work->kinv, work->Pnnz);\n")
        f.write("   qoco_custom_scale_arrayf(work->c, work->c, work->kinv, work->n);\n")
        f.write("   qoco_custom_ew_product(work->c, work->Dinvruiz, work->c, work->n);\n")
        f.write("   qoco_custom_ew_product(work->b, work->Einvruiz, work->b, work->p);\n")
        f.write("   qoco_custom_ew_product(work->h, work->Finvruiz, work->h, work->m);\n")
        f.write("}\n\n")

    f.write("void KKT_perm_product(double* x, double* y, Workspace* work) {\n")
    for i in range(N):
        f.write("   y[%i] = " % i)
        for j in range(N):
            if write_Kelem(
                f,
                i,
                j,
                n,
                m,
                p,
                P,
                A,
                G,
                perm,
                Wsparse2dense,
                False,
                True,
            ):
                f.write(" * x[%i]" % j)
                f.write(" + ")
        f.write("0;\n")
    f.write("}\n")

    # f.write("void KKT_product(double* x, double* y, Workspace* work) {\n")
    # for i in range(N):
    #     f.write("   y[%i] = " % i)
    #     for j in range(N):
    #         if write_Kelem(
    #             f,
    #             i,
    #             j,
    #             n,
    #             m,
    #             p,
    #             P,
    #             A,
    #             G,
    #             np.linspace(0, N - 1, N, dtype=np.int32),
    #             Wsparse2dense,
    #             False,
    #             True,
    #         ):
    #             f.write(" * x[%i]" % j)
    #             f.write(" + ")
    #     f.write("0;\n")
    # f.write("}\n")

    # f.write("// computes \|y-K*x\|_\inf where K is the KKT matrix.\n")
    # f.write("double kkt_solve_verify(Workspace* work, double* x, double* y) {\n")
    # f.write("   KKT_product(x, work->xyzbuff, work);\n")
    # f.write("   for (int i = 0; i < work->n + work->m + work->p; ++i) {\n")
    # f.write("       work->xyzbuff2[i] = work->xyzbuff[i] - y[i];\n")
    # f.write("   }\n")
    # f.write("   double res = qoco_custom_inf_norm(work->xyzbuff2, work->n + work->m + work->p);\n")
    # f.write("   return res;\n")
    # f.write("}\n")

    f.write("void compute_kkt_residual(Workspace* work) {\n")

    f.write("   // Zero out NT Block.\n")
    f.write("   for (int i = 0; i < work->Wnnz; ++i) {\n")
    f.write("       work->WtW[i] = 0.0;\n")
    f.write("   }\n")

    f.write("   // Load [x;y;z] into xyzbuff.\n")
    f.write("   for (int i = 0; i < work->n; ++i) {\n")
    f.write("       work->xyzbuff[i] = work->x[i];\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < work->p; ++i) {\n")
    f.write("       work->xyzbuff[i + work->n] = work->y[i];\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < work->m; ++i) {\n")
    f.write("       work->xyzbuff[i + work->n + work->p] = work->z[i];\n")
    f.write("   }\n\n")
    f.write("   // Permute xyzbuff and store into xyz.\n")
    f.write("   for (int i = 0; i < work->n + work->p + work->m; ++i) {\n")
    f.write("       work->xyz[i] = work->xyzbuff[work->perm[i]];\n")
    f.write("   }\n\n")
    f.write("   KKT_perm_product(work->xyz, work->xyzbuff, work);\n\n")
    f.write("   // Permute xyzbuff and store into xyz.\n")
    f.write("   for (int i = 0; i < work->n + work->p + work->m; ++i) {\n")
    f.write("       work->kkt_res[work->perm[i]] = work->xyzbuff[i];\n")
    f.write("   }\n\n")
    f.write("   // Add [c;-b;-h+s].\n")
    f.write("   for (int i = 0; i < work->n; ++i) {\n")
    f.write("       work->kkt_res[i] += work->c[i];\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < work->p; ++i) {\n")
    f.write("       work->kkt_res[i + work->n] -= work->b[i];\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < work->m; ++i) {\n")
    f.write(
        "       work->kkt_res[i + work->n + work->p] += (work->s[i] - work->h[i]);\n"
    )
    f.write("   }\n\n")
    f.write("}\n\n")

    f.write("void construct_kkt_aff_rhs(Workspace* work) {\n")
    f.write(
        "   copy_and_negate_arrayf(work->kkt_res, work->kkt_rhs, work->n + work->p + work->m);\n"
    )
    f.write("   nt_multiply(work->W, work->lambda, work->ubuff1);\n")
    f.write("   for (int i = 0; i < work->m; ++i) {\n")
    f.write("       work->kkt_rhs[work->n + work->p + i] += work->ubuff1[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void construct_kkt_comb_rhs(Workspace* work) {\n")
    f.write(
        "   copy_and_negate_arrayf(work->kkt_res, work->kkt_rhs, work->n + work->p + work->m);\n"
    )
    f.write("   nt_multiply(work->Winv, work->Ds, work->ubuff1);\n")
    f.write("   nt_multiply(work->W, &work->xyz[work->n + work->p], work->ubuff2);\n")
    f.write(
        "   cone_product(work->ubuff1, work->ubuff2, work->ubuff3, work->l, work->nsoc, work->q);\n"
    )
    f.write("   double sm = work->sigma * work->mu;\n")
    f.write("   int idx = 0;\n")
    f.write("   for (idx = 0; idx < work->l; ++idx) {\n")
    f.write("       work->ubuff3[idx] -= sm;\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < work->nsoc; ++i) {\n")
    f.write("       work->ubuff3[idx] -= sm;\n")
    f.write("       idx += work->q[i];\n")
    f.write("   }\n")
    f.write(
        "   cone_product(work->lambda, work->lambda, work->ubuff1, work->l, work->nsoc, work->q);\n"
    )
    f.write("   for (int i = 0; i < work->m; ++i) {\n")
    f.write("       work->Ds[i] = -work->ubuff1[i] - work->ubuff3[i];\n")
    f.write("   }\n")
    f.write(
        "   cone_division(work->lambda, work->Ds, work->ubuff2, work->l, work->nsoc, work->q);\n"
    )
    f.write("   nt_multiply(work->W, work->ubuff2, work->ubuff1);\n")
    f.write("   for (int i = 0; i < work->m; ++i) {\n")
    f.write("       work->kkt_rhs[work->n + work->p + i] -= work->ubuff1[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void predictor_corrector(Workspace* work) {\n")
    f.write("   ldl(work);\n\n")
    f.write("   // Construct rhs for affine scaling direction.\n")
    f.write("   construct_kkt_aff_rhs(work);\n\n")
    # f.write("   qoco_custom_copy_arrayf(work->kkt_rhs, work->xyzbuff3, work->n + work->m + work->p);\n")
    f.write("   // Solve KKT system to get affine scaling direction.\n")
    f.write("   kkt_solve(work);\n\n")
    # f.write("   double res = kkt_solve_verify(work, work->xyz, work->xyzbuff3);\n")
    # f.write("   printf(\"%f\", res);\n")
    f.write("   // Compute Dsaff.\n")
    f.write("   nt_multiply(work->W, &work->xyz[work->n + work->p], work->ubuff1);\n")
    f.write("   for (int i = 0; i < work->m; ++i) {\n")
    f.write("       work->ubuff1[i] = -work->lambda[i] - work->ubuff1[i];\n")
    f.write("   }\n")
    f.write("   nt_multiply(work->W, work->ubuff1, work->Ds);\n\n")
    f.write("   compute_centering(work);\n")
    f.write("   construct_kkt_comb_rhs(work);\n")
    f.write("   kkt_solve(work);\n\n")
    f.write(
        "   // Check if solution has NaNs. If NaNs are present, early exit and set a to 0.0 to trigger reduced tolerance optimality checks.\n"
    )
    f.write("   for (int i = 0; i < work->n + work->m + work->p; ++i) {\n")
    f.write("       if (isnan(work->xyz[i])) {\n")
    f.write("           work->a = 0.0;\n")
    f.write("           return;\n")
    f.write("       }\n")
    f.write("   }\n")
    f.write("   // Compute Dz.\n")
    f.write(
        "   cone_division(work->lambda, work->Ds, work->ubuff1, work->l, work->nsoc, work->q);\n"
    )
    f.write("   nt_multiply(work->W, &work->xyz[work->n + work->p], work->ubuff2);\n")
    f.write("   for (int i = 0; i < work->m; ++i) {\n")
    f.write("       work->ubuff3[i] = work->ubuff1[i] - work->ubuff2[i];\n")
    f.write("   }\n")
    f.write("   nt_multiply(work->W, work->ubuff3, work->Ds);\n\n")
    f.write("   // Compute step-size.\n")
    f.write(
        "   double a = qoco_min(linesearch(work->s, work->Ds, 0.99, work), linesearch(work->z, &work->xyz[work->n + work->p], 0.99, work));\n"
    )
    f.write("   work->a = a;\n\n")
    f.write("   // Update iterate.\n")
    f.write("   for (int i = 0; i < work->n; ++i) {\n")
    f.write("       work->x[i] += a * work->xyz[i];\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < work->m; ++i) {\n")
    f.write("       work->s[i] += a * work->Ds[i];\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < work->p; ++i) {\n")
    f.write("       work->y[i] += a * work->xyz[work->n + i];\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < work->m; ++i) {\n")
    f.write("       work->z[i] += a * work->xyz[work->n + work->p + i];\n")
    f.write("   }\n")
    f.write("}\n\n")
    f.close()


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
    # Write header.
    f = open(solver_dir + "/utils.h", "a")
    write_license(f)
    f.write("#ifndef QOCO_CUSTOM_UTILS_H\n")
    f.write("#define QOCO_CUSTOM_UTILS_H\n\n")
    f.write('#include "workspace.h"\n\n')
    f.write("#define qoco_abs(x) ((x)<0 ? -(x) : (x))\n")
    f.write("#define safe_div(a, b) (qoco_abs(a) > 1e-15) ? (a / b) : 1e16\n")
    f.write("#include <math.h>\n")
    f.write("#define qoco_sqrt(a) sqrt(a)\n")
    f.write("#define qoco_min(a, b) (((a) < (b)) ? (a) : (b))\n")
    f.write("#define qoco_max(a, b) (((a) > (b)) ? (a) : (b))\n\n")

    f.write("enum qoco_custom_solve_status {\n")
    f.write("   QOCO_CUSTOM_UNSOLVED = 0,\n")
    f.write("   QOCO_CUSTOM_SOLVED,\n")
    f.write("   QOCO_CUSTOM_SOLVED_INACCURATE,\n")
    f.write("   QOCO_CUSTOM_NUMERICAL_ERROR,\n")
    f.write("   QOCO_CUSTOM_MAX_ITER,\n")
    f.write("};\n\n")

    f.write("static const char *QOCO_CUSTOM_SOLVE_STATUS_MESSAGE[] = {\n")
    f.write('   "unsolved",\n')
    f.write('   "solved",\n')
    f.write('   "solved inaccurately",\n')
    f.write('   "numerical problems",\n')
    f.write('   "maximum iterations reached",\n')
    f.write("};\n\n")

    f.write("void update_P(Workspace* work, double* P_new);\n")
    f.write("void update_A(Workspace* work, double* A_new);\n")
    f.write("void update_G(Workspace* work, double* G_new);\n")
    f.write("void update_c(Workspace* work, double* c_new);\n")
    f.write("void update_b(Workspace* work, double* b_new);\n")
    f.write("void update_h(Workspace* work, double* h_new);\n")

    f.write("void load_data(Workspace* work);\n")
    f.write("void set_default_settings(Workspace* work);\n")
    f.write("void qoco_custom_copy_arrayf(double* x, double* y, int n);\n")
    f.write("void copy_and_negate_arrayf(double* x, double* y, int n);\n")
    f.write("double qoco_custom_dot(double* x, double* y, int n);\n")
    f.write("double qoco_custom_inf_norm(double* x, int n);\n")
    if generate_ruiz:
        f.write("void KKTrow_inf_norm(double* norm, Workspace* work);\n")
        f.write("void ruiz_scale_KKT(double* d, Workspace* work);\n")
        f.write("void Pinf_norm(double* norm, Workspace* work);\n")
    f.write("void qoco_custom_qoco_custom_Px(double* x, double* y, Workspace* work);\n")
    f.write("void qoco_custom_Ax(double* x, double* y, Workspace* work);\n")
    f.write("void Gx(double* x, double* y, Workspace* work);\n")
    f.write("void Atx(double* x, double* y, Workspace* work);\n")
    f.write("void Gtx(double* x, double* y, Workspace* work);\n")
    f.write("void qoco_custom_scale_arrayf(double* x, double* y, double s, int n);\n")
    f.write("void qoco_custom_ew_product(double* x, double* y, double* z, int n);\n")
    f.write("void qoco_custom_axpy(double* x, double* y, double* z, double a, int n);\n")
    f.write("unsigned char check_stopping(Workspace* work);\n")
    f.write("void copy_solution(Workspace* work);\n")
    f.write("void unscale_solution(Workspace* work);\n")
    f.write("#ifndef DISABLE_PRINTING\n")
    f.write("#include <stdio.h>\n")
    f.write("void print_header(Workspace* work);\n")
    f.write("void print_footer(Workspace* work);\n")
    f.write("void log_iter(Workspace* work);\n")
    f.write("#endif\n")
    f.write("#endif")
    f.close()

    # Write source.
    N = n + m + p
    Pnnz = len(P.data) if P is not None else 0
    Annz = len(A.data) if A is not None else 0
    Gnnz = len(G.data) if G is not None else 0

    f = open(solver_dir + "/utils.c", "a")
    write_license(f)
    f.write('#include "utils.h"\n\n')
    f.write("void update_P(Workspace* work, double* P_new) {\n")
    f.write("   for (int i = 0; i < %d; ++i) {\n" % Pnnz)
    f.write("       work->P[i] = P_new[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void update_A(Workspace* work, double* A_new) {\n")
    f.write("   for (int i = 0; i < %d; ++i) {\n" % Annz)
    f.write("       work->A[i] = A_new[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void update_G(Workspace* work, double* G_new) {\n")
    f.write("   for (int i = 0; i < %d; ++i) {\n" % Gnnz)
    f.write("       work->G[i] = G_new[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void update_c(Workspace* work, double* c_new) {\n")
    f.write("   for (int i = 0; i < %d; ++i) {\n" % n)
    f.write("       work->c[i] = c_new[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void update_b(Workspace* work, double* b_new) {\n")
    f.write("   for (int i = 0; i < %d; ++i) {\n" % p)
    f.write("       work->b[i] = b_new[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void update_h(Workspace* work, double* h_new) {\n")
    f.write("   for (int i = 0; i < %d; ++i) {\n" % m)
    f.write("       work->h[i] = h_new[i];\n")
    f.write("   }\n")
    f.write("}\n\n")


    f.write("void load_data(Workspace* work) {\n")
    f.write("   work->n = %d;\n" % n)
    f.write("   work->m = %d;\n" % m)
    f.write("   work->p = %d;\n" % p)

    for i in range(Pnnz):
        f.write("   work->P[%i] = %.17g;\n" % (i, P.data[i]))
    f.write("\n")

    for i in range(len(c)):
        f.write("   work->c[%i] = %.17g;\n" % (i, c[i]))
    f.write("\n")

    for i in range(Annz):
        f.write("   work->A[%i] = %.17g;\n" % (i, A.data[i]))
    f.write("\n")

    for i in range(len(b)):
        f.write("   work->b[%i] = %.17g;\n" % (i, b[i]))
    f.write("\n")

    for i in range(Gnnz):
        f.write("   work->G[%i] = %.17g;\n" % (i, G.data[i]))
    f.write("\n")

    for i in range(len(h)):
        f.write("   work->h[%i] = %.17g;\n" % (i, h[i]))
    f.write("\n")
    f.write("   work->l = %d;\n" % l)
    f.write("   work->nsoc = %d;\n" % nsoc)

    for i in range(len(q)):
        f.write("   work->q[%i] = %d;\n" % (i, q[i]))
    f.write("\n")

    for i in range(len(perm)):
        f.write("   work->perm[%i] = %d;\n" % (i, perm[i]))
    f.write("\n")
    f.write("   work->Pnnz = %i;\n" % Pnnz)
    f.write("   work->Wnnz = %i;\n" % Wnnz)
    f.write("   work->mu = 0.0;\n")
    f.write("   work->sigma = 0.0;\n")
    f.write("   work->a = 0.0;\n")
    f.write("   work->sol.iters = 0;\n")
    f.write("   work->sol.pres = 0;\n")
    f.write("   work->sol.dres = 0;\n")
    f.write("   work->sol.gap = 0;\n")
    f.write("   work->sol.obj = 0;\n")
    f.write("   work->sol.status = QOCO_CUSTOM_UNSOLVED;\n")
    f.write("}\n\n")

    f.write("void set_default_settings(Workspace* work) {\n")
    f.write("   work->settings.max_iters = 200;\n")
    f.write("   work->settings.bisect_iters = 5;\n")
    if generate_ruiz:
        f.write("   work->settings.ruiz_iters = 5;\n")
    else:
        f.write("   work->settings.ruiz_iters = 0;\n")
    f.write("   work->settings.iter_ref_iters = 1;\n")
    f.write("   work->settings.kkt_static_reg = 1e-8;\n")
    f.write("   work->settings.kkt_dynamic_reg = 1e-8;\n")
    f.write("   work->settings.abstol = 1e-7;\n")
    f.write("   work->settings.reltol = 1e-7;\n")
    f.write("   work->settings.abstol_inacc = 1e-5;\n")
    f.write("   work->settings.reltol_inacc = 1e-5;\n")
    f.write("   work->settings.verbose = 0;\n")
    f.write("}\n\n")

    f.write("void qoco_custom_copy_arrayf(double* x, double* y, int n) {\n")
    f.write("   for (int i = 0; i < n; ++i) {\n")
    f.write("      y[i] = x[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void copy_and_negate_arrayf(double* x, double* y, int n) {\n")
    f.write("   for (int i = 0; i < n; ++i) {\n")
    f.write("      y[i] = -x[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("double qoco_custom_dot(double* x, double* y, int n) {\n")
    f.write("   double ans = 0;\n")
    f.write("      for (int i = 0; i < n; ++i) {\n")
    f.write("         ans += x[i] * y[i];\n")
    f.write("      }\n")
    f.write("   return ans;\n")
    f.write("}\n\n")

    f.write("double qoco_custom_inf_norm(double* x, int n) {\n")
    f.write("   double norm = 0.0;\n")
    f.write("   double xi;\n")
    f.write("   for (int i = 0; i < n; ++i) {\n")
    f.write("       xi = qoco_abs(x[i]);\n")
    f.write("       norm = qoco_max(norm , xi);\n")
    f.write("   }\n")
    f.write("   return norm;\n")
    f.write("}\n\n")

    if generate_ruiz:
        f.write("void KKTrow_inf_norm(double* norm, Workspace* work) {\n")
        f.write("   // Zero out NT Block.\n")
        f.write("   for (int i = 0; i < work->Wnnz; ++i) {\n")
        f.write("       work->WtW[i] = 0.0;\n")
        f.write("   }\n")
        for i in range(N):
            for j in range(N):
                if write_Kelem(
                    f,
                    i,
                    j,
                    n,
                    m,
                    p,
                    P,
                    A,
                    G,
                    np.linspace(0, N - 1, N, dtype=np.int32),
                    Wsparse2dense,
                    False,
                    False,
                ):
                    f.write("   norm[%i] = qoco_max(norm[%i], qoco_abs(" % (i, i))
                    write_Kelem(
                        f,
                        i,
                        j,
                        n,
                        m,
                        p,
                        P,
                        A,
                        G,
                        np.linspace(0, N - 1, N, dtype=np.int32),
                        Wsparse2dense,
                        False,
                        True,
                    )
                    f.write("));\n")
        f.write("}\n\n")

        f.write("void ruiz_scale_KKT(double* d, Workspace* work) {\n")
        for i in range(n):
            for j in range(i, N):
                if write_Kelem(
                    f,
                    i,
                    j,
                    n,
                    m,
                    p,
                    P,
                    A,
                    G,
                    np.linspace(0, N - 1, N, dtype=np.int32),
                    Wsparse2dense,
                    False,
                    False,
                ):
                    f.write("   ")
                    write_Kelem(
                        f,
                        i,
                        j,
                        n,
                        m,
                        p,
                        P,
                        A,
                        G,
                        np.linspace(0, N - 1, N, dtype=np.int32),
                        Wsparse2dense,
                        False,
                        True,
                    )
                    f.write(" *= d[%i] * d[%i];\n" % (i, j))
        f.write("}\n\n")

        f.write("void Pinf_norm(double* norm, Workspace* work) {\n")
        f.write("   // Clear norm buffer.\n")
        f.write("   for (int i = 0; i < work->n; ++i) {\n")
        f.write("       norm[i] = 0.0;\n")
        f.write("   }\n")
        for i in range(n):
            for j in range(n):
                if write_Kelem(
                    f,
                    i,
                    j,
                    n,
                    m,
                    p,
                    P,
                    A,
                    G,
                    np.linspace(0, N - 1, N, dtype=np.int32),
                    Wsparse2dense,
                    False,
                    False,
                ):
                    f.write("   norm[%i] = qoco_max(norm[%i], qoco_abs(" % (i, i))
                    write_Kelem(
                        f,
                        i,
                        j,
                        n,
                        m,
                        p,
                        P,
                        A,
                        G,
                        np.linspace(0, N - 1, N, dtype=np.int32),
                        Wsparse2dense,
                        False,
                        True,
                    )
                    f.write("));\n")
        f.write("}\n")

    f.write("void qoco_custom_Px(double* x, double* y, Workspace* work) {\n")
    for i in range(n):
        f.write("   y[%i] = " % i)
        for j in range(n):
            if write_Kelem(
                f,
                i,
                j,
                n,
                m,
                p,
                P,
                A,
                G,
                np.linspace(0, N - 1, N, dtype=np.int32),
                Wsparse2dense,
                False,
                True,
            ):
                f.write(" * x[%i]" % j)
                f.write(" + ")
        f.write("0;\n")
    f.write("}\n\n")

    f.write("void qoco_custom_Ax(double* x, double* y, Workspace* work) {\n")
    for i in range(p):
        f.write("   y[%i] = " % i)
        for j in range(n):
            if write_Kelem(
                f,
                i + n,
                j,
                n,
                m,
                p,
                P,
                A,
                G,
                np.linspace(0, N - 1, N, dtype=np.int32),
                Wsparse2dense,
                False,
                True,
            ):
                f.write(" * x[%i]" % j)
                f.write(" + ")
        f.write("0;\n")
    f.write("}\n\n")

    f.write("void Gx(double* x, double* y, Workspace* work) {\n")
    for i in range(m):
        f.write("   y[%i] = " % i)
        for j in range(n):
            if write_Kelem(
                f,
                i + n + p,
                j,
                n,
                m,
                p,
                P,
                A,
                G,
                np.linspace(0, N - 1, N, dtype=np.int32),
                Wsparse2dense,
                False,
                True,
            ):
                f.write(" * x[%i]" % j)
                f.write(" + ")
        f.write("0;\n")
    f.write("}\n\n")

    f.write("void Atx(double* x, double* y, Workspace* work) {\n")
    for i in range(n):
        f.write("   y[%i] = " % i)
        for j in range(p):
            if write_Kelem(
                f,
                i,
                j + n,
                n,
                m,
                p,
                P,
                A,
                G,
                np.linspace(0, N - 1, N, dtype=np.int32),
                Wsparse2dense,
                False,
                True,
            ):
                f.write(" * x[%i]" % j)
                f.write(" + ")
        f.write("0;\n")
    f.write("}\n\n")

    f.write("void Gtx(double* x, double* y, Workspace* work) {\n")
    for i in range(n):
        f.write("   y[%i] = " % i)
        for j in range(m):
            if write_Kelem(
                f,
                i,
                j + n + p,
                n,
                m,
                p,
                P,
                A,
                G,
                np.linspace(0, N - 1, N, dtype=np.int32),
                Wsparse2dense,
                False,
                True,
            ):
                f.write(" * x[%i]" % j)
                f.write(" + ")
        f.write("0;\n")
    f.write("}\n\n")

    f.write("void qoco_custom_scale_arrayf(double* x, double* y, double s, int n) {\n")
    f.write("   for (int i = 0; i < n; ++i) {\n")
    f.write("       y[i] = s * x[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void qoco_custom_ew_product(double* x, double* y, double* z, int n) {\n")
    f.write("   for (int i = 0; i < n; ++i) {\n")
    f.write("       z[i] = x[i] * y[i];\n")
    f.write("   }\n")
    f.write("}\n")

    f.write("void qoco_custom_axpy(double* x, double* y, double* z, double a, int n) {\n")
    f.write("   for (int i = 0; i < n; ++i) {\n")
    f.write("       z[i] = a * x[i] + y[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("unsigned char check_stopping(Workspace* work) {\n")
    f.write("   // Compute objective.\n")
    f.write("   double obj = qoco_custom_dot(work->c, work->x, work->n);\n")
    f.write("   qoco_custom_Px(work->x, work->xbuff, work);\n")
    f.write("   double xqoco_custom_Px = qoco_custom_dot(work->x, work->xbuff, work->n);\n")
    f.write("   obj += 0.5 * qoco_custom_dot(work->x, work->xbuff, work->n);\n")
    f.write("   work->sol.obj = obj * work->kinv;\n")
    f.write("   qoco_custom_ew_product(work->xbuff, work->Dinvruiz, work->xbuff, work->n);\n")
    f.write("   double qoco_custom_Pxinf = qoco_custom_inf_norm(work->xbuff, work->n);\n\n")

    f.write("   // Compute primal residual, dual residual, and duality gap.\n")

    f.write(
        "   qoco_custom_ew_product(&work->kkt_res[work->n], work->Einvruiz, work->xbuff, work->p);\n"
    )
    f.write("   double eq_res = qoco_custom_inf_norm(work->xbuff, work->p);\n")
    f.write(
        "   qoco_custom_ew_product(&work->kkt_res[work->n + work->p], work->Finvruiz, work->ubuff1, work->m);\n"
    )
    f.write("   double conic_res = qoco_custom_inf_norm(work->ubuff1, work->m);\n")
    f.write("   double pres = qoco_max(eq_res, conic_res);\n")
    f.write("   qoco_custom_ew_product(work->kkt_res, work->Dinvruiz, work->xbuff, work->n);\n")
    f.write("   qoco_custom_scale_arrayf(work->xbuff, work->xbuff, work->kinv, work->n);\n")
    f.write("   double dres = qoco_custom_inf_norm(work->xbuff, work->n);\n")
    f.write("   work->sol.pres = pres;\n")
    f.write("   work->sol.dres = dres;\n\n")
    f.write("   qoco_custom_ew_product(work->s, work->Fruiz, work->ubuff1, work->m);\n")
    f.write("   qoco_custom_ew_product(work->z, work->Fruiz, work->ubuff2, work->m);\n")
    f.write("   double gap = qoco_custom_dot(work->ubuff1, work->ubuff2, work->m);\n")
    f.write("   gap *= work->kinv;\n")
    f.write("   work->sol.gap = gap;\n\n")

    f.write("   qoco_custom_ew_product(work->Einvruiz, work->b, work->xbuff, work->p);\n")
    f.write("   double binf = work->p > 0 ? qoco_custom_inf_norm(work->xbuff, work->p) : 0;\n\n")

    f.write("   qoco_custom_ew_product(work->Fruiz, work->s, work->ubuff1, work->m);\n")
    f.write("   double sinf = work->m > 0 ? qoco_custom_inf_norm(work->ubuff1, work->m) : 0;\n\n")

    f.write("   qoco_custom_ew_product(work->Dinvruiz, work->x, work->xbuff, work->n);\n")
    f.write("   double cinf = work->n > 0 ? qoco_custom_inf_norm(work->xbuff, work->n) : 0;\n\n")

    f.write("   qoco_custom_ew_product(work->Finvruiz, work->h, work->ubuff3, work->m);\n")
    f.write("   double hinf = work->m > 0 ? qoco_custom_inf_norm(work->ubuff3, work->m) : 0;\n\n")

    f.write("   Gtx(work->z, work->xbuff, work);\n")
    f.write("   qoco_custom_ew_product(work->xbuff, work->Dinvruiz, work->xbuff, work->n);\n")
    f.write("   double Gtzinf = qoco_custom_inf_norm(work->xbuff, work->n);\n\n")

    f.write("   Atx(work->y, work->xbuff, work);\n")
    f.write("   qoco_custom_ew_product(work->xbuff, work->Dinvruiz, work->xbuff, work->n);\n")
    f.write("   double Atyinf = qoco_custom_inf_norm(work->xbuff, work->n);\n\n")

    f.write("   Gx(work->x, work->ubuff1, work);\n")
    f.write("   qoco_custom_ew_product(work->ubuff1, work->Finvruiz, work->ubuff1, work->m);\n")

    f.write("   double Gxinf = qoco_custom_inf_norm(work->ubuff1, work->m);\n")
    # Using xbuff instead of adding a ybuff, since n >= p
    f.write("   qoco_custom_Ax(work->x, work->xbuff, work);\n")
    f.write("   qoco_custom_ew_product(work->xbuff, work->Einvruiz, work->xbuff, work->p);\n")
    f.write("   double Axinf = qoco_custom_inf_norm(work->xbuff, work->p);\n\n")

    f.write("   // Compute max{Axinf, binf, Gxinf, hinf, sinf}.\n")
    f.write("   double pres_rel = qoco_max(Axinf, binf);\n")
    f.write("   pres_rel = qoco_max(pres_rel, Gxinf);\n")
    f.write("   pres_rel = qoco_max(pres_rel, hinf);\n")
    f.write("   pres_rel = qoco_max(pres_rel, sinf);\n\n")
    f.write("   // Compute max{qoco_custom_Pxinf, Atyinf, Gtzinf, cinf}.\n")
    f.write("   double dres_rel = qoco_max(qoco_custom_Pxinf, Atyinf);\n")
    f.write("   dres_rel = qoco_max(dres_rel, Gtzinf);\n")
    f.write("   dres_rel = qoco_max(dres_rel, cinf);\n")
    f.write("   dres_rel *= work->kinv;\n")

    f.write("   // Compute max{sinf, zinf}.\n")
    f.write("   double ctx = qoco_custom_dot(work->c, work->x, work->n);\n")
    f.write("   double bty = qoco_custom_dot(work->b, work->y, work->p);\n")
    f.write("   double htz = qoco_custom_dot(work->h, work->z, work->m);\n\n")

    f.write("   double pobj = 0.5 * xqoco_custom_Px + ctx;\n")
    f.write("   double dobj = -0.5 * xqoco_custom_Px - bty - htz;\n")
    f.write("   pobj = qoco_abs(pobj);\n")
    f.write("   dobj = qoco_abs(dobj);\n\n")

    f.write("   double gap_rel = qoco_max(1, pobj);\n")
    f.write("   gap_rel = qoco_max(gap_rel, dobj);\n\n")

    f.write(
        "   // If the solver stalled (a = 0) check if low tolerance stopping criteria is met.\n "
    )
    f.write("  if(work->a < 1e-8) {\n")
    f.write(
        "      if (pres < work->settings.abstol_inacc + work->settings.reltol_inacc * pres_rel && dres < work->settings.abstol_inacc + work->settings.reltol_inacc * dres_rel && work->sol.gap < work->settings.abstol_inacc + work->settings.reltol_inacc * gap_rel) {\n"
    )
    f.write("           work->sol.status = QOCO_CUSTOM_SOLVED_INACCURATE;\n")
    f.write("           return 1;\n")
    f.write("      }\n")
    f.write("      else {\n")
    f.write("           work->sol.status = QOCO_CUSTOM_NUMERICAL_ERROR;\n")
    f.write("           return 1;\n")
    f.write("      }\n")
    f.write("   }\n")

    f.write(
        "   if (pres < work->settings.abstol + work->settings.reltol * pres_rel && dres < work->settings.abstol + work->settings.reltol * dres_rel && work->sol.gap < work->settings.abstol + work->settings.reltol * gap_rel) {\n"
    )
    f.write("      work->sol.status = QOCO_CUSTOM_SOLVED;\n")
    f.write("      return 1;\n")
    f.write("   }\n")
    f.write("   return 0;\n")
    f.write("}\n\n")

    f.write("void copy_solution(Workspace* work) {\n")
    f.write("   qoco_custom_copy_arrayf(work->x, work->sol.x, work->n);\n")
    f.write("   qoco_custom_copy_arrayf(work->s, work->sol.s, work->m);\n")
    f.write("   qoco_custom_copy_arrayf(work->y, work->sol.y, work->p);\n")
    f.write("   qoco_custom_copy_arrayf(work->z, work->sol.z, work->m);\n")
    f.write("}\n\n")

    f.write("void unscale_solution(Workspace* work) {\n")
    f.write("   qoco_custom_ew_product(work->x, work->Druiz, work->x, work->n);\n")
    f.write("   qoco_custom_ew_product(work->s, work->Finvruiz, work->s, work->m);\n")
    f.write("   qoco_custom_ew_product(work->y, work->Eruiz, work->y, work->p);\n")
    f.write("   qoco_custom_scale_arrayf(work->y, work->y, work->kinv, work->p);\n")
    f.write("   qoco_custom_ew_product(work->z, work->Fruiz, work->z, work->m);\n")
    f.write("   qoco_custom_scale_arrayf(work->z, work->z, work->kinv, work->m);\n")
    f.write("}\n")

    Pnnz = len(P.data) if P is not None else 0
    Annz = len(A.data) if A is not None else 0
    Gnnz = len(G.data) if G is not None else 0

    f.write("void print_header(Workspace* work) {\n")
    f.write("#ifndef DISABLE_PRINTING\n")
    f.write('   printf("\\n");\n')
    f.write(
        '   printf("+-------------------------------------------------------+\\n");\n'
    )
    f.write(
        '   printf("|              QOCO Custom Generated Solver             |\\n");\n'
    )
    f.write(
        '   printf("|             (c) Govind M. Chari, 2025                 |\\n");\n'
    )
    f.write(
        '   printf("|    University of Washington Autonomous Controls Lab   |\\n");\n'
    )
    f.write(
        '   printf("+-------------------------------------------------------+\\n");\n'
    )
    f.write(
        '   printf("| Problem Data:                                         |\\n");\n'
    )
    f.write(
        '   printf("|     variables:        %-9d                       |\\n");\n' % n
    )
    f.write(
        '   printf("|     constraints:      %-9d                       |\\n");\n'
        % (l + p + nsoc)
    )
    f.write(
        '   printf("|     eq constraints:   %-9d                       |\\n");\n' % p
    )
    f.write(
        '   printf("|     ineq constraints: %-9d                       |\\n");\n' % l
    )
    f.write(
        '   printf("|     soc constraints:  %-9d                       |\\n");\n' % nsoc
    )
    f.write(
        '   printf("|     nnz(P):           %-9d                       |\\n");\n' % Pnnz
    )
    f.write(
        '   printf("|     nnz(A):           %-9d                       |\\n");\n' % Annz
    )
    f.write(
        '   printf("|     nnz(G):           %-9d                       |\\n");\n' % Gnnz
    )
    f.write(
        '   printf("| Solver Settings:                                      |\\n");\n'
    )
    f.write(
        '   printf("|     max_iters: %-3d abstol: %3.2e reltol: %3.2e  |\\n", work->settings.max_iters, work->settings.abstol, work->settings.reltol);\n'
    )
    f.write(
        '   printf("|     abstol_inacc: %3.2e reltol_inacc: %3.2e     |\\n", work->settings.abstol_inacc, work->settings.reltol_inacc);\n'
    )
    f.write(
        '   printf("|     bisect_iters: %-2d iter_ref_iters: %-2d               |\\n", work->settings.bisect_iters, work->settings.iter_ref_iters);\n'
    )
    f.write(
        '   printf("|     ruiz_iters: %-2d kkt_static_reg: %3.2e           |\\n", work->settings.ruiz_iters, work->settings.kkt_static_reg);\n'
    )
    f.write(
        '   printf("|     kkt_dynamic_reg: %3.2e                         |\\n", work->settings.kkt_dynamic_reg);\n'
    )
    f.write(
        '   printf("+-------------------------------------------------------+\\n");\n'
    )
    f.write('   printf("\\n");\n')
    f.write(
        '   printf("+--------+-----------+------------+------------+------------+-----------+-----------+\\n");\n'
    )
    f.write(
        '   printf("|  Iter  |   Pcost   |    Pres    |    Dres    |     Gap    |     Mu    |    Step   |\\n");\n'
    )
    f.write(
        '   printf("+--------+-----------+------------+------------+------------+-----------+-----------+\\n");\n'
    )
    f.write("#endif\n")
    f.write("}\n\n")

    f.write("void print_footer(Workspace* work) {\n")
    f.write("#ifndef DISABLE_PRINTING\n")
    f.write(
        '   printf("\\nstatus: %s ", QOCO_CUSTOM_SOLVE_STATUS_MESSAGE[work->sol.status]);\n'
    )
    f.write('   printf("\\nnumber of iterations: %d ", work->sol.iters);\n')
    f.write('   printf("\\nobjective: %f ", work->sol.obj);\n')
    f.write("#endif\n")
    f.write("}\n\n")

    f.write("void log_iter(Workspace* work) {\n")
    f.write("#ifndef DISABLE_PRINTING\n")
    f.write(
        'printf("|   %2d   | %+.2e | %+.3e | %+.3e | %+.3e | %+.2e |   %.3f   |\\n",work->sol.iters, work->sol.obj, work->sol.pres, work->sol.dres, work->sol.gap, work->mu, work->a);\n'
    )
    f.write(
        'printf("+--------+-----------+------------+------------+------------+-----------+-----------+\\n");'
    )
    f.write("\n#endif\n")
    f.write("}\n")
    f.close()


def generate_solver(solver_dir, m, Wsparse2dense, generate_ruiz):
    f = open(solver_dir + "/qoco_custom.h", "a")
    write_license(f)
    f.write("#ifndef QOCO_CUSTOM_H\n")
    f.write("#define QOCO_CUSTOM_H\n\n")
    f.write('#include "cone.h"\n')
    f.write('#include "kkt.h"\n')
    f.write('#include "ldl.h"\n')
    f.write('#include "utils.h"\n')
    f.write('#include "workspace.h"\n\n')
    f.write("void qoco_custom_solve(Workspace* work);\n")
    f.write("#endif")
    f.close()

    f = open(solver_dir + "/qoco_custom.c", "a")
    write_license(f)
    f.write('#include "qoco_custom.h"\n\n')
    f.write("void initialize_ipm(Workspace* work) {\n")
    f.write(
        "   // Need to be set to 1.0 not 0.0 due to low tolerance stopping criteria checks\n"
    )
    f.write(
        "   // which only occur when a = 0.0. If a is set to 0.0 then the low tolerance\n"
    )
    f.write("   // stopping criteria check would be triggered.\n")
    f.write("   work->a = 1.0;\n")
    f.write("   // Set NT block to I.\n")
    f.write("   for (int i = 0; i < work->Wnnz; ++i) {\n")
    f.write("       work->WtW[i] = 0.0;\n")
    f.write("   }\n\n")

    # Initialize Ruiz data if Ruiz is disabled, since downstream functions use Druiz etc. so they should be set to all ones to prevent issues.
    if not generate_ruiz:
        f.write("   for (int i = 0; i < work->n; ++i) {\n")
        f.write("       work->Druiz[i] = 1.0;\n")
        f.write("       work->Dinvruiz[i] = 1.0;\n")
        f.write("   }\n")
        f.write("   for (int i = 0; i < work->p; ++i) {\n")
        f.write("       work->Eruiz[i] = 1.0;\n")
        f.write("       work->Einvruiz[i] = 1.0;\n")
        f.write("   }\n")
        f.write("   for (int i = 0; i < work->m; ++i) {\n")
        f.write("       work->Fruiz[i] = 1.0;\n")
        f.write("       work->Finvruiz[i] = 1.0;\n")
        f.write("   }\n")
        f.write("   work->k = 1.0;\n")
        f.write("   work->kinv = 1.0;\n\n")
    for i in range(m):
        f.write("   work->WtW[%i] = 1.0;\n" % Wsparse2dense[i * m + i])

    f.write("\n   // kkt_rhs = [-c;b;h].\n")
    f.write("   for(int i = 0; i < work->n; ++i) {\n")
    f.write("       work->kkt_rhs[i] = -work->c[i];\n")
    f.write("   }\n")
    f.write("   for(int i = 0; i < work->p; ++i) {\n")
    f.write("       work->kkt_rhs[work->n + i] = work->b[i];\n")
    f.write("   }\n")
    f.write("   for(int i = 0; i < work->m; ++i) {\n")
    f.write("       work->kkt_rhs[work->n + work->p + i] = work->h[i];\n")
    f.write("   }\n\n")
    f.write("   ldl(work);\n")
    f.write("   kkt_solve(work);\n")
    f.write("   qoco_custom_copy_arrayf(work->xyz, work->x, work->n);\n")
    f.write("   qoco_custom_copy_arrayf(&work->xyz[work->n], work->y, work->p);\n")
    f.write("   qoco_custom_copy_arrayf(&work->xyz[work->n + work->p], work->z, work->m);\n")
    f.write(
        "   copy_and_negate_arrayf(&work->xyz[work->n + work->p], work->s, work->m);\n"
    )
    f.write("   bring2cone(work->s, work->l, work->nsoc, work->q);\n")
    f.write("   bring2cone(work->z, work->l, work->nsoc, work->q);\n")
    f.write("}\n\n")

    f.write("void qoco_custom_solve(Workspace* work) {\n")
    f.write("   if (work->settings.verbose) {\n")
    f.write("       print_header(work);\n")
    f.write("   }\n")
    if generate_ruiz:
        f.write("   ruiz_equilibration(work);\n")
    f.write("   initialize_ipm(work);\n")
    f.write("   for (int i = 1; i < work->settings.max_iters; ++i) {\n")
    f.write("      compute_kkt_residual(work);\n")
    f.write("      compute_mu(work);\n")
    f.write("      if (check_stopping(work)) {\n")
    f.write("           unscale_solution(work);\n")
    if generate_ruiz:
        f.write("           unequilibrate_data(work);\n")
    f.write("           copy_solution(work);\n")
    f.write("           if (work->settings.verbose) {\n")
    f.write("               print_footer(work);\n")
    f.write("           }\n")
    f.write("         return;\n")
    f.write("      }\n")
    f.write("      compute_nt_scaling(work);\n")
    f.write("      compute_lambda(work);\n")
    f.write("      compute_WtW(work);\n")
    f.write("      predictor_corrector(work);\n")
    f.write("      work->sol.iters = i;\n")
    f.write("      if (work->settings.verbose) {\n")
    f.write("           log_iter(work);\n")
    f.write("      }\n")
    f.write("   }\n")
    if generate_ruiz:
        f.write("   unequilibrate_data(work);\n")
    f.write("   work->sol.status = QOCO_CUSTOM_MAX_ITER;\n")
    f.write("   if (work->settings.verbose) {\n")
    f.write("       print_footer(work);\n")
    f.write("   }\n")
    f.write("}\n\n")
    f.close()


def generate_runtest(solver_dir):
    f = open(solver_dir + "/runtest.c", "a")
    write_license(f)
    f.write("#include <stdio.h>\n")
    f.write("#ifndef IS_WINDOWS\n")
    f.write("#include <time.h>\n")
    f.write("#endif\n")
    f.write('#include "qoco_custom.h"\n\n')
    f.write("int main() {\n")
    f.write("   Workspace work;\n")
    f.write("   set_default_settings(&work);\n")
    f.write("   work.settings.verbose = 1;\n")
    f.write("   double N = 1000;\n")
    f.write("   double solve_time_sec = 1e10;\n")
    f.write("   for (int i = 0; i < N; ++i) {\n")
    f.write("#ifndef IS_WINDOWS\n")
    f.write("       struct timespec start, end;\n")
    f.write("       clock_gettime(CLOCK_MONOTONIC, &start);\n")
    f.write("#endif\n")
    f.write("       load_data(&work);\n")
    f.write("       qoco_custom_solve(&work);\n")
    f.write("#ifndef IS_WINDOWS\n")
    f.write("       clock_gettime(CLOCK_MONOTONIC, &end);\n")
    f.write(
        "       double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;\n"
    )
    f.write("       solve_time_sec = qoco_min(solve_time_sec, elapsed_time);\n")
    f.write("#endif\n")
    f.write("   }\n")
    f.write("#ifndef IS_WINDOWS\n")
    f.write('   printf("\\nSolvetime: %.9f ms", 1e3 * solve_time_sec);\n')
    f.write("#endif\n")
    f.write('   FILE *file = fopen("result.bin", "wb");\n')
    f.write("   fwrite(&work.sol.status, sizeof(unsigned char), 1, file);\n")
    f.write("   fwrite(&work.sol.obj, sizeof(double), 1, file);\n")
    f.write("   fwrite(&solve_time_sec, sizeof(double), 1, file);\n")
    f.write("   fclose(file);\n")
    f.write('   printf("\\nobj: %f", work.sol.obj);\n')

    f.write("}")

    f.close()
