# Copyright (c) 2024, Govind M. Chari <govindchari1@gmail.com>
# This source code is licensed under the BSD 3-Clause License

import numpy as np
from scipy import sparse
from qocogen.codegen import _generate_solver
import time


def generate_solver(
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
    output_dir=".",
    name="qoco_custom",
    generate_ruiz=False,
):

    Psp = P.astype(np.float64) if P is not None else None
    Asp = A.astype(np.float64) if A is not None else None
    Gsp = G.astype(np.float64) if G is not None else None

    if P is not None:
        P = sparse.triu(P, format="csc").astype(np.float64)

    if c is not None:
        c = c.astype(np.float64)

    if A is not None:
        A = A.astype(np.float64)

    if b is not None:
        b = b.astype(np.float64)
    else:
        b = np.zeros((0), np.float64)

    if G is not None:
        G = G.astype(np.float64)

    if h is not None:
        h = h.astype(np.float64)
    else:
        h = np.zeros((0), np.float64)

    if q is not None:
        if not isinstance(q, np.ndarray):
            q = np.array(q)
        q = q.astype(np.int32)
    else:
        q = np.zeros((0), np.int32)

    start_time = time.time()
    _generate_solver(
        n,
        m,
        p,
        Psp,
        c,
        Asp,
        b,
        Gsp,
        h,
        l,
        nsoc,
        q,
        output_dir,
        name,
        generate_ruiz,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    mins, secs = divmod(elapsed_time, 60)
    if mins > 0:
        formatted_time = f"{int(mins):03} mins {secs:06.3f} secs"
    else:
        formatted_time = f"{secs:06.3f} secs"
    print(f"\nQOCOGEN Generation Time: {formatted_time}")
