"""Shared test utilities for Phase 2 — CSR builders, DILU reference, SpSV reference."""
from __future__ import annotations

import numpy as np


def dense_to_csr(A: np.ndarray):
    """Dense → CSR triplet (int32, int32, float64)."""
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    n = A.shape[0]
    rows, cols = np.nonzero(A)
    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    vals = A[rows, cols].astype(np.float64, copy=True)
    col_idx = cols.astype(np.int32, copy=True)
    nnz = len(vals)
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    for r in rows:
        row_ptr[r + 1] += 1
    np.cumsum(row_ptr, out=row_ptr)
    assert row_ptr[-1] == nnz
    return (
        np.ascontiguousarray(row_ptr),
        np.ascontiguousarray(col_idx),
        np.ascontiguousarray(vals),
    )


def laplacian_3d_7point(nx: int, ny: int, nz: int):
    """Build the 3-D 7-point Dirichlet Laplacian CSR directly (no dense)."""
    n = nx * ny * nz
    row_ptr_list = [0]
    col_idx_list = []
    values_list = []

    def idx(i, j, k):
        return (k * ny + j) * nx + i

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = idx(i, j, k)
                # Collect (col, val) then sort by col for cuSPARSE-ascending order.
                row_entries = [(p, 6.0)]
                for di, dj, dk in ((-1, 0, 0), (1, 0, 0),
                                   (0, -1, 0), (0, 1, 0),
                                   (0, 0, -1), (0, 0, 1)):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                        row_entries.append((idx(ii, jj, kk), -1.0))
                row_entries.sort(key=lambda e: e[0])
                for c, v in row_entries:
                    col_idx_list.append(c)
                    values_list.append(v)
                row_ptr_list.append(len(col_idx_list))

    row_ptr = np.asarray(row_ptr_list, dtype=np.int32)
    col_idx = np.asarray(col_idx_list, dtype=np.int32)
    values = np.asarray(values_list, dtype=np.float64)
    return (
        np.ascontiguousarray(row_ptr),
        np.ascontiguousarray(col_idx),
        np.ascontiguousarray(values),
    )


def dilu_d_reference(row_ptr: np.ndarray, col_idx: np.ndarray,
                     values: np.ndarray) -> np.ndarray:
    """Serial NumPy reference for D_* computation (math doc §2.1)."""
    n = int(row_ptr.shape[0]) - 1
    d = np.zeros(n, dtype=np.float64)
    # Build quick (r, c) -> k lookup per row for finding a_{ki}.
    for i in range(n):
        # diagonal
        a_ii = None
        for k_p in range(int(row_ptr[i]), int(row_ptr[i + 1])):
            if int(col_idx[k_p]) == i:
                a_ii = float(values[k_p])
                break
        if a_ii is None:
            raise ValueError(f"row {i} missing diagonal")
        di = a_ii
        for k_p in range(int(row_ptr[i]), int(row_ptr[i + 1])):
            k_col = int(col_idx[k_p])
            if k_col >= i:
                continue
            a_ik = float(values[k_p])
            # look up a_{k_col, i} in row k_col
            a_ki = 0.0
            for q in range(int(row_ptr[k_col]), int(row_ptr[k_col + 1])):
                if int(col_idx[q]) == i:
                    a_ki = float(values[q])
                    break
            if a_ki == 0.0:
                continue
            di -= (a_ik * a_ki) / d[k_col]
        d[i] = di
    return d


def dilu_apply_reference(row_ptr: np.ndarray, col_idx: np.ndarray,
                         values: np.ndarray, d_star: np.ndarray,
                         r: np.ndarray) -> np.ndarray:
    """Serial NumPy reference for z = M_DILU^{-1} r.

    Matches the cuSPARSE path exactly (math doc §2.2): y = (D_*+L)^{-1} r,
    then z = (D_*+U)^{-1} (D_* * y).
    """
    n = int(row_ptr.shape[0]) - 1
    # Forward: y[i] = (r[i] - sum_{k<i} a_ik y[k]) / d_i
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        acc = 0.0
        for p in range(int(row_ptr[i]), int(row_ptr[i + 1])):
            c = int(col_idx[p])
            if c >= i:
                continue
            acc += float(values[p]) * y[c]
        y[i] = (float(r[i]) - acc) / float(d_star[i])

    # Middle: rhs[i] = d_i * y[i]
    rhs = d_star * y

    # Backward: z[i] = (rhs[i] - sum_{k>i} a_ik z[k]) / d_i
    z = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        acc = 0.0
        for p in range(int(row_ptr[i]), int(row_ptr[i + 1])):
            c = int(col_idx[p])
            if c <= i:
                continue
            acc += float(values[p]) * z[c]
        z[i] = (float(rhs[i]) - acc) / float(d_star[i])
    return z


def tolerance(n: int, nnz_per_row: int, y_ref: np.ndarray,
              safety: float = 200.0) -> float:
    """Phase 2 tolerance: safety * (n * nnz_per_row) * eps * ||y||_∞.

    More generous than Phase 1's kernel-only reduction tolerance because
    triangular solves have error that grows with n (backward substitution
    accumulates along the dependency chain).
    """
    eps = np.finfo(np.float64).eps
    return safety * n * nnz_per_row * eps * max(float(np.max(np.abs(y_ref))), 1.0)


def spmv(row_ptr, col_idx, values, x):
    """Serial CSR SpMV."""
    n = int(row_ptr.shape[0]) - 1
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        acc = 0.0
        for p in range(int(row_ptr[i]), int(row_ptr[i + 1])):
            acc += float(values[p]) * float(x[int(col_idx[p])])
        y[i] = acc
    return y
