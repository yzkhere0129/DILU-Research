"""Shared test utilities for Phase 3.

Keeps the Phase 2 `_harness` idioms (CSR builders, reference DILU) and adds:
  - `dilu_d_reference_permuted`: DILU factorization on the permuted matrix
    (T4-equiv reference, §5.1 math doc).
  - `dilu_apply_reference_permuted`: full M⁻¹ r on the permuted matrix
    (T6-equiv reference, §5.1).
  - `stiff_laplacian_3d`: copied verbatim from Phase 2 T7 so Phase 3 uses
    the exact same matrix (no sneaky drift).

We do NOT import from `dilu.cusparse.tests._harness` — the brief says Phase 3
is lexically separate.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Small CSR builders.
# ---------------------------------------------------------------------------
def dense_to_csr(A: np.ndarray):
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
    """3-D 7-point Dirichlet Laplacian CSR (ascending col_idx within rows)."""
    row_ptr_list = [0]
    col_idx_list = []
    values_list = []

    def idx(i, j, k):
        return (k * ny + j) * nx + i

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = idx(i, j, k)
                entries = [(p, 6.0)]
                for di, dj, dk in ((-1, 0, 0), (1, 0, 0),
                                   (0, -1, 0), (0, 1, 0),
                                   (0, 0, -1), (0, 0, 1)):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                        entries.append((idx(ii, jj, kk), -1.0))
                entries.sort(key=lambda e: e[0])
                for c, v in entries:
                    col_idx_list.append(c)
                    values_list.append(v)
                row_ptr_list.append(len(col_idx_list))
    return (
        np.ascontiguousarray(np.asarray(row_ptr_list, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(col_idx_list, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(values_list, dtype=np.float64)),
    )


def stiff_laplacian_3d(nx: int, ny: int, nz: int, contrast: float):
    """3-D 7-point Laplacian with coefficient jump along z-axis midplane.

    Verbatim copy of `_stiff_laplacian_3d` in
    `dilu/cusparse/tests/test_t7_pcg_stiff.py` so Phase 3 T7-equiv uses the
    EXACT same matrix (§5.2 math doc).
    """
    kcoef = np.ones(nz, dtype=np.float64)
    kcoef[nz // 2:] = contrast

    row_ptr_list = [0]
    col_idx_list = []
    values_list = []

    def idx(i, j, k):
        return (k * ny + j) * nx + i

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = idx(i, j, k)
                diag = 0.0
                offdiags = []
                for di, dj, dk in ((-1, 0, 0), (1, 0, 0),
                                   (0, -1, 0), (0, 1, 0),
                                   (0, 0, -1), (0, 0, 1)):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if not (0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz):
                        # Dirichlet ghost: coef = local k.
                        diag += kcoef[k]
                        continue
                    fk = 2.0 * kcoef[k] * kcoef[kk] / (kcoef[k] + kcoef[kk])
                    diag += fk
                    offdiags.append((idx(ii, jj, kk), -fk))
                entries = [(p, diag)] + offdiags
                entries.sort(key=lambda e: e[0])
                for c, v in entries:
                    col_idx_list.append(c)
                    values_list.append(v)
                row_ptr_list.append(len(col_idx_list))
    return (
        np.ascontiguousarray(np.asarray(row_ptr_list, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(col_idx_list, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(values_list, dtype=np.float64)),
    )


# ---------------------------------------------------------------------------
# Serial DILU reference on a CSR (any ordering — called with permuted Ã).
# ---------------------------------------------------------------------------
def dilu_d_reference(row_ptr: np.ndarray, col_idx: np.ndarray,
                     values: np.ndarray) -> np.ndarray:
    """Serial NumPy reference for D_* on the CSR as-ordered (math §1.2 eq 1.3).

    When `(row_ptr, col_idx, values)` is the PERMUTED matrix Ã, the output is
    the multi-color D̃_* referenced by (1.7).
    """
    n = int(row_ptr.shape[0]) - 1
    d = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a_ii = None
        for k_p in range(int(row_ptr[i]), int(row_ptr[i + 1])):
            if int(col_idx[k_p]) == i:
                a_ii = float(values[k_p]); break
        if a_ii is None:
            raise ValueError(f"row {i} missing diagonal")
        di = a_ii
        for k_p in range(int(row_ptr[i]), int(row_ptr[i + 1])):
            k_col = int(col_idx[k_p])
            if k_col >= i:
                continue
            a_ik = float(values[k_p])
            a_ki = 0.0
            for q in range(int(row_ptr[k_col]), int(row_ptr[k_col + 1])):
                if int(col_idx[q]) == i:
                    a_ki = float(values[q]); break
            if a_ki == 0.0:
                continue
            di -= (a_ik * a_ki) / d[k_col]
        d[i] = di
    return d


def dilu_apply_reference(row_ptr: np.ndarray, col_idx: np.ndarray,
                         values: np.ndarray, d_star: np.ndarray,
                         r: np.ndarray) -> np.ndarray:
    """Serial NumPy reference for z = M_DILU^{-1} r on the CSR as-ordered.

    For Phase 3, call with the PERMUTED (Ã, D̃_*, r̃) to get ẑ; then
    inverse-permute ẑ to original-index space for comparison. This is exactly
    the §5.1 T6-equiv contract.
    """
    n = int(row_ptr.shape[0]) - 1
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        acc = 0.0
        for p in range(int(row_ptr[i]), int(row_ptr[i + 1])):
            c = int(col_idx[p])
            if c >= i:
                continue
            acc += float(values[p]) * y[c]
        y[i] = (float(r[i]) - acc) / float(d_star[i])

    rhs = d_star * y

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
    eps = np.finfo(np.float64).eps
    return safety * n * nnz_per_row * eps * max(float(np.max(np.abs(y_ref))), 1.0)
