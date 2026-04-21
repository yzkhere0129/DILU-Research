"""Shared test utilities for Phase 4 — CSR builders, reference PCG.

Reuses the same 3-D Laplacian and stiff-Laplacian generators as the Phase 2
tests. We do NOT import from `dilu.cusparse.tests._harness` because that
module sets its own sys.path and conftest state; we inline the small helpers
here to keep the Phase 4 test tree self-contained.
"""
from __future__ import annotations

import numpy as np


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

    return (
        np.ascontiguousarray(np.asarray(row_ptr_list, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(col_idx_list, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(values_list, dtype=np.float64)),
    )


def stiff_laplacian_3d(nx: int, ny: int, nz: int, contrast: float):
    """3-D 7-point Laplacian with coefficient jump along the z midplane.

    Matches the T7 stiff pattern used by Phase 2 `test_t7_pcg_stiff.py`:
    k(z) = 1 for z < nz/2, k(z) = `contrast` for z >= nz/2. Face coefficient
    is the harmonic mean of the two adjacent cells' k; diagonal is the sum
    of incident-face coefficients (including Dirichlet ghost cells).
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
                        face_coef = kcoef[k]
                        diag += face_coef
                        continue
                    fk = 2.0 * kcoef[k] * kcoef[kk] / (kcoef[k] + kcoef[kk])
                    diag += fk
                    offdiags.append((idx(ii, jj, kk), -fk))
                row_entries = [(p, diag)] + offdiags
                row_entries.sort(key=lambda e: e[0])
                for c, v in row_entries:
                    col_idx_list.append(c)
                    values_list.append(v)
                row_ptr_list.append(len(col_idx_list))

    return (
        np.ascontiguousarray(np.asarray(row_ptr_list, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(col_idx_list, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(values_list, dtype=np.float64)),
    )


def csr_spmv_np(row_ptr, col_idx, values, x):
    """Serial CSR SpMV (NumPy). Used for reference residual computation."""
    n = int(row_ptr.shape[0]) - 1
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        acc = 0.0
        for p in range(int(row_ptr[i]), int(row_ptr[i + 1])):
            acc += float(values[p]) * float(x[int(col_idx[p])])
        y[i] = acc
    return y
