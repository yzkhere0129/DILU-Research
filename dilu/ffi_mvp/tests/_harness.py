"""Shared test utilities: CSR assembly, tolerance check, matrix builders."""
from __future__ import annotations

import numpy as np


def dense_to_csr(A: np.ndarray):
    """Convert a 2-D dense ndarray to CSR triplets (int32, int32, float64).

    Zero entries are dropped. C-contiguous outputs (per CLAUDE.md).
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    n = A.shape[0]
    rows, cols = np.nonzero(A)
    # Sort by row, then col, to get canonical CSR ordering.
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


def laplacian_1d(n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A[i, i] = 2.0
        if i > 0:
            A[i, i - 1] = -1.0
        if i < n - 1:
            A[i, i + 1] = -1.0
    return A


def laplacian_3d_7point(nx: int, ny: int, nz: int) -> np.ndarray:
    """Build the 3-D 7-point Laplacian (Dirichlet, uniform h=1) on an nx*ny*nz grid.

    Diagonal = 6, off-diagonals to ±x/±y/±z neighbors = -1. Small n only;
    dense build is O(n^2) memory — keep n below a few thousand.
    """
    n = nx * ny * nz
    A = np.zeros((n, n), dtype=np.float64)

    def idx(i, j, k):
        return (k * ny + j) * nx + i

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = idx(i, j, k)
                A[p, p] = 6.0
                for di, dj, dk in ((-1, 0, 0), (1, 0, 0),
                                   (0, -1, 0), (0, 1, 0),
                                   (0, 0, -1), (0, 0, 1)):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                        A[p, idx(ii, jj, kk)] = -1.0
    return A


def diag_dominant_random(n: int, seed: int, diag_spread: float = 1e3) -> np.ndarray:
    """Strictly diagonally dominant matrix with diagonal spread `diag_spread`.

    diag[i] ~ U[1, diag_spread]. Off-diagonals sparse random in [-0.1, 0.1],
    then diagonal is inflated to enforce strict row-sum dominance. Useful for
    T3: catches dtype-precision bugs where small entries lose to large ones.
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=np.float64)
    # Sparsify off-diagonals: ~ 5 per row on average.
    density = min(5.0 / n, 1.0)
    mask = rng.random((n, n)) < density
    np.fill_diagonal(mask, False)
    offd = rng.uniform(-0.1, 0.1, size=(n, n))
    A = np.where(mask, offd, 0.0)
    # Symmetrize to be close to AM matrix structure.
    A = 0.5 * (A + A.T)
    # Diagonal drawn with wide spread; bump it above row-sum of |offd| to
    # guarantee M-matrix-like dominance.
    diag = rng.uniform(1.0, diag_spread, size=n)
    row_abs_sum = np.sum(np.abs(A), axis=1)
    diag = np.maximum(diag, row_abs_sum + 1.0)
    np.fill_diagonal(A, diag)
    return A


def tolerance_bound(nnz_per_row: int, y_ref: np.ndarray) -> float:
    """Design-doc §2.4 tolerance: 10 * nnz_per_row * eps * ||y_ref||_inf."""
    eps = np.finfo(np.float64).eps
    return 10.0 * nnz_per_row * eps * max(float(np.max(np.abs(y_ref))), 1.0)
