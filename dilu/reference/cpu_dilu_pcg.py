"""CPU Traditional DILU-PCG reference for `docs/benchmark/CANONICAL_CASE.md`.

Single-threaded scipy/numpy driver that reproduces the Phase 2 T7 stiff
7-point Poisson at 128^3 and runs vanilla left-to-right DILU-PCG to tol=1e-10.
Produces the THIRD data point for the CANONICAL 128^3 table alongside
cuSPARSE-DILU (Phase 2) and multicolor-DILU (Phase 3). Iter count must
equal 186 -- DILU math is identical to Phase 2, only serial execution differs.

Hard rules (task brief): OMP/MKL threads=1 before scipy import; matrix bit-
identical to dilu/amgx/tests/_harness.py::stiff_laplacian_3d; RHS seed=0;
vanilla serial DILU factor (Phase 2 T4 recurrence); float64; no jax, no GPU.
"""
# ---- Thread pinning (must precede numpy/scipy import) --------------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular


def build_stiff_laplacian_3d(n: int, contrast: float = 100.0):
    """Vectorized build of Phase 2 T7 stiff 3-D 7-point Laplacian (n^3).
    Bit-identical to `dilu/amgx/tests/_harness.py::stiff_laplacian_3d`:
    natural-order rows, ascending cols, harmonic-mean face coeffs with
    k(z)=1 for z<n/2 else contrast, Dirichlet ghost adds +kcoef[k] to diag.
    """
    nx = ny = nz = n
    N = nx * ny * nz
    kcoef = np.ones(nz, dtype=np.float64); kcoef[nz // 2:] = contrast

    I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                          indexing="ij")
    # Transpose so C-order ravel gives (k*ny+j)*nx+i = natural order.
    I = np.transpose(I, (2, 1, 0)); J = np.transpose(J, (2, 1, 0))
    K = np.transpose(K, (2, 1, 0))
    flat = lambda Ii, Jj, Kk: (Kk * ny + Jj) * nx + Ii

    # accum_order matches reference _harness loop for bit-identical diag FP;
    # dir_order yields already-ascending column indices per row.
    accum_order = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),
                   (0, 0, -1), (0, 0, 1)]
    dir_order = [(0, 0, -1), (0, -1, 0), (-1, 0, 0),
                 (1, 0, 0), (0, 1, 0), (0, 0, 1)]

    recs = {}
    diag = np.zeros((nz, ny, nx), dtype=np.float64)
    for d in accum_order:
        di, dj, dk = d
        I2, J2, K2 = I + di, J + dj, K + dk
        mask = ((I2 >= 0) & (I2 < nx) & (J2 >= 0) & (J2 < ny)
                & (K2 >= 0) & (K2 < nz))
        Kc, K2c = np.clip(K, 0, nz - 1), np.clip(K2, 0, nz - 1)
        fk = 2.0 * kcoef[Kc] * kcoef[K2c] / (kcoef[Kc] + kcoef[K2c])
        col = flat(np.clip(I2, 0, nx - 1), np.clip(J2, 0, ny - 1), K2c)
        # Per-neighbour diag accumulation: fk (interior) or kcoef[k] (ghost).
        diag += np.where(mask, fk, kcoef[K])
        recs[d] = (mask.ravel(order="C"), col.ravel(order="C"),
                   (-fk).ravel(order="C"))

    # CSR assembly.
    count = np.ones(N, dtype=np.int64)
    for d in dir_order:
        count += recs[d][0].astype(np.int64)
    row_ptr = np.zeros(N + 1, dtype=np.int32)
    np.cumsum(count, out=row_ptr[1:])
    nnz = int(row_ptr[-1])
    col_idx = np.empty(nnz, dtype=np.int32)
    values = np.empty(nnz, dtype=np.float64)
    cursor = row_ptr[:-1].copy()
    diag_flat = diag.ravel(order="C")
    p_self = np.arange(N, dtype=np.int32)
    for d in dir_order[:3]:  # col < self
        m, c, v = recs[d]
        idx = np.nonzero(m)[0]
        pos = cursor[idx]; cursor[idx] = pos + 1
        col_idx[pos] = c[idx]; values[pos] = v[idx]
    col_idx[cursor] = p_self; values[cursor] = diag_flat
    cursor = cursor + 1
    for d in dir_order[3:]:  # col > self
        m, c, v = recs[d]
        idx = np.nonzero(m)[0]
        pos = cursor[idx]; cursor[idx] = pos + 1
        col_idx[pos] = c[idx]; values[pos] = v[idx]

    return (np.ascontiguousarray(row_ptr),
            np.ascontiguousarray(col_idx),
            np.ascontiguousarray(values))


def dilu_factor(row_ptr, col_idx, values):
    """Serial left-to-right DILU factor: D_*[i] = A_ii - sum_{j<i} A_ij*(1/D_*[j])*A_ji.
    Python loop; at 128^3 takes ~12 s, acceptable because it's outside PCG.
    """
    n = int(row_ptr.shape[0]) - 1
    d_star = np.empty(n, dtype=np.float64)
    for i in range(n):
        start, end = int(row_ptr[i]), int(row_ptr[i + 1])
        acc = 0.0; aii = 0.0
        for p in range(start, end):
            j = int(col_idx[p])
            if j == i:
                aii = float(values[p])
            elif j < i:
                rs, re = int(row_ptr[j]), int(row_ptr[j + 1])
                lo, hi = rs, re  # binary search: cols ascending
                while lo < hi:
                    mid = (lo + hi) // 2
                    if int(col_idx[mid]) < i: lo = mid + 1
                    else: hi = mid
                if lo < re and int(col_idx[lo]) == i:
                    acc += float(values[p]) * float(values[lo]) / d_star[j]
        d_star[i] = aii - acc
    return d_star


def dilu_apply_setup(A_csr, d_star):
    """Build L_tri = strictly-lower(A) + diag(D_*) and U_tri similarly."""
    L = sp.tril(A_csr, k=-1, format="csr")
    U = sp.triu(A_csr, k=+1, format="csr")
    D = sp.diags(d_star, format="csr")
    return (L + D).tocsr(), (U + D).tocsr()


def dilu_apply(L_tri, U_tri, d_star, r):
    """z = M^{-1} r with M = (D_*+L) D_*^{-1} (D_*+U)."""
    y = spsolve_triangular(L_tri, r, lower=True, overwrite_b=False)
    return spsolve_triangular(U_tri, d_star * y, lower=False, overwrite_b=True)


def pcg(A_csr, b, apply_M_inv, tol=1e-10, max_iter=500):
    """PCG, timing the entire loop body from iter-0 to convergence."""
    n = b.shape[0]
    t0 = time.perf_counter()
    x = np.zeros(n, dtype=np.float64)
    r = b - A_csr.dot(x)
    b_norm = max(float(np.linalg.norm(b)), 1e-300)
    z = apply_M_inv(r); p = z.copy()
    rz = float(np.dot(r, z))
    for it in range(max_iter):
        rnorm = float(np.linalg.norm(r))
        if rnorm / b_norm < tol:
            return x, it, True, rnorm, time.perf_counter() - t0
        Ap = A_csr.dot(p)
        alpha = rz / float(np.dot(p, Ap))
        x += alpha * p; r -= alpha * Ap
        z = apply_M_inv(r)
        rz_new = float(np.dot(r, z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, max_iter, False, float(np.linalg.norm(r)), time.perf_counter() - t0


def main():
    N = 128
    print(f"numpy {np.__version__}  scipy {scipy.__version__}  "
          f"OMP={os.environ.get('OMP_NUM_THREADS')}  "
          f"MKL={os.environ.get('MKL_NUM_THREADS')}")
    print(f"[build] stiff Laplacian {N}^3 ...")
    t = time.perf_counter()
    row_ptr, col_idx, values = build_stiff_laplacian_3d(N, contrast=100.0)
    print(f"[build] done in {time.perf_counter()-t:.2f} s, nnz={values.size}")
    n = N ** 3
    A = sp.csr_matrix((values, col_idx, row_ptr), shape=(n, n))
    print("[factor] computing D_* (serial recurrence) ...")
    t = time.perf_counter()
    d_star = dilu_factor(row_ptr, col_idx, values)
    t_factor = time.perf_counter() - t
    print(f"[factor] D_* in {t_factor:.3f} s  min={d_star.min():.3e}  "
          f"max={d_star.max():.3e}")
    L_tri, U_tri = dilu_apply_setup(A, d_star)
    apply_fn = lambda r: dilu_apply(L_tri, U_tri, d_star, r)
    rng = np.random.default_rng(0)
    b = np.ascontiguousarray(rng.standard_normal(n).astype(np.float64))
    t = time.perf_counter(); _ = apply_fn(b)
    t_first_apply = time.perf_counter() - t
    print(f"[apply] first-call wall = {t_first_apply*1e3:.2f} ms")
    print("[pcg] running DILU-PCG, tol=1e-10, max_iter=500 ...")
    x, iters, conv, rnorm, t_pcg = pcg(A, b, apply_fn, tol=1e-10, max_iter=500)
    rel = rnorm / float(np.linalg.norm(b))
    print(f"[pcg] iters={iters}  conv={conv}  rel_res={rel:.3e}  "
          f"wall={t_pcg:.3f} s")
    print(f"\n== Summary ==\n  iters                 {iters}\n"
          f"  factor wall (s)       {t_factor:.3f}\n"
          f"  first apply wall (ms) {t_first_apply*1e3:.2f}\n"
          f"  PCG wall (s)          {t_pcg:.3f}\n"
          f"  final rel residual    {rel:.3e}")
    if iters != 186:
        print(f"[WARN] iters={iters} != 186; matrix has drifted from "
              f"Phase 2 T7 -- halt and investigate.")


if __name__ == "__main__":
    main()
