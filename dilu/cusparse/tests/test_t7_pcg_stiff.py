"""T7: DILU-PCG on a stiff 3-D problem. Assert DILU-PCG iterations < Jacobi-PCG.

Grid = 16^3 = 4096 (well under 32^3 cap).
Stiffness: diagonal contrast factor 100 between two halves (z < 8 vs z >= 8).
This is a "mini-AM" setup — the point is to compare DILU vs Jacobi iteration
counts, not to match a specific absolute number.
"""
import conftest  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp

from dilu.cusparse.python import build_diag_offset, Plan


def _stiff_laplacian_3d(nx, ny, nz, contrast):
    """3-D 7-point Laplacian with coefficient jump along z-axis midplane.

    k(z) = 1 for z < nz/2, k(z) = contrast for z >= nz/2.
    Face coefficient = harmonic mean. Diagonal = sum of face coefficients.
    """
    n = nx * ny * nz
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
                        # Dirichlet ghost: treat coeff as local k
                        face_coef = kcoef[k]
                        diag += face_coef
                        continue
                    # harmonic mean of kcoef[k] and kcoef[kk]
                    fk = 2.0 * kcoef[k] * kcoef[kk] / (kcoef[k] + kcoef[kk])
                    diag += fk
                    offdiags.append((idx(ii, jj, kk), -fk))
                # Sort by column for cuSPARSE.
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


def _spmv_gpu(row_ptr, col_idx, values, x):
    """Dense-style SpMV using jnp.segment_sum."""
    n = x.shape[0]
    nnz = values.shape[0]
    k = jnp.arange(nnz, dtype=jnp.int32)
    row_of = jnp.searchsorted(row_ptr[1:], k, side="right")
    prods = values * x[col_idx]
    return jax.ops.segment_sum(prods, row_of, num_segments=n)


def _pcg(apply_M_inv, spmv, b, tol=1e-8, max_iter=1000):
    """Plain PCG. Returns (x, niter, final_rnorm)."""
    n = b.shape[0]
    x = jnp.zeros(n, dtype=jnp.float64)
    r = b - spmv(x)
    r.block_until_ready()
    b_norm = float(jnp.linalg.norm(b))
    z = apply_M_inv(r)
    p = z
    rz = float(jnp.dot(r, z))
    for it in range(max_iter):
        rnorm = float(jnp.linalg.norm(r))
        if rnorm / (b_norm + 1e-300) < tol:
            return x, it, rnorm
        Ap = spmv(p)
        alpha = rz / float(jnp.dot(p, Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        z = apply_M_inv(r)
        rz_new = float(jnp.dot(r, z))
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    return x, max_iter, float(jnp.linalg.norm(r))


def test_t7_pcg_stiff_dilu_beats_jacobi():
    nx = ny = nz = 16
    contrast = 100.0
    row_ptr, col_idx, values = _stiff_laplacian_3d(nx, ny, nz, contrast)
    n = nx * ny * nz
    diag_offset = build_diag_offset(row_ptr, col_idx)

    rng = np.random.default_rng(0)
    b_h = rng.standard_normal(n).astype(np.float64)

    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    do = jax.device_put(jnp.asarray(diag_offset))
    b_d = jax.device_put(jnp.asarray(b_h))

    def spmv(x):
        return _spmv_gpu(rp, ci, vv, x)

    # Jacobi-PCG: M^{-1} r = r / diag
    diag_vec = jnp.asarray(values[diag_offset])
    def jacobi_apply(r):
        return r / diag_vec

    _, iter_jacobi, res_jacobi = _pcg(jacobi_apply, spmv, b_d,
                                       tol=1e-8, max_iter=2000)
    print(f"Jacobi-PCG: {iter_jacobi} iters, res={res_jacobi:.3e}")

    # DILU-PCG
    with Plan(rp, ci, vv, do) as plan:
        d_star = plan.factor(vv)
        d_star.block_until_ready()

        def dilu_apply(r):
            return plan.apply(vv, d_star, r)

        _, iter_dilu, res_dilu = _pcg(dilu_apply, spmv, b_d,
                                       tol=1e-8, max_iter=2000)
        print(f"DILU-PCG:   {iter_dilu} iters, res={res_dilu:.3e}")

    assert iter_dilu < iter_jacobi, (
        f"DILU must beat Jacobi: DILU {iter_dilu} vs Jacobi {iter_jacobi}")
    print(f"T7 OK: DILU iter {iter_dilu} < Jacobi iter {iter_jacobi}")


if __name__ == "__main__":
    test_t7_pcg_stiff_dilu_beats_jacobi()
