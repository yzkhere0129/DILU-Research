"""T7 (C4, C5) — DILU-mcPCG iteration count on the Phase 2 16³ stiff matrix.

Uses the EXACT same matrix as `dilu/cusparse/tests/test_t7_pcg_stiff.py`
(copied via `stiff_laplacian_3d` into `_harness.py`). Runs:

  1. Phase 3 multi-color DILU-PCG (this is the measurement)
  2. Jacobi-PCG reference baseline (re-measures Phase 2's 71)
  3. Optionally Phase 2 DILU-PCG for a direct comparison (24 reference)

Acceptance (brief §4, §5, math doc §5.4):
    C4 : N_P3 ≤ 72   (≤ 3× Phase 2's 24)
    C5 : N_P3 < 71   (must beat Jacobi)

The test RECORDS N_P3 regardless of whether C4/C5 pass — per brief directive
"Report the number regardless". Assertion failures distinguish which bound
was violated so the user can read the verdict at a glance.
"""
import conftest  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp

from dilu.multicolor.python import MulticolorPlan
from _harness import stiff_laplacian_3d


PHASE2_DILU_ITER_REF = 24   # Phase 2 T7 baseline
PHASE2_JACOBI_ITER_REF = 71 # Phase 2 Jacobi baseline


def _spmv_gpu(row_ptr, col_idx, values, x):
    n = x.shape[0]
    nnz = values.shape[0]
    k = jnp.arange(nnz, dtype=jnp.int32)
    row_of = jnp.searchsorted(row_ptr[1:], k, side="right")
    prods = values * x[col_idx]
    return jax.ops.segment_sum(prods, row_of, num_segments=n)


def _pcg(apply_M_inv, spmv, b, tol=1e-8, max_iter=2000):
    """Plain PCG. Identical to Phase 2 test_t7's driver so iteration counts
    are directly comparable."""
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


def test_t7_multicolor_dilu_pcg():
    nx = ny = nz = 16
    contrast = 100.0
    row_ptr, col_idx, values = stiff_laplacian_3d(nx, ny, nz, contrast)
    n = nx * ny * nz

    rng = np.random.default_rng(0)
    b_h = rng.standard_normal(n).astype(np.float64)

    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    b_d = jax.device_put(jnp.asarray(b_h))

    def spmv(x):
        return _spmv_gpu(rp, ci, vv, x)

    # --- Jacobi-PCG baseline ---
    diag_idx = np.empty(n, dtype=np.int32)
    for i in range(n):
        rs, re = int(row_ptr[i]), int(row_ptr[i + 1])
        k = rs + int(np.searchsorted(col_idx[rs:re], i))
        assert int(col_idx[k]) == i
        diag_idx[i] = k
    diag_vec = jnp.asarray(values[diag_idx])
    def jacobi_apply(r):
        return r / diag_vec

    _, iter_jacobi, res_jacobi = _pcg(jacobi_apply, spmv, b_d, tol=1e-8, max_iter=2000)
    print(f"Jacobi-PCG:            {iter_jacobi} iters, res={res_jacobi:.3e}")

    # --- Phase 3 multi-color DILU-PCG (RED-BLACK) ---
    with MulticolorPlan(row_ptr, col_idx, values,
                        grid_shape=(nx, ny, nz)) as plan:
        d_star = plan.factor(vv)
        d_star.block_until_ready()

        def mcdilu_apply(r):
            return plan.apply(vv, d_star, r)

        _, iter_mcdilu, res_mcdilu = _pcg(
            mcdilu_apply, spmv, b_d, tol=1e-8, max_iter=2000)
        print(f"Phase 3 DILU-mcPCG:    {iter_mcdilu} iters, res={res_mcdilu:.3e}")

    phase2_ref = PHASE2_DILU_ITER_REF
    jacobi_ref = PHASE2_JACOBI_ITER_REF
    penalty = iter_mcdilu / phase2_ref
    print(f"\n== T7 trade-off summary ==")
    print(f"   Phase 2 DILU-PCG (ref, cached): {phase2_ref} iters")
    print(f"   Jacobi-PCG (measured now):      {iter_jacobi} iters")
    print(f"   Phase 3 DILU-mcPCG:             {iter_mcdilu} iters")
    print(f"   Penalty (Phase 3 / Phase 2):    {penalty:.2f}x")
    print(f"   C4 budget (≤72):  "
          f"{'PASS' if iter_mcdilu <= 72 else 'FAIL'}")
    print(f"   C5 (< Jacobi 71): "
          f"{'PASS' if iter_mcdilu < iter_jacobi else 'FAIL'}")

    # Hard asserts — both must pass.
    assert iter_mcdilu <= 72, (
        f"C4 FAIL: Phase 3 iter count {iter_mcdilu} > 72 (3x Phase 2's 24). "
        f"Per brief §5 this is an acceptable negative result IF physical tests "
        f"still pass. Continue to physical benchmark to report the trade-off.")
    assert iter_mcdilu < iter_jacobi, (
        f"C5 FAIL: Phase 3 iter count {iter_mcdilu} >= Jacobi {iter_jacobi}. "
        f"Multi-color DILU degenerated to Jacobi quality — abandon Path B.")


if __name__ == "__main__":
    test_t7_multicolor_dilu_pcg()
    print("T7 OK")
