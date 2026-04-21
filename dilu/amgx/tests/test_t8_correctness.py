"""T8 — AMG-PCG vs DILU-PCG correctness on 16^3 stiff (contrast 100).

Acceptance (design §6.1):
    |x_AMG - x_DILU|_inf / |x_DILU|_inf <= 1e-8

Phase 2 DILU-PCG is treated as the reference. Both solvers converge the same
linear system Ax=b (SPD) to tol 1e-10; different preconditioners reach the
same unique solution within float64 noise.
"""
import conftest  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp

from dilu.amgx.python import Plan as AmgxPlan, CLASSICAL_V_CYCLE, with_tolerance
from dilu.amgx.tests._harness import stiff_laplacian_3d

# Phase 2 cuSPARSE DILU stack (reference).
from dilu.cusparse.python import Plan as CusparsePlan, build_diag_offset


def _spmv_gpu(row_ptr, col_idx, values, x):
    n = x.shape[0]
    nnz = values.shape[0]
    k = jnp.arange(nnz, dtype=jnp.int32)
    row_of = jnp.searchsorted(row_ptr[1:], k, side="right")
    prods = values * x[col_idx]
    return jax.ops.segment_sum(prods, row_of, num_segments=n)


def _pcg_dilu(plan, spmv, b, tol=1e-10, max_iter=1000):
    n = b.shape[0]
    x = jnp.zeros(n, dtype=jnp.float64)
    r = b - spmv(x)
    r.block_until_ready()
    b_norm = float(jnp.linalg.norm(b))
    d_star = plan.factor(plan._values)  # use seeded values
    z = plan.apply(plan._values, d_star, r)
    p = z
    rz = float(jnp.dot(r, z))
    for it in range(max_iter):
        rnorm = float(jnp.linalg.norm(r))
        if rnorm / (b_norm + 1e-300) < tol:
            return np.asarray(x), it, rnorm
        Ap = spmv(p)
        alpha = rz / float(jnp.dot(p, Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        z = plan.apply(plan._values, d_star, r)
        rz_new = float(jnp.dot(r, z))
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    return np.asarray(x), max_iter, float(jnp.linalg.norm(r))


def test_t8_correctness_16cubed():
    nx = ny = nz = 16
    contrast = 100.0
    row_ptr, col_idx, values = stiff_laplacian_3d(nx, ny, nz, contrast)
    n = nx * ny * nz

    rng = np.random.default_rng(42)
    b_h = rng.standard_normal(n).astype(np.float64)

    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    b_d = jax.device_put(jnp.asarray(b_h))

    # Phase 2 DILU-PCG reference.
    do_np = build_diag_offset(row_ptr, col_idx)
    do = jax.device_put(jnp.asarray(do_np))
    with CusparsePlan(rp, ci, vv, do) as dilu_plan:
        def spmv(x):
            return _spmv_gpu(rp, ci, vv, x)
        x_dilu, iter_dilu, res_dilu = _pcg_dilu(dilu_plan, spmv, b_d,
                                                 tol=1e-10, max_iter=2000)
    print(f"DILU-PCG: {iter_dilu} iters, res={res_dilu:.3e}, "
          f"||x||inf={np.max(np.abs(x_dilu)):.3e}")

    # Phase 4 AMG-PCG.
    cfg = with_tolerance(CLASSICAL_V_CYCLE, tol=1e-10, max_iters=200)
    with AmgxPlan(rp, ci, vv, cfg) as amgx_plan:
        x_amgx_arr, iters_arr, status_arr = amgx_plan.solve(b_d)
        x_amgx = np.asarray(x_amgx_arr)
        iter_amgx = int(iters_arr[0])
        status_amgx = int(status_arr[0])

    # AMGx-reported status: 0 SUCCESS, 1 FAILED, 2 DIVERGED, 3 NOT_CONVERGED.
    assert status_amgx == 0, f"AMGx status {status_amgx} (expected 0=SUCCESS)"
    print(f"AMG-PCG:  {iter_amgx} iters, status={status_amgx}, "
          f"||x||inf={np.max(np.abs(x_amgx)):.3e}")

    rel_err = np.max(np.abs(x_amgx - x_dilu)) / max(np.max(np.abs(x_dilu)), 1.0)
    print(f"T8: max|x_AMG - x_DILU|_inf / |x_DILU|_inf = {rel_err:.3e}")
    assert rel_err <= 1e-8, f"rel err {rel_err:.3e} > 1e-8"
    print("T8 PASS")


if __name__ == "__main__":
    test_t8_correctness_16cubed()
