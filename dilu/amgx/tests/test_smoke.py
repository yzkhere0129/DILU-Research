"""Smoke test: run amgx_setup + amgx_solve on a tiny 8^3 Laplacian.

Catches API-surface bugs before we hit the real T8-T13 acceptance suite.
Runs as the first Phase-4 end-to-end test; a failure here halts the rest.
"""
import conftest  # noqa: F401

import os
import numpy as np
import jax
import jax.numpy as jnp

from dilu.amgx.python import Plan, CLASSICAL_V_CYCLE
from dilu.amgx.tests._harness import laplacian_3d_7point, csr_spmv_np


def test_smoke_8cubed():
    nx = ny = nz = 8
    row_ptr, col_idx, values = laplacian_3d_7point(nx, ny, nz)
    n = nx * ny * nz

    rng = np.random.default_rng(0)
    b = rng.standard_normal(n).astype(np.float64)

    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    b_d = jax.device_put(jnp.asarray(b))
    x0_d = jax.device_put(jnp.zeros(n, dtype=jnp.float64))

    print(f"smoke: n={n}, nnz={values.shape[0]}")

    with Plan(rp, ci, vv, CLASSICAL_V_CYCLE) as plan:
        x, iters_arr, status_arr = plan.solve(b_d, x0_d)
        x.block_until_ready()
        x_host = np.asarray(x)
        iters = int(iters_arr[0])
        status = int(status_arr[0])

    # Residual check
    r = b - csr_spmv_np(row_ptr, col_idx, values, x_host)
    rnorm = float(np.linalg.norm(r))
    bnorm = float(np.linalg.norm(b))
    rel = rnorm / (bnorm + 1e-300)

    print(f"smoke: iters={iters}, status={status}, relres={rel:.3e}")
    assert status == 0, f"AMGx status={status} (0=SUCCESS)"
    assert rel <= 1e-8, f"relative residual {rel:.3e} > 1e-8"
    assert iters <= 30, f"iters={iters} unexpectedly high on trivial 8^3"
    print("smoke: PASS")


if __name__ == "__main__":
    test_smoke_8cubed()
