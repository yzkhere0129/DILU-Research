"""T5: SpSV on a purely diagonal matrix. DILU reduces to Jacobi.

For A diagonal, L = U = 0, so M_DILU = D, D_* = D, and
  z = M^{-1} r = r / diag.

Failure here rules out cuSPARSE itself and isolates our scatter / fill-mode
configuration — see arch §4.3 for the reasoning (scatter-diag trap).
"""
import conftest  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp

from dilu.cusparse.python import (
    dilu_factor, cusparse_dilu_apply, build_diag_offset, Plan,
)


def test_t5_diagonal():
    n = 2000  # stays under the T5 ≤5k cap
    rng = np.random.default_rng(0)
    diag = rng.uniform(1.0, 100.0, size=n).astype(np.float64)
    r_h = rng.standard_normal(n).astype(np.float64)

    # Assemble A = diag(d) in CSR.
    row_ptr = np.arange(n + 1, dtype=np.int32)
    col_idx = np.arange(n, dtype=np.int32)
    values = diag.copy()
    diag_offset = build_diag_offset(row_ptr, col_idx)

    row_ptr_d = jax.device_put(jnp.asarray(row_ptr))
    col_idx_d = jax.device_put(jnp.asarray(col_idx))
    values_d = jax.device_put(jnp.asarray(values))
    diag_offset_d = jax.device_put(jnp.asarray(diag_offset))
    r_d = jax.device_put(jnp.asarray(r_h))

    # D_* = diag
    d_star = dilu_factor(row_ptr_d, col_idx_d, values_d, diag_offset_d)
    d_star.block_until_ready()
    d_star_h = np.asarray(d_star)
    assert np.max(np.abs(d_star_h - diag)) < 1e-12

    # Analyze + apply.
    with Plan(row_ptr_d, col_idx_d, values_d, diag_offset_d) as plan:
        z = plan.apply(values_d, d_star, r_d)
        z.block_until_ready()
        z_h = np.asarray(z)

    z_ref = r_h / diag
    err = float(np.max(np.abs(z_h - z_ref)))
    tol = 100.0 * np.finfo(np.float64).eps * max(float(np.max(np.abs(z_ref))), 1.0)
    print(f"T5 diagonal: n={n}, max|Δz|={err:.3e}, tol={tol:.3e}")
    assert err <= tol, f"T5 failed: err {err:.3e} > tol {tol:.3e}"


if __name__ == "__main__":
    test_t5_diagonal()
    print("T5 OK")
