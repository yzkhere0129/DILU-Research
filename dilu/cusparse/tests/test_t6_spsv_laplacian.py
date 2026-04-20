"""T6: DILU apply on 3-D 7-point Laplacian. Compare to dense NumPy reference of M^{-1} r.

Grid capped at 8^3 = 512 per the §6 T6 brief (≤ 32^3); we use 8^3 for a
small, densifiable reference and 12^3 for a larger-but-still-safe sanity.
"""
import conftest  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp

from dilu.cusparse.python import dilu_factor, build_diag_offset, Plan
from _harness import laplacian_3d_7point, dilu_d_reference, dilu_apply_reference, tolerance


def _run_laplacian(nx, ny, nz, name):
    row_ptr, col_idx, values = laplacian_3d_7point(nx, ny, nz)
    n = nx * ny * nz
    diag_offset = build_diag_offset(row_ptr, col_idx)

    rng = np.random.default_rng(42)
    r_h = rng.standard_normal(n).astype(np.float64)

    row_ptr_d = jax.device_put(jnp.asarray(row_ptr))
    col_idx_d = jax.device_put(jnp.asarray(col_idx))
    values_d = jax.device_put(jnp.asarray(values))
    diag_offset_d = jax.device_put(jnp.asarray(diag_offset))
    r_d = jax.device_put(jnp.asarray(r_h))

    with Plan(row_ptr_d, col_idx_d, values_d, diag_offset_d) as plan:
        d_star = plan.factor(values_d)
        z = plan.apply(values_d, d_star, r_d)
        z.block_until_ready()
        z_gpu = np.asarray(z)

    # Reference: serial NumPy.
    d_ref = dilu_d_reference(row_ptr, col_idx, values)
    z_ref = dilu_apply_reference(row_ptr, col_idx, values, d_ref, r_h)

    abs_err = float(np.max(np.abs(z_gpu - z_ref)))
    # 3-D 7-point: max nnz/row = 7. n is the critical factor (triangular solve
    # accumulates along chain of length L_max ≈ nx+ny+nz).
    tol = tolerance(n, 7, z_ref, safety=500.0)
    print(f"{name}: n={n}, max|Δz|={abs_err:.3e}, tol={tol:.3e}, "
          f"||z_ref||_inf={np.max(np.abs(z_ref)):.3e}")
    assert abs_err <= tol, f"{name} failed: err {abs_err:.3e} > tol {tol:.3e}"


def test_t6_1_laplacian_8cubed():
    _run_laplacian(8, 8, 8, "T6.1 lap 8^3")


def test_t6_2_laplacian_12cubed():
    _run_laplacian(12, 12, 12, "T6.2 lap 12^3")


if __name__ == "__main__":
    test_t6_1_laplacian_8cubed()
    test_t6_2_laplacian_12cubed()
    print("T6 OK")
