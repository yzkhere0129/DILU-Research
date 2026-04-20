"""T6 (C3) — DILU-apply on 3-D Laplacian vs serial reference on Ã.

Per math doc §5.1, the reference for Phase 3 is serial DILU-apply on the
PERMUTED matrix (`_harness.dilu_apply_reference` fed Ã, D̃_*, r̃). Inverse-
permute ẑ back to original index space and compare to Plan.apply's z.

Grid capped at 12³ per Phase 2 T6 bounds (and the brief's §HARDWARE SAFETY
T6-equiv: grid ≤ 16³).
"""
import conftest  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp

from dilu.multicolor.python import MulticolorPlan
from dilu.multicolor.python.coloring import red_black_color
from dilu.multicolor.python.permute import permute_csr
from _harness import (
    laplacian_3d_7point, dilu_d_reference, dilu_apply_reference, tolerance,
)


def _run(nx, ny, nz, name):
    row_ptr, col_idx, values = laplacian_3d_7point(nx, ny, nz)
    N = nx * ny * nz

    rng = np.random.default_rng(42)
    r_h = rng.standard_normal(N).astype(np.float64)

    # Host reference on the permuted matrix.
    perm, iperm, _, _ = red_black_color(nx, ny, nz)
    rp_t, ci_t, vv_t, _ = permute_csr(row_ptr, col_idx, values, perm, iperm)
    d_tilde_ref = dilu_d_reference(rp_t, ci_t, vv_t)
    r_tilde = r_h[perm]
    z_tilde_ref = dilu_apply_reference(rp_t, ci_t, vv_t, d_tilde_ref, r_tilde)
    # Inverse permute: z[old] = z_tilde[iperm[old]].
    z_ref = z_tilde_ref[iperm]

    # GPU via Plan.
    r_d = jax.device_put(jnp.asarray(r_h))
    values_d = jax.device_put(jnp.asarray(values))
    with MulticolorPlan(row_ptr, col_idx, values,
                        grid_shape=(nx, ny, nz)) as plan:
        d_star = plan.factor(values_d)
        z = plan.apply(values_d, d_star, r_d)
        z.block_until_ready()
        z_gpu = np.asarray(z)

    abs_err = float(np.max(np.abs(z_gpu - z_ref)))
    # 7-point stencil: nnz_per_row ≤ 7. Safety 500 matches Phase 2 T6.
    tol = tolerance(N, 7, z_ref, safety=500.0)
    print(f"{name}: N={N}, max|Δz|={abs_err:.3e}, tol={tol:.3e}, "
          f"||z_ref||_inf={np.max(np.abs(z_ref)):.3e}")
    assert abs_err <= tol, f"{name} failed: err {abs_err:.3e} > tol {tol:.3e}"


def test_t6_1_lap_8cubed():
    _run(8, 8, 8, "T6.1 lap 8^3")


def test_t6_2_lap_12cubed():
    _run(12, 12, 12, "T6.2 lap 12^3")


def test_t6_3_lap_16cubed():
    _run(16, 16, 16, "T6.3 lap 16^3")


if __name__ == "__main__":
    test_t6_1_lap_8cubed()
    test_t6_2_lap_12cubed()
    test_t6_3_lap_16cubed()
    print("T6 OK")
