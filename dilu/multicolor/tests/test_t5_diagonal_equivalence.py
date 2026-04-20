"""T5 (C2) — On a purely diagonal A, DILU ≡ Jacobi.

With A = diag(d), no edges exist, so every coloring is trivially valid
(each node could be its own color; our greedy produces exactly 1 color since
nothing conflicts). DILU reduces to z = r / d — must be bit-identical to
Phase 2's Plan.apply on the same problem.
"""
import conftest  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp

from dilu.multicolor.python import MulticolorPlan


def test_t5_diagonal_reduces_to_jacobi():
    n = 2000
    rng = np.random.default_rng(0)
    diag = rng.uniform(1.0, 100.0, size=n).astype(np.float64)
    r_h = rng.standard_normal(n).astype(np.float64)

    # A = diag(d) in CSR: one entry per row, column = row.
    row_ptr = np.arange(n + 1, dtype=np.int32)
    col_idx = np.arange(n, dtype=np.int32)
    values = diag.copy()

    r_d = jax.device_put(jnp.asarray(r_h))
    values_d = jax.device_put(jnp.asarray(values))

    # No grid_shape → falls through to greedy_color_csr. With zero edges,
    # every vertex greedy-colors to 0, giving n_colors = 1.
    with MulticolorPlan(row_ptr, col_idx, values) as plan:
        print(f"plan on diag: N={plan.N}, n_colors={plan.n_colors}, "
              f"color_offsets={plan.color_offsets.tolist()}")
        assert plan.n_colors == 1, (
            f"expected 1 color on edge-free graph, got {plan.n_colors}")
        d_star = plan.factor(values_d)
        d_star.block_until_ready()
        d_star_h = np.asarray(d_star)
        # D̃_* should equal diag (on diag matrix, no subtractions in the recursion).
        assert np.max(np.abs(d_star_h - diag)) < 1e-12, \
            "d_star != diag on diagonal A"

        z = plan.apply(values_d, d_star, r_d)
        z.block_until_ready()
        z_h = np.asarray(z)

    z_ref = r_h / diag
    err = float(np.max(np.abs(z_h - z_ref)))
    tol = 100.0 * np.finfo(np.float64).eps * max(float(np.max(np.abs(z_ref))), 1.0)
    print(f"T5 diagonal: n={n}, max|Δz|={err:.3e}, tol={tol:.3e}")
    assert err <= tol, f"T5 failed: err {err:.3e} > tol {tol:.3e}"


if __name__ == "__main__":
    test_t5_diagonal_reduces_to_jacobi()
    print("T5 OK")
