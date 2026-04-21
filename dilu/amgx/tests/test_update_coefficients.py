"""update_coefficients correctness — values-only refresh vs fresh setup.

Protocol:
  1. Build stiff Laplacian with contrast c1=10. setup + solve -> x1.
  2. Mutate values to contrast c2=100 (pattern unchanged).
     Call plan.update_coefficients(new_values), then plan.solve -> x2.
  3. Build a fresh plan (fresh setup) with c2=100 values. solve -> x_ref.
  4. |x2 - x_ref|_inf / |x_ref|_inf <= tol.

tol chosen at 1e-6 (looser than correctness T8's 1e-8) because AMG hierarchy
from c1 values is NOT the optimal hierarchy for c2; update_coefficients only
refreshes the SMOOTHER state, not the coarsening graph. Matches the
mathematical contract (math doc §5.4: refresh ~= 2-4x solve cost, collapse
in convergence rate if coefficients change too much).
"""
import conftest  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp

from dilu.amgx.python import Plan as AmgxPlan, CLASSICAL_V_CYCLE, with_tolerance
from dilu.amgx.tests._harness import stiff_laplacian_3d


def test_update_coefficients_16cubed():
    nx = ny = nz = 16
    # Two contrasts sharing the SAME sparsity pattern.
    rp1, ci1, v1 = stiff_laplacian_3d(nx, ny, nz, contrast=10.0)
    rp2, ci2, v2 = stiff_laplacian_3d(nx, ny, nz, contrast=100.0)
    assert np.array_equal(rp1, rp2), "pattern mismatch in row_ptr"
    assert np.array_equal(ci1, ci2), "pattern mismatch in col_idx"
    n = nx * ny * nz

    rng = np.random.default_rng(7)
    b_h = rng.standard_normal(n).astype(np.float64)

    rp = jax.device_put(jnp.asarray(rp1))
    ci = jax.device_put(jnp.asarray(ci1))
    vv1 = jax.device_put(jnp.asarray(v1))
    vv2 = jax.device_put(jnp.asarray(v2))
    b_d = jax.device_put(jnp.asarray(b_h))

    cfg = with_tolerance(CLASSICAL_V_CYCLE, tol=1e-10, max_iters=200)

    # (1) Setup on contrast=10 values.
    with AmgxPlan(rp, ci, vv1, cfg) as plan:
        x1_arr, iters1_arr, status1_arr = plan.solve(b_d)
        x1_arr.block_until_ready()
        print(f"step1 (contrast=10):  iters={int(iters1_arr[0])}, "
              f"status={int(status1_arr[0])}")

        # (2) Refresh coefficients to contrast=100 (pattern unchanged).
        plan.update_coefficients(vv2)
        x2_arr, iters2_arr, status2_arr = plan.solve(b_d)
        x2_arr.block_until_ready()
        x2 = np.asarray(x2_arr)
        print(f"step2 (refresh to 100): iters={int(iters2_arr[0])}, "
              f"status={int(status2_arr[0])}")
        assert int(status2_arr[0]) == 0, "post-refresh solve failed"

    # (3) Fresh setup on contrast=100.
    with AmgxPlan(rp, ci, vv2, cfg) as plan_ref:
        xref_arr, itersref_arr, statusref_arr = plan_ref.solve(b_d)
        xref_arr.block_until_ready()
        x_ref = np.asarray(xref_arr)
        print(f"step3 (fresh, contrast=100): iters={int(itersref_arr[0])}, "
              f"status={int(statusref_arr[0])}")

    rel_err = np.max(np.abs(x2 - x_ref)) / max(np.max(np.abs(x_ref)), 1.0)
    print(f"update_coefficients: |x2 - x_ref|_inf / |x_ref|_inf = {rel_err:.3e}")
    # Both reached tol 1e-10 convergence; solutions match the underlying
    # SPD system's unique solution (different Krylov paths, same root).
    assert rel_err <= 1e-6, f"update_coefficients drift {rel_err} > 1e-6"
    print("update_coefficients PASS")


if __name__ == "__main__":
    test_update_coefficients_16cubed()
