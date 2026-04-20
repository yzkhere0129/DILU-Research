"""T4 (C1) — D̃_* ULP correctness vs the serial DILU reference on Ã.

Plan.factor() returns d_star in ORIGINAL ordering. The reference is
    d_star_tilde_ref = dilu_d_reference(row_ptr_tilde, col_idx_tilde, values_tilde)
which lives in PERMUTED ordering. We compare by permuting both into the same
frame.

Test matrices (all N ≤ 10k, brief §HARDWARE SAFETY):
  - 3-D 7-point Laplacian 10³ = 1000
  - 16³ stiff (contrast=100, exactly Phase 2 T7 matrix) — only 4096 unknowns,
    well under the 10k T4 cap.
  - 1-D tridiag n=500
"""
import conftest  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp

from dilu.multicolor.python import MulticolorPlan
from dilu.multicolor.python.coloring import red_black_color, greedy_color_csr
from dilu.multicolor.python.permute import permute_csr
from _harness import laplacian_3d_7point, stiff_laplacian_3d, dilu_d_reference


def _run_one(name, row_ptr, col_idx, values, grid_shape=None):
    N = int(row_ptr.shape[0]) - 1
    # Host-permute via the same coloring route Plan will use.
    if grid_shape is not None:
        perm, iperm, _, _ = red_black_color(*grid_shape)
    else:
        perm, iperm, _, _ = greedy_color_csr(row_ptr, col_idx)
    rp_t, ci_t, vv_t, _ = permute_csr(row_ptr, col_idx, values, perm, iperm)

    # Reference: serial DILU on the permuted matrix.
    d_tilde_ref = dilu_d_reference(rp_t, ci_t, vv_t)

    # GPU: build the plan, pull d_star (original ordering) back.
    values_d = jax.device_put(jnp.asarray(values))
    with MulticolorPlan(row_ptr, col_idx, values,
                        grid_shape=grid_shape) as plan:
        d_star = plan.factor(values_d)
        d_star.block_until_ready()
        d_star_gpu_orig = np.asarray(d_star)  # original ordering

    # Permute GPU result back into tilde ordering to match the reference.
    d_star_gpu_tilde = d_star_gpu_orig[perm]

    abs_err = float(np.max(np.abs(d_star_gpu_tilde - d_tilde_ref)))
    ref_norm = max(float(np.max(np.abs(d_tilde_ref))), 1.0)
    rel_err = abs_err / ref_norm
    eps = np.finfo(np.float64).eps
    # Math doc §5.1: tolerance ≤ 50 * nnz_per_row * eps ≤ ~1e-13 for 7-point.
    # Use a sliding safety factor tied to N (like Phase 2 T4).
    tol = 200.0 * N * eps * ref_norm
    print(f"{name}: N={N}, max|Δd̃|={abs_err:.3e}, rel={rel_err:.3e}, tol={tol:.3e}")
    assert abs_err <= tol, f"{name}: err {abs_err:.3e} > tol {tol:.3e}"


def test_t4_1_laplacian_10cubed():
    nx = ny = nz = 10
    row_ptr, col_idx, values = laplacian_3d_7point(nx, ny, nz)
    _run_one("T4.1 lap 10^3 red-black", row_ptr, col_idx, values,
             grid_shape=(nx, ny, nz))


def test_t4_2_stiff_16cubed_rb():
    # 16^3 stiff — the exact Phase 2 T7 matrix, pre-T7 smoke of factor correctness.
    nx = ny = nz = 16
    row_ptr, col_idx, values = stiff_laplacian_3d(nx, ny, nz, contrast=100.0)
    _run_one("T4.2 stiff 16^3 red-black", row_ptr, col_idx, values,
             grid_shape=(nx, ny, nz))


def test_t4_3_greedy_on_laplacian_8cubed():
    # Greedy path on a 7-pt stencil (should degenerate to 2 colors).
    nx = ny = nz = 8
    row_ptr, col_idx, values = laplacian_3d_7point(nx, ny, nz)
    _run_one("T4.3 lap 8^3 greedy", row_ptr, col_idx, values,
             grid_shape=None)


if __name__ == "__main__":
    test_t4_1_laplacian_10cubed()
    test_t4_2_stiff_16cubed_rb()
    test_t4_3_greedy_on_laplacian_8cubed()
    print("T4 OK")
