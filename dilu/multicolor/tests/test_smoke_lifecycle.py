"""Smoke test: load .so, build plan, apply once, release. No correctness check.

This catches: library-not-found, FFI registration issues, kernel-launch
errors, and plan-buffer leaks at the coarsest level.
"""
import conftest  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp

from dilu.multicolor.python import MulticolorPlan
from _harness import laplacian_3d_7point


def test_smoke_lifecycle_7pt_small():
    nx, ny, nz = 4, 4, 4
    N = nx * ny * nz
    row_ptr, col_idx, values = laplacian_3d_7point(nx, ny, nz)

    rng = np.random.default_rng(0)
    r_h = rng.standard_normal(N).astype(np.float64)
    r_d = jax.device_put(jnp.asarray(r_h))
    values_d = jax.device_put(jnp.asarray(values))

    with MulticolorPlan(row_ptr, col_idx, values,
                        grid_shape=(nx, ny, nz)) as plan:
        print(f"plan: N={plan.N}, nnz={plan.nnz}, n_colors={plan.n_colors}, "
              f"color_offsets={plan.color_offsets.tolist()}")
        d_star = plan.factor(values_d)
        d_star.block_until_ready()
        d_star_h = np.asarray(d_star)
        print(f"d_star[:8] = {d_star_h[:8]}")
        assert np.all(np.isfinite(d_star_h)), "d_star has NaN/Inf"

        z = plan.apply(values_d, d_star, r_d)
        z.block_until_ready()
        z_h = np.asarray(z)
        print(f"z[:8] = {z_h[:8]}")
        assert np.all(np.isfinite(z_h)), "z has NaN/Inf"

    print("smoke OK")


if __name__ == "__main__":
    test_smoke_lifecycle_7pt_small()
