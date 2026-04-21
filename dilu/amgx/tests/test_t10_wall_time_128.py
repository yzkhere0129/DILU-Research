"""T10 — AMG-PCG total wall time on 128^3 stiff.

Acceptance (design §6.3):
    T_AMG_total < 0.5 * T_DILU_128 = 0.5 * 28.47 s = 14.24 s
(Phase 2 baseline: 28.47 s per phase3_scaling_64_128.md.)

`T_AMG_total` = setup time + solve time from the Python caller's point of
view. This is the realistic "what the user pays per solve on a fresh
pattern" number.

Shares its subprocess with T9 when run from pytest (same matrix construction
is expensive) but is left as a separate test file because of the one-test-
per-subprocess rule for VRAM accounting on the 4 GB card.
"""
import conftest  # noqa: F401

import time
import numpy as np
import jax
import jax.numpy as jnp

from dilu.amgx.python import Plan as AmgxPlan, CLASSICAL_V_CYCLE, with_tolerance
from dilu.amgx.tests._harness import stiff_laplacian_3d

DILU_BASELINE_128 = 28.47  # s — from docs/benchmark/phase3_scaling_64_128.md


def test_t10_wall_time_128cubed():
    nx = ny = nz = 128
    contrast = 100.0
    row_ptr, col_idx, values = stiff_laplacian_3d(nx, ny, nz, contrast)
    n = nx * ny * nz

    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))

    rng = np.random.default_rng(42)
    b_h = rng.standard_normal(n).astype(np.float64)
    b_d = jax.device_put(jnp.asarray(b_h))

    cfg = with_tolerance(CLASSICAL_V_CYCLE, tol=1e-10, max_iters=200)

    t0 = time.time()
    with AmgxPlan(rp, ci, vv, cfg) as plan:
        t_setup_end = time.time()
        setup_wall = t_setup_end - t0
        x_arr, iters_arr, status_arr = plan.solve(b_d)
        x_arr.block_until_ready()
        t_solve_end = time.time()
        solve_wall = t_solve_end - t_setup_end
        total_wall = t_solve_end - t0
        iters = int(iters_arr[0])
        status = int(status_arr[0])

    speedup = DILU_BASELINE_128 / total_wall
    print(f"T10: setup={setup_wall:.2f}s, solve={solve_wall:.2f}s, "
          f"total={total_wall:.2f}s, iters={iters}, status={status}")
    print(f"T10: DILU_baseline={DILU_BASELINE_128:.2f}s, "
          f"speedup={speedup:.2f}x, budget=14.24s")

    assert status == 0, f"status {status} (0=SUCCESS)"
    assert total_wall < 0.5 * DILU_BASELINE_128, (
        f"AMG total {total_wall:.2f}s >= 0.5 * DILU {DILU_BASELINE_128:.2f}s")
    print(f"T10 PASS: {total_wall:.2f}s < {0.5*DILU_BASELINE_128:.2f}s")


if __name__ == "__main__":
    test_t10_wall_time_128cubed()
