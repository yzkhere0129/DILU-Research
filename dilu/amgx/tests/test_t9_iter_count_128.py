"""T9 — AMG-PCG iteration count on 128^3 stiff (contrast 100).

Acceptance (design §6.2):
    N_AMG <= 50 iterations to reach tol=1e-10 relative residual.
Phase 2 DILU-PCG needed 186 iters on the same problem (phase3_scaling_64_128).

This test is VRAM-tight (128^3 setup+solve peak ~2.5 GB on a 4 GB card).
If OOM, halts and reports partial.
"""
import conftest  # noqa: F401

import os
import time
import numpy as np
import jax
import jax.numpy as jnp

from dilu.amgx.python import Plan as AmgxPlan, CLASSICAL_V_CYCLE, with_tolerance
from dilu.amgx.tests._harness import stiff_laplacian_3d


def _nvidia_smi_mem():
    """Return (used MiB, free MiB) from nvidia-smi."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader,nounits"], text=True).strip()
        used_s, free_s = out.split("\n")[0].split(",")
        return int(used_s.strip()), int(free_s.strip())
    except Exception:
        return None, None


def test_t9_iter_count_128cubed():
    nx = ny = nz = 128
    contrast = 100.0
    used0, free0 = _nvidia_smi_mem()
    print(f"T9 pre: VRAM used={used0} MiB, free={free0} MiB")

    # Build matrix on host. 128^3 build takes ~30 s due to Python triple loop.
    t0 = time.time()
    row_ptr, col_idx, values = stiff_laplacian_3d(nx, ny, nz, contrast)
    n = nx * ny * nz
    print(f"T9 build_csr: n={n}, nnz={values.shape[0]}, build_wall={time.time()-t0:.1f}s")

    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))

    rng = np.random.default_rng(42)
    b_h = rng.standard_normal(n).astype(np.float64)
    b_d = jax.device_put(jnp.asarray(b_h))

    cfg = with_tolerance(CLASSICAL_V_CYCLE, tol=1e-10, max_iters=200)
    t_setup = time.time()
    with AmgxPlan(rp, ci, vv, cfg) as plan:
        used_post, free_post = _nvidia_smi_mem()
        print(f"T9 post-setup: VRAM used={used_post} MiB, free={free_post} MiB, "
              f"setup_wall={time.time()-t_setup:.2f}s")
        t_solve = time.time()
        x_arr, iters_arr, status_arr = plan.solve(b_d)
        x_arr.block_until_ready()
        solve_wall = time.time() - t_solve
        iters = int(iters_arr[0])
        status = int(status_arr[0])

    print(f"T9: iters={iters}, status={status}, solve_wall={solve_wall:.2f}s")
    assert status == 0, f"AMGx status {status} (0=SUCCESS)"
    assert iters <= 50, f"T9 iter count {iters} > 50 budget"
    print(f"T9 PASS: {iters} iters <= 50 (Phase 2 DILU required 186)")


if __name__ == "__main__":
    test_t9_iter_count_128cubed()
