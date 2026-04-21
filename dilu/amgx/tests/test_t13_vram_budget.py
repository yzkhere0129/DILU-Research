"""T13 — Peak VRAM on 128^3 setup+solve.

Acceptance (design §6.6):
    peak delta <= 2 GB per nvidia-smi (3050 has 3 GB headroom after desktop).
"""
import conftest  # noqa: F401

import subprocess
import time

import numpy as np
import jax
import jax.numpy as jnp

from dilu.amgx.python import (
    Plan as AmgxPlan, CLASSICAL_V_CYCLE, AGGRESSIVE_COARSENING, with_tolerance)
from dilu.amgx.tests._harness import stiff_laplacian_3d


def _vram_used_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], text=True).strip()
    return int(out.split("\n")[0].strip())


def _run_once(cfg_json, label):
    nx = ny = nz = 128
    contrast = 100.0
    row_ptr, col_idx, values = stiff_laplacian_3d(nx, ny, nz, contrast)
    n = nx * ny * nz

    used_pre = _vram_used_mib()
    print(f"T13 [{label}] pre: VRAM used={used_pre} MiB")

    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    rng = np.random.default_rng(42)
    b = jax.device_put(jnp.asarray(rng.standard_normal(n)))

    with AmgxPlan(rp, ci, vv, cfg_json) as plan:
        used_after_setup = _vram_used_mib()
        print(f"T13 [{label}] post-setup: VRAM used={used_after_setup} MiB, "
              f"delta={used_after_setup - used_pre} MiB")
        x, iters_arr, status_arr = plan.solve(b)
        x.block_until_ready()
        used_during = _vram_used_mib()
        iters = int(iters_arr[0])
        print(f"T13 [{label}] during-solve: VRAM used={used_during} MiB, "
              f"delta={used_during - used_pre} MiB, iters={iters}")

    peak_delta_mib = max(used_after_setup, used_during) - used_pre
    print(f"T13 [{label}] peak delta = {peak_delta_mib} MiB, iters = {iters}")
    return peak_delta_mib, iters


def test_t13_vram_budget_128cubed():
    """Measure VRAM at 128^3 for both CLASSICAL and AGGRESSIVE configs.

    Per design §11.2, AGGRESSIVE_COARSENING is the documented memory-lean
    fallback for 128^3 on a 4 GB card. CLASSICAL often exceeds the 2 GB
    target on our RTX 3050 Laptop due to D2 interpolator hierarchy cost;
    AGGRESSIVE (D1 + aggressive_levels=2) is what the design spec
    explicitly calls out as the in-budget path.
    """
    cfg_classical = with_tolerance(CLASSICAL_V_CYCLE, tol=1e-10, max_iters=200)
    cfg_aggr = with_tolerance(AGGRESSIVE_COARSENING, tol=1e-10, max_iters=200)

    # CLASSICAL first — expected to exceed 2 GB on this hardware.
    peak_cl, iters_cl = _run_once(cfg_classical, "CLASSICAL")
    # Must release VRAM between runs so the second run starts clean.
    import jax as _jax; _jax.clear_caches()
    import gc; gc.collect()

    peak_ag, iters_ag = _run_once(cfg_aggr, "AGGRESSIVE")

    print(f"\nT13 summary:")
    print(f"  CLASSICAL: peak {peak_cl} MiB, iters {iters_cl}")
    print(f"  AGGRESSIVE: peak {peak_ag} MiB, iters {iters_ag}")

    # Acceptance: at least ONE config meets the 2 GB budget at 128^3.
    # If neither does, we report the stretch situation per design §11.2.
    assert min(peak_cl, peak_ag) <= 2048, (
        f"Both configs exceed 2 GB budget: "
        f"CLASSICAL={peak_cl} MiB, AGGRESSIVE={peak_ag} MiB")
    if peak_cl <= 2048:
        print(f"T13 PASS: CLASSICAL fits in 2 GB ({peak_cl} MiB)")
    else:
        print(f"T13 PASS (stretch-budget): CLASSICAL={peak_cl} MiB > 2 GB, "
              f"but AGGRESSIVE={peak_ag} MiB <= 2 GB — use AGGRESSIVE at 128^3.")


if __name__ == "__main__":
    test_t13_vram_budget_128cubed()
