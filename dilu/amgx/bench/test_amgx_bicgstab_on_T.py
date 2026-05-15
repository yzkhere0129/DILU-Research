"""Quick test: AMGx BICGSTAB+AMG-DIAGSCALED on a single T matrix.

T_corr0 is non-symmetric (asymmetry rel ~1e-6 small but nonzero) so PCG
isn't appropriate. AMGx has CLASSICAL_V_DIAGSCALED_BICGSTAB pre-configured.

This script confirms:
  1. AMGx BICGSTAB can converge on T (not divergent)
  2. tol=1e-12 + 1 IR gives ε-machine truth (rel_resid < 1e-13)
  3. Wall time on 5060 (~few seconds expected)

If WORKS → use AMGx BICGSTAB @ 1e-12 + 1 IR as truth for T-phase, bypassing
SuperLU which is far too slow on Xeon (>20 min/matrix).

Usage on lab 5060:
  source ~/jax-pip-venv/bin/activate
  cd ~/DILU-Research
  PYTHONPATH=. python dilu/amgx/bench/test_amgx_bicgstab_on_T.py
"""
from __future__ import annotations
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import time
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from jax import config as _jc
_jc.update("jax_enable_x64", True)
import jax.numpy as jnp

from dilu.amgx.python import (
    Plan, CLASSICAL_V_DIAGSCALED_BICGSTAB, with_tolerance,
)


CASE = "/home/manyxu/cases/single_track_dump"
TS = "4.1e-07"  # mid-melt, asymmetry rel 6.7e-7


def main():
    p = f"{CASE}/postProcessing/matrices/{TS}/T_corr0/"
    print(f"# Loading T_corr0 matrix from {p}")
    t0 = time.time()
    A = sio.mmread(p + "A.mm").tocsr()
    b = np.asarray(sio.mmread(p + "b.mm")).flatten()
    x_OF = np.asarray(sio.mmread(p + "x_final.mm")).flatten()
    print(f"  load: {time.time()-t0:.1f}s  N={A.shape[0]}  nnz={A.nnz}")
    print(f"  diag mean: {A.diagonal().mean():.3e}")
    asym = sp.linalg.norm(A - A.T, "fro") / sp.linalg.norm(A, "fro")
    print(f"  asymmetry rel: {asym:.3e}")
    print(f"  x_OF range: [{x_OF.min():.1f}, {x_OF.max():.1f}] K")

    rp = jnp.asarray(A.indptr.astype(np.int32))
    ci = jnp.asarray(A.indices.astype(np.int32))
    vv = jnp.asarray(A.data.astype(np.float64))
    bd = jnp.asarray(b)

    # === Test 1: AMGx BICGSTAB @ tol=1e-8 (engineering accuracy) ===
    print("\n--- Test 1: AMGx BICGSTAB @ tol=1e-8 ---")
    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED_BICGSTAB, 1e-8, max_iters=2000)
    t0 = time.time()
    plan = Plan(rp, ci, vv, cfg)
    t_setup = time.time() - t0
    print(f"  setup: {t_setup*1000:.0f}ms")
    t0 = time.time()
    x_jax, it_d, st_d = plan.solve(bd, jnp.zeros_like(bd))
    x_jax.block_until_ready()
    t_solve = time.time() - t0
    iters = int(np.asarray(it_d)[0])
    status = int(np.asarray(st_d)[0])
    x = np.asarray(x_jax)
    rel_resid = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))
    diff_OF = np.abs(x - x_OF)
    print(f"  solve: {t_solve*1000:.0f}ms  iter={iters}  status={status}")
    print(f"  rel_resid (Ax-b): {rel_resid:.3e}")
    print(f"  max diff vs x_OF: {diff_OF.max():.3e} K  median: {np.median(diff_OF):.3e}")
    print(f"  Verdict @ 1e-8: {'CONVERGED ✓' if status == 0 else 'FAILED status='+str(status)}")
    plan.release()

    # === Test 2: AMGx BICGSTAB @ tol=1e-12 + 1 IR (truth-level) ===
    print("\n--- Test 2: AMGx BICGSTAB @ tol=1e-12 + 1 IR ---")
    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED_BICGSTAB, 1e-12, max_iters=3000)
    t0 = time.time()
    plan = Plan(rp, ci, vv, cfg)
    t_setup = time.time() - t0
    print(f"  setup: {t_setup*1000:.0f}ms")
    t0 = time.time()
    x_jax, it_d, st_d = plan.solve(bd, jnp.zeros_like(bd))
    x_jax.block_until_ready()
    t_solve = time.time() - t0
    iters_main = int(np.asarray(it_d)[0])
    status = int(np.asarray(st_d)[0])
    x = np.asarray(x_jax)
    # 1 IR step
    r = b - A @ x
    t0 = time.time()
    d_jax, _, _ = plan.solve(jnp.asarray(r), jnp.zeros_like(bd))
    d_jax.block_until_ready()
    t_ir = time.time() - t0
    x = x + np.asarray(d_jax)
    rel_resid = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))
    diff_OF = np.abs(x - x_OF)
    total_wall = t_setup + t_solve + t_ir
    print(f"  solve: {t_solve*1000:.0f}ms  iter={iters_main}  status={status}")
    print(f"  IR step: {t_ir*1000:.0f}ms")
    print(f"  TOTAL wall: {total_wall*1000:.0f}ms")
    print(f"  rel_resid (Ax-b) after IR: {rel_resid:.3e}")
    print(f"  max diff vs x_OF: {diff_OF.max():.3e} K  median: {np.median(diff_OF):.3e}")
    if rel_resid < 1e-12:
        print(f"  Verdict @ 1e-12+IR: TRUTH-LEVEL ✓ (use as T truth)")
    elif rel_resid < 1e-8:
        print(f"  Verdict @ 1e-12+IR: GOOD (rel_resid < 1e-8) but not ε-machine")
    else:
        print(f"  Verdict @ 1e-12+IR: NOT CONVERGED")
    plan.release()


if __name__ == "__main__":
    main()
