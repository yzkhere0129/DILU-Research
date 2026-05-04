"""Wall-time benchmark: AMGx vs OpenFOAM N=32 on similarly-sized pd matrices.

Reference baseline (from docs/benchmark/OPENFOAM_SCALING_20260503.md):
  OpenFOAM laserMeltFoam, spot_melt_150W (500K cells), Xeon Gold 5120 N=32 cores:
    pd_corr0 median wall = 44.1 ms (12.86× speedup vs N=1 = 567ms)

This script measures AMGx + 1 IR step on senior's 512K-cell pd matrices
(closest available match to OF's 500K) under 3 modes:

  1. Fresh setup per matrix     (worst case, every solve includes AMG hierarchy build)
  2. Amortized (single Plan)    (build hierarchy once, reuse for all 21 matrices)
  3. Amortized + IR             (the precision-grade config from AMGX_PRECISION_20260504.md)

Outputs per-matrix table + summary stats.
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import jax
from jax import config as _jc
_jc.update("jax_enable_x64", True)
import jax.numpy as jnp

from dilu.amgx.python import (
    Plan, CLASSICAL_V_DIAGSCALED, with_tolerance,
    amgx_solve_with_refinement,
)
from dilu.amgx.bench.senior_data_loader import load_step, list_available
from dilu.amgx.bench.sweep_amgx_senior_data import normalize_sign


OF_BASELINE_MS = 44.1   # from scaling test: N=32 lab 28-core, spot_melt_150W 500K
OF_BASELINE_NCELLS = 500_000
OF_BASELINE_NCORES = 32
OF_BASELINE_LABEL = "OpenFOAM PCG-DIC, 32 MPI ranks (28 phys + 4 SMT), Xeon Gold 5120 (lab)"


def bench_fresh(bundles):
    """Mode 1: rebuild Plan every matrix (includes setup)."""
    rows = []
    for b in bundles:
        A_raw, b_raw = b.A, b.b
        A, src, _ = normalize_sign(A_raw, b_raw)
        rp = jnp.asarray(A.indptr.astype(np.int32))
        ci = jnp.asarray(A.indices.astype(np.int32))
        vv = jnp.asarray(A.data.astype(np.float64))
        b_d = jnp.asarray(src)
        x0_d = jnp.zeros_like(b_d)
        cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, 1e-12, max_iters=500)

        t0 = time.time()
        with Plan(rp, ci, vv, cfg) as plan:
            t_setup = time.time() - t0
            t1 = time.time()
            x, iters, status = plan.solve(b_d, x0_d)
            x.block_until_ready()
            t_solve = time.time() - t1
        rows.append(dict(step=b.step, corr=b.corr,
                         iters=int(iters[0]), status=int(status[0]),
                         setup_ms=t_setup * 1e3, solve_ms=t_solve * 1e3,
                         total_ms=(t_setup + t_solve) * 1e3))
        gc.collect()
    return rows


def bench_amortized(bundles, with_ir=False):
    """Mode 2/3: build Plan once on first matrix, reuse for rest via update_coefficients."""
    rows = []
    plan = None
    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, 1e-12, max_iters=500)

    for i, b in enumerate(bundles):
        A_raw, b_raw = b.A, b.b
        A, src, _ = normalize_sign(A_raw, b_raw)
        rp = jnp.asarray(A.indptr.astype(np.int32))
        ci = jnp.asarray(A.indices.astype(np.int32))
        vv = jnp.asarray(A.data.astype(np.float64))
        b_d = jnp.asarray(src)
        x0_d = jnp.zeros_like(b_d)
        b_norm2 = float(np.linalg.norm(src))
        if b_norm2 == 0: b_norm2 = 1.0

        if plan is None:
            t0 = time.time()
            plan = Plan(rp, ci, vv, cfg)
            t_setup = time.time() - t0
            t_update = 0.0
        else:
            t_setup = 0.0
            t0 = time.time()
            plan.update_coefficients(vv)
            t_update = time.time() - t0

        t1 = time.time()
        x_jax, iters, status = plan.solve(b_d, x0_d)
        x_jax.block_until_ready()
        t_primary_solve = time.time() - t1
        x = np.asarray(x_jax)

        # Optional IR step
        t_ir = 0.0
        if with_ir:
            r = A @ x - src
            r_d = jnp.asarray(-r)
            zero = jnp.zeros_like(r_d)
            t2 = time.time()
            delta_jax, _, _ = plan.solve(r_d, zero)
            delta_jax.block_until_ready()
            t_ir = time.time() - t2
            x = x + np.asarray(delta_jax)

        rows.append(dict(step=b.step, corr=b.corr,
                         iters=int(iters[0]),
                         setup_ms=t_setup * 1e3,
                         update_ms=t_update * 1e3,
                         solve_ms=t_primary_solve * 1e3,
                         ir_ms=t_ir * 1e3,
                         total_ms=(t_setup + t_update + t_primary_solve + t_ir) * 1e3,
                         is_first=(i == 0)))

    if plan is not None:
        plan.release()
    return rows


def main():
    avail = list_available()
    print(f"Loading {len(avail)} senior bundles (80³=512K mesh, pd matrices)...")
    bundles = []
    for step, corr in avail:
        b = load_step(step, corr)
        if b is not None:
            bundles.append(b)
    print(f"Loaded {len(bundles)} bundles, N={bundles[0].n}, nnz={bundles[0].A.nnz}")
    print()

    print("=" * 80)
    print("Mode 1: FRESH setup per matrix (worst case)")
    print("=" * 80)
    fresh = bench_fresh(bundles)
    for r in fresh[:3]:
        print(f"  step{r['step']}/c{r['corr']}: setup={r['setup_ms']:.1f}ms  "
              f"solve={r['solve_ms']:.1f}ms  iter={r['iters']}  total={r['total_ms']:.1f}ms")
    print(f"  ...")
    setup_med = np.median([r["setup_ms"] for r in fresh])
    solve_med = np.median([r["solve_ms"] for r in fresh])
    total_med = np.median([r["total_ms"] for r in fresh])
    print(f"  median: setup={setup_med:.1f}ms  solve={solve_med:.1f}ms  total={total_med:.1f}ms")
    print()

    print("=" * 80)
    print("Mode 2: AMORTIZED (1 setup + N updates)")
    print("=" * 80)
    amort = bench_amortized(bundles, with_ir=False)
    print(f"  first  : setup={amort[0]['setup_ms']:.1f}ms  solve={amort[0]['solve_ms']:.1f}ms")
    print(f"  next 3 : ", end="")
    for r in amort[1:4]:
        print(f"upd={r['update_ms']:.1f}+solve={r['solve_ms']:.1f}={r['total_ms']:.1f}ms  ", end="")
    print()
    upd_med = np.median([r["update_ms"] for r in amort if not r["is_first"]])
    solve_med2 = np.median([r["solve_ms"] for r in amort if not r["is_first"]])
    total_med2 = np.median([r["total_ms"] for r in amort if not r["is_first"]])
    print(f"  median (excl first): update={upd_med:.1f}ms  solve={solve_med2:.1f}ms  "
          f"total={total_med2:.1f}ms")
    print()

    print("=" * 80)
    print("Mode 3: AMORTIZED + 1 IR step (precision config)")
    print("=" * 80)
    amort_ir = bench_amortized(bundles, with_ir=True)
    upd_med_ir = np.median([r["update_ms"] for r in amort_ir if not r["is_first"]])
    solve_med_ir = np.median([r["solve_ms"] for r in amort_ir if not r["is_first"]])
    ir_med = np.median([r["ir_ms"] for r in amort_ir if not r["is_first"]])
    total_med_ir = np.median([r["total_ms"] for r in amort_ir if not r["is_first"]])
    print(f"  median: update={upd_med_ir:.1f}ms  solve={solve_med_ir:.1f}ms  "
          f"ir={ir_med:.1f}ms  total={total_med_ir:.1f}ms")
    print()

    print("=" * 80)
    print("HEAD-TO-HEAD SUMMARY (per pd matrix solve, 512K cells)")
    print("=" * 80)
    print(f"{'config':50s} {'wall (ms)':>11s} {'vs OF N=32':>12s}")
    print("-" * 78)
    print(f"{OF_BASELINE_LABEL:50s} {OF_BASELINE_MS:>11.1f} {'1.00× (ref)':>12s}")
    print(f"{'AMGx fresh setup (per matrix)':50s} {total_med:>11.1f} "
          f"{f'{OF_BASELINE_MS/total_med:.2f}×':>12s}")
    print(f"{'AMGx amortized (no IR)':50s} {total_med2:>11.1f} "
          f"{f'{OF_BASELINE_MS/total_med2:.2f}×':>12s}")
    print(f"{'AMGx amortized + 1 IR (precision config)':50s} {total_med_ir:>11.1f} "
          f"{f'{OF_BASELINE_MS/total_med_ir:.2f}×':>12s}")

    out = Path("/home/yzk/DILU-Research/dilu/amgx/bench/bench_amgx_vs_of_n32_results.json")
    out.write_text(json.dumps(dict(
        date=datetime.now().isoformat(),
        of_baseline_ms=OF_BASELINE_MS,
        of_baseline_label=OF_BASELINE_LABEL,
        n_bundles=len(bundles),
        n_cells=int(bundles[0].n),
        nnz=int(bundles[0].A.nnz),
        mode_fresh=fresh,
        mode_amortized=amort,
        mode_amortized_ir=amort_ir,
    ), indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
