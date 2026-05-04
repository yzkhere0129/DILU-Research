"""3-way wall-time comparison on senior's pd matrices.

  Implementation               Hardware                      Algorithm
  ---------------------------- ----------------------------- ------------------
  1. OpenFOAM original         lab Xeon 28 phys cores N=32   PCG + DIC (native)
  2. Our openfoam_cpu replica  this machine (or lab 1 core)  PBiCG + DILU (port)
  3. AMGx                      this machine (RTX 3050)       PCG + classical AMG

(1) requires lab machine — emit a tar package + Allrun script.
(2) and (3) run here, output JSON.

NOTE: pd matrices are SPD (after sign flip). OpenFOAM uses PCG+DIC on them.
Our replica is PBiCG+DILU — it converges on SPD but takes ~2× iters of PCG.
We measure honestly and label the algorithm mismatch.
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

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
)
from dilu.amgx.bench.senior_data_loader import load_step, list_available
from dilu.amgx.bench.sweep_amgx_senior_data import normalize_sign

from dilu.openfoam_cpu.python.ldu import csr_to_ldu
from dilu.openfoam_cpu.python.pbicg import solve as our_pbicg_solve


def time_amgx_amortized(bundles, do_ir: bool = False):
    """AMGx with amortized Plan; return per-matrix dict list."""
    rows = []
    plan = None
    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, 1e-12, max_iters=500)
    for i, b in enumerate(bundles):
        A_raw, b_raw = b.A, b.b
        A, src, _ = normalize_sign(A_raw, b_raw)
        rp = jnp.asarray(A.indptr.astype(np.int32))
        ci = jnp.asarray(A.indices.astype(np.int32))
        vv = jnp.asarray(A.data.astype(np.float64))
        b_d = jnp.asarray(src); x0_d = jnp.zeros_like(b_d)
        b_norm = float(np.linalg.norm(src)); b_norm = b_norm or 1.0

        t_setup = t_update = t_solve = t_ir = 0.0
        if plan is None:
            t0 = time.time()
            plan = Plan(rp, ci, vv, cfg)
            t_setup = time.time() - t0
        else:
            t0 = time.time()
            plan.update_coefficients(vv)
            t_update = time.time() - t0

        t0 = time.time()
        x_jax, iters, status = plan.solve(b_d, x0_d)
        x_jax.block_until_ready()
        t_solve = time.time() - t0
        x = np.asarray(x_jax)

        if do_ir:
            r = A @ x - src
            r_d = jnp.asarray(-r); zero = jnp.zeros_like(r_d)
            t0 = time.time()
            d_jax, _, _ = plan.solve(r_d, zero)
            d_jax.block_until_ready()
            t_ir = time.time() - t0
            x = x + np.asarray(d_jax)

        rows.append(dict(step=b.step, corr=b.corr,
                         iters=int(iters[0]),
                         setup_ms=t_setup * 1e3,
                         update_ms=t_update * 1e3,
                         solve_ms=t_solve * 1e3,
                         ir_ms=t_ir * 1e3,
                         total_ms=(t_setup + t_update + t_solve + t_ir) * 1e3,
                         is_first=(i == 0),
                         x_first10=x[:10].tolist()))
    if plan is not None:
        plan.release()
    return rows


def time_our_replica(bundles):
    """Our PBiCG-DILU on each pd matrix. Note algorithm mismatch with OF (PCG-DIC)."""
    rows = []
    for b in bundles:
        A_raw, b_raw = b.A, b.b
        A, src, _ = normalize_sign(A_raw, b_raw)
        ldu = csr_to_ldu(A)
        x0 = np.zeros_like(src)
        t0 = time.time()
        perf = our_pbicg_solve(ldu, src, x0,
                               tolerance=1e-12, rel_tol=0.0,
                               min_iter=1, max_iter=2000)
        t_solve = time.time() - t0
        rows.append(dict(step=b.step, corr=b.corr,
                         iters=perf.n_iterations,
                         converged=perf.converged,
                         solve_ms=t_solve * 1e3,
                         total_ms=t_solve * 1e3,
                         x_first10=perf.x[:10].tolist()))
    return rows


def main():
    avail = list_available()
    print(f"Loading {len(avail)} senior pd bundles (80³=512K, nnz=3.55M)...")
    bundles = []
    for step, corr in avail:
        b = load_step(step, corr)
        if b is not None:
            bundles.append(b)
    print(f"  loaded {len(bundles)}")
    print()

    OF_BASELINE_MS = 44.1  # lab N=32 from scaling test, 500K cells
    OF_LABEL = "OpenFOAM PCG+DIC, lab Xeon Gold 5120 N=32 (28 phys + 4 SMT)"

    print("=" * 90)
    print("Method 2: Our openfoam_cpu replica (PBiCG+DILU, single-thread numpy/numba)")
    print("=" * 90)
    print("(WARNING: PBiCG+DILU != OpenFOAM PCG+DIC on pd; same converged x but ~2× iters)")
    rep = time_our_replica(bundles)
    rep_solve = np.median([r["solve_ms"] for r in rep])
    rep_iter = int(np.median([r["iters"] for r in rep]))
    n_conv = sum(1 for r in rep if r.get("converged"))
    print(f"  N={len(rep)}  median solve={rep_solve:.1f}ms  median iters={rep_iter}  "
          f"converged={n_conv}/{len(rep)}")
    print()

    print("=" * 90)
    print("Method 3: AMGx amortized (this machine GPU)")
    print("=" * 90)
    am = time_amgx_amortized(bundles, do_ir=False)
    am_med = np.median([r["total_ms"] for r in am if not r["is_first"]])
    am_iter = int(np.median([r["iters"] for r in am]))
    print(f"  N={len(am)}  median total (excl first)={am_med:.1f}ms  median iters={am_iter}")
    print()

    print("=" * 90)
    print("Method 3b: AMGx amortized + 1 IR (precision config)")
    print("=" * 90)
    am_ir = time_amgx_amortized(bundles, do_ir=True)
    am_ir_med = np.median([r["total_ms"] for r in am_ir if not r["is_first"]])
    print(f"  N={len(am_ir)}  median total (excl first)={am_ir_med:.1f}ms")
    print()

    print("=" * 90)
    print(f"3-WAY HEAD-TO-HEAD (per pd solve, 512K cells, senior data)")
    print("=" * 90)
    print(f"{'method':60s} {'wall (ms)':>11s} {'speedup vs OF':>14s}")
    print("-" * 90)
    print(f"{OF_LABEL:60s} {OF_BASELINE_MS:>11.1f} {'1.00× (ref)':>14s}")
    print(f"{'Our PBiCG+DILU replica (this machine, single-thread)':60s} "
          f"{rep_solve:>11.1f} {OF_BASELINE_MS/rep_solve:>14.3f}×")
    print(f"{'AMGx amortized (this machine GPU)':60s} "
          f"{am_med:>11.1f} {OF_BASELINE_MS/am_med:>14.3f}×")
    print(f"{'AMGx amortized + 1 IR (machine-precision config)':60s} "
          f"{am_ir_med:>11.1f} {OF_BASELINE_MS/am_ir_med:>14.3f}×")

    out = Path("/home/yzk/DILU-Research/dilu/amgx/bench/threeway_senior_results.json")
    out.write_text(json.dumps(dict(
        date=datetime.now().isoformat(),
        n_bundles=len(bundles),
        n_cells=int(bundles[0].n),
        nnz=int(bundles[0].A.nnz),
        of_baseline_ms=OF_BASELINE_MS,
        of_label=OF_LABEL,
        replica_rows=rep,
        amgx_rows=am,
        amgx_ir_rows=am_ir,
        summary=dict(
            of_baseline=OF_BASELINE_MS,
            replica_median=rep_solve,
            amgx_median=am_med,
            amgx_ir_median=am_ir_med,
        ),
    ), indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
