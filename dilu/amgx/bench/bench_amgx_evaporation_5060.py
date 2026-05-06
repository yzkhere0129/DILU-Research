"""AMGx benchmark on senior's NEW Evaporation dataset (57 matrices, t=700ns–1.06μs).

Run on lab 5060 GPU. Measures:
  - Wall time per pd solve (3 modes: fresh / amortized / amortized+IR)
  - Precision: ||x_AMGx - x_xref||∞ / ||x_xref||∞ + ||A·x - b||₂/||b||₂
  - Per-step timing distribution → identify variability across evaporation phases

Output:
  /tmp/amgx_5060_evaporation_bench.{log,json}

Usage:
  cd ~/DILU-Research
  python -u -m dilu.amgx.bench.bench_amgx_evaporation_5060
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

from jax import config as _jc
_jc.update("jax_enable_x64", True)
import jax.numpy as jnp

from dilu.amgx.python import (
    Plan, CLASSICAL_V_DIAGSCALED, with_tolerance,
    amgx_solve_with_refinement,
)
from dilu.amgx.bench.senior_data_loader import (
    set_dataset, list_available, load_step,
)
from dilu.amgx.bench.sweep_amgx_senior_data import normalize_sign


def main():
    set_dataset("evaporation")
    avail = list_available()
    print(f"Loading {len(avail)} evaporation bundles ...", flush=True)
    bundles = []
    for s, c in avail:
        b = load_step(s, c)
        if b is not None:
            bundles.append(b)
    print(f"  loaded {len(bundles)} bundles, N={bundles[0].n}, nnz={bundles[0].A.nnz}",
          flush=True)
    print(f"  step range: {min(b.step for b in bundles)} to {max(b.step for b in bundles)}",
          flush=True)
    print()

    # ---- Mode 1: Fresh setup per matrix (worst case)
    print("=" * 78)
    print("Mode 1: AMGx FRESH setup per matrix")
    print("=" * 78)
    fresh_rows = []
    for b in bundles:
        A, src, _ = normalize_sign(b.A, b.b)
        rp = jnp.asarray(A.indptr.astype(np.int32))
        ci = jnp.asarray(A.indices.astype(np.int32))
        vv = jnp.asarray(A.data.astype(np.float64))
        b_d = jnp.asarray(src); x0_d = jnp.zeros_like(b_d)
        cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, 1e-12, max_iters=500)
        t0 = time.time()
        with Plan(rp, ci, vv, cfg) as plan:
            t_setup = time.time() - t0
            t1 = time.time()
            x, iters, status = plan.solve(b_d, x0_d)
            x.block_until_ready()
            t_solve = time.time() - t1
            x_h = np.asarray(x)
        fresh_rows.append(dict(
            step=b.step, corr=b.corr,
            iters=int(iters[0]), status=int(status[0]),
            setup_ms=t_setup*1e3, solve_ms=t_solve*1e3,
            total_ms=(t_setup + t_solve)*1e3,
        ))
    setup_med = np.median([r["setup_ms"] for r in fresh_rows])
    solve_med = np.median([r["solve_ms"] for r in fresh_rows])
    total_med = np.median([r["total_ms"] for r in fresh_rows])
    print(f"  median: setup={setup_med:.1f}ms  solve={solve_med:.1f}ms  total={total_med:.1f}ms",
          flush=True)
    print()

    # ---- Mode 2: Amortized
    print("=" * 78)
    print("Mode 2: AMGx AMORTIZED (1 setup + N updates)")
    print("=" * 78)
    amort_rows = []
    plan = None
    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, 1e-12, max_iters=500)
    for i, b in enumerate(bundles):
        A, src, _ = normalize_sign(b.A, b.b)
        rp = jnp.asarray(A.indptr.astype(np.int32))
        ci = jnp.asarray(A.indices.astype(np.int32))
        vv = jnp.asarray(A.data.astype(np.float64))
        b_d = jnp.asarray(src); x0_d = jnp.zeros_like(b_d)

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
        x, iters, status = plan.solve(b_d, x0_d)
        x.block_until_ready()
        t_solve = time.time() - t1
        amort_rows.append(dict(
            step=b.step, corr=b.corr, iters=int(iters[0]),
            setup_ms=t_setup*1e3, update_ms=t_update*1e3, solve_ms=t_solve*1e3,
            total_ms=(t_setup + t_update + t_solve)*1e3,
            is_first=(i == 0),
        ))
    if plan is not None:
        plan.release()
    upd_med = np.median([r["update_ms"] for r in amort_rows if not r["is_first"]])
    sol_med = np.median([r["solve_ms"] for r in amort_rows if not r["is_first"]])
    tot_med = np.median([r["total_ms"] for r in amort_rows if not r["is_first"]])
    print(f"  median (excl first): update={upd_med:.1f}ms  solve={sol_med:.1f}ms  total={tot_med:.1f}ms",
          flush=True)
    print()

    # ---- Mode 3: Amortized + 1 IR (precision config)
    print("=" * 78)
    print("Mode 3: AMGx AMORTIZED + 1 IR (machine-precision config)")
    print("=" * 78)
    ir_rows = []
    for b in bundles:
        A, src, _ = normalize_sign(b.A, b.b)
        denom_xref = max(float(np.max(np.abs(b.x_ref))), 1e-300)
        t0 = time.time()
        res = amgx_solve_with_refinement(
            A, src, np.zeros_like(src),
            eq_kind="pd", tol=1e-12, n_refine=1, max_iters=500)
        t_total = time.time() - t0
        x_h = res["x"]
        rel_vs_xref = float(np.max(np.abs(x_h - b.x_ref)) / denom_xref)
        rel_resid = res["rel_residual"]
        ir_rows.append(dict(
            step=b.step, corr=b.corr,
            iters=res["primary_iters"],
            total_ms=t_total*1e3,
            rel_resid=rel_resid,
            rel_vs_xref=rel_vs_xref,
        ))
    ir_med = np.median([r["total_ms"] for r in ir_rows])
    ir_resid_max = max(r["rel_resid"] for r in ir_rows)
    ir_xref_max = max(r["rel_vs_xref"] for r in ir_rows)
    print(f"  median total: {ir_med:.1f}ms")
    print(f"  rel_resid max: {ir_resid_max:.2e}")
    print(f"  rel_vs_xref max: {ir_xref_max:.2e}", flush=True)
    print()

    # ---- Summary
    print("=" * 78)
    print("SUMMARY (per pd solve, 512K cells, 57 evaporation matrices)")
    print("=" * 78)
    print(f"{'method':50s} {'median (ms)':>12s}")
    print("-" * 64)
    print(f"{'AMGx fresh setup':50s} {total_med:>12.1f}")
    print(f"{'AMGx amortized':50s} {tot_med:>12.1f}")
    print(f"{'AMGx amortized + 1 IR (precision)':50s} {ir_med:>12.1f}")
    print()

    out = Path("/tmp/amgx_5060_evaporation_bench.json")
    out.write_text(json.dumps(dict(
        date=datetime.now().isoformat(),
        dataset="evaporation",
        n_bundles=len(bundles),
        mode_fresh=fresh_rows,
        mode_amortized=amort_rows,
        mode_amortized_ir=ir_rows,
    ), indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
