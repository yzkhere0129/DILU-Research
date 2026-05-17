"""AMGx tolerance sweep on senior pd matrices.

For each (tol, n_refine) config, time AMGx (amortized, 1 setup + 20 updates)
and record per-matrix wall + final residual + iter count.

Output: tol_sweep_results.json + console table.
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
from dilu.amgx.bench.senior_data_loader import load_step, list_available
from dilu.amgx.bench.sweep_amgx_senior_data import normalize_sign


def time_amortized(bundles, tol: float, n_refine: int):
    """Run AMGx amortized over bundles with given (tol, n_refine).
    Returns per-matrix dict list."""
    rows = []
    plan = None
    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, tol, max_iters=500)

    for i, b in enumerate(bundles):
        A, src, _ = normalize_sign(b.A, b.b)
        rp = jnp.asarray(A.indptr.astype(np.int32))
        ci = jnp.asarray(A.indices.astype(np.int32))
        vv = jnp.asarray(A.data.astype(np.float64))
        b_d = jnp.asarray(src)
        x0_d = jnp.zeros_like(b_d)

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

        t0 = time.time()
        x_jax, iters, status = plan.solve(b_d, x0_d)
        x_jax.block_until_ready()
        t_solve = time.time() - t0
        x = np.asarray(x_jax)

        t_ir = 0.0
        for _ in range(n_refine):
            r = A @ x - src
            r_d = jnp.asarray(-r)
            zero = jnp.zeros_like(r_d)
            t0 = time.time()
            d_jax, _, _ = plan.solve(r_d, zero)
            d_jax.block_until_ready()
            t_ir += time.time() - t0
            x = x + np.asarray(d_jax)

        # Compute precision
        r_final = A @ x - src
        rel_resid = float(np.linalg.norm(r_final) / max(np.linalg.norm(src), 1e-300))
        denom_xref = max(float(np.max(np.abs(b.x_ref))), 1e-300)
        rel_vs_xref = float(np.max(np.abs(x - b.x_ref)) / denom_xref)

        rows.append(dict(
            step=b.step, corr=b.corr, is_first=(i == 0),
            iters=int(iters[0]),
            setup_ms=t_setup * 1e3, update_ms=t_update * 1e3,
            solve_ms=t_solve * 1e3, ir_ms=t_ir * 1e3,
            total_ms=(t_setup + t_update + t_solve + t_ir) * 1e3,
            rel_resid=rel_resid, rel_vs_xref=rel_vs_xref,
        ))

    if plan is not None:
        plan.release()
    return rows


def main():
    avail = list_available()
    print(f"Loading {len(avail)} senior pd bundles...")
    bundles = [load_step(s, c) for s, c in avail]
    bundles = [b for b in bundles if b is not None]
    print(f"  loaded {len(bundles)}")

    configs = [
        (1e-14, 0), (1e-14, 1),
        (1e-12, 0), (1e-12, 1),
        (1e-10, 0), (1e-10, 1),
        (1e-8,  0),
        (1e-6,  0),
        (1e-4,  0),
    ]

    print()
    print(f"{'tol':>10s} {'IR':>3s} {'iter':>5s} "
          f"{'update':>8s} {'solve':>8s} {'ir':>6s} {'total':>8s} "
          f"{'rel_resid':>10s} {'rel_vs_xref':>12s}")
    print("-" * 90)

    summary = []
    for tol, n_ir in configs:
        rows = time_amortized(bundles, tol, n_ir)
        # Median over non-first matrices (skip Plan setup overhead)
        non_first = [r for r in rows if not r["is_first"]]
        upd_med = np.median([r["update_ms"] for r in non_first])
        solve_med = np.median([r["solve_ms"] for r in non_first])
        ir_med = np.median([r["ir_ms"] for r in non_first])
        total_med = np.median([r["total_ms"] for r in non_first])
        iter_med = int(np.median([r["iters"] for r in non_first]))
        rel_resid_max = max([r["rel_resid"] for r in non_first])
        rel_xref_max = max([r["rel_vs_xref"] for r in non_first])

        print(f"{tol:>10.0e} {n_ir:>3d} {iter_med:>5d} "
              f"{upd_med:>8.1f} {solve_med:>8.1f} {ir_med:>6.1f} {total_med:>8.1f} "
              f"{rel_resid_max:>10.2e} {rel_xref_max:>12.2e}")

        summary.append(dict(
            tol=tol, n_refine=n_ir,
            iter_med=iter_med,
            update_ms=upd_med, solve_ms=solve_med,
            ir_ms=ir_med, total_ms=total_med,
            rel_resid_max=rel_resid_max,
            rel_vs_xref_max=rel_xref_max,
            rows=rows,
        ))

    out = Path("/home/yzk/DILU-Research/dilu/amgx/bench/tol_sweep_results.json")
    out.write_text(json.dumps(dict(
        date=datetime.now().isoformat(),
        n_bundles=len(bundles),
        configs=summary,
    ), indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
