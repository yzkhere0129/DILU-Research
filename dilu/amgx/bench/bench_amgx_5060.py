"""Lab 5060 — run AMGx on senior pd matrices as a pure "matrix calculator"
benchmark.  No OpenFOAM, no CFD context.

Runs on BOTH datasets (initial + evaporation) so we can compare wall and
precision side-by-side with bench_pcg_xeon.py running on the lab Xeon.

Per case we apply the same preprocessing as the Xeon bench:
  1. normalize_sign     — flip A,b so diag is positive
  2. project to col(A)  — b ← b - mean(b)  (Neumann compatibility)

Three solve modes:
  fresh     : new AMGx setup per matrix (cold-start)
  amortized : 1 setup + N coefficient updates (warm)
  +1 IR     : amortized + one iterative-refinement pass (machine precision)

Output: /tmp/bench_amgx_5060.{json,log}

Run on lab 5060:
    cd ~/DILU-Research && git pull origin main
    python3 -u -m dilu.amgx.bench.bench_amgx_5060 \\
        > /tmp/bench_amgx_5060.log 2>&1
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
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


def project_col_space(b: np.ndarray) -> np.ndarray:
    return b - b.mean()


def actual_rel_resid(A, x, b) -> float:
    Ax = A @ x
    return float(np.linalg.norm(Ax - b) / max(np.linalg.norm(b), 1e-300))


def run_dataset(name: str, *, tol: float, max_iter: int) -> dict:
    print(f"\n{'='*78}")
    print(f"Dataset: {name}")
    print(f"{'='*78}", flush=True)
    set_dataset(name)
    avail = list_available()
    print(f"  {len(avail)} bundles available", flush=True)
    bundles = []
    for s, c in avail:
        bundle = load_step(s, c)
        if bundle is None:
            continue
        A_pos, b_pos, _ = normalize_sign(bundle.A, bundle.b)
        b_proj = project_col_space(b_pos)
        bundles.append((bundle.step, bundle.corr, A_pos, b_proj, bundle.x_ref))
    if not bundles:
        return dict(fresh=[], amortized=[], ir1=[])
    n = bundles[0][2].shape[0]
    print(f"  N={n}, nnz={bundles[0][2].nnz}, "
          f"steps {min(b[0] for b in bundles)}…{max(b[0] for b in bundles)}",
          flush=True)

    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, tol, max_iters=max_iter)

    # ---- Mode 1: Fresh setup per matrix
    print(f"\n  [Mode 1: FRESH setup per matrix]", flush=True)
    fresh = []
    for step, corr, A, b, _ in bundles:
        rp = jnp.asarray(A.indptr.astype(np.int32))
        ci = jnp.asarray(A.indices.astype(np.int32))
        vv = jnp.asarray(A.data.astype(np.float64))
        b_d = jnp.asarray(b); x0_d = jnp.zeros_like(b_d)
        t0 = time.time()
        with Plan(rp, ci, vv, cfg) as plan:
            t_setup = time.time() - t0
            t1 = time.time()
            x, iters, status = plan.solve(b_d, x0_d)
            x.block_until_ready()
            t_solve = time.time() - t1
            x_h = np.asarray(x)
        rel_resid = actual_rel_resid(A, x_h, b)
        fresh.append(dict(
            step=step, corr=corr,
            iters=int(iters[0]), status=int(status[0]),
            setup_ms=t_setup*1e3, solve_ms=t_solve*1e3,
            total_ms=(t_setup + t_solve)*1e3,
            rel_resid_actual=rel_resid,
        ))
    print(f"    median: setup={np.median([r['setup_ms'] for r in fresh]):.1f}ms  "
          f"solve={np.median([r['solve_ms'] for r in fresh]):.1f}ms  "
          f"total={np.median([r['total_ms'] for r in fresh]):.1f}ms  "
          f"resid_max={max(r['rel_resid_actual'] for r in fresh):.2e}",
          flush=True)

    # ---- Mode 2: Amortized
    print(f"\n  [Mode 2: AMORTIZED — 1 setup + (N-1) updates]", flush=True)
    amort = []
    plan = None
    for i, (step, corr, A, b, _) in enumerate(bundles):
        rp = jnp.asarray(A.indptr.astype(np.int32))
        ci = jnp.asarray(A.indices.astype(np.int32))
        vv = jnp.asarray(A.data.astype(np.float64))
        b_d = jnp.asarray(b); x0_d = jnp.zeros_like(b_d)
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
        x_h = np.asarray(x)
        rel_resid = actual_rel_resid(A, x_h, b)
        amort.append(dict(
            step=step, corr=corr,
            iters=int(iters[0]), status=int(status[0]),
            setup_ms=t_setup*1e3, update_ms=t_update*1e3,
            solve_ms=t_solve*1e3,
            total_ms=(t_setup + t_update + t_solve)*1e3,
            is_first=(i == 0),
            rel_resid_actual=rel_resid,
        ))
    if plan is not None:
        plan.release()
    after_first = [r for r in amort if not r["is_first"]]
    if after_first:
        print(f"    median (excl first): "
              f"update={np.median([r['update_ms'] for r in after_first]):.1f}ms  "
              f"solve={np.median([r['solve_ms'] for r in after_first]):.1f}ms  "
              f"total={np.median([r['total_ms'] for r in after_first]):.1f}ms  "
              f"resid_max={max(r['rel_resid_actual'] for r in amort):.2e}",
              flush=True)

    # ---- Mode 3: Amortized + 1 IR
    print(f"\n  [Mode 3: AMORTIZED + 1 IR — machine-precision target]", flush=True)
    ir1 = []
    for step, corr, A, b, x_ref in bundles:
        t0 = time.time()
        res = amgx_solve_with_refinement(
            A, b, np.zeros_like(b),
            eq_kind="pd", tol=tol, n_refine=1, max_iters=max_iter)
        t_total = time.time() - t0
        rel_resid = actual_rel_resid(A, res["x"], b)
        ir1.append(dict(
            step=step, corr=corr,
            iters=res["primary_iters"],
            total_ms=t_total*1e3,
            rel_resid_internal=res["rel_residual"],
            rel_resid_actual=rel_resid,
        ))
    print(f"    median total: {np.median([r['total_ms'] for r in ir1]):.1f}ms  "
          f"resid_max={max(r['rel_resid_actual'] for r in ir1):.2e}",
          flush=True)

    return dict(fresh=fresh, amortized=amort, ir1=ir1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--datasets", default="initial,evaporation")
    ap.add_argument("--out", default="/tmp/bench_amgx_5060.json")
    args = ap.parse_args()

    print(f"bench_amgx_5060 — AMGx as matrix calculator")
    print(f"  tol={args.tol:.0e}, max_iter={args.max_iter}")
    print(f"  datasets: {args.datasets}")

    out = dict(date=datetime.now().isoformat(),
               tol=args.tol, max_iter=args.max_iter,
               datasets={})
    for name in args.datasets.split(","):
        out["datasets"][name.strip()] = run_dataset(
            name.strip(), tol=args.tol, max_iter=args.max_iter)

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
