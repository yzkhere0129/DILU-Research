"""5060 AMGx tolerance sweep: wall + iter at several tolerances.

Use senior's initial period matrices (21 bundles, cold-start, AMGx CAN
solve them — we measured rel_resid 1e-15 in earlier byte_match work).

Sweeps tol = 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4 across the 21 cases.
For each (tol, case): record AMGx setup_ms, solve_ms, total_ms, iter,
rel_resid_actual.

Output: /tmp/amgx_tol_sweep.json + plot data

Run on lab 5060:
    cd ~/DILU-Research && git pull origin main
    python3 -u -m dilu.amgx.bench.bench_amgx_tol_sweep \\
        --npz-dir dilu/amgx/bench/lab_deploy/initial_period_npz \\
        > /tmp/amgx_tol_sweep.log 2>&1
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
from scipy.sparse import csr_matrix

from jax import config as _jc
_jc.update("jax_enable_x64", True)
import jax.numpy as jnp

from dilu.amgx.python import (
    Plan, CLASSICAL_V_DIAGSCALED, with_tolerance,
)
from dilu.amgx.bench.sweep_amgx_senior_data import normalize_sign


TOLS = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]


def load_bundle(npz_path: Path):
    z = np.load(npz_path)
    n = int(z["n"][0])
    A = csr_matrix((z["A_data"], z["A_indices"], z["A_indptr"]),
                    shape=(n, n))
    return A, z["b"], z["x_ref"], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True,
                     help="dir with bundle_pd_*.npz")
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--out", default="/tmp/amgx_tol_sweep.json")
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir).expanduser()
    files = sorted(npz_dir.glob("bundle_pd_*.npz"))
    print(f"Loading {len(files)} bundles from {npz_dir}", flush=True)
    bundles = []
    for f in files:
        A, b, x_ref, n = load_bundle(f)
        # Sign-flip to match AMGx pos-diag convention
        A_pos, b_pos, _ = normalize_sign(A, b)
        bundles.append((f.stem, A_pos, b_pos, x_ref, n))
    print(f"  loaded {len(bundles)} bundles, N={bundles[0][4]}, "
          f"nnz={bundles[0][1].nnz}", flush=True)

    results = []
    for tol in TOLS:
        print(f"\n{'='*60}\nTOL = {tol:.0e}\n{'='*60}", flush=True)
        cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, tol,
                              max_iters=args.max_iter)
        for label, A, b, x_ref, n in bundles:
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

            rel_resid = float(np.linalg.norm(A @ x_h - b)
                              / max(np.linalg.norm(b), 1e-300))

            r = dict(
                tol=tol, label=label, n=n,
                iters=int(iters[0]), status=int(status[0]),
                setup_ms=t_setup * 1e3,
                solve_ms=t_solve * 1e3,
                total_ms=(t_setup + t_solve) * 1e3,
                rel_resid_actual=rel_resid,
            )
            results.append(r)
            print(f"  {label}: iter={r['iters']:>4d}  "
                  f"solve={r['solve_ms']:>7.1f}ms  "
                  f"setup={r['setup_ms']:>7.1f}ms  "
                  f"resid={r['rel_resid_actual']:.2e}", flush=True)

        # Per-tol summary
        rows_this_tol = [r for r in results if r["tol"] == tol]
        med_solve = np.median([r["solve_ms"] for r in rows_this_tol])
        med_iter = int(np.median([r["iters"] for r in rows_this_tol]))
        max_resid = max(r["rel_resid_actual"] for r in rows_this_tol)
        print(f"\n  ↳ tol={tol:.0e}  median solve_ms={med_solve:.1f}  "
              f"median iter={med_iter}  max actual_resid={max_resid:.2e}",
              flush=True)

    # Output
    Path(args.out).write_text(json.dumps(dict(
        date=datetime.now().isoformat(),
        machine="lab-5060",
        npz_dir=str(npz_dir),
        n_bundles=len(bundles),
        tols=TOLS,
        results=results,
    ), indent=2))
    print(f"\nWrote {args.out}")

    # Quick console summary table
    print(f"\n{'='*60}\nSUMMARY: AMGx wall vs tolerance\n{'='*60}")
    print(f"{'tol':>10s}  {'med iter':>10s}  {'med solve_ms':>14s}  "
          f"{'max actual_resid':>18s}")
    print("-" * 60)
    for tol in TOLS:
        rs = [r for r in results if r["tol"] == tol]
        med_solve = np.median([r["solve_ms"] for r in rs])
        med_iter = int(np.median([r["iters"] for r in rs]))
        max_r = max(r["rel_resid_actual"] for r in rs)
        print(f"{tol:>10.0e}  {med_iter:>10d}  {med_solve:>14.2f}  {max_r:>18.2e}")


if __name__ == "__main__":
    main()
