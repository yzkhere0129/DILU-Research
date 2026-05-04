"""3-way precision comparison on senior pd matrices.

For each of 21 senior pd matrices, compute:
  rel_vs_xref(method)   = ||x_method − x_ref||∞ / ||x_ref||∞
  rel_vs_truth(method)  = ||x_method − x_truth||∞ / ||x_truth||∞

Where x_truth = AMGx with tol=1e-14 + 3 IR (verified to match scipy spsolve
to <1e-15 on cases where both fit in memory).

3 methods:
  1. OpenFOAM PCG-DIC (x_ref dumped by senior, OF tol=1e-8)
  2. Our PBiCG-DILU replica   (this machine, single-thread numpy/numba)
  3. AMGx amortized + 1 IR    (this machine GPU)
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

from dilu.amgx.python import amgx_solve_with_refinement
from dilu.amgx.bench.senior_data_loader import load_step, list_available
from dilu.amgx.bench.sweep_amgx_senior_data import normalize_sign

from dilu.openfoam_cpu.python.ldu import csr_to_ldu
from dilu.openfoam_cpu.python.pbicg import solve as our_pbicg_solve


def main():
    avail = list_available()
    print(f"Loading {len(avail)} senior bundles...")
    bundles = []
    for s, c in avail:
        b = load_step(s, c)
        if b: bundles.append(b)
    print(f"  loaded {len(bundles)}")

    rows = []
    print()
    print(f"{'step':>4s} {'corr':>4s} "
          f"{'OF_resid_2':>11s} "
          f"{'AMGx_truth':>11s} {'AMGx_xref':>10s} "
          f"{'rep_truth':>10s} {'rep_xref':>10s} {'rep_iter':>8s} {'rep_t(s)':>8s}")
    print("-" * 95)

    for b in bundles:
        A_raw, b_raw, xref = b.A, b.b, b.x_ref
        A, src, _ = normalize_sign(A_raw, b_raw)

        # Truth proxy: AMGx tight + 3 IR (verified ≤ 5e-15 vs scipy spsolve)
        res_truth = amgx_solve_with_refinement(A, src, np.zeros_like(src),
            eq_kind="pd", tol=1e-14, n_refine=3, max_iters=1000)
        x_truth = res_truth["x"]
        denom_truth = max(float(np.max(np.abs(x_truth))), 1e-300)
        denom_xref = max(float(np.max(np.abs(xref))), 1e-300)

        # Method 1: OF (xref) precision
        of_resid_2 = float(np.linalg.norm(A @ xref - src)
                           / max(np.linalg.norm(src), 1e-300))
        of_vs_truth = float(np.max(np.abs(xref - x_truth)) / denom_truth)

        # Method 2: AMGx amortized + 1 IR
        res_amgx = amgx_solve_with_refinement(A, src, np.zeros_like(src),
            eq_kind="pd", tol=1e-12, n_refine=1, max_iters=500)
        x_amgx = res_amgx["x"]
        amgx_vs_truth = float(np.max(np.abs(x_amgx - x_truth)) / denom_truth)
        amgx_vs_xref = float(np.max(np.abs(x_amgx - xref)) / denom_xref)

        # Method 3: replica
        ldu = csr_to_ldu(A)
        t0 = time.time()
        perf = our_pbicg_solve(ldu, src, np.zeros_like(src),
            tolerance=1e-12, rel_tol=0.0, min_iter=1, max_iter=2000)
        rep_t = time.time() - t0
        x_rep = perf.x
        rep_vs_truth = float(np.max(np.abs(x_rep - x_truth)) / denom_truth)
        rep_vs_xref = float(np.max(np.abs(x_rep - xref)) / denom_xref)

        print(f"{b.step:>4d} {b.corr:>4d} "
              f"{of_resid_2:>11.2e} "
              f"{amgx_vs_truth:>11.2e} {amgx_vs_xref:>10.2e} "
              f"{rep_vs_truth:>10.2e} {rep_vs_xref:>10.2e} "
              f"{perf.n_iterations:>8d} {rep_t:>8.2f}", flush=True)

        rows.append(dict(
            step=b.step, corr=b.corr,
            of_resid_2=of_resid_2,
            of_vs_truth=of_vs_truth,
            amgx_vs_truth=amgx_vs_truth,
            amgx_vs_xref=amgx_vs_xref,
            rep_vs_truth=rep_vs_truth,
            rep_vs_xref=rep_vs_xref,
            rep_iter=perf.n_iterations,
            rep_solve_s=rep_t,
        ))

    print()
    print("=" * 90)
    print("SUMMARY (max / median / min over 21 senior pd matrices)")
    print("=" * 90)
    def stat(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return f"max={max(vals):.2e}  med={sorted(vals)[len(vals)//2]:.2e}  min={min(vals):.2e}"
    print(f"  OpenFOAM (xref)     vs truth : {stat('of_vs_truth')}")
    print(f"  AMGx +1 IR          vs truth : {stat('amgx_vs_truth')}")
    print(f"  Our PBiCG-DILU repl vs truth : {stat('rep_vs_truth')}")
    print()
    print(f"  AMGx           vs OF (xref)  : {stat('amgx_vs_xref')}")
    print(f"  Replica        vs OF (xref)  : {stat('rep_vs_xref')}")

    out = Path("/home/yzk/DILU-Research/dilu/amgx/bench/threeway_precision_results.json")
    out.write_text(json.dumps(dict(date=datetime.now().isoformat(),
                                    n=len(rows), rows=rows), indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
