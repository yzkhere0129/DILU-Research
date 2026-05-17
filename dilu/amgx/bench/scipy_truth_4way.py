"""4-way comparison vs scipy LU direct solve (the absolute truth).

Tests if AMGx + our replica share a hidden bug. If both differ from scipy truth
in the SAME direction as xref, there's a common pipeline issue (sign, data
loading, indexing). If both match scipy and only xref drifts, xref is the
outlier.

scipy.sparse.linalg.spsolve does NOT use any tolerance — it's direct LU
factorization (= mathematical truth up to fp64 round-off ~1e-16).
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
from scipy.sparse.linalg import spsolve

from jax import config as _jc
_jc.update("jax_enable_x64", True)

from dilu.amgx.python import amgx_solve_with_refinement
from dilu.amgx.bench.senior_data_loader import load_step, list_available
from dilu.amgx.bench.sweep_amgx_senior_data import normalize_sign

from dilu.openfoam_cpu.python.ldu import csr_to_ldu
from dilu.openfoam_cpu.python.pbicg import solve as our_pbicg_solve


def main():
    avail = list_available()
    bundles = [load_step(s, c) for s, c in avail]
    bundles = [b for b in bundles if b is not None]
    # Only do first 3 bundles (spsolve is memory-heavy on 512K)
    bundles = bundles[:3]
    print(f"4-way comparison vs scipy LU truth on {len(bundles)} senior pd bundles")
    print(f"(spsolve is memory-heavy ~10GB per matrix on 512K, hence small subset)\n")

    print(f"{'step':>4s} {'corr':>4s}  "
          f"{'spsolve_s':>10s}  "
          f"{'truth_resid':>12s}  "
          f"{'AMGx vs truth':>14s}  "
          f"{'replica vs truth':>17s}  "
          f"{'xref vs truth':>14s}  "
          f"{'AMGx vs xref':>13s}")
    print("-" * 120)

    rows = []
    for b in bundles:
        A_raw, b_raw, xref = b.A, b.b, b.x_ref
        A, src, _ = normalize_sign(A_raw, b_raw)

        # 1. scipy LU truth (direct, no tolerance)
        try:
            t0 = time.time()
            x_truth = spsolve(A.tocsc(), src)
            t_truth = time.time() - t0
        except Exception as e:
            print(f"step{b.step}/c{b.corr}: spsolve failed: {e}")
            continue

        truth_resid = float(np.linalg.norm(A @ x_truth - src) / max(np.linalg.norm(src), 1e-300))
        denom = max(float(np.max(np.abs(x_truth))), 1e-300)

        # 2. AMGx tight + IR
        res_amgx = amgx_solve_with_refinement(
            A, src, np.zeros_like(src),
            eq_kind="pd", tol=1e-14, n_refine=3, max_iters=1000)
        x_amgx = res_amgx["x"]

        # 3. Our replica at tol=1e-12
        ldu = csr_to_ldu(A)
        perf = our_pbicg_solve(ldu, src, np.zeros_like(src),
                                tolerance=1e-12, rel_tol=0.0,
                                min_iter=1, max_iter=2000)
        x_replica = perf.x

        # Distances vs truth
        rel_amgx = float(np.max(np.abs(x_amgx - x_truth)) / denom)
        rel_replica = float(np.max(np.abs(x_replica - x_truth)) / denom)
        rel_xref = float(np.max(np.abs(xref - x_truth)) / denom)
        rel_amgx_vs_xref = float(np.max(np.abs(x_amgx - xref)) /
                                  max(float(np.max(np.abs(xref))), 1e-300))

        print(f"{b.step:>4d} {b.corr:>4d}  "
              f"{t_truth:>10.1f}  "
              f"{truth_resid:>12.2e}  "
              f"{rel_amgx:>14.2e}  "
              f"{rel_replica:>17.2e}  "
              f"{rel_xref:>14.2e}  "
              f"{rel_amgx_vs_xref:>13.2e}",
              flush=True)

        rows.append(dict(step=b.step, corr=b.corr,
                         truth_solve_s=t_truth,
                         truth_resid=truth_resid,
                         rel_amgx_vs_truth=rel_amgx,
                         rel_replica_vs_truth=rel_replica,
                         rel_xref_vs_truth=rel_xref,
                         rel_amgx_vs_xref=rel_amgx_vs_xref))

    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    if not rows:
        print("(no successful runs)")
        return

    am_vs_t = [r["rel_amgx_vs_truth"] for r in rows]
    rep_vs_t = [r["rel_replica_vs_truth"] for r in rows]
    xref_vs_t = [r["rel_xref_vs_truth"] for r in rows]

    print(f"AMGx     vs truth:  max = {max(am_vs_t):.2e}  median = {sorted(am_vs_t)[len(am_vs_t)//2]:.2e}")
    print(f"Replica  vs truth:  max = {max(rep_vs_t):.2e}  median = {sorted(rep_vs_t)[len(rep_vs_t)//2]:.2e}")
    print(f"xref     vs truth:  max = {max(xref_vs_t):.2e}  median = {sorted(xref_vs_t)[len(xref_vs_t)//2]:.2e}")

    print()
    print("DECISION TREE:")
    if max(am_vs_t) < 1e-6 and max(rep_vs_t) < 1e-6 and max(xref_vs_t) > 1e-3:
        print("  → AMGx + replica BOTH close to truth, xref FAR")
        print("  → No shared bug. xref is the outlier (senior dump issue).")
    elif max(am_vs_t) > 1e-3 and max(rep_vs_t) > 1e-3 and max(xref_vs_t) > 1e-3:
        print("  → AMGx + replica + xref ALL far from truth")
        print("  → SHARED BUG in our pipeline (sign/loading/indexing)!")
    elif max(am_vs_t) < 1e-6 and max(rep_vs_t) > 1e-3:
        print("  → AMGx close to truth, replica drifts")
        print("  → Replica algorithm bug (PBiCG-DILU on pd?)")
    else:
        print("  → mixed signals, interpret manually")

    out = Path("/home/yzk/DILU-Research/dilu/amgx/bench/scipy_truth_4way_results.json")
    out.write_text(json.dumps(dict(date=datetime.now().isoformat(),
                                    n=len(rows), rows=rows), indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
