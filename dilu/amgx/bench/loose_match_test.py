"""Test: run AMGx and replica at the SAME tol=1e-8 as OF used, see if byte-match xref."""
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
    bundles = [load_step(s, c) for s, c in avail]
    bundles = [b for b in bundles if b is not None]
    print(f"All solvers @ tol=1e-8 (matching OF default), N={len(bundles)} matrices")
    print(f"{'step':>4s} {'corr':>4s}  "
          f"{'AMGx_loose vs xref':>18s}  "
          f"{'replica_loose vs xref':>21s}  "
          f"{'AMGx_loose vs replica_loose':>27s}  "
          f"{'AMGx_tight vs xref':>18s}")
    print("-" * 120)

    rows = []
    for b in bundles:
        A_raw, b_raw, xref = b.A, b.b, b.x_ref
        A, src, _ = normalize_sign(A_raw, b_raw)
        denom_xref = max(float(np.max(np.abs(xref))), 1e-300)

        # AMGx at tol=1e-8 (match OF)
        res_amgx_loose = amgx_solve_with_refinement(
            A, src, np.zeros_like(src),
            eq_kind="pd", tol=1e-8, n_refine=0, max_iters=500)
        x_amgx_loose = res_amgx_loose["x"]

        # AMGx tight + IR (the truth-quality run for reference)
        res_amgx_tight = amgx_solve_with_refinement(
            A, src, np.zeros_like(src),
            eq_kind="pd", tol=1e-14, n_refine=3, max_iters=1000)
        x_amgx_tight = res_amgx_tight["x"]

        # Replica at tol=1e-8
        ldu = csr_to_ldu(A)
        perf = our_pbicg_solve(ldu, src, np.zeros_like(src),
                                tolerance=1e-8, rel_tol=0.0,
                                min_iter=1, max_iter=2000)
        x_rep_loose = perf.x

        rel_amgx_loose_vs_xref = float(np.max(np.abs(x_amgx_loose - xref)) / denom_xref)
        rel_rep_loose_vs_xref = float(np.max(np.abs(x_rep_loose - xref)) / denom_xref)
        rel_amgx_loose_vs_rep_loose = float(np.max(np.abs(x_amgx_loose - x_rep_loose)) /
                                             max(float(np.max(np.abs(x_amgx_loose))), 1e-300))
        rel_amgx_tight_vs_xref = float(np.max(np.abs(x_amgx_tight - xref)) / denom_xref)

        print(f"{b.step:>4d} {b.corr:>4d}  "
              f"{rel_amgx_loose_vs_xref:>18.2e}  "
              f"{rel_rep_loose_vs_xref:>21.2e}  "
              f"{rel_amgx_loose_vs_rep_loose:>27.2e}  "
              f"{rel_amgx_tight_vs_xref:>18.2e}", flush=True)

        rows.append(dict(
            step=b.step, corr=b.corr,
            amgx_loose_iter=res_amgx_loose["primary_iters"],
            amgx_loose_resid=res_amgx_loose["rel_residual"],
            replica_loose_iter=perf.n_iterations,
            replica_loose_resid=float(np.linalg.norm(A @ x_rep_loose - src) /
                                       max(np.linalg.norm(src), 1e-300)),
            rel_amgx_loose_vs_xref=rel_amgx_loose_vs_xref,
            rel_replica_loose_vs_xref=rel_rep_loose_vs_xref,
            rel_amgx_loose_vs_replica_loose=rel_amgx_loose_vs_rep_loose,
            rel_amgx_tight_vs_xref=rel_amgx_tight_vs_xref,
        ))

    print("\n" + "=" * 90)
    print("SUMMARY (max / median / min over 21 matrices)")
    print("=" * 90)
    cols = [
        ("AMGx loose (tol=1e-8) vs xref", "rel_amgx_loose_vs_xref"),
        ("Replica loose (tol=1e-8) vs xref", "rel_replica_loose_vs_xref"),
        ("AMGx loose vs replica loose", "rel_amgx_loose_vs_replica_loose"),
        ("AMGx tight (tol=1e-14+IR) vs xref [for ref]", "rel_amgx_tight_vs_xref"),
    ]
    print(f"{'metric':<50s} {'max':>10s} {'median':>10s} {'min':>10s}")
    print("-" * 85)
    for name, key in cols:
        vals = [r[key] for r in rows]
        mx = max(vals); md = sorted(vals)[len(vals)//2]; mn = min(vals)
        print(f"{name:<50s} {mx:>10.2e} {md:>10.2e} {mn:>10.2e}")

    out = Path("/home/yzk/DILU-Research/dilu/amgx/bench/loose_match_results.json")
    out.write_text(json.dumps(dict(date=datetime.now().isoformat(),
                                    n=len(rows), rows=rows), indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
