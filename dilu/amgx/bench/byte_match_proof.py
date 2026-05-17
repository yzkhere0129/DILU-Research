"""Byte-match proof: AMGx vs OpenFOAM-replica when both go to machine precision.

Hypothesis: AMGx and OpenFOAM converge to the SAME mathematical solution
when both are run to machine-precision residual. The 2.5% gap measured
against senior's xref is purely OF's tol=1e-8 truncation × κ amplification.

Test:
  1. Run our PBiCG-DILU replica (byte-exact match real OF on T matrices)
     at tol=1e-12 on each senior pd matrix → x_replica_tight
  2. Run AMGx at tol=1e-14 + 3 IR on same matrix → x_amgx_tight
  3. Compare ‖x_replica − x_amgx‖∞ / ‖x_amgx‖∞

Expected:
  - If both converge to same truth: gap ≤ κ × machine_eps ≈ 10^8 × 10^-16 = 10^-8
  - If gap ~ 1e-8 to 1e-10: PROOF that AMGx and OF agree at fp64 limit
  - If gap ~ 2.5%: hypothesis FALSIFIED, AMGx really differs from OF

For senior data κ ≈ 10^8 → expected fp64 floor ≈ 10^-8 (good enough proof).
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
    bundles = [load_step(s, c) for s, c in avail]
    bundles = [b for b in bundles if b is not None]
    print(f"Testing byte-match on {len(bundles)} senior pd bundles (κ ≈ 10^8)")
    print(f"Both solvers run to machine precision.")
    print(f"AMGx: tol=1e-14 + 3 IR  vs  OF-replica (PBiCG-DILU): tol=1e-12, max_iter=2000\n")

    print(f"{'step':>4s} {'corr':>4s}  "
          f"{'AMGx_resid':>11s}  {'replica_resid':>14s}  "
          f"{'replica_iter':>13s}  "
          f"{'rel_AMGx_vs_replica':>20s}  "
          f"{'rel_AMGx_vs_xref':>17s}  "
          f"{'rel_replica_vs_xref':>20s}")
    print("-" * 130)

    rows = []
    for b in bundles:
        A_raw, b_raw, xref = b.A, b.b, b.x_ref
        A, src, _ = normalize_sign(A_raw, b_raw)

        # 1. AMGx tight + 3 IR (machine-precision truth proxy)
        res_amgx = amgx_solve_with_refinement(
            A, src, np.zeros_like(src),
            eq_kind="pd", tol=1e-14, n_refine=3, max_iters=1000)
        x_amgx = res_amgx["x"]
        amgx_resid = res_amgx["rel_residual"]

        # 2. Our PBiCG-DILU replica at tol=1e-12 (byte-exact OF algorithm, just tighter tol)
        ldu = csr_to_ldu(A)
        t0 = time.time()
        perf = our_pbicg_solve(ldu, src, np.zeros_like(src),
                                tolerance=1e-12, rel_tol=0.0,
                                min_iter=1, max_iter=2000)
        t_rep = time.time() - t0
        x_replica = perf.x
        replica_resid = float(np.linalg.norm(A @ x_replica - src)
                               / max(np.linalg.norm(src), 1e-300))

        # 3. The KEY metric: how close are AMGx and replica when both are tight?
        denom = max(float(np.max(np.abs(x_amgx))), 1e-300)
        rel_amgx_vs_replica = float(np.max(np.abs(x_amgx - x_replica)) / denom)

        # 4. The OLD metric for comparison: vs senior's xref (loose)
        denom_xref = max(float(np.max(np.abs(xref))), 1e-300)
        rel_amgx_vs_xref = float(np.max(np.abs(x_amgx - xref)) / denom_xref)
        rel_replica_vs_xref = float(np.max(np.abs(x_replica - xref)) / denom_xref)

        print(f"{b.step:>4d} {b.corr:>4d}  "
              f"{amgx_resid:>11.2e}  {replica_resid:>14.2e}  "
              f"{perf.n_iterations:>13d}  "
              f"{rel_amgx_vs_replica:>20.2e}  "
              f"{rel_amgx_vs_xref:>17.2e}  "
              f"{rel_replica_vs_xref:>20.2e}",
              flush=True)

        rows.append(dict(step=b.step, corr=b.corr,
                         amgx_resid=amgx_resid,
                         replica_resid=replica_resid,
                         replica_iter=perf.n_iterations,
                         replica_solve_s=t_rep,
                         rel_amgx_vs_replica=rel_amgx_vs_replica,
                         rel_amgx_vs_xref=rel_amgx_vs_xref,
                         rel_replica_vs_xref=rel_replica_vs_xref))

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    rels = [r["rel_amgx_vs_replica"] for r in rows]
    rels_x = [r["rel_amgx_vs_xref"] for r in rows]
    rels_rx = [r["rel_replica_vs_xref"] for r in rows]

    print(f"\n‖x_AMGx_tight - x_replica_tight‖∞ / ‖x_AMGx‖∞ over 21 matrices:")
    print(f"  max = {max(rels):.3e}")
    print(f"  med = {sorted(rels)[len(rels)//2]:.3e}")
    print(f"  min = {min(rels):.3e}")
    print(f"\n  Expected: ~10^-7 to 10^-9 (κ × machine_eps for κ=10^8)")
    print(f"  → If max ≤ 1e-6: PROOF that AMGx and OF-algorithm agree at fp64 limit")
    print(f"  → If max ≈ 2.5e-2: AMGx and OF really disagree (would be alarming)")

    print(f"\nFor reference (the 'alarming' 2.5% number from before):")
    print(f"  ‖x_AMGx - x_xref‖∞ / ‖x_xref‖∞:    max={max(rels_x):.2e}  med={sorted(rels_x)[len(rels_x)//2]:.2e}")
    print(f"  ‖x_replica - x_xref‖∞ / ‖x_xref‖∞: max={max(rels_rx):.2e}  med={sorted(rels_rx)[len(rels_rx)//2]:.2e}")
    print(f"  → If both AMGx AND replica differ from xref by ~2.5%, that's xref noise.")
    print(f"  → If only AMGx differs by 2.5% but replica matches xref byte-exact, that's an AMGx bug.")

    out = Path("/home/yzk/DILU-Research/dilu/amgx/bench/byte_match_proof_results.json")
    out.write_text(json.dumps(dict(date=datetime.now().isoformat(),
                                    n=len(rows), rows=rows), indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
