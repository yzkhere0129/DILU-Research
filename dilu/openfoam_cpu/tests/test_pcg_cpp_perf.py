"""Run the C++-backed PCG (DIC) solver on senior pd matrices.

Quick perf + sanity check — for each pd dump:
  - pcg_cpp.solve from x0=0, tol=1e-12
  - report iter / final_residual / wall
  - compare ‖x_ours - x_ref‖∞ / ‖x_ref‖∞ where x_ref is the OF xref

Note: senior xref was OF tol=1e-8. Setting our tol smaller will produce
a more accurate solution — the relative gap vs xref reflects xref's
truncation, not our error (proven by tol_sweep_senior.py).

Acceptance gate (plan §1.6): median wall ≤ 250ms over 78 senior pd cases.

Run:
    /home/yzk/jax-env/bin/python -m dilu.openfoam_cpu.tests.test_pcg_cpp_perf
"""
from __future__ import annotations

import sys
import time

import numpy as np

from dilu.amgx.bench.senior_data_loader import set_dataset, list_available, load_step
from dilu.openfoam_cpu.python.ldu import csr_to_ldu
from dilu.openfoam_cpu.python import pcg


def main() -> int:
    rows = []
    print("PCG (C++ DIC) on senior pd matrices, tol=1e-12, x0=0\n")
    print(f"{'case':>22s}  {'iter':>6s}  {'r_init':>10s}  {'r_final':>10s}  "
          f"{'wall_ms':>8s}  {'rel_xref':>10s}  flags")
    print("-" * 90)

    for ds in ("initial", "evaporation"):
        set_dataset(ds)
        for step, corr in list_available():
            bundle = load_step(step, corr)
            ldu = csr_to_ldu(bundle.A)
            x0 = np.zeros_like(bundle.b)

            t0 = time.time()
            res = pcg.solve(ldu, bundle.b, x0,
                             tolerance=1e-12, min_iter=1, max_iter=2000)
            wall = time.time() - t0

            denom = max(float(np.abs(bundle.x_ref).max()), 1e-300)
            rel_xref = float(np.abs(res.x - bundle.x_ref).max() / denom)
            flags = ""
            if res.singular: flags += "S"
            if not res.converged: flags += "X"
            label = f"{ds[:5]:>5s}/{step:>3d}/{corr}"
            print(f"{label:>22s}  {res.n_iterations:>6d}  "
                  f"{res.initial_residual:>10.2e}  {res.final_residual:>10.2e}  "
                  f"{wall*1e3:>8.1f}  {rel_xref:>10.2e}  {flags}")
            rows.append(dict(
                ds=ds, step=step, corr=corr,
                iters=res.n_iterations,
                r_init=res.initial_residual, r_final=res.final_residual,
                wall_ms=wall*1e3, rel_xref=rel_xref,
                singular=res.singular, converged=res.converged,
            ))

    walls = np.array([r["wall_ms"] for r in rows])
    iters = np.array([r["iters"]   for r in rows])
    rels  = np.array([r["rel_xref"] for r in rows])

    print()
    print("=" * 70)
    print(f"Cases:                {len(rows)}")
    print(f"Iter   median/max:    {int(np.median(iters))} / {iters.max()}")
    print(f"Wall   median/max:    {np.median(walls):.1f}ms / {walls.max():.1f}ms")
    print(f"rel_xref median/max:  {np.median(rels):.2e} / {rels.max():.2e}")
    print(f"P1 perf gate (med ≤ 250ms): "
          f"{'PASS' if np.median(walls) <= 250 else 'FAIL'}  "
          f"({np.median(walls):.0f}ms)")
    print("=" * 70)

    return 0 if np.median(walls) <= 250 else 1


if __name__ == "__main__":
    sys.exit(main())
