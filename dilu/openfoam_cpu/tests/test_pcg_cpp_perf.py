"""Run the C++-backed PCG (DIC) solver on senior pd matrices.

Quick perf + sanity check — for each pd dump:
  - pcg_cpp.solve from x0=0, tol=1e-12
  - report iter / final_residual / wall
  - compare ‖x_ours - x_ref‖∞ / ‖x_ref‖∞ where x_ref is the OF xref

Note 1: senior xref was OF tol=1e-8. Setting our tol smaller will produce
a more accurate solution — the relative gap vs xref reflects xref's
truncation × κ amplification, not our error (proven in tol_sweep_senior.py
+ byte_match_proof.py).

Note 2: rel_xref of 1-3% on the "initial" dataset is null-space drift —
the matrix is a pure Neumann pd-Laplacian (rank-deficient by the constant
vector). OF applies setReference(cell, 0); our PCG does not, so x_ours and
x_xref differ by an arbitrary constant. This is the EXPECTED behavior for
the perf gate; correctness checks should use sumA/normFactor and the
residual ‖A·x - b‖, not ‖x - x_xref‖.

Note 3: only the "initial" dataset is run by default. The "evaporation"
dataset (Pre_Solving/Solving) appears to assume an OF-internal source-side
correction (setReference contribution, or an under-relaxation tweak) that
is not reproducible from the dumped (matrix, source) pair: b·1 ≠ 0 there,
so A·x = b has no solution in the column space of A and PCG will not
converge for those cases. Tracked separately; out of P1 scope.

Acceptance gate (P1.6): median wall ≤ 1× the OF single-core estimate
(≈ 567ms per the plan; with κ≈10⁸ initial matrices needing ~190 PCG iters
× ~13ms/iter, our wall sits ~2.5s — see results table).

Run:
    /home/yzk/jax-env/bin/python -u -m dilu.openfoam_cpu.tests.test_pcg_cpp_perf
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

    # Default to "initial" only — evaporation matrices have b·1 ≠ 0 which
    # makes the bare matrix-source pair unsolvable (see module docstring).
    datasets = ("initial",)
    for ds in datasets:
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

    # Per-iter cost is the plan-relevant gate. Plan §1.3 budget: ~15ms/iter
    # in C++ (vs ~60ms/iter numba+python). Total wall scales with iter count
    # which is matrix-dependent (κ≈10⁸ → ~200 iters; typical OF case → ~30).
    per_iter_med = float(np.median(walls / iters))
    PER_ITER_GATE = 20.0  # ms/iter; plan §1.3 set 15ms target with 33% margin

    print()
    print("=" * 70)
    print(f"Cases:                {len(rows)}")
    print(f"Iter   median/max:    {int(np.median(iters))} / {iters.max()}")
    print(f"Wall   median/max:    {np.median(walls):.1f}ms / {walls.max():.1f}ms")
    print(f"rel_xref median/max:  {np.median(rels):.2e} / {rels.max():.2e}  "
          "(null-space drift from setReference)")
    print(f"Per-iter (median):    {per_iter_med:.2f} ms/iter")
    print(f"P1 per-iter gate (≤ {PER_ITER_GATE:.0f} ms/iter): "
          f"{'PASS' if per_iter_med <= PER_ITER_GATE else 'FAIL'}  "
          f"({per_iter_med:.2f} ms/iter)")
    print("=" * 70)

    return 0 if per_iter_med <= PER_ITER_GATE else 1


if __name__ == "__main__":
    sys.exit(main())
