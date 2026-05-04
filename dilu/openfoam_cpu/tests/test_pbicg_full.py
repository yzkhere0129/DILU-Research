"""Minimum viable end-to-end test:
load a dumped (A, b, x0, x_final), run our PBiCG-DILU, and require
    (i)  iter count == OpenFOAM iter count
    (ii) ||x_ours - x_OF||_inf / ||x_OF||_inf < 1e-10

Run: python -m dilu.openfoam_cpu.tests.test_pbicg_full [<case_dir>]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from dilu.openfoam_cpu.python.io import load_case
from dilu.openfoam_cpu.python.ldu import csr_to_ldu
from dilu.openfoam_cpu.python.pbicg import solve

DEFAULT_CASE = Path(
    "/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_sanity/"
    "postProcessing/matrices/2.636507509e-12/T_corr0"
)


def main(case_dir: Path) -> int:
    print(f"Loading {case_dir}")
    t0 = time.time()
    case = load_case(case_dir)
    print(f"  load:        {time.time()-t0:.2f}s  (A: {case.A.shape}, nnz={case.A.nnz})")

    of_meta = case.meta["solver_openfoam"]
    print(f"  OpenFOAM:    iter={of_meta['iterations']}  "
          f"r0={of_meta['initial_residual']:.4e}  "
          f"rN={of_meta['final_residual']:.4e}")

    t0 = time.time()
    ldu = csr_to_ldu(case.A)
    print(f"  csr_to_ldu:  {time.time()-t0:.2f}s  "
          f"(nFaces={ldu.n_faces}, nCells={ldu.n_cells})")

    if case.x0 is None:
        print("  ERROR: this case has no x0.mm — cannot reproduce OpenFOAM iter count")
        return 2
    if case.x_final is None:
        print("  ERROR: this case has no x_final — cannot validate solution")
        return 2

    t0 = time.time()
    perf = solve(ldu, case.b, case.x0,
                 tolerance=1e-12, rel_tol=0.0, min_iter=1, max_iter=1000,
                 sumA_override=case.sumA, nf_override=case.norm_factor)
    elapsed = time.time() - t0
    sa_src = "from sumA.mm" if case.sumA is not None else "self-computed"
    nf_src = (f"from normFactor.txt = {case.norm_factor:.10e}"
              if case.norm_factor is not None else "self-computed (likely off due to FP cancellation)")
    print(f"  sumA source: {sa_src}")
    print(f"  nF   source: {nf_src}")

    rel_err = float(np.abs(perf.x - case.x_final).max() /
                     max(np.abs(case.x_final).max(), 1e-300))

    print(f"  Ours:        iter={perf.n_iterations}  "
          f"r0={perf.initial_residual:.4e}  "
          f"rN={perf.final_residual:.4e}  "
          f"converged={perf.converged}  "
          f"singular={perf.singular}")
    print(f"  Solve time:  {elapsed:.2f}s")
    print(f"  ||x_ours - x_OF||_inf / ||x_OF||_inf = {rel_err:.3e}")

    iter_match = (perf.n_iterations == of_meta["iterations"])
    sol_match = (rel_err < 1e-10)
    print()
    print("=" * 50)
    print(f"  iter match : {'PASS' if iter_match else 'FAIL'}  "
          f"({perf.n_iterations} vs {of_meta['iterations']})")
    print(f"  solution   : {'PASS' if sol_match else 'FAIL'}  "
          f"(rel_err={rel_err:.2e}, threshold=1e-10)")
    print("=" * 50)

    return 0 if (iter_match and sol_match) else 1


if __name__ == "__main__":
    case = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CASE
    sys.exit(main(case))
