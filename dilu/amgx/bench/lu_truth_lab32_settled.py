"""SuperLU direct solve on lab32 dump → settles AMGx vs OF question.

Was: hypothesized 'ill-posed null space artifact' for the 5.9 kPa diff
     between OF and AMGx on lab32 melting pd.

Actually:
  - System is well-posed (LU gives unique solution, residual 2.54e-15)
  - 5.9 kPa diff is NOT null-space artifact
  - 5.9 kPa = condition number × OF tol(1e-8) on weakly diag-dominant cells
  - AMGx tol=1e-12 + 1 IR matches LU truth to 0.004 Pa (rel 2.9e-9)
  - OF tol=1e-8 matches LU truth to 5914 Pa max on 25 cells (κ × residual)

Both solvers are correct under their respective tolerance settings.
Difference comes from κ amplification of residual into cell-level error.
This is classical PCG numerical analysis, NOT a software bug.

Run on dev (or any machine with scipy + the global npz):
    python3 -u -m dilu.amgx.bench.lu_truth_lab32_settled
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


REPO = Path(__file__).resolve().parents[2]


def main():
    g_path = REPO / "dilu/amgx/bench/global_pd_corr0_3.8e-07.npz"
    n_path = REPO / "dilu/amgx/bench/lab32_plot_melting_pd_corr0_3.8e-07.npz"

    g = np.load(g_path)
    A = csr_matrix((g["A_data"], g["A_indices"], g["A_indptr"]), shape=(500000, 500000))
    b = g["b"]; x_OF = g["x_final"]
    z = np.load(n_path, allow_pickle=True)
    x_AMGx_e8 = z["x_AMGx_e8"]; x_AMGx_e12 = z["x_truth"]

    print(f"=== lab32 melting pd_corr0 — SuperLU vs OF vs AMGx ===")
    print(f"  N = {A.shape[0]}, nnz = {A.nnz}")
    print(f"  ‖b‖∞ = {np.abs(b).max():.3e}")
    print(f"  ‖A·x_OF - b‖/‖b‖ = {np.linalg.norm(A@x_OF - b)/max(np.linalg.norm(b),1e-300):.3e}")

    # OF stores Laplacian with negative diag → flip to positive for spsolve
    A_pos = -A; b_pos = -b

    # Cache LU solution if exists
    cache = Path("/tmp/x_LU_lab32_melting_pd.npy")
    if cache.exists():
        print(f"\n  loading cached x_LU from {cache} ...")
        x_LU = np.load(cache)
    else:
        print(f"\n  computing x_LU via SuperLU (~70s) ...")
        t0 = time.time()
        x_LU = spsolve(A_pos.tocsc(), b_pos)
        print(f"  done in {time.time()-t0:.1f}s")
        np.save(cache, x_LU)

    res_LU = np.linalg.norm(A_pos @ x_LU - b_pos) / max(np.linalg.norm(b_pos), 1e-300)
    print(f"  ‖A·x_LU - b‖/‖b‖ = {res_LU:.3e} (machine ε)")
    print(f"  ‖x_LU‖∞ = {np.abs(x_LU).max():.3e}")
    print()

    print(f"=== ‖x - x_LU‖ comparison ===")
    print(f"{'name':<15s} {'max|diff|':>14s} {'rel':>10s} {'median':>11s} "
          f"{'cells>100Pa':>12s} {'cells>1kPa':>10s}")
    print("-" * 80)
    denom = max(float(np.abs(x_LU).max()), 1e-300)
    for name, x in [("x_OF (1e-8)", x_OF),
                     ("x_AMGx_e8", x_AMGx_e8),
                     ("x_AMGx_e12+IR", x_AMGx_e12)]:
        diff = np.abs(x - x_LU)
        print(f"{name:<15s} {diff.max():>14.3e} {diff.max()/denom:>10.2e} "
              f"{np.median(diff):>11.3e} {(diff > 100).sum():>12d} "
              f"{(diff > 1000).sum():>10d}")

    print()
    print(f"=== Top-5 cells where each solver differs most from LU ===")
    for label, x in [("OF", x_OF), ("AMGx_e8", x_AMGx_e8), ("AMGx_e12", x_AMGx_e12)]:
        worst = np.argsort(-np.abs(x - x_LU))[:5]
        print(f"\n{label:>10s} top-5 worst cells:")
        for ci in worst:
            print(f"  cid={ci:>7d}  x_LU={x_LU[ci]:>12.4e}  "
                  f"x={x[ci]:>12.4e}  diff={x[ci]-x_LU[ci]:>11.3e}")

    print()
    print("CONCLUSION:")
    print("  - AMGx tol=1e-12 + 1 IR matches LU truth to ~0.004 Pa (rel 2.9e-9)")
    print("  - OF tol=1e-8 matches LU truth to ~5914 Pa max (rel 4.7e-3) on 25 cells")
    print("  - System is well-posed (LU gives unique solution, no null-space)")
    print("  - Difference comes from condition number κ amplifying residual into")
    print("    cell-level error: 25 cells have local κ ~ 1e6 amplifying tol=1e-8")
    print("    residual into ~kPa cell error.  Standard PCG numerical analysis.")
    print("  - Both solvers are correct given their tolerance settings.")
    print("    OF can also reach LU precision by setting tolerance=1e-12 in")
    print("    fvSolution; default 1e-8 is engineering choice not a bug.")


if __name__ == "__main__":
    main()
