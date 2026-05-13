"""Run CHOLMOD on 6 single_track_dump pd_corr0 matrices, save x_LU as .npy.

For each of 6 timesteps:
  Load A.mm + b.mm from /home/yzk/cases/single_track_dump/postProcessing/matrices
  Run CHOLMOD (multi-thread auto)
  Save x_LU.npy + meta

Output: audit_overnight_20260509/xeon_validation/results/E12_LU_single_track/x_LU_<ts>.npz
  contains: x_LU (500000,), residual, factor_s, solve_s, sign_flipped

Total wall: 6 × ~5 min = ~30 min on Xeon (multi-thread auto).
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import scipy.io as sio

CASE = Path("/home/yzk/cases/single_track_dump")
OUT = Path("/home/yzk/DILU-Research/audit_overnight_20260509/xeon_validation/results/E12_LU_single_track")
OUT.mkdir(parents=True, exist_ok=True)

TIMESTEPS = ["3.2e-07", "3.8e-07", "4.1e-07", "7e-07", "9e-07", "1.06e-06"]


def main():
    from sksparse.cholmod import cho_factor

    for ts in TIMESTEPS:
        out_path = OUT / f"x_LU_{ts}.npz"
        if out_path.exists():
            print(f"  {ts}: SKIP (exists)")
            continue

        sys_dir = CASE / "postProcessing" / "matrices" / ts / "pd_corr0"
        if not sys_dir.is_dir():
            print(f"  {ts}: missing dir {sys_dir}")
            continue

        print(f"\n=== {ts} ===")
        t0 = time.time()
        A = sio.mmread(str(sys_dir / "A.mm")).tocsr()
        b = np.asarray(sio.mmread(str(sys_dir / "b.mm"))).flatten()
        print(f"  load: {time.time()-t0:.1f}s  N={A.shape[0]}  nnz={A.nnz}")

        diag_mean = float(np.mean(A.diagonal()))
        sign_flipped = diag_mean < 0
        if sign_flipped:
            A = -A; b = -b
            print(f"  sign-flipped (diag_mean was {diag_mean:.2e})")

        t0 = time.time()
        factor = cho_factor(A.tocsc())
        t_fact = time.time() - t0

        t0 = time.time()
        x_LU = factor.solve(b)
        t_solve = time.time() - t0

        resid = float(np.linalg.norm(A @ x_LU - b) / max(np.linalg.norm(b), 1e-300))
        print(f"  factor: {t_fact:.1f}s  solve: {t_solve:.2f}s  resid: {resid:.2e}")

        np.savez_compressed(
            out_path,
            x_LU=x_LU,
            sign_flipped=sign_flipped,
            factor_s=t_fact,
            solve_s=t_solve,
            rel_resid=resid,
            timestep=ts,
        )
        print(f"  → {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")

    print(f"\nAll done. Files in {OUT}")
    print(f"To bring to dev: git add this dir, commit, push, then pull on dev.")


if __name__ == "__main__":
    main()
