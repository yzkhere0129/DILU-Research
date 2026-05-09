"""SuperLU/CHOLMOD direct solve on single_track_dump (real LPBF physics).

Verifies AMGx vs LU **across all 6 timesteps** of the working single-track
case (rays>0, real melt+vapor physics, well-conditioned matrix).

Compares 4 solvers per timestep:
  X_LU            = scipy.sparse.linalg.spsolve  (or CHOLMOD if available)
  X_OF            = OF DICPCG @ tol=1e-8 (read from x_final.mm)
  X_AMGx_e8       = AMGx PCG @ tol=1e-8
  X_AMGx_e12+IR   = AMGx PCG @ tol=1e-12 + 1 IR step (read from npz x_truth)

Output:
  - console table per timestep
  - dilu/amgx/bench/lu_truth_single_track_results.json

Run on dev (or lab Xeon for more RAM headroom):
    cd ~/DILU-Research && source ~/jax-env/bin/activate
    python3 -u -m dilu.amgx.bench.lu_truth_single_track \\
        --case ~/single_track_dump

Or on lab Xeon with case dir:
    python3 -u -m dilu.amgx.bench.lu_truth_single_track \\
        --case ~/cases/single_track_dump

Each timestep ~70s SuperLU (or ~5s CHOLMOD). 6 timesteps total ~7 min.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Optional CHOLMOD
try:
    from sksparse.cholmod import cholesky as cholmod_factor
    HAVE_CHOLMOD = True
except ImportError:
    HAVE_CHOLMOD = False


# Map timestep → phase tag (matches single_*.npz filenames)
PHASE_FOR_TIME = {
    "3.2e-07":  "melting",
    "3.8e-07":  "melting",
    "4.1e-07":  "melting",
    "7e-07":    "evap_early",
    "9e-07":    "evap",
    "1.06e-06": "evap_late",
}


def solve_lu(A_csr, b, prefer_cholmod=True):
    """Return (x_LU, wall_seconds, method_name)."""
    if HAVE_CHOLMOD and prefer_cholmod:
        try:
            t0 = time.time()
            factor = cholmod_factor(A_csr.tocsc())
            x = factor(b)
            return x, time.time() - t0, "CHOLMOD"
        except Exception as e:
            print(f"  CHOLMOD failed: {e}, falling back to SuperLU")
    t0 = time.time()
    x = spsolve(A_csr.tocsc(), b)
    return x, time.time() - t0, "SuperLU"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True,
                     help="path to single_track_dump (containing postProcessing/matrices)")
    ap.add_argument("--out-json",
                     default="dilu/amgx/bench/lu_truth_single_track_results.json")
    ap.add_argument("--npz-dir",
                     default="dilu/amgx/bench")
    args = ap.parse_args()

    case = Path(args.case).expanduser()
    matrices_root = case / "postProcessing" / "matrices"
    if not matrices_root.exists():
        raise SystemExit(f"Not found: {matrices_root}")

    npz_dir = Path(args.npz_dir).expanduser()

    timesteps = sorted(PHASE_FOR_TIME.keys(),
                        key=lambda t: float(t))   # 3.2e-07 first

    print(f"Case dir: {case}")
    print(f"npz dir : {npz_dir}")
    print(f"LU method: {'CHOLMOD' if HAVE_CHOLMOD else 'SuperLU'}")
    print(f"\nProcessing {len(timesteps)} timesteps × pd_corr0")
    print("=" * 100)

    results = []
    for t in timesteps:
        phase = PHASE_FOR_TIME[t]
        eq_dir = matrices_root / t / "pd_corr0"
        if not eq_dir.exists():
            print(f"\n  [skip] {t}: no dump dir")
            continue
        print(f"\n[{phase} / t={t} / pd_corr0]")

        # Read raw A, b, x_OF
        A = sio.mmread(str(eq_dir / "A.mm")).tocsr()
        b = sio.mmread(str(eq_dir / "b.mm")).flatten()
        x_OF = sio.mmread(str(eq_dir / "x_final.mm")).flatten()
        n = A.shape[0]
        print(f"  N={n}, nnz={A.nnz}")

        # Sign-flip (OF stores Laplacian with negative diag)
        diag = A.diagonal()
        all_neg = np.all(diag <= 0) and np.any(diag < 0)
        if all_neg:
            A_pos, b_pos = -A, -b
            print(f"  sign-flipped (positive diag)")
        else:
            A_pos, b_pos = A, b

        # Load AMGx solutions from npz
        npz_path = npz_dir / f"single_{phase}_pd_corr0_{t}.npz"
        if not npz_path.exists():
            print(f"  [warn] missing npz: {npz_path}")
            continue
        z = np.load(npz_path, allow_pickle=True)
        x_AMGx_e8 = z["x_AMGx_e8"]
        x_AMGx_e12 = z["x_truth"]
        meta = json.loads(str(z["meta"][0]))

        # LU direct solve
        print(f"  computing X_LU ...", flush=True)
        x_LU, wall_LU, method = solve_lu(A_pos, b_pos)
        res_LU = float(np.linalg.norm(A_pos @ x_LU - b_pos)
                        / max(np.linalg.norm(b_pos), 1e-300))
        print(f"  ✓ {method} done in {wall_LU:.1f}s, "
              f"‖A·x_LU - b‖/‖b‖ = {res_LU:.3e}")
        denom = max(float(np.abs(x_LU).max()), 1e-300)

        # Compare each solver to LU
        comparisons = {}
        for name, x in [("x_OF", x_OF), ("x_AMGx_e8", x_AMGx_e8),
                          ("x_AMGx_e12_IR", x_AMGx_e12)]:
            diff = np.abs(x - x_LU)
            comparisons[name] = {
                "max_diff_Pa": float(diff.max()),
                "rel_max": float(diff.max() / denom),
                "median_diff_Pa": float(np.median(diff)),
                "cells_above_100Pa": int((diff > 100).sum()),
                "cells_above_1kPa": int((diff > 1000).sum()),
            }

        # Console summary
        print(f"  ‖x_LU‖∞ = {denom:.3e} Pa")
        print(f"  {'name':<18s} {'max|diff|':>12s} {'rel':>10s} "
              f"{'median':>11s} {'>100Pa':>8s} {'>1kPa':>7s}")
        for name, r in comparisons.items():
            print(f"  {name:<18s} {r['max_diff_Pa']:>12.3e} "
                  f"{r['rel_max']:>10.2e} {r['median_diff_Pa']:>11.3e} "
                  f"{r['cells_above_100Pa']:>8d} {r['cells_above_1kPa']:>7d}")

        results.append({
            "timestep": t,
            "phase": phase,
            "n": int(n),
            "nnz": int(A.nnz),
            "lu_method": method,
            "lu_wall_s": wall_LU,
            "lu_rel_resid": res_LU,
            "x_LU_inf_norm": denom,
            "OF_dump_consistency": meta.get("rel_OF_consistency"),
            "amgx_truth_iters": meta["amgx_truth"]["iters"],
            "amgx_truth_rel_resid": meta["amgx_truth"]["rel_resid_actual"],
            "amgx_e8_iters": meta["amgx_e8"]["iters"],
            "amgx_e8_rel_resid": meta["amgx_e8"]["rel_resid_actual"],
            "comparisons_vs_LU": comparisons,
        })

    # Save JSON
    out_path = Path(args.out_json).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "case_dir": str(case),
        "lu_method": "CHOLMOD" if HAVE_CHOLMOD else "SuperLU",
        "n_timesteps": len(results),
        "results": results,
    }, indent=2))
    print(f"\n→ Wrote {out_path}")

    # Final cross-timestep summary
    print(f"\n{'='*100}")
    print(f"OVERALL SUMMARY (max over all {len(results)} timesteps)")
    print(f"{'='*100}")
    for solver in ["x_OF", "x_AMGx_e8", "x_AMGx_e12_IR"]:
        max_rel = max(r["comparisons_vs_LU"][solver]["rel_max"] for r in results)
        max_abs = max(r["comparisons_vs_LU"][solver]["max_diff_Pa"] for r in results)
        max_n100 = max(r["comparisons_vs_LU"][solver]["cells_above_100Pa"]
                        for r in results)
        print(f"  {solver:<18s}: max_rel={max_rel:.2e}, max_abs={max_abs:.2e} Pa, "
              f"max cells > 100 Pa = {max_n100}")

    print(f"\nVERDICT:")
    amgx_e12_max = max(r["comparisons_vs_LU"]["x_AMGx_e12_IR"]["rel_max"]
                        for r in results)
    if amgx_e12_max < 1e-7:
        print(f"  ★ AMGx + IR matches LU truth to rel<{amgx_e12_max:.0e} "
              f"across all {len(results)} timesteps — AMGx is verified correct.")
    else:
        print(f"  ⚠ AMGx + IR max rel diff vs LU = {amgx_e12_max:.2e} — "
              f"larger than expected, investigate.")


if __name__ == "__main__":
    main()
