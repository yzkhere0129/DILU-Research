"""E09 — Lanczos κ(A) for 6 single_track + 1 lab32 matrices.

For each:
  σ_max = eigsh(A_pos, k=1, which='LA')
  σ_min = eigsh(A_pos, k=1, sigma=0, which='LM')   # shift-invert
  κ = σ_max / σ_min
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    record_environment, write_result_json,
    normalize_sign, load_raw_matrix, find_npz,
)

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def estimate_kappa(A_pos, tol=1e-3, maxiter=200):
    """Return (sigma_max, sigma_min, kappa) via Lanczos.

    Note: per H11/A021 finding, sigma_min from Lanczos shift-invert can converge
    to fp64 noise floor on near-singular matrices. Cross-check with smallest_10
    via svds (P8 closeout upgrade).
    """
    sigma_max = float(eigsh(A_pos, k=1, which='LA',
                             return_eigenvectors=False,
                             tol=tol, maxiter=maxiter)[0])
    try:
        sigma_min = float(eigsh(A_pos, k=1, sigma=0, which='LM',
                                 return_eigenvectors=False,
                                 tol=tol, maxiter=maxiter)[0])
    except Exception as e:
        # fallback to SA
        sigma_min = float(eigsh(A_pos, k=1, which='SA',
                                 return_eigenvectors=False,
                                 tol=tol, maxiter=maxiter)[0])
    kappa = abs(sigma_max / max(abs(sigma_min), 1e-300))
    return sigma_max, sigma_min, kappa


def smallest_k_singulars(A_pos, k=10, tol=1e-3):
    """P8: svds k=10 to characterize near-null structure.

    Returns sorted ascending list of k smallest singular values, plus
    a 'has_near_null' boolean (True if smallest 5 all < 1e-12).
    """
    from scipy.sparse.linalg import svds
    try:
        _, sv, _ = svds(A_pos.tocsr(), k=k, which='SM', tol=tol)
        sv_sorted = sorted([float(s) for s in sv])
        has_near_null = bool(all(s < 1e-12 for s in sv_sorted[:5]))
        return sv_sorted, has_near_null
    except Exception as e:
        return [None] * k, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(args.npz_dir)

    case_dir_candidates = [
        Path("/home/yzk/cases/single_track_dump"),
        Path("/home/yzk/single_track_dump"),
    ]
    case_dir = next((c for c in case_dir_candidates
                     if (c / "postProcessing/matrices").exists()), None)
    if case_dir is None:
        sys.exit(f"FAIL: no case dir found")

    timesteps = find_npz(npz_dir)
    env = record_environment()
    summary = []

    # 6 single_track
    for time_str, phase, _ in timesteps:
        rep_dir = output_dir / f"single_track_{time_str}"
        result_path = rep_dir / "result.json"
        if args.resume and result_path.exists():
            print(f"single_track {time_str}: SKIP"); continue

        A, b, _ = load_raw_matrix(case_dir, time_str)
        A_pos, _, _ = normalize_sign(A, b)
        print(f"\nsingle_track {phase} t={time_str}: estimating κ + smallest 10 σ ...")
        t0 = time.time()
        sigma_max, sigma_min, kappa = estimate_kappa(A_pos)
        wall_lanczos = time.time() - t0
        # P8 closeout: also run svds k=10 for near-null subspace characterization
        t1 = time.time()
        smallest_10, has_near_null = smallest_k_singulars(A_pos, k=10)
        wall_svds = time.time() - t1
        diag = A_pos.diagonal()
        record = {
            "expt_id": "E09",
            "case": "single_track",
            "phase": phase,
            "timestep": time_str,
            "n": int(A_pos.shape[0]),
            "nnz": int(A_pos.nnz),
            "sigma_max_lanczos": sigma_max,
            "sigma_min_lanczos": sigma_min,
            "kappa_lanczos": kappa,
            "smallest_10_sigmas_svds": smallest_10,
            "has_near_null_subspace": has_near_null,
            "near_null_dim_estimate": sum(1 for s in smallest_10 if s is not None and s < 1e-12),
            "diag_max": float(np.abs(diag).max()),
            "diag_min": float(np.abs(diag).min()),
            "diag_spread": float(np.abs(diag).max() / max(np.abs(diag).min(), 1e-300)),
            "lanczos_wall_s": wall_lanczos,
            "svds_wall_s": wall_svds,
            "env": env,
        }
        write_result_json(result_path, record)
        summary.append(record)
        print(f"  Lanczos: σ_max={sigma_max:.3e}  σ_min={sigma_min:.3e}  κ={kappa:.3e}  ({wall_lanczos:.1f}s)")
        print(f"  svds:   smallest_10 σ ∈ [{smallest_10[0]:.2e}, {smallest_10[-1]:.2e}]  near_null={has_near_null}  ({wall_svds:.1f}s)")
        if args.smoke: break

    # lab32 matrix (already cached as global_pd_corr0_3.8e-07.npz)
    lab32_path = Path("/home/yzk/DILU-Research/dilu/amgx/bench/global_pd_corr0_3.8e-07.npz")
    if lab32_path.exists() and not args.smoke:
        rep_dir = output_dir / "lab32_melt_380ns"
        result_path = rep_dir / "result.json"
        if not (args.resume and result_path.exists()):
            g = np.load(lab32_path)
            A_lab32 = csr_matrix((g["A_data"], g["A_indices"], g["A_indptr"]),
                                  shape=(500000, 500000))
            A_lab32_pos = -A_lab32
            print(f"\nlab32 melt-380ns: estimating κ + smallest 10 σ ...")
            t0 = time.time()
            sigma_max, sigma_min, kappa = estimate_kappa(A_lab32_pos)
            wall_lanczos = time.time() - t0
            t1 = time.time()
            smallest_10, has_near_null = smallest_k_singulars(A_lab32_pos, k=10)
            wall_svds = time.time() - t1
            record = {
                "expt_id": "E09",
                "case": "lab32",
                "phase": "melting",
                "timestep": "3.8e-07",
                "n": 500000,
                "nnz": int(A_lab32_pos.nnz),
                "sigma_max_lanczos": sigma_max,
                "sigma_min_lanczos": sigma_min,
                "kappa_lanczos": kappa,
                "smallest_10_sigmas_svds": smallest_10,
                "has_near_null_subspace": has_near_null,
                "near_null_dim_estimate": sum(1 for s in smallest_10 if s is not None and s < 1e-12),
                "lanczos_wall_s": wall_lanczos,
                "svds_wall_s": wall_svds,
                "env": env,
            }
            write_result_json(result_path, record)
            summary.append(record)
            print(f"  Lanczos: σ_max={sigma_max:.3e}  σ_min={sigma_min:.3e}  κ={kappa:.3e}  ({wall_lanczos:.1f}s)")
            print(f"  svds:   smallest_10 σ ∈ [{smallest_10[0]:.2e}, {smallest_10[-1]:.2e}]  near_null={has_near_null}  ({wall_svds:.1f}s)")

    # Summary
    aggr_path = output_dir / "aggregate.json"
    with open(aggr_path, "w") as f:
        json.dump({"expt_id": "E09", "n_matrices": len(summary), "results": summary}, f, indent=2)
    print(f"\n[E09] aggregate → {aggr_path}")
    print("\nκ + near-null summary:")
    print(f"{'case':<20s} {'phase':<11s} {'t':<11s} {'σ_max':>11s} {'σ_min(L)':>11s} {'κ(L)':>11s} {'near-null':>11s}")
    for r in summary:
        print(f"{r['case']:<20s} {r['phase']:<11s} {r['timestep']:<11s} "
              f"{r['sigma_max_lanczos']:>11.3e} {r['sigma_min_lanczos']:>11.3e} "
              f"{r['kappa_lanczos']:>11.3e} {str(r['has_near_null_subspace']):>11s}")


if __name__ == "__main__":
    main()
