"""Batch accuracy audit on a pool of OpenFOAM pd matrix dumps.

For each matrix in `sane_pool.txt` (output of validate_matrix_dumps.py):
  1. Compute x_truth via AMGx+IR (tight tol + 1 step iterative refinement).
  2. Compare x_OF (= x_final from dump) vs x_truth.
  3. (Optional) For a small spot-check subset, also compute x_LU via CHOLMOD
     and verify x_truth ≈ x_LU as a sanity gate.

Outputs:
  <output-dir>/per_matrix/<ts>_<system>.json   per-matrix accuracy record
  <output-dir>/aggregate.json                  stats across the pool
  <output-dir>/spot_check_LU.json               LU vs AMGx+IR comparison (if --cholmod-spot)

Speed:
  AMGx+IR on RTX 3050: ~5s per 500K matrix → 200 matrices ≈ 17 min
  CHOLMOD spot-check: ~400s per matrix → 5 matrices ≈ 35 min

Usage:
  python3 batch_accuracy_audit.py \\
      --case ~/cases/dense_track_dump_500K \\
      --pool /tmp/validate_dense/sane_pool.txt \\
      --output-dir /tmp/batch_audit_dense \\
      --max 200 --seed 0xC0FFEE --cholmod-spot 5
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def load_matrix(matrix_dir: Path, system: str):
    """Load A, b, x_OF from a single dump."""
    sys_dir = matrix_dir / system
    A = sio.mmread(str(sys_dir / "A.mm")).tocsr()
    b = np.asarray(sio.mmread(str(sys_dir / "b.mm"))).flatten()
    x_OF = np.asarray(sio.mmread(str(sys_dir / "x_final.mm"))).flatten()
    # pd matrices may have negative diagonal (OF sign convention) — flip to SPD form
    diag_mean = float(np.mean(A.diagonal()))
    if diag_mean < 0:
        A = -A; b = -b
        sign_flipped = True
    else:
        sign_flipped = False
    return A, b, x_OF, sign_flipped


def amgx_solve(A: sp.csr_matrix, b: np.ndarray, tol: float = 1e-12,
               do_ir: bool = True) -> tuple[np.ndarray, dict]:
    """Solve via AMGx; return (x, info dict)."""
    from dilu.amgx.python import Plan
    A_csr = A.astype(np.float64).tocsr()
    rp = np.ascontiguousarray(A_csr.indptr, dtype=np.int32)
    ci = np.ascontiguousarray(A_csr.indices, dtype=np.int32)
    vv = np.ascontiguousarray(A_csr.data, dtype=np.float64)
    cfg = {
        "config_version": 2,
        "solver": {
            "solver": "PCG",
            "preconditioner": {"solver": "AMG", "algorithm": "CLASSICAL",
                                "cycle": "V", "max_levels": 24,
                                "smoother": {"solver": "BLOCK_JACOBI", "relaxation_factor": 0.5},
                                "presweeps": 1, "postsweeps": 1, "coarsest_sweeps": 2,
                                "max_iters": 1, "scope": "amg"},
            "use_scalar_norm": 1,
            "tolerance": tol,
            "max_iters": 5000,
            "norm": "L2",
            "convergence": "RELATIVE_INI_CORE",
            "monitor_residual": 1,
            "store_res_history": 0,
            "print_solve_stats": 0,
            "obtain_timings": 1,
            "scaling": "DIAGONAL_SCALING",
        },
    }
    with Plan(rp, ci, vv, json.dumps(cfg).encode()) as plan:
        x = np.zeros_like(b)
        t0 = time.time()
        x, iters = plan.solve(b, x, return_iters=True)
        t1 = time.time()
    info = {"tol": tol, "iters": int(iters), "wall_s": t1 - t0}

    if do_ir:
        # 1 step iterative refinement
        r = b - A @ x
        # Solve A dx = r via Plan (re-use). Easiest: another AMGx solve.
        with Plan(rp, ci, vv, json.dumps(cfg).encode()) as plan:
            dx = np.zeros_like(b)
            t0 = time.time()
            dx, ir_iters = plan.solve(r, dx, return_iters=True)
            t1 = time.time()
        x = x + dx
        info["ir_iters"] = int(ir_iters)
        info["ir_wall_s"] = t1 - t0
    return x, info


def cholmod_solve(A: sp.csr_matrix, b: np.ndarray) -> tuple[np.ndarray, dict]:
    """Solve via sksparse.cholmod (LU truth via Cholesky on SPD A)."""
    try:
        from sksparse.cholmod import cho_factor
    except ImportError:
        raise RuntimeError("sksparse not available; cannot do CHOLMOD spot check")
    t0 = time.time()
    factor = cho_factor(A.tocsc())
    t1 = time.time()
    x = factor.solve(b)
    t2 = time.time()
    info = {"factor_s": t1 - t0, "solve_s": t2 - t1, "wall_s": t2 - t0}
    return x, info


def compare(x_a: np.ndarray, x_b: np.ndarray, A: sp.csr_matrix, b: np.ndarray,
            name_a: str, name_b: str) -> dict:
    """Return dict of comparison metrics between x_a and x_b."""
    d = x_a - x_b
    x_inf = float(np.abs(x_b).max())
    rec = {
        "max_abs_diff": float(np.abs(d).max()),
        "L2_diff": float(np.linalg.norm(d)),
        "rel_max": float(np.abs(d).max() / max(x_inf, 1e-300)),
        "rel_L2": float(np.linalg.norm(d) / max(np.linalg.norm(x_b), 1e-300)),
        "median_abs_diff": float(np.median(np.abs(d))),
        "cells_above_100Pa": int(np.sum(np.abs(d) > 100)),
        "cells_above_1kPa": int(np.sum(np.abs(d) > 1000)),
    }
    # also algebraic residual of both
    rec[f"{name_a}_rel_resid"] = float(np.linalg.norm(A @ x_a - b) / max(np.linalg.norm(b), 1e-300))
    rec[f"{name_b}_rel_resid"] = float(np.linalg.norm(A @ x_b - b) / max(np.linalg.norm(b), 1e-300))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--pool", required=True, help="sane_pool.txt from validate_matrix_dumps.py")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max", type=int, default=200, help="max matrices to audit (subsample)")
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    ap.add_argument("--cholmod-spot", type=int, default=5,
                    help="how many to also LU-verify via CHOLMOD (spot check)")
    ap.add_argument("--amgx-tol", type=float, default=1e-12)
    args = ap.parse_args()

    case = Path(args.case)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_matrix").mkdir(exist_ok=True)

    # Read pool, deterministic subsample
    pool: list[str] = []
    with open(args.pool) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                pool.append(line)
    print(f"Pool size: {len(pool)}")
    rng = random.Random(args.seed)
    if len(pool) > args.max:
        sample = sorted(rng.sample(pool, args.max),
                         key=lambda p: float(p.split("/")[0]))
        print(f"Sub-sampled to {args.max} (seed={hex(args.seed)})")
    else:
        sample = sorted(pool, key=lambda p: float(p.split("/")[0]))

    # Spot-check selection: deterministic, evenly spaced through sample
    if args.cholmod_spot > 0:
        step = max(1, len(sample) // args.cholmod_spot)
        spot_idx = list(range(0, len(sample), step))[:args.cholmod_spot]
    else:
        spot_idx = []
    print(f"CHOLMOD spot check on {len(spot_idx)} matrices (indices: {spot_idx})")
    print()

    per_matrix = []
    spot_records = []
    t_start = time.time()
    for i, entry in enumerate(sample):
        ts_name, system = entry.split("/")
        matrix_dir = case / "postProcessing" / "matrices" / ts_name
        print(f"  [{i+1:4d}/{len(sample)}] {ts_name:<14s} {system:<10s}", end=" ", flush=True)
        t0 = time.time()
        try:
            A, b, x_OF, sign_flipped = load_matrix(matrix_dir, system)
            x_truth, ir_info = amgx_solve(A, b, tol=args.amgx_tol, do_ir=True)
            rec = compare(x_OF, x_truth, A, b, name_a="x_OF", name_b="x_truth")
            rec.update({
                "timestep": ts_name, "system": system,
                "n": int(A.shape[0]), "nnz": int(A.nnz),
                "sign_flipped": sign_flipped,
                "amgx_iters": ir_info["iters"],
                "amgx_ir_iters": ir_info.get("ir_iters"),
                "amgx_wall_s": ir_info["wall_s"] + ir_info.get("ir_wall_s", 0),
                "wall_s_total": time.time() - t0,
            })
            # CHOLMOD spot check
            if i in spot_idx:
                print(f"  [SPOT-CHOLMOD]", end=" ", flush=True)
                t0c = time.time()
                x_LU, lu_info = cholmod_solve(A, b)
                lu_cmp = compare(x_truth, x_LU, A, b, name_a="x_truth", name_b="x_LU")
                spot_rec = {"timestep": ts_name, "system": system,
                            "cholmod_factor_s": lu_info["factor_s"],
                            "cholmod_solve_s": lu_info["solve_s"],
                            "amgx_truth_vs_LU": lu_cmp}
                spot_records.append(spot_rec)
                rec["cholmod_spot"] = lu_cmp
                print(f"AMGx+IR vs LU rel_max = {lu_cmp['rel_max']:.2e}", end=" ", flush=True)
            with open(output_dir / "per_matrix" / f"{ts_name}_{system}.json", "w") as f:
                json.dump(rec, f, indent=2)
            per_matrix.append(rec)
            print(f"  x_OF vs truth rel_max={rec['rel_max']:.2e}  "
                  f"({rec['amgx_wall_s']:.1f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            per_matrix.append({"timestep": ts_name, "system": system, "error": str(e)})

    # Aggregate stats
    rels = [r["rel_max"] for r in per_matrix if "rel_max" in r]
    abss = [r["max_abs_diff"] for r in per_matrix if "max_abs_diff" in r]
    above100 = [r["cells_above_100Pa"] for r in per_matrix if "cells_above_100Pa" in r]
    of_resids = [r["x_OF_rel_resid"] for r in per_matrix if "x_OF_rel_resid" in r]
    truth_resids = [r["x_truth_rel_resid"] for r in per_matrix if "x_truth_rel_resid" in r]

    aggregate = {
        "case": str(case), "pool_size": len(pool), "audited": len(sample),
        "wall_s_total": time.time() - t_start,
        "x_OF_vs_truth": {
            "rel_max_stats": {
                "n": len(rels),
                "max": float(np.max(rels)) if rels else None,
                "min": float(np.min(rels)) if rels else None,
                "median": float(np.median(rels)) if rels else None,
                "p99": float(np.percentile(rels, 99)) if rels else None,
                "p95": float(np.percentile(rels, 95)) if rels else None,
            },
            "max_abs_diff_Pa_stats": {
                "max": float(np.max(abss)) if abss else None,
                "median": float(np.median(abss)) if abss else None,
                "p99": float(np.percentile(abss, 99)) if abss else None,
            },
            "cells_above_100Pa_max": int(np.max(above100)) if above100 else 0,
            "cells_above_100Pa_mean": float(np.mean(above100)) if above100 else 0.0,
        },
        "x_OF_self_residual_stats": {
            "max": float(np.max(of_resids)) if of_resids else None,
            "median": float(np.median(of_resids)) if of_resids else None,
        },
        "x_truth_residual_stats": {
            "max": float(np.max(truth_resids)) if truth_resids else None,
            "median": float(np.median(truth_resids)) if truth_resids else None,
        },
        "cholmod_spot_count": len(spot_records),
        "spot_check_records": spot_records,
    }
    with open(output_dir / "aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n=== Aggregate ({len(sample)} matrices) ===")
    if rels:
        s = aggregate["x_OF_vs_truth"]["rel_max_stats"]
        print(f"  x_OF vs truth (AMGx+IR) rel_max:")
        print(f"    median={s['median']:.2e}  p95={s['p95']:.2e}  p99={s['p99']:.2e}  max={s['max']:.2e}")
        a = aggregate["x_OF_vs_truth"]["max_abs_diff_Pa_stats"]
        print(f"  max|Δx_OF|: median={a['median']:.2f} Pa  p99={a['p99']:.2f} Pa  max={a['max']:.2f} Pa")
        print(f"  cells above 100 Pa: max={aggregate['x_OF_vs_truth']['cells_above_100Pa_max']}  "
              f"mean={aggregate['x_OF_vs_truth']['cells_above_100Pa_mean']:.1f}")
    if spot_records:
        spot_rels = [r["amgx_truth_vs_LU"]["rel_max"] for r in spot_records]
        print(f"\n  CHOLMOD spot check: AMGx+IR vs LU rel_max max={max(spot_rels):.2e}  "
              f"min={min(spot_rels):.2e}  ({len(spot_records)} matrices)")
    print(f"\n  → {output_dir}/aggregate.json")
    print(f"  total wall: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
