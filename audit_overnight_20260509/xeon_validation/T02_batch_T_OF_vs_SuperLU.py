"""T02 — Batch SuperLU truth on T_corr0 matrices (non-symmetric, can't use CHOLMOD).

For each matrix in --pool:
  1. Load A.mm, b.mm, x_final.mm (x_OF)
  2. Compute x_LU via scipy.sparse.linalg.splu (SuperLU)
  3. Verify ‖A·x_LU - b‖ / ‖b‖ < 1e-12 (ε-machine truth)
  4. Compare x_OF vs x_LU: max|Δ|, rel, L2, cells > 0.1 K / > 1 K
  5. Save per-matrix JSON + x_LU npz

Output:
  <out>/aggregate.json                    overall stats over all matrices
  <out>/per_matrix/<ts>_T_corr0.json      per-matrix results
  <out>/x_LU_npz/x_LU_T_<ts>.npz          x_LU for downstream replay/diff plots

Different from E11 (which used CHOLMOD on SPD pd):
  - T is non-symmetric → must use general LU (splu), not CHOLMOD
  - SuperLU stores both L and U (no Cholesky shortcut) → ~2× factor cost
  - No supernodal optimization → ~10-15 min/matrix on 500K Xeon
  - No sign-flip needed (T diagonal is already positive)

Usage:
  # 6-matrix pilot
  PYTHONPATH=. python audit_overnight_20260509/xeon_validation/T02_batch_T_OF_vs_SuperLU.py \\
      --case ~/cases/single_track_dump \\
      --pool audit_overnight_20260509/xeon_validation/results/T01_single_track_T/sane_pool_T.txt \\
      --output-dir audit_overnight_20260509/xeon_validation/results/T02_single_track_T_LU \\
      --max 6

  # Full 384-matrix batch (after T01 on dense_track produces sane_pool_T)
  unset OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS  # let BLAS multi-thread if it can
  nohup ~/jax-env/bin/python3 audit_overnight_20260509/xeon_validation/T02_batch_T_OF_vs_SuperLU.py \\
      --case ~/cases/dense_track_dump_500K \\
      --pool audit_overnight_20260509/xeon_validation/results/T01_dense_T/sane_pool_T.txt \\
      --output-dir audit_overnight_20260509/xeon_validation/results/T02_dense_T_LU \\
      --max 384 \\
      > /tmp/T02_full.log 2>&1 &
  echo $! > /tmp/T02.pid
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def read_pool(pool_path: Path) -> list[str]:
    rels = []
    for line in pool_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rels.append(line)
    rels.sort(key=lambda s: float(s.split("/")[0]))
    return rels


def load_matrix(case: Path, rel: str):
    sys_dir = case / "postProcessing" / "matrices" / rel
    A = sio.mmread(str(sys_dir / "A.mm")).tocsr()
    b = np.asarray(sio.mmread(str(sys_dir / "b.mm"))).flatten()
    x_OF = np.asarray(sio.mmread(str(sys_dir / "x_final.mm"))).flatten()
    return A, b, x_OF


def superlu_solve(A_csr, b):
    """SuperLU LU factor + solve for non-symmetric matrix.
    Returns x_LU, dict of timings.
    """
    from scipy.sparse.linalg import splu

    A_csc = A_csr.tocsc()
    t0 = time.time()
    lu = splu(A_csc)
    t_factor = time.time() - t0

    t0 = time.time()
    x = lu.solve(b)
    t_solve = time.time() - t0

    return x, {"factor_s": t_factor, "solve_s": t_solve,
                "total_s": t_factor + t_solve}


def compare_x(x_OF, x_LU, A, b):
    diff = x_OF - x_LU
    abs_diff = np.abs(diff)
    L2_diff = float(np.linalg.norm(diff))
    L2_x = float(np.linalg.norm(x_LU))
    inf_x = float(np.max(np.abs(x_LU)))
    cmp = {
        "max_abs_diff_K": float(abs_diff.max()),
        "median_abs_diff": float(np.median(abs_diff)),
        "L2_diff": L2_diff,
        "rel_max":   float(abs_diff.max() / max(inf_x, 1e-300)),
        "rel_L2":    float(L2_diff / max(L2_x, 1e-300)),
        "cells_above_0_1K": int(np.sum(abs_diff > 0.1)),
        "cells_above_1K":   int(np.sum(abs_diff > 1.0)),
        "cells_above_10K":  int(np.sum(abs_diff > 10.0)),
        "cells_above_100K": int(np.sum(abs_diff > 100.0)),
        "x_LU_inf_norm": inf_x,
        "x_LU_L2_norm":  L2_x,
        "x_LU_min_K":    float(x_LU.min()),
        "x_LU_max_K":    float(x_LU.max()),
        "x_OF_rel_resid": float(np.linalg.norm(A @ x_OF - b) / max(np.linalg.norm(b), 1e-300)),
        "x_LU_rel_resid": float(np.linalg.norm(A @ x_LU - b) / max(np.linalg.norm(b), 1e-300)),
    }
    return cmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max", type=int, default=0,
                     help="cap number of matrices (0 = all in pool)")
    ap.add_argument("--seed", default="0xC0FFEE", help="seed for shuffle (hex or int)")
    ap.add_argument("--no-save-x-LU", action="store_true",
                     help="skip saving x_LU npz (just compare stats; saves disk)")
    args = ap.parse_args()

    case = Path(args.case).expanduser()
    pool = Path(args.pool).expanduser()
    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    per_dir = out / "per_matrix"
    per_dir.mkdir(exist_ok=True)
    if not args.no_save_x_LU:
        x_dir = out / "x_LU_npz"
        x_dir.mkdir(exist_ok=True)

    rels = read_pool(pool)
    if args.max > 0 and len(rels) > args.max:
        # shuffle for representative sample
        try:
            seed = int(args.seed, 16) if args.seed.startswith("0x") else int(args.seed)
        except ValueError:
            seed = hash(args.seed) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rels), size=args.max, replace=False)
        idx.sort()
        rels = [rels[i] for i in idx]

    print(f"# Batch SuperLU truth on {len(rels)} T matrices from {case}")
    print(f"# pool: {pool}")
    print(f"# OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS','unset')}  "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','unset')}  "
          f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS','unset')}")

    t_start = time.time()
    n_errors = 0
    rec_aggregate = []

    for i, rel in enumerate(rels):
        per_path = per_dir / f"{rel.replace('/', '_')}.json"
        if per_path.exists():
            try:
                rec = json.loads(per_path.read_text())
                print(f"  [{i+1}/{len(rels)}] {rel}: SKIP (already done)")
                rec_aggregate.append(rec)
                continue
            except Exception:
                pass  # re-do if json broken

        rec = {"timestep": rel.split("/")[0], "system": rel.split("/")[1] if "/" in rel else "T_corr0",
                "rel": rel}
        print(f"\n=== [{i+1}/{len(rels)}] {rel} ===")
        try:
            t0 = time.time()
            A, b, x_OF = load_matrix(case, rel)
            t_load = time.time() - t0
            rec["n"] = int(A.shape[0])
            rec["nnz"] = int(A.nnz)
            rec["load_s"] = t_load
            print(f"  load: {t_load:.1f}s  N={A.shape[0]} nnz={A.nnz}")

            x_LU, times = superlu_solve(A, b)
            rec.update({f"lu_{k}": v for k, v in times.items()})
            print(f"  factor: {times['factor_s']:.1f}s  solve: {times['solve_s']:.2f}s  "
                  f"total: {times['total_s']:.1f}s")

            cmp = compare_x(x_OF, x_LU, A, b)
            rec["comparison_OF_vs_LU"] = cmp
            print(f"  err_max: {cmp['max_abs_diff_K']:.3e} K  rel: {cmp['rel_max']:.3e}")
            print(f"  cells>0.1K: {cmp['cells_above_0_1K']}  >1K: {cmp['cells_above_1K']}  "
                  f">10K: {cmp['cells_above_10K']}  >100K: {cmp['cells_above_100K']}")
            print(f"  x_LU rel_resid: {cmp['x_LU_rel_resid']:.2e}  "
                  f"x_OF rel_resid: {cmp['x_OF_rel_resid']:.2e}")

            per_path.write_text(json.dumps(rec, indent=2))

            if not args.no_save_x_LU:
                np.savez_compressed(
                    out / "x_LU_npz" / f"x_LU_T_{rec['timestep']}.npz",
                    x_LU=x_LU,
                    rel_resid=cmp['x_LU_rel_resid'],
                    factor_s=times['factor_s'],
                    solve_s=times['solve_s'],
                    timestep=rec['timestep'],
                )

            rec_aggregate.append(rec)
            del A, b, x_OF, x_LU
            gc.collect()
        except Exception as e:
            n_errors += 1
            print(f"  ERROR: {type(e).__name__}: {e}")
            rec["error"] = f"{type(e).__name__}: {e}"
            per_path.write_text(json.dumps(rec, indent=2))

        if (i+1) % 5 == 0 or i+1 == len(rels):
            cum = time.time() - t_start
            eta_min = (cum / (i+1)) * (len(rels) - i - 1) / 60
            print(f"\n  PROGRESS: {i+1}/{len(rels)}  errors={n_errors}  "
                  f"cum={cum/60:.1f}min  eta={eta_min:.1f}min")

    # Aggregate
    if rec_aggregate:
        oks = [r for r in rec_aggregate if "comparison_OF_vs_LU" in r]
        if oks:
            cmps = [r["comparison_OF_vs_LU"] for r in oks]
            agg = {
                "case": str(case),
                "n_audited": len(oks),
                "n_errors": n_errors,
                "wall_s_total": time.time() - t_start,
                "wall_s_mean_per_matrix": (time.time() - t_start) / max(len(oks), 1),
                "OF_vs_LU_summary": {
                    "max_abs_diff_K": {
                        "min":    float(min(c["max_abs_diff_K"] for c in cmps)),
                        "max":    float(max(c["max_abs_diff_K"] for c in cmps)),
                        "mean":   float(np.mean([c["max_abs_diff_K"] for c in cmps])),
                        "median": float(np.median([c["max_abs_diff_K"] for c in cmps])),
                        "p95":    float(np.percentile([c["max_abs_diff_K"] for c in cmps], 95)),
                        "p99":    float(np.percentile([c["max_abs_diff_K"] for c in cmps], 99)),
                    },
                    "rel_max_stats": {
                        "min":    float(min(c["rel_max"] for c in cmps)),
                        "max":    float(max(c["rel_max"] for c in cmps)),
                        "mean":   float(np.mean([c["rel_max"] for c in cmps])),
                        "median": float(np.median([c["rel_max"] for c in cmps])),
                    },
                    "cells_above_0_1K_max":  int(max(c["cells_above_0_1K"]  for c in cmps)),
                    "cells_above_1K_max":    int(max(c["cells_above_1K"]    for c in cmps)),
                    "cells_above_10K_max":   int(max(c["cells_above_10K"]   for c in cmps)),
                    "matrices_any_above_1K": sum(1 for c in cmps if c["cells_above_1K"] > 0),
                    "x_OF_rel_resid_max":    float(max(c["x_OF_rel_resid"] for c in cmps)),
                    "x_LU_rel_resid_max":    float(max(c["x_LU_rel_resid"] for c in cmps)),
                },
            }
            (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
            print(f"\n=== AGGREGATE ===")
            print(f"  n_audited: {agg['n_audited']}  errors: {agg['n_errors']}")
            print(f"  err_max K: mean={agg['OF_vs_LU_summary']['max_abs_diff_K']['mean']:.2f}  "
                  f"max={agg['OF_vs_LU_summary']['max_abs_diff_K']['max']:.2f}  "
                  f"p99={agg['OF_vs_LU_summary']['max_abs_diff_K']['p99']:.2f}")
            print(f"  rel_max: median={agg['OF_vs_LU_summary']['rel_max_stats']['median']:.2e}  "
                  f"max={agg['OF_vs_LU_summary']['rel_max_stats']['max']:.2e}")
            print(f"  cells>1K max: {agg['OF_vs_LU_summary']['cells_above_1K_max']}, "
                  f"matrices_any: {agg['OF_vs_LU_summary']['matrices_any_above_1K']}/{len(cmps)}")
            print(f"  x_LU rel_resid max: {agg['OF_vs_LU_summary']['x_LU_rel_resid_max']:.2e} "
                  f"(should be < 1e-12 for truth)")
            print(f"\n→ {out}/aggregate.json")
            print(f"→ {out}/per_matrix/ ({len(rec_aggregate)} files)")


if __name__ == "__main__":
    main()
