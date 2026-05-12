"""E11 — Batch OF-vs-LU truth comparison on ~200 validated matrices.

For each matrix in sane_pool.txt:
  1. Load A.mm, b.mm, x_final.mm (x_OF) from <case>/postProcessing/matrices/<ts>/pd_corr0/
  2. Compute x_LU via sksparse.cholmod (MULTI-THREAD — no BLAS pin)
  3. Compare x_OF vs x_LU: max|Δ|, rel, L2, cells > 100Pa / 1kPa

Different from E07 which used 6 single_track matrices — E11 uses ~200 from
dense_track_dump_500K to give statistically-defensible OF accuracy claim.

Speed: multi-thread CHOLMOD on 56-core Xeon expected 30-60 s/matrix
       200 matrices ≈ 2-3 hours total wall.

Usage on Xeon:
  unset OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS    # enable multi-thread
  ~/jax-env/bin/python3 audit_overnight_20260509/xeon_validation/E11_batch_OF_vs_LU.py \\
      --case ~/cases/dense_track_dump_500K \\
      --pool audit_overnight_20260509/xeon_validation/results/E10_validate_dense_pd/sane_pool.txt \\
      --output-dir audit_overnight_20260509/xeon_validation/results/E11 \\
      --max 200 --seed 0xC0FFEE
"""
from __future__ import annotations
import argparse
import gc
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
    sys_dir = matrix_dir / system
    A = sio.mmread(str(sys_dir / "A.mm")).tocsr()
    b = np.asarray(sio.mmread(str(sys_dir / "b.mm"))).flatten()
    x_OF = np.asarray(sio.mmread(str(sys_dir / "x_final.mm"))).flatten()
    # pd matrices may have negative diagonal — flip for SPD
    diag_mean = float(np.mean(A.diagonal()))
    sign_flipped = diag_mean < 0
    if sign_flipped:
        A = -A; b = -b
    return A, b, x_OF, sign_flipped


def cholmod_solve(A, b):
    from sksparse.cholmod import cho_factor
    t0 = time.time()
    factor = cho_factor(A.tocsc())
    t1 = time.time()
    x = factor.solve(b)
    t2 = time.time()
    return x, {"factor_s": t1 - t0, "solve_s": t2 - t1, "wall_s": t2 - t0}


def compare(x_OF, x_LU, A, b):
    d = x_OF - x_LU
    x_LU_inf = float(np.abs(x_LU).max())
    return {
        "max_abs_diff_Pa": float(np.abs(d).max()),
        "rel_max": float(np.abs(d).max() / max(x_LU_inf, 1e-300)),
        "L2_diff": float(np.linalg.norm(d)),
        "rel_L2": float(np.linalg.norm(d) / max(np.linalg.norm(x_LU), 1e-300)),
        "median_abs_diff": float(np.median(np.abs(d))),
        "cells_above_100Pa": int(np.sum(np.abs(d) > 100)),
        "cells_above_1kPa": int(np.sum(np.abs(d) > 1000)),
        "cells_above_10kPa": int(np.sum(np.abs(d) > 10000)),
        "x_LU_inf_norm": x_LU_inf,
        "x_LU_L2_norm": float(np.linalg.norm(x_LU)),
        "x_OF_rel_resid": float(np.linalg.norm(A @ x_OF - b) / max(np.linalg.norm(b), 1e-300)),
        "x_LU_rel_resid": float(np.linalg.norm(A @ x_LU - b) / max(np.linalg.norm(b), 1e-300)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=0xC0FFEE)
    ap.add_argument("--resume", action="store_true",
                    help="skip matrices already with result.json (resume after crash)")
    args = ap.parse_args()

    case = Path(args.case)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    per_matrix_dir = output_dir / "per_matrix"; per_matrix_dir.mkdir(exist_ok=True)

    # Pool
    pool = []
    with open(args.pool) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                pool.append(line)
    print(f"Pool size: {len(pool)}")

    # Subsample
    rng = random.Random(args.seed)
    if len(pool) > args.max:
        sample = sorted(rng.sample(pool, args.max), key=lambda p: float(p.split("/")[0]))
        print(f"Sub-sampled to {args.max} (seed={hex(args.seed)})")
    else:
        sample = sorted(pool, key=lambda p: float(p.split("/")[0]))
    print(f"Will audit {len(sample)} matrices on case {case}")
    print(f"Thread config: OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS','unset')}  "
          f"OMP={os.environ.get('OMP_NUM_THREADS','unset')}  "
          f"MKL={os.environ.get('MKL_NUM_THREADS','unset')}")
    print()

    per_matrix = []
    t_start = time.time()
    for i, entry in enumerate(sample):
        ts_name, system = entry.split("/")
        result_path = per_matrix_dir / f"{ts_name}_{system}.json"
        if args.resume and result_path.exists():
            try:
                per_matrix.append(json.load(open(result_path)))
                print(f"  [{i+1:4d}/{len(sample)}] {ts_name:<14s} SKIP (resume)")
                continue
            except Exception:
                pass

        matrix_dir = case / "postProcessing" / "matrices" / ts_name
        try:
            t_load0 = time.time()
            A, b, x_OF, sign_flipped = load_matrix(matrix_dir, system)
            t_load = time.time() - t_load0
            gc.collect()

            x_LU, info = cholmod_solve(A, b)
            cmp = compare(x_OF, x_LU, A, b)

            rec = {
                "timestep": ts_name, "system": system,
                "n": int(A.shape[0]), "nnz": int(A.nnz),
                "sign_flipped": sign_flipped,
                "load_s": t_load,
                "lu_factor_s": info["factor_s"],
                "lu_solve_s": info["solve_s"],
                "lu_wall_s": info["wall_s"],
                "comparison_OF_vs_LU": cmp,
            }
            with open(result_path, "w") as f: json.dump(rec, f, indent=2)
            per_matrix.append(rec)

            elapsed = time.time() - t_start
            eta = (elapsed / (i+1)) * (len(sample) - i - 1) if i > 0 else 0
            print(f"  [{i+1:4d}/{len(sample)}] {ts_name:<14s}  "
                  f"factor={info['factor_s']:.1f}s  solve={info['solve_s']:.2f}s  "
                  f"max|Δ|={cmp['max_abs_diff_Pa']:.1f}Pa  rel={cmp['rel_max']:.2e}  "
                  f">100Pa={cmp['cells_above_100Pa']}  [ETA {eta/60:.0f}min]")
        except Exception as e:
            print(f"  [{i+1:4d}/{len(sample)}] {ts_name:<14s}  ERROR: {e}")
            per_matrix.append({"timestep": ts_name, "system": system, "error": str(e)})

    # Aggregate
    valid = [r for r in per_matrix if "comparison_OF_vs_LU" in r]
    if valid:
        rels = np.array([r["comparison_OF_vs_LU"]["rel_max"] for r in valid])
        abss = np.array([r["comparison_OF_vs_LU"]["max_abs_diff_Pa"] for r in valid])
        c100 = np.array([r["comparison_OF_vs_LU"]["cells_above_100Pa"] for r in valid])
        c1k  = np.array([r["comparison_OF_vs_LU"]["cells_above_1kPa"] for r in valid])
        c10k = np.array([r["comparison_OF_vs_LU"]["cells_above_10kPa"] for r in valid])
        of_rs = np.array([r["comparison_OF_vs_LU"]["x_OF_rel_resid"] for r in valid])
        lu_rs = np.array([r["comparison_OF_vs_LU"]["x_LU_rel_resid"] for r in valid])
        walls = np.array([r["lu_wall_s"] for r in valid])

        agg = {
            "case": str(case),
            "n_audited": len(valid),
            "n_errors": len(per_matrix) - len(valid),
            "wall_s_total": time.time() - t_start,
            "wall_s_mean_per_matrix": float(walls.mean()),
            "thread_config": {
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "unset"),
            },
            "OF_vs_LU_summary": {
                "rel_max_stats": {
                    "n": len(rels),
                    "min": float(rels.min()), "max": float(rels.max()),
                    "mean": float(rels.mean()), "median": float(np.median(rels)),
                    "p95": float(np.percentile(rels, 95)),
                    "p99": float(np.percentile(rels, 99)),
                },
                "max_abs_diff_Pa_stats": {
                    "min": float(abss.min()), "max": float(abss.max()),
                    "mean": float(abss.mean()), "median": float(np.median(abss)),
                    "p99": float(np.percentile(abss, 99)),
                },
                "cells_above_100Pa": {"max": int(c100.max()), "mean": float(c100.mean()),
                                       "matrices_with_any": int(np.sum(c100 > 0))},
                "cells_above_1kPa":  {"max": int(c1k.max()), "mean": float(c1k.mean()),
                                       "matrices_with_any": int(np.sum(c1k > 0))},
                "cells_above_10kPa": {"max": int(c10k.max()), "mean": float(c10k.mean()),
                                       "matrices_with_any": int(np.sum(c10k > 0))},
                "x_OF_rel_resid_max": float(of_rs.max()),
                "x_LU_rel_resid_max": float(lu_rs.max()),
            }
        }
        with open(output_dir / "aggregate.json", "w") as f:
            json.dump(agg, f, indent=2)

        print(f"\n=== E11 aggregate ({len(valid)} matrices) ===")
        s = agg["OF_vs_LU_summary"]["rel_max_stats"]
        print(f"  OF vs LU rel_max:   median={s['median']:.2e}  p95={s['p95']:.2e}  "
              f"p99={s['p99']:.2e}  max={s['max']:.2e}")
        a = agg["OF_vs_LU_summary"]["max_abs_diff_Pa_stats"]
        print(f"  max|Δ|:             median={a['median']:.2f}Pa  p99={a['p99']:.2f}Pa  "
              f"max={a['max']:.2f}Pa")
        print(f"  cells >100Pa:       matrices with any: {agg['OF_vs_LU_summary']['cells_above_100Pa']['matrices_with_any']}/{len(valid)}")
        print(f"  cells >1kPa:        matrices with any: {agg['OF_vs_LU_summary']['cells_above_1kPa']['matrices_with_any']}/{len(valid)}")
        print(f"  cells >10kPa:       matrices with any: {agg['OF_vs_LU_summary']['cells_above_10kPa']['matrices_with_any']}/{len(valid)}")
        print(f"  mean CHOLMOD wall:  {walls.mean():.1f}s per matrix (multi-thread)")
        print(f"  total wall:         {(time.time() - t_start)/60:.1f} min")
        print(f"  → {output_dir}/aggregate.json")


if __name__ == "__main__":
    main()
