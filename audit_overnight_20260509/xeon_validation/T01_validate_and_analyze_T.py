"""T01 — Validate + analyze T_corr0 matrices on any OF dump case.

For each matrix in --pool (or --all-in-case):
  1. Load A.mm, b.mm, x_final.mm
  2. Finiteness + nnz + diag sanity
  3. Asymmetry: ‖A - A^T‖_F / ‖A‖_F   (T has advection → expect 1e-2 to 1e-1)
  4. upper/lower ratio
  5. b/x_OF range sanity
  6. OF self-consistency: ‖A·x_OF - b‖ / ‖b‖

Usage:
  # 6-matrix pilot on single_track_dump
  PYTHONPATH=. python audit_overnight_20260509/xeon_validation/T01_validate_and_analyze_T.py \\
      --case ~/cases/single_track_dump \\
      --output-dir audit_overnight_20260509/xeon_validation/results/T01_single_track_T

  # Full 384-matrix run on dense_track_dump_500K (after OF re-dump with T)
  PYTHONPATH=. python audit_overnight_20260509/xeon_validation/T01_validate_and_analyze_T.py \\
      --case ~/cases/dense_track_dump_500K \\
      --output-dir audit_overnight_20260509/xeon_validation/results/T01_dense_T

Output:
  <out>/per_matrix.json     full per-matrix records
  <out>/summary.json        aggregate stats
  <out>/sane_pool_T.txt     timesteps where all checks pass (for downstream batch jobs)
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def discover_timesteps(case: Path, system: str = "T_corr0") -> list[str]:
    """Find all timesteps under case/postProcessing/matrices/<ts>/<system>/."""
    root = case / "postProcessing" / "matrices"
    if not root.is_dir():
        raise FileNotFoundError(f"No postProcessing/matrices under {case}")
    found = []
    for child in root.iterdir():
        if not child.is_dir(): continue
        if (child / system / "A.mm").exists():
            found.append(child.name)
    found.sort(key=lambda s: float(s))
    return found


def analyze_one(case: Path, ts: str, system: str = "T_corr0"):
    sys_dir = case / "postProcessing" / "matrices" / ts / system
    rec = {"ts": ts, "system": system}
    t0 = time.time()
    A = sio.mmread(str(sys_dir / "A.mm")).tocsr()
    b = np.asarray(sio.mmread(str(sys_dir / "b.mm"))).flatten()
    x_OF = np.asarray(sio.mmread(str(sys_dir / "x_final.mm"))).flatten()
    rec["load_s"] = time.time() - t0
    rec["n"] = int(A.shape[0])
    rec["nnz"] = int(A.nnz)

    diag = A.diagonal()
    rec["diag_min"] = float(diag.min())
    rec["diag_max"] = float(diag.max())
    rec["diag_mean"] = float(diag.mean())
    rec["any_zero_diag"] = bool(np.any(diag == 0))
    rec["any_neg_diag"] = bool(np.any(diag < 0))

    rec["A_finite"] = bool(np.all(np.isfinite(A.data)))
    rec["b_finite"] = bool(np.all(np.isfinite(b)))
    rec["x_OF_finite"] = bool(np.all(np.isfinite(x_OF)))

    AT = A.T
    diff = A - AT
    n_A = sp.linalg.norm(A, "fro")
    n_diff = sp.linalg.norm(diff, "fro")
    rec["asymmetry_rel"] = float(n_diff / max(n_A, 1e-300))

    Atriu = sp.triu(A, k=1)
    Atril = sp.tril(A, k=-1)
    rec["max_upper_abs"] = float(np.max(np.abs(Atriu.data))) if Atriu.nnz else 0.0
    rec["max_lower_abs"] = float(np.max(np.abs(Atril.data))) if Atril.nnz else 0.0
    rec["upper_over_lower"] = (
        rec["max_upper_abs"] / max(rec["max_lower_abs"], 1e-300)
    )

    rec["b_min"] = float(b.min())
    rec["b_max"] = float(b.max())
    rec["b_inf_norm"] = float(np.max(np.abs(b)))
    rec["b_l2_norm"] = float(np.linalg.norm(b))

    rec["x_OF_min"] = float(x_OF.min())
    rec["x_OF_max"] = float(x_OF.max())
    rec["x_OF_inf_norm"] = float(np.max(np.abs(x_OF)))

    r = A @ x_OF - b
    rec["OF_rel_resid"] = float(np.linalg.norm(r) / max(np.linalg.norm(b), 1e-300))
    rec["OF_max_resid_abs"] = float(np.max(np.abs(r)))

    meta_p = sys_dir / "metadata.json"
    if meta_p.exists():
        rec["metadata"] = json.loads(meta_p.read_text())

    rec["total_s"] = time.time() - t0

    # Sanity verdict
    sane = (rec["A_finite"] and rec["b_finite"] and rec["x_OF_finite"]
            and not rec["any_zero_diag"]
            and rec["b_inf_norm"] > 0
            and 100.0 <= rec["x_OF_max"] <= 1e5   # 100K - 100kK physical T range
            and rec["OF_rel_resid"] < 1e-5)
    rec["sane"] = bool(sane)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--system", default="T_corr0")
    ap.add_argument("--max", type=int, default=0,
                     help="cap number of matrices (0 = all)")
    ap.add_argument("--ts-list", default=None,
                     help="comma-separated explicit timestep list (overrides --max)")
    args = ap.parse_args()

    case = Path(args.case).expanduser()
    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if args.ts_list:
        timesteps = [t.strip() for t in args.ts_list.split(",")]
    else:
        timesteps = discover_timesteps(case, args.system)
        if args.max > 0:
            timesteps = timesteps[:args.max]

    print(f"# T_corr0 analysis on {len(timesteps)} matrices from {case}")

    results = []
    t_start = time.time()
    for i, ts in enumerate(timesteps):
        try:
            rec = analyze_one(case, ts, args.system)
        except FileNotFoundError as e:
            print(f"  [{i+1}/{len(timesteps)}] {ts}: MISSING ({e})")
            continue
        results.append(rec)
        if (i+1) % 10 == 0 or i+1 == len(timesteps) or len(timesteps) <= 12:
            cum = time.time() - t_start
            print(f"  [{i+1}/{len(timesteps)}] ts={ts:>12}  "
                  f"asym={rec['asymmetry_rel']:.2e}  "
                  f"OF_res={rec['OF_rel_resid']:.2e}  "
                  f"x_max={rec['x_OF_max']:.0f}K  "
                  f"sane={rec['sane']}  cum={cum:.1f}s")

    (out / "per_matrix.json").write_text(json.dumps(results, indent=2))

    if not results:
        print(f"\nNo matrices found.")
        return

    sanes = [r for r in results if r["sane"]]
    summary = {
        "n_matrices_total": len(results),
        "n_sane": len(sanes),
        "case": str(case),
        "system": args.system,
        "asymmetry_rel": {
            "min": float(min(r["asymmetry_rel"] for r in results)),
            "max": float(max(r["asymmetry_rel"] for r in results)),
            "mean": float(np.mean([r["asymmetry_rel"] for r in results])),
            "median": float(np.median([r["asymmetry_rel"] for r in results])),
        },
        "OF_rel_resid": {
            "max": float(max(r["OF_rel_resid"] for r in results)),
            "median": float(np.median([r["OF_rel_resid"] for r in results])),
        },
        "x_OF_max_K_range": [
            float(min(r["x_OF_max"] for r in results)),
            float(max(r["x_OF_max"] for r in results)),
        ],
        "wall_total_s": time.time() - t_start,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # sane_pool_T.txt for downstream batch jobs
    pool_lines = [
        f"# Sane T matrices passing all checks ({args.system})",
        f"# case: {case}",
        f"# generated: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"# n_sane / n_total = {len(sanes)} / {len(results)}",
    ]
    for r in sanes:
        pool_lines.append(f"{r['ts']}/{args.system}")
    (out / "sane_pool_T.txt").write_text("\n".join(pool_lines) + "\n")

    print(f"\n=== SUMMARY ===")
    print(f"  total: {summary['n_matrices_total']},  sane: {summary['n_sane']}")
    print(f"  asymmetry: median={summary['asymmetry_rel']['median']:.2e}  "
          f"range [{summary['asymmetry_rel']['min']:.2e}, "
          f"{summary['asymmetry_rel']['max']:.2e}]")
    print(f"  OF rel_resid: median={summary['OF_rel_resid']['median']:.2e}  "
          f"max={summary['OF_rel_resid']['max']:.2e}")
    print(f"  x_OF max-T: {summary['x_OF_max_K_range']}")
    print(f"  wall: {summary['wall_total_s']:.1f}s")
    print(f"\n→ {out}/per_matrix.json")
    print(f"→ {out}/summary.json")
    print(f"→ {out}/sane_pool_T.txt")


if __name__ == "__main__":
    main()
