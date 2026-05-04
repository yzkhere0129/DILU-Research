"""Global norm analysis: replace per-matrix median with whole-dataset norms.

Computes:
  1. Global L2:  sqrt(Σ_i ‖x_s_i − x_OF_i‖²) / sqrt(Σ_i ‖x_OF_i‖²)
  2. Global L∞:  max_i(‖x_s_i − x_OF_i‖∞) / max_i(‖x_OF_i‖∞)
  3. Weighted L2: Σ_i ‖x_s_i − x_OF_i‖₂ / Σ_i ‖x_OF_i‖₂  (L1 of L2s)

vs the old per-matrix median metric.

Usage:
    python3 global_norm_analysis.py <root>
    python3 global_norm_analysis.py <root> --solver amgx_classical_v_diagscaled_amortized
    python3 global_norm_analysis.py <root> --eq pd
    python3 global_norm_analysis.py <root> --csv norms.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def collect_x(root: Path, solver_filter=None, eq_filter=None):
    """Walk <root>/<TS>/<eq>_corr*/results/ and load x_s + x_OF pairs.

    Returns dict: (solver, eq) -> list of (ts, x_s, x_OF) tuples.
    """
    groups = defaultdict(list)

    for results_dir in sorted(root.glob("*/*_corr*/results")):
        ts = results_dir.parent.parent.name
        eq_dir = results_dir.parent.name
        eq = eq_dir.split("_corr")[0]

        if eq_filter and eq.lower() != eq_filter.lower():
            continue

        # Load x_OF from the parent directory (the npz or mm data)
        matrix_dir = results_dir.parent
        x_of = None

        # Try loading from npz
        npz_files = list(matrix_dir.glob("*.npz"))
        if npz_files:
            try:
                data = np.load(npz_files[0])
                if "x_final" in data:
                    x_of = data["x_final"]
            except Exception:
                pass

        # Try loading from mm format
        if x_of is None:
            x_of_file = matrix_dir / "x_final.mm"
            if x_of_file.exists():
                try:
                    from scipy.io import mmread
                    x_of = mmread(str(x_of_file)).toarray().ravel()
                except Exception:
                    pass

        if x_of is None:
            continue

        for x_file in sorted(results_dir.glob("*_x.npy")):
            solver = x_file.stem.replace("_x", "")

            if solver_filter and solver not in solver_filter:
                continue

            try:
                x_s = np.load(x_file)
            except Exception:
                continue

            groups[(solver, eq)].append((ts, x_s, x_of))

    return dict(groups)


def compute_global_norms(pairs: list[tuple]) -> dict:
    """Compute global norms from list of (ts, x_s, x_OF) tuples."""
    if not pairs:
        return {}

    # Accumulators for global L2
    sum_sq_diff = 0.0
    sum_sq_of = 0.0
    sum_abs_diff_l2 = 0.0   # for weighted L2
    sum_abs_of_l2 = 0.0

    # For global L∞
    max_diff_inf = 0.0
    max_of_inf = 0.0

    # Per-matrix stats for comparison
    per_matrix_rel = []

    for ts, x_s, x_of in pairs:
        diff = x_s - x_of

        # L2 accumulators
        sum_sq_diff += float(np.dot(diff, diff))
        sum_sq_of += float(np.dot(x_of, x_of))

        # L2 per-matrix for weighted
        d_l2 = float(np.linalg.norm(diff))
        o_l2 = float(np.linalg.norm(x_of))
        sum_abs_diff_l2 += d_l2
        sum_abs_of_l2 += o_l2

        # L∞
        d_inf = float(np.max(np.abs(diff)))
        o_inf = float(np.max(np.abs(x_of)))
        max_diff_inf = max(max_diff_inf, d_inf)
        max_of_inf = max(max_of_inf, o_inf)

        # Per-matrix L∞ relative (same as old rel_vs_OF)
        if o_inf > 0:
            per_matrix_rel.append(d_inf / o_inf)

    denom_l2 = np.sqrt(sum_sq_of)
    denom_inf = max_of_inf

    return {
        "n_matrices": len(pairs),
        "global_l2": np.sqrt(sum_sq_diff) / denom_l2 if denom_l2 > 0 else 0.0,
        "global_linf": max_diff_inf / denom_inf if denom_inf > 0 else 0.0,
        "weighted_l2": sum_abs_diff_l2 / sum_abs_of_l2 if sum_abs_of_l2 > 0 else 0.0,
        # Old metric for comparison
        "median_per_matrix_linf": float(np.median(per_matrix_rel)) if per_matrix_rel else 0.0,
        "max_per_matrix_linf": float(np.max(per_matrix_rel)) if per_matrix_rel else 0.0,
        "p95_per_matrix_linf": float(np.percentile(per_matrix_rel, 95)) if per_matrix_rel else 0.0,
        "p99_per_matrix_linf": float(np.percentile(per_matrix_rel, 99)) if per_matrix_rel else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="Global norm analysis")
    ap.add_argument("root", type=Path)
    ap.add_argument("--solver", nargs="+", help="Filter solvers")
    ap.add_argument("--eq", choices=["pd", "T"])
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    groups = collect_x(args.root, args.solver, args.eq)

    if not groups:
        print("No x vectors found. Re-run drivers without --no-save-x.")
        sys.exit(1)

    print(f"\n## Global Norm Analysis\n")
    print(f"| {'solver':<52s} | {'eq':3s} | {'N':>4s} | "
          f"{'global_L2':>10s} | {'global_L∞':>10s} | {'weighted_L2':>11s} | "
          f"{'med_per_mat':>11s} | {'max_per_mat':>11s} |")
    print("|" + "|".join(["-" * (w + 2) for w in [54, 5, 6, 12, 12, 13, 13, 13]]) + "|")

    rows = []
    for (solver, eq), pairs in sorted(groups.items()):
        norms = compute_global_norms(pairs)
        print(
            f"| {solver:<52s} | {eq:3s} | {norms['n_matrices']:>4d} | "
            f"{norms['global_l2']:>10.3e} | {norms['global_linf']:>10.3e} | "
            f"{norms['weighted_l2']:>11.3e} | "
            f"{norms['median_per_matrix_linf']:>11.3e} | "
            f"{norms['max_per_matrix_linf']:>11.3e} |"
        )
        rows.append({"solver": solver, "eq": eq, **norms})

    if args.csv:
        import csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
