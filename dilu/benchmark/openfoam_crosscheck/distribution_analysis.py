"""Distribution analysis tool for OpenFOAM cross-check benchmark results.

Replaces the median-only reporting in aggregate_results.py with full distribution
statistics (min / mean / std / P95 / P99 / max / argmax_timestamp).

Usage:
    python3 distribution_analysis.py <root>
    python3 distribution_analysis.py <root> --solver-config amgx_classical_v_diagscaled cusparse
    python3 distribution_analysis.py <root> --eq pd
    python3 distribution_analysis.py <root> --csv out.csv

Writes:
    <root>/distribution.csv   (tidy per-matrix rows; use --csv to override)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect(root: Path,
            solver_filter: list[str] | None = None,
            eq_filter: str | None = None) -> dict[tuple[str, str], list[dict]]:
    """Walk <root>/<TS>/<eq>_corr*/results/*.json and group by (solver, eq).

    Returns a dict mapping (solver_label, equation) -> list of per-matrix dicts,
    each with keys: ts, rel_vs_OF, selfres, solve_s, iter, status.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for results_dir in sorted(root.glob("*/*_corr*/results")):
        ts = results_dir.parent.parent.name          # "2.25322e-06"
        eq_dir = results_dir.parent.name             # "pd_corr0"
        eq = eq_dir.split("_corr")[0]               # "pd" or "T"

        if eq_filter is not None and eq.lower() != eq_filter.lower():
            continue

        for jf in sorted(results_dir.glob("*.json")):
            solver = jf.stem   # e.g. "amgx_classical_v_diagscaled"

            if solver_filter and solver not in solver_filter:
                continue

            try:
                with open(jf) as fh:
                    data = json.load(fh)
            except Exception as exc:
                print(f"WARN  failed reading {jf}: {exc}", file=sys.stderr)
                continue

            # Skip skipped / error entries
            status = data.get("status", "")
            if isinstance(status, str) and status.startswith("skip"):
                continue

            row = {
                "ts": ts,
                "solver": solver,
                "eq": eq,
                "rel_vs_OF": data.get("rel_vs_OF"),
                "selfres": data.get("rel_residual"),
                "solve_s": data.get("solve_s"),
                "iters": data.get("iters"),
                "status": status,
                "converged": (
                    status == "ok"
                    or data.get("converged") is True
                    or data.get("status_code") == 0
                ),
            }
            groups[(solver, eq)].append(row)

    return dict(groups)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(values: list[float]) -> dict:
    """Return distribution statistics for a list of floats."""
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    argmax_idx = int(np.argmax(arr))
    return {
        "n": len(arr),
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "P95": float(np.percentile(arr, 95)),
        "P99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "argmax_idx": argmax_idx,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _fmt(v, digits=2):
    """Format float in scientific notation or return '—' for None/NaN."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{digits}e}"


def print_markdown_table(groups: dict[tuple[str, str], list[dict]]) -> None:
    """Print a markdown table with full distribution stats for rel_vs_OF."""
    header = (
        f"| {'solver':<52s} | {'eq':3s} | {'N':>4s} | "
        f"{'min':>9s} | {'median':>9s} | {'mean':>9s} | "
        f"{'P95':>9s} | {'P99':>9s} | **{'max':^9s}** | "
        f"{'argmax_ts':>14s} |"
    )
    sep = "|" + "|".join(["-" * (w + 2) for w in [54, 5, 6, 11, 11, 11, 11, 11, 13, 16]]) + "|"

    print("\n## Distribution of rel_vs_OF (‖x_solver − x_OF‖_∞ / ‖x_OF‖_∞)\n")
    print(header)
    print(sep)

    for (solver, eq), rows in sorted(groups.items()):
        rv = [r["rel_vs_OF"] for r in rows if r.get("rel_vs_OF") is not None]
        if not rv:
            continue
        stats = compute_stats(rv)
        argmax_ts = rows[stats["argmax_idx"]]["ts"] if "argmax_idx" in stats else "?"
        print(
            f"| {solver:<52s} | {eq:3s} | {stats['n']:>4d} | "
            f"{_fmt(stats['min']):>9s} | {_fmt(stats['median']):>9s} | "
            f"{_fmt(stats['mean']):>9s} | "
            f"{_fmt(stats['P95']):>9s} | {_fmt(stats['P99']):>9s} | "
            f"**{_fmt(stats['max']):^9s}** | "
            f"{argmax_ts:>14s} |"
        )


def print_selfres_table(groups: dict[tuple[str, str], list[dict]]) -> None:
    """Print selfres distribution for validation."""
    header = (
        f"| {'solver':<52s} | {'eq':3s} | {'N':>4s} | "
        f"{'selfres_min':>12s} | {'selfres_med':>12s} | {'selfres_max':>12s} |"
    )
    sep = "|" + "|".join(["-" * (w + 2) for w in [54, 5, 6, 14, 14, 14]]) + "|"

    print("\n## Self-residual distribution (‖A·x − b‖ / ‖b‖, sanity check)\n")
    print(header)
    print(sep)

    for (solver, eq), rows in sorted(groups.items()):
        sr = [r["selfres"] for r in rows if r.get("selfres") is not None]
        if not sr:
            continue
        stats = compute_stats(sr)
        print(
            f"| {solver:<52s} | {eq:3s} | {stats['n']:>4d} | "
            f"{_fmt(stats['min']):>12s} | {_fmt(stats['median']):>12s} | "
            f"{_fmt(stats['max']):>12s} |"
        )


def flag_failures(groups: dict[tuple[str, str], list[dict]]) -> list[dict]:
    """Identify entries where BOTH rel_vs_OF and selfres are in the worst 5%.

    Returns list of flagged rows.
    """
    flagged = []
    for (solver, eq), rows in groups.items():
        rv = [r["rel_vs_OF"] for r in rows if r.get("rel_vs_OF") is not None]
        sr = [r["selfres"] for r in rows if r.get("selfres") is not None]
        if not rv or not sr:
            continue
        p95_rv = float(np.percentile(rv, 95))
        p95_sr = float(np.percentile(sr, 95))
        for r in rows:
            if (r.get("rel_vs_OF") is not None and r.get("selfres") is not None
                    and r["rel_vs_OF"] > p95_rv
                    and r["selfres"] > p95_sr):
                flagged.append({**r, "p95_rel_vs_OF": p95_rv, "p95_selfres": p95_sr})
    return flagged


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(groups: dict[tuple[str, str], list[dict]], out_path: Path) -> None:
    """Save a tidy per-matrix CSV with all fields."""
    fieldnames = [
        "solver", "eq", "ts",
        "rel_vs_OF", "selfres", "solve_s", "iters",
        "status", "converged",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for (solver, eq), rows in sorted(groups.items()):
            for row in rows:
                writer.writerow({**row, "solver": solver, "eq": eq})
    print(f"\nWrote tidy CSV: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Distribution analysis for OpenFOAM cross-check results"
    )
    ap.add_argument("root", type=Path,
                    help="Root directory containing <TS>/<eq>_corr*/results/")
    ap.add_argument("--solver-config", nargs="+", dest="solver_config",
                    metavar="SOLVER",
                    help="Include only these solver labels (stem of JSON filename)")
    ap.add_argument("--eq", choices=["pd", "T"],
                    help="Filter to a single equation (pd or T)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="CSV output path (default: <root>/distribution.csv)")
    args = ap.parse_args()

    csv_path = args.csv or (args.root / "distribution.csv")

    groups = collect(args.root,
                     solver_filter=args.solver_config,
                     eq_filter=args.eq)

    if not groups:
        print("No results found under", args.root)
        sys.exit(1)

    total = sum(len(v) for v in groups.values())
    print(f"Loaded {total} result entries across {len(groups)} (solver, eq) pairs.")

    print_markdown_table(groups)
    print_selfres_table(groups)

    flagged = flag_failures(groups)
    if flagged:
        print(f"\n## WARNING: {len(flagged)} entries with BOTH rel_vs_OF AND selfres in worst 5%\n")
        for f in flagged:
            print(f"  solver={f['solver']} eq={f['eq']} ts={f['ts']} "
                  f"rel_vs_OF={_fmt(f['rel_vs_OF'])} selfres={_fmt(f['selfres'])}")
    else:
        print("\n## Failure flags: none (no entries have both rel_vs_OF and selfres in worst 5%)")

    save_csv(groups, csv_path)


if __name__ == "__main__":
    main()
