"""Aggregate per-matrix solver results into summary statistics.

Walks <root>/<time>/<eq>_corr<k>/results/<solver>.json files, prints
distribution stats (median / max / min iter count, solve time, etc.)
per (solver, equation) pair. Useful after running driver_*.py over many
matrices to see the overall picture without combing through 150 JSONs.

Usage:
    python3 aggregate_results.py <root>
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()

    # Walk: <root>/<time>/<eq>_corr<k>/results/<solver>.json
    all_results = defaultdict(list)  # (solver, eq) -> list of dicts

    for results_dir in args.root.glob("*/*_corr*/results"):
        eq_dir = results_dir.parent.name      # e.g. "pd_corr0"
        eq = eq_dir.split("_corr")[0]          # "pd" or "T"
        time_str = results_dir.parent.parent.name
        for jf in results_dir.glob("*.json"):
            solver = jf.stem    # "cusparse", "amgx_classical_v", "scipy", etc.
            try:
                with open(jf) as fh:
                    data = json.load(fh)
                data["_time"] = time_str
                all_results[(solver, eq)].append(data)
            except Exception as e:
                print(f"WARN  failed reading {jf}: {e}")

    if not all_results:
        print("No results found")
        return

    print(f"\nFound {sum(len(v) for v in all_results.values())} result entries "
          f"across {len(all_results)} (solver, eq) pairs.\n")

    print(f"{'solver':<26s} {'eq':<3s} {'#':>4s} {'conv':>5s}  "
          f"{'iter med':>9s} {'iter max':>9s}  "
          f"{'solve_s med':>12s} {'solve_s max':>12s}  "
          f"{'rel_res med':>12s}  {'rel_vs_OF med':>14s}")
    print("-" * 124)

    rows = []
    for (solver, eq), entries in sorted(all_results.items()):
        n = len(entries)
        converged = sum(1 for e in entries
                        if e.get("status") == "ok"
                        or e.get("converged") is True
                        or (e.get("status_code") == 0))
        iters = [e["iters"] for e in entries if e.get("iters") is not None]
        solve_s = [e["solve_s"] for e in entries if e.get("solve_s") is not None]
        rel_res = [e["rel_residual"] for e in entries if e.get("rel_residual") is not None]
        rel_OF = [e["rel_vs_OF"] for e in entries if e.get("rel_vs_OF") is not None]

        def med(xs):
            return statistics.median(xs) if xs else float('nan')
        def mx(xs):
            return max(xs) if xs else float('nan')

        row = {
            "solver": solver, "eq": eq, "n": n, "converged": converged,
            "iters_med": med(iters), "iters_max": mx(iters),
            "solve_s_med": med(solve_s), "solve_s_max": mx(solve_s),
            "rel_res_med": med(rel_res),
            "rel_vs_OF_med": med(rel_OF) if rel_OF else None,
        }
        rows.append(row)

        of_str = f"{row['rel_vs_OF_med']:>14.2e}" if row['rel_vs_OF_med'] is not None else f"{'—':>14s}"
        print(f"{solver:<26s} {eq:<3s} {n:>4d} {converged:>3d}/{n:<2d}  "
              f"{row['iters_med']:>9.0f} {row['iters_max']:>9.0f}  "
              f"{row['solve_s_med']:>12.2f} {row['solve_s_max']:>12.2f}  "
              f"{row['rel_res_med']:>12.2e}  {of_str}")

    # Save aggregate JSON
    out = args.root / "aggregate.json"
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
