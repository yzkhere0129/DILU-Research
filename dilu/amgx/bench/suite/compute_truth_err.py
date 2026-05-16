"""Post-process: compute x_solver - x_truth difference per matrix per protocol.

Reads x_all (saved when --save-x in run_benchmark.py) and computes:
  - max |x_protocol - x_truth|  (Pa for pd, K for T)
  - max |x_OF - x_truth|
  - cells exceeding 0.1, 1, 10, 100 unit thresholds

Uses amortized_e12_IR as truth (rel_resid 1e-16 ε-machine).
Saves err per matrix into <results>/<eq>/precision_err.json.

Usage:
  PYTHONPATH=. python dilu/amgx/bench/suite/compute_truth_err.py \\
      --suite ~/benchmark_suite_v1 \\
      --results results_HOSTNAME_DATE
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def load_x_all(npz_path: Path):
    d = np.load(npz_path, allow_pickle=False)
    if "x_all" not in d.files:
        raise ValueError(f"{npz_path} has no x_all (was --save-x used?)")
    return d["x_all"], list(d["ts"]), list(d["phase"])


def load_x_OF(suite: Path, eq: str, phase: str, ts: str):
    short = {"pd": "pd", "T": "T"}[eq]
    p = suite / "matrices_npz" / phase / f"ts_{ts}" / f"{short}.npz"
    return np.load(p)["x_OF"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite",   required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--truth-protocol", default="amortized_e12_IR",
                     help="which protocol's x_all to use as truth (default amortized_e12_IR)")
    args = ap.parse_args()

    suite = Path(args.suite).expanduser()
    results = Path(args.results).expanduser()
    summary = json.loads((results / "summary.json").read_text())

    UNIT = {"pd": "Pa", "T": "K"}

    for eq in summary.keys():
        eq_dir = results / eq
        truth_npz = eq_dir / f"replay_{args.truth_protocol}.npz"
        try:
            x_truth_all, ts_list, phase_list = load_x_all(truth_npz)
        except ValueError as e:
            print(f"  {eq}: skip — {e}"); continue
        n = len(ts_list)
        unit = UNIT.get(eq, "?")
        print(f"\n=== {eq} ({n} matrices, truth={args.truth_protocol}, unit={unit}) ===")

        err_per_matrix = []
        for k in range(n):
            phase = phase_list[k]; ts = ts_list[k]
            x_truth = x_truth_all[k]
            x_OF = load_x_OF(suite, eq, phase, ts)
            entry = {"ts": ts, "phase": phase, "x_truth_inf": float(np.max(np.abs(x_truth)))}

            # x_OF vs truth
            d = np.abs(x_OF - x_truth)
            entry["x_OF"] = {
                "max":     float(d.max()),
                "median":  float(np.median(d)),
                "rel_max": float(d.max() / max(np.max(np.abs(x_truth)), 1e-300)),
                "cells_gt_0.1":  int(np.sum(d > 0.1)),
                "cells_gt_1":    int(np.sum(d > 1.0)),
                "cells_gt_10":   int(np.sum(d > 10.0)),
                "cells_gt_100":  int(np.sum(d > 100.0)),
            }

            # Each protocol vs truth
            for proto in summary[eq].keys():
                if proto == args.truth_protocol: continue
                proto_npz = eq_dir / f"replay_{proto}.npz"
                try:
                    x_p_all, ts_p, _ = load_x_all(proto_npz)
                except ValueError:
                    continue
                if ts_p[k] != ts:
                    print(f"  ts mismatch! {ts_p[k]} != {ts}"); continue
                x_p = x_p_all[k]
                d = np.abs(x_p - x_truth)
                entry[proto] = {
                    "max":     float(d.max()),
                    "median":  float(np.median(d)),
                    "rel_max": float(d.max() / max(np.max(np.abs(x_truth)), 1e-300)),
                    "cells_gt_0.1":  int(np.sum(d > 0.1)),
                    "cells_gt_1":    int(np.sum(d > 1.0)),
                    "cells_gt_10":   int(np.sum(d > 10.0)),
                    "cells_gt_100":  int(np.sum(d > 100.0)),
                }
            err_per_matrix.append(entry)

        out_json = eq_dir / "precision_err.json"
        out_json.write_text(json.dumps({
            "eq": eq, "unit": unit, "truth_protocol": args.truth_protocol,
            "n": n, "per_matrix": err_per_matrix,
        }, indent=2))
        print(f"→ {out_json}")

        # Aggregate
        for src in ["x_OF"] + [p for p in summary[eq] if p != args.truth_protocol]:
            valid = [m[src] for m in err_per_matrix if src in m]
            if not valid: continue
            mx = [v["max"] for v in valid]
            print(f"  {src:<24} max_max={max(mx):.3e}{unit:<3}  "
                  f"mean_max={np.mean(mx):.3e}{unit:<3}  "
                  f"median_max={np.median(mx):.3e}{unit:<3}")


if __name__ == "__main__":
    main()
