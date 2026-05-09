"""E08 — Per-timestep iter + residual diagnostics + headline ratios.

Reads E02-E07 result.json files; produces:
  - diagnostics.csv (per-row data for plotting)
  - result.json (aggregated summary)
  - speedup_ratios.json (key ratios that settle the headline claims)

This is post-processing; no solver runs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def collect_per_rep(rep_dir: Path):
    """For E02 / E05: list of per-rep result.json dicts."""
    out = []
    for r in rep_dir.iterdir():
        if not r.is_dir(): continue
        rj = r / "result.json"
        if not rj.exists(): continue
        with open(rj) as f: out.append(json.load(f))
    return out


def collect_seq(rep_dir: Path):
    """For E03 / E04 / E06: list of full-sequence result dicts (each has per_step)."""
    out = []
    for r in rep_dir.iterdir():
        if not r.is_dir(): continue
        rj = r / "result.json"
        if not rj.exists(): continue
        with open(rj) as f: out.append(json.load(f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--npz-dir", default=None)
    args = ap.parse_args()

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir) if args.results_dir else output_dir.parent

    # === Collect ===
    e02 = collect_per_rep(results_dir / "E02") if (results_dir / "E02").exists() else []
    e03 = collect_seq(results_dir / "E03") if (results_dir / "E03").exists() else []
    e04 = collect_seq(results_dir / "E04") if (results_dir / "E04").exists() else []
    e05 = collect_per_rep(results_dir / "E05") if (results_dir / "E05").exists() else []
    e06 = collect_seq(results_dir / "E06") if (results_dir / "E06").exists() else []
    e07_aggr_path = results_dir / "E07" / "aggregate.json"
    e07_aggr = json.loads(e07_aggr_path.read_text()) if e07_aggr_path.exists() else {}
    e09_aggr_path = results_dir / "E09" / "aggregate.json"
    e09_aggr = json.loads(e09_aggr_path.read_text()) if e09_aggr_path.exists() else {}
    e01_path = results_dir / "E01" / "01" / "result.json"
    e01 = json.loads(e01_path.read_text()) if e01_path.exists() else {}

    # === Diagnostics CSV ===
    rows = []
    for d in e02:
        rows.append({
            "expt": "E02", "rep": d.get("rep"), "timestep": d.get("timestep"),
            "phase": d.get("phase"), "iter_count": d.get("iter_count"),
            "rel_resid": d.get("rel_resid_actual"), "wall_s": d.get("wall_seconds"),
            "step": -1,
        })
    for d in e03 + e04 + e06:
        for s in d.get("per_step", []):
            rows.append({
                "expt": d.get("expt_id", "?"),
                "rep": d.get("rep"),
                "step": s.get("step"),
                "timestep": s.get("timestep"),
                "phase": s.get("phase"),
                "iter_count": s.get("iter_count"),
                "rel_resid": s.get("rel_resid"),
                "wall_s": s.get("total_ms", 0) / 1000,
            })
    for d in e05:
        rows.append({
            "expt": "E05", "rep": d.get("rep"), "timestep": d.get("timestep"),
            "phase": d.get("phase"), "iter_count": -1,
            "rel_resid": d.get("rel_resid_actual"),
            "wall_s": d.get("wall_seconds"),
            "step": -1,
        })
    csv_path = output_dir / "diagnostics.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow(r)

    # === Headline ratios ===
    ratios = {}

    # E02 mean wall × 6 ≈ AMGx_fresh_total_for_6_steps
    if e02:
        amgx_fresh_per = float(np.mean([d["wall_seconds"] for d in e02]))
        ratios["amgx_fresh_per_step_s_mean"] = amgx_fresh_per
        ratios["amgx_fresh_total_6step_estimate_s"] = amgx_fresh_per * 6

    # E03 mean total
    if e03:
        amgx_amort_total = float(np.mean([d["wall_seconds"] for d in e03]))
        ratios["amgx_amortized_total_s_mean"] = amgx_amort_total
        if e02:
            ratios["amortized_speedup_vs_fresh"] = (amgx_fresh_per * 6) / amgx_amort_total

    # E04 mean total
    if e04:
        amgx_warm_total = float(np.mean([d["wall_seconds"] for d in e04]))
        ratios["amgx_amortized_warm_total_s_mean"] = amgx_warm_total
        if e02:
            ratios["warm_amortized_speedup_vs_fresh"] = (amgx_fresh_per * 6) / amgx_warm_total

    # E05 mean wall × 6
    if e05:
        chol_fresh_per = float(np.mean([d["wall_seconds"] for d in e05]))
        ratios["cholmod_fresh_per_step_s_mean"] = chol_fresh_per
        ratios["cholmod_fresh_total_6step_estimate_s"] = chol_fresh_per * 6

    # E06 mean total
    if e06:
        chol_sym_total = float(np.mean([d["wall_seconds"] for d in e06]))
        ratios["cholmod_symbolic_reuse_total_s_mean"] = chol_sym_total
        if e05:
            ratios["cholmod_symbolic_speedup_vs_fresh"] = (chol_fresh_per * 6) / chol_sym_total

    # AMGx warm vs CHOLMOD fresh — the headline production-style comparison
    if e04 and e05:
        ratios["AMGx_warm_vs_LU_fresh_speedup"] = (chol_fresh_per * 6) / amgx_warm_total
    if e04 and e06:
        ratios["AMGx_warm_vs_LU_symbolic_speedup"] = chol_sym_total / amgx_warm_total

    # E01 OF wall — would need to estimate per-pd-solve from log parsing
    if e01.get("step_wall_stats"):
        ratios["of_step_wall_mean_s"] = e01["step_wall_stats"].get("mean")
        ratios["of_pd_iter_mean"] = e01.get("pd_iter_per_step_stats", {}).get("mean")
        ratios["of_total_iter_mean"] = e01.get("total_iter_per_step_stats", {}).get("mean")
        ratios["of_approx_per_iter_ms"] = e01.get("approx_per_iter_ms")

    # OF vs AMGx_warm: per-step
    if e04 and e01.get("step_wall_stats", {}).get("mean"):
        # AMGx warm mean per step (steps 1..N, excluding setup step 0)
        amort_step1n = []
        for d in e04:
            ps = d.get("per_step", [])
            if len(ps) > 1:
                amort_step1n.extend([s["total_ms"] / 1000 for s in ps[1:]])
        if amort_step1n:
            ratios["amgx_warm_per_step_amortized_s_mean"] = float(np.mean(amort_step1n))
            ratios["amgx_warm_vs_OF_per_step_speedup"] = (
                e01["step_wall_stats"]["mean"] / float(np.mean(amort_step1n))
            )

    # === Build summary ===
    out = {
        "expt_id": "E08",
        "n_rows": len(rows),
        "n_e02": len(e02),
        "n_e03": len(e03),
        "n_e04": len(e04),
        "n_e05": len(e05),
        "n_e06": len(e06),
        "e07_summary": e07_aggr,
        "e09_summary": e09_aggr,
        "e01_present": bool(e01),
        "headline_ratios": ratios,
        "rows": rows,
    }
    with open(output_dir / "result.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(output_dir / "speedup_ratios.json", "w") as f:
        json.dump(ratios, f, indent=2)

    print(f"\nE08 aggregate ({len(rows)} rows):")
    print(f"  E02 reps: {len(e02)}, E03 reps: {len(e03)}, E04 reps: {len(e04)}")
    print(f"  E05 reps: {len(e05)}, E06 reps: {len(e06)}")
    print(f"  E07: {'YES' if e07_aggr else 'no'}, E09: {'YES' if e09_aggr else 'no'}, E01: {'YES' if e01 else 'no'}")
    print(f"\nHeadline ratios:")
    for k, v in ratios.items():
        if isinstance(v, float):
            print(f"  {k:<55s}: {v:.3f}")
        else:
            print(f"  {k:<55s}: {v}")


if __name__ == "__main__":
    main()
