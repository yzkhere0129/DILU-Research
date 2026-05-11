"""Compare E0X result.json files to expected_results_template.json.

Walks results-dir for each E0X, extracts key metrics, compares to predicted
range. Outputs summary.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def collect_e02(rep_dir: Path):
    walls = []; iters = []; resids = []
    for r in rep_dir.iterdir():
        if not r.is_dir(): continue
        rj = r / "result.json"
        if not rj.exists(): continue
        d = json.loads(rj.read_text())
        walls.append(d["wall_seconds"])
        iters.append(d["iter_count"])
        resids.append(d["rel_resid_actual"])
    return walls, iters, resids


def collect_seq(rep_dir: Path):
    """For E03/E04/E06 — list of total_wall_s and per-step lists."""
    totals = []
    per_step_walls = []
    per_step_iters = []
    for r in rep_dir.iterdir():
        if not r.is_dir(): continue
        rj = r / "result.json"
        if not rj.exists(): continue
        d = json.loads(rj.read_text())
        totals.append(d["wall_seconds"])
        per_step_walls.append([s["total_ms"] for s in d.get("per_step", [])])
        per_step_iters.append([s.get("iter_count") or 0 for s in d.get("per_step", [])])
    return totals, per_step_walls, per_step_iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--expected", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    expected_path = Path(args.expected)
    output_path = Path(args.output)

    expected = json.loads(expected_path.read_text()) if expected_path.exists() else {}
    summary = ["# Xeon Validation Summary\n",
                f"results_dir: `{results_dir}`",
                f"expected_template: `{expected_path}`\n",
                "---\n"]

    for E in ("E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09"):
        e_dir = results_dir / E
        summary.append(f"\n## {E}")
        if not e_dir.exists():
            summary.append(f"  NOT RUN")
            continue

        if E == "E01":
            rj = e_dir / "01" / "result.json"
            if rj.exists():
                d = json.loads(rj.read_text())
                summary.append(f"  status: {d.get('status')}")
                if d.get("step_wall_stats"):
                    s = d["step_wall_stats"]
                    summary.append(f"  mean step wall: {s.get('mean'):.2f} s, median: {s.get('median'):.2f} s")
                summary.append(f"  approx ms/iter: {d.get('approx_per_iter_ms')}")

        elif E in ("E02", "E05"):
            walls, iters, resids = collect_e02(e_dir)
            if walls:
                summary.append(f"  N reps: {len(walls)}")
                summary.append(f"  wall: mean={np.mean(walls):.2f}s median={np.median(walls):.2f}s "
                                f"std={np.std(walls):.2f}s")
                summary.append(f"  iter: mean={np.mean(iters):.0f}")
                summary.append(f"  resid: max={max(resids):.2e}")

        elif E in ("E03", "E04", "E06"):
            totals, per_step_walls, per_step_iters = collect_seq(e_dir)
            if totals:
                summary.append(f"  N reps: {len(totals)}")
                summary.append(f"  total wall: mean={np.mean(totals):.2f}s median={np.median(totals):.2f}s")
                if per_step_walls:
                    psw = np.array(per_step_walls)  # [reps, steps]
                    summary.append(f"  step 0 (setup): {psw[:, 0].mean():.0f}ms")
                    if psw.shape[1] > 1:
                        summary.append(f"  steps 1..N (amortized): mean={psw[:, 1:].mean():.0f}ms")

        elif E == "E07":
            agg = e_dir / "aggregate.json"
            if agg.exists():
                d = json.loads(agg.read_text())
                for sname in ["OF", "AMGx_e8", "AMGx_e12_IR"]:
                    if f"{sname}_max_rel_max" in d:
                        summary.append(f"  {sname:<14s}: max rel diff vs LU = {d[f'{sname}_max_rel_max']:.3e}")

        elif E == "E09":
            agg = e_dir / "aggregate.json"
            if agg.exists():
                d = json.loads(agg.read_text())
                for r in d.get("results", []):
                    # E09 stores kappa_lanczos + smallest_10_sigmas_svds (P8 upgrade).
                    kappa = r.get("kappa_lanczos", r.get("kappa"))
                    sigma_min = r.get("sigma_min_lanczos")
                    near_null = r.get("has_near_null_subspace")
                    bits = []
                    if kappa is not None:
                        bits.append(f"κ_L={kappa:.3e}")
                    if sigma_min is not None:
                        bits.append(f"σ_min={sigma_min:.3e}")
                    if near_null is not None:
                        bits.append(f"near_null={near_null}")
                    summary.append(f"  {r.get('case','?'):<14s} {r.get('phase','?'):<10s} "
                                    f"t={r.get('timestep','?')}: " + "  ".join(bits))

    # Compare to expected
    summary.append("\n---\n## Comparison to expected\n")
    if expected:
        for pred in expected.get("predictions", []):
            eid = pred.get("expt_id", "?")
            metric = pred.get("metric", "?")
            val_pred = pred.get("predicted_value")
            range_pred = pred.get("predicted_range", [None, None])
            summary.append(f"\n### {eid} / {metric}")
            summary.append(f"  Predicted: {val_pred} ∈ [{range_pred[0]}, {range_pred[1]}]")
            summary.append(f"  Rationale: {pred.get('rationale', '(none)')}")

    output_path.write_text("\n".join(summary))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
