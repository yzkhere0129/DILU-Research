#!/usr/bin/env python3
"""Parse log.run for per-step + per-equation OF solve timings.

Extracts:
- Time = X (physical timestep)
- ExecutionTime = Y (cumulative wall)
- DICPCG / DILUPBiCG: Solving for <eq>, ... iter=N, finalRes=R

Computes per-step wall delta and aggregates per-equation iter / residual stats.

Output:
  postProcessing/solve_timings_summary.json
  postProcessing/solve_timings_per_step.csv

Usage:
  python3 parse_solve_timings.py [path/to/log.run]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import defaultdict


def parse(log_path: Path):
    text = log_path.read_text()
    lines = text.split("\n")

    # State
    cur_time = None
    cur_step_idx = -1
    last_exec_time = 0.0
    pending_solves = []   # solves seen since last "Time = " line

    # Output
    per_step = []         # list of dicts
    per_eq_iters = defaultdict(list)
    per_eq_resid = defaultdict(list)

    re_time   = re.compile(r"^Time = (\S+)$")
    re_exec   = re.compile(r"^ExecutionTime = (\S+) s")
    re_solve  = re.compile(r"(DICPCG|DILUPBiCG|GAMG|smoothSolver):\s*Solving for (\w+),"
                            r"\s*Initial residual = (\S+),\s*Final residual = (\S+),"
                            r"\s*No Iterations (\d+)")

    for ln in lines:
        m = re_time.match(ln)
        if m:
            # New step starts
            new_time = float(m.group(1))
            if cur_time is not None and pending_solves:
                # Emit previous step's record
                per_step.append({
                    "time": cur_time,
                    "step_idx": cur_step_idx,
                    "wall_delta_s": None,   # filled in after exec line
                    "solves": pending_solves,
                })
            cur_time = new_time
            cur_step_idx += 1
            pending_solves = []
            continue

        m = re_solve.search(ln)
        if m:
            algorithm, eq, init_r, final_r, n_iter = m.groups()
            pending_solves.append({
                "algorithm": algorithm,
                "equation": eq,
                "init_residual": float(init_r),
                "final_residual": float(final_r),
                "n_iter": int(n_iter),
            })
            per_eq_iters[eq].append(int(n_iter))
            per_eq_resid[eq].append(float(final_r))
            continue

        m = re_exec.match(ln)
        if m:
            new_exec = float(m.group(1))
            if per_step and per_step[-1]["wall_delta_s"] is None:
                per_step[-1]["wall_delta_s"] = new_exec - last_exec_time
                per_step[-1]["wall_cum_s"] = new_exec
            last_exec_time = new_exec

    # Final pending step
    if cur_time is not None and pending_solves:
        per_step.append({
            "time": cur_time,
            "step_idx": cur_step_idx,
            "wall_delta_s": None,
            "wall_cum_s": last_exec_time,
            "solves": pending_solves,
        })

    return per_step, per_eq_iters, per_eq_resid


def main():
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("log.run")
    if not log_path.exists():
        sys.exit(f"log not found: {log_path}")

    per_step, per_eq_iters, per_eq_resid = parse(log_path)

    if not per_step:
        sys.exit("no timesteps parsed — log may be empty or format unrecognized")

    # Summary
    import numpy as np
    total_wall = per_step[-1].get("wall_cum_s") or 0.0
    total_steps = len(per_step)
    walls = [s.get("wall_delta_s") for s in per_step if s.get("wall_delta_s") is not None]
    summary = {
        "log_path": str(log_path),
        "total_steps": total_steps,
        "total_wall_s": total_wall,
        "mean_wall_per_step_s": float(np.mean(walls)) if walls else None,
        "median_wall_per_step_s": float(np.median(walls)) if walls else None,
        "first_time": per_step[0]["time"],
        "last_time": per_step[-1]["time"],
        "per_eq": {
            eq: {
                "n_solves": len(iters),
                "iter_min": int(min(iters)),
                "iter_max": int(max(iters)),
                "iter_mean": float(np.mean(iters)),
                "iter_median": float(np.median(iters)),
                "resid_min": float(min(per_eq_resid[eq])),
                "resid_max": float(max(per_eq_resid[eq])),
                "resid_median": float(np.median(per_eq_resid[eq])),
            }
            for eq, iters in per_eq_iters.items()
        },
    }

    out_dir = log_path.parent / "postProcessing"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "solve_timings_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # Per-step CSV
    out_csv = out_dir / "solve_timings_per_step.csv"
    with out_csv.open("w") as f:
        f.write("step,time,wall_cum_s,wall_delta_s,n_solves,solve_summary\n")
        for s in per_step:
            solve_str = "; ".join(f"{x['equation']}({x['n_iter']}, {x['final_residual']:.2e})"
                                    for x in s["solves"])
            f.write(f"{s['step_idx']},{s['time']},"
                    f"{s.get('wall_cum_s',''):.3f},"
                    f"{s.get('wall_delta_s',''):.4f},"
                    f"{len(s['solves'])},\"{solve_str}\"\n")

    print(f"=== OF Solve Timings Summary ===")
    print(f"  log file:        {log_path}")
    print(f"  total steps:     {summary['total_steps']}")
    print(f"  total wall:      {summary['total_wall_s']:.1f} s ({summary['total_wall_s']/3600:.2f} h)")
    print(f"  mean wall/step:  {summary['mean_wall_per_step_s']:.3f} s")
    print(f"  median wall/step:{summary['median_wall_per_step_s']:.3f} s")
    print(f"")
    print(f"  Per-equation:")
    for eq, st in summary["per_eq"].items():
        print(f"    {eq:>15s}: {st['n_solves']:>5d} solves, "
              f"iter median={st['iter_median']:.1f} (min {st['iter_min']}, max {st['iter_max']}), "
              f"resid median={st['resid_median']:.2e}")
    print(f"")
    print(f"  Wrote: {out_json}")
    print(f"  Wrote: {out_csv}")


if __name__ == "__main__":
    main()
