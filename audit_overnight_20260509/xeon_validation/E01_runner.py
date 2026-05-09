"""E01 — OF DICPCG actual per-solve wall via solverInfo function object.

This MODIFIES the OF case (system/controlDict) to add solverInfo,
re-runs laserMeltFoam for the full 1.2 μs case (~7h single-core),
parses postProcessing/solverInfo/<t>/solverInfo.dat for per-solve wall.

WARNING: Long-running. Requires --of-case path. Idempotent if controlDict already
has solverInfo and a previous result.json exists.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import record_environment, write_result_json


SOLVER_INFO_BLOCK = """
functions
{
    solverInfo
    {
        type            solverInfo;
        libs            (utilityFunctionObjects);
        fields          (pd T U alpha.material);
        writeControl    timeStep;
        writeInterval   1;
    }
}
"""


def add_solverInfo_to_controlDict(controlDict_path: Path):
    """Idempotently add functions block to system/controlDict."""
    text = controlDict_path.read_text()
    if "solverInfo" in text:
        print(f"  controlDict already has solverInfo block")
        return
    # Backup
    bak = controlDict_path.with_suffix(controlDict_path.suffix + ".pre_E01")
    if not bak.exists():
        shutil.copy2(controlDict_path, bak)
    # Append functions block (find a safe spot — e.g. end of file)
    if not text.rstrip().endswith("//"):
        text = text.rstrip() + "\n"
    text += SOLVER_INFO_BLOCK
    controlDict_path.write_text(text)
    print(f"  added solverInfo to {controlDict_path}")


def parse_solverInfo(case_dir: Path):
    """Walk postProcessing/solverInfo/<t>/solverInfo.dat and aggregate per-equation per-step wall."""
    si_dir = case_dir / "postProcessing" / "solverInfo"
    if not si_dir.exists():
        return []
    rows = []
    for time_dir in sorted(si_dir.iterdir(), key=lambda p: float(p.name) if p.name.replace('.', '').replace('e', '').replace('-', '').replace('+', '').isdigit() else 0):
        dat = time_dir / "solverInfo.dat"
        if not dat.exists(): continue
        # Format depends on OF version; typically:
        # Time pd_solver pd_initial_residual pd_final_residual pd_no_iterations ... [+ wall?]
        # But solverInfo doesn't include wall by default. We may need to parse log.run for that.
        with open(dat) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line: continue
                rows.append(line)
    return rows


def parse_log_run_for_wall(log_path: Path):
    """Parse log.run for Time = X / ExecutionTime = Y per timestep,
    plus DICPCG: Solving for pd, ... iter=N, finalRes=R lines.

    Returns list of dicts with: time, exec_time, pd_solves (list of {iter, init_res, final_res})
    """
    re_time = re.compile(r"^Time = (\S+)$")
    re_exec = re.compile(r"^ExecutionTime = (\S+) s")
    re_solve = re.compile(
        r"^(DICPCG|DILUPBiCG|GAMG|smoothSolver):\s*Solving for (\w+),"
        r"\s*Initial residual = (\S+),\s*Final residual = (\S+),"
        r"\s*No Iterations (\d+)"
    )

    rows = []
    cur_time = None
    cur_solves = []
    last_exec = 0.0

    if not log_path.exists():
        return []
    for line in log_path.read_text().split("\n"):
        m = re_time.match(line)
        if m:
            if cur_time is not None:
                rows.append({"time": cur_time, "solves": cur_solves})
            cur_time = float(m.group(1))
            cur_solves = []
            continue
        m = re_solve.search(line)
        if m:
            cur_solves.append({
                "algorithm": m.group(1),
                "equation": m.group(2),
                "init_residual": float(m.group(3)),
                "final_residual": float(m.group(4)),
                "iterations": int(m.group(5)),
            })
            continue
        m = re_exec.match(line)
        if m and rows:
            new_exec = float(m.group(1))
            rows[-1]["exec_time_at_step_end"] = new_exec
            rows[-1]["wall_delta_s"] = new_exec - last_exec
            last_exec = new_exec
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--of-case", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-rerun", action="store_true",
                     help="skip the laserMeltFoam re-run, just parse existing log.run")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    of_case = Path(args.of_case)
    if not of_case.exists():
        sys.exit(f"FAIL: --of-case {of_case} not found")

    env = record_environment()
    result_path = output_dir / "01" / "result.json"

    if args.resume and result_path.exists():
        print(f"E01: result.json exists, SKIP (resume)"); return

    # Step 1: add solverInfo (idempotent)
    controlDict = of_case / "system" / "controlDict"
    if not args.skip_rerun:
        add_solverInfo_to_controlDict(controlDict)
        # Step 2: re-run laserMeltFoam (clean + run)
        # Don't clean if --resume — assume user wants to keep partial dump
        if not args.smoke:
            print(f"  Re-running laserMeltFoam (~7h)...")
            print(f"  Command: cd {of_case} && nohup laserMeltFoam > log.run 2>&1")
            print(f"  WARNING: this script does NOT actually launch the run by default.")
            print(f"  Edit launching logic if you want this script to invoke laserMeltFoam.")
            # Actually we must NOT spawn an OF run inside this Python script
            # because the user may want to control it. Just signal what to do:
            print(f"")
            print(f"  Action required: launch laserMeltFoam manually:")
            print(f"    cd {of_case}")
            print(f"    nohup laserMeltFoam > log.run 2>&1 &")
            print(f"    disown")
            print(f"  Then re-run E01 with --skip-rerun once it completes.")

    # Step 3: parse log.run
    log_path = of_case / "log.run"
    if not log_path.exists():
        print(f"WARN: {log_path} not found. Cannot parse OF wall yet.")
        payload = {
            "expt_id": "E01",
            "status": "WAITING_FOR_LOG",
            "of_case": str(of_case),
            "env": env,
        }
        write_result_json(result_path, payload)
        return

    rows = parse_log_run_for_wall(log_path)
    print(f"  parsed {len(rows)} timesteps from log.run")

    # Aggregate pd_corr0 walls (note: log doesn't tell us per-solve wall directly,
    # only step total; we compute pd-iter share)
    import numpy as np
    pd_iters_per_step = []
    total_iters_per_step = []
    walls_per_step = []
    pd_solves = []  # individual pd_corr0 entries with iter
    for r in rows:
        if "wall_delta_s" not in r: continue
        step_iters_pd = sum(s["iterations"] for s in r["solves"]
                             if s["equation"] == "pd")
        step_iters_total = sum(s["iterations"] for s in r["solves"])
        # Approximate pd-share of wall (assume per-iter cost equal across solvers — coarse)
        if step_iters_total > 0:
            pd_share = step_iters_pd / step_iters_total
            pd_wall = r["wall_delta_s"] * pd_share
        else:
            pd_wall = 0
        for s in r["solves"]:
            if s["equation"] == "pd":
                pd_solves.append({
                    "time": r["time"],
                    "iter": s["iterations"],
                    "final_residual": s["final_residual"],
                    "approx_wall_s": pd_wall * s["iterations"] / max(step_iters_pd, 1),
                })
        walls_per_step.append(r["wall_delta_s"])
        pd_iters_per_step.append(step_iters_pd)
        total_iters_per_step.append(step_iters_total)

    payload = {
        "expt_id": "E01",
        "status": "PARSED",
        "of_case": str(of_case),
        "n_timesteps_in_log": len(rows),
        "total_steps_with_wall": len(walls_per_step),
        "n_pd_solves": len(pd_solves),
        "step_wall_stats": {
            "mean": float(np.mean(walls_per_step)) if walls_per_step else None,
            "median": float(np.median(walls_per_step)) if walls_per_step else None,
            "min": float(min(walls_per_step)) if walls_per_step else None,
            "max": float(max(walls_per_step)) if walls_per_step else None,
        },
        "pd_iter_per_step_stats": {
            "mean": float(np.mean(pd_iters_per_step)) if pd_iters_per_step else None,
            "median": float(np.median(pd_iters_per_step)) if pd_iters_per_step else None,
        },
        "total_iter_per_step_stats": {
            "mean": float(np.mean(total_iters_per_step)) if total_iters_per_step else None,
            "median": float(np.median(total_iters_per_step)) if total_iters_per_step else None,
        },
        "approx_per_iter_ms": (
            float(np.mean(walls_per_step)) / max(float(np.mean(total_iters_per_step)), 1)
            * 1000 if walls_per_step else None
        ),
        "pd_solves_sample": pd_solves[:50],
        "env": env,
    }
    write_result_json(result_path, payload)
    print(f"\n[E01] parsed; mean step wall = {payload['step_wall_stats']['mean']:.2f}s, "
          f"approx ms/iter = {payload.get('approx_per_iter_ms')}")


if __name__ == "__main__":
    main()
