"""Parse E01 (laserMeltFoam 300-step run) outputs to settle C006.

Reads:
  audit_overnight_20260509/xeon_validation/results/E01/log.E01
  audit_overnight_20260509/xeon_validation/results/E01/solverInfo/1.2e-06/solverInfo.dat

Extracts per-timestep:
  - simulation time
  - pd iter counts (3 PISO correctors per timestep, from log.E01)
  - pd initial/final residuals (from solverInfo.dat)
  - ExecutionTime cumulative + delta (from log.E01)

Then computes:
  - per-step wall (ExecutionTime delta)
  - per-step total pd iters (sum of 3 correctors)
  - estimated ms per pd iter (upper bound: wall_step / pd_iters_total_step)

Outputs:
  audit_overnight_20260509/xeon_validation/results/E01/analysis.json
  docs/benchmark/figures/xeon_E01_pd_wall.png
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/yzk/DILU-Research/audit_overnight_20260509/xeon_validation/results/E01")
LOG = ROOT / "log.E01"
SI = ROOT / "solverInfo" / "1.2e-06" / "solverInfo.dat"
OUT_JSON = ROOT / "analysis.json"
OUT_FIG = Path("/home/yzk/DILU-Research/docs/benchmark/figures/xeon_E01_pd_wall.png")
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)


def parse_log(log_path: Path) -> list[dict]:
    """Walk through log.E01.  Each timestep is delimited by 'Time = X' followed by
    1-3 'Solving for pd' lines and one 'ExecutionTime = X s ClockTime = Y s' line."""
    records: list[dict] = []
    current: dict | None = None
    pd_re = re.compile(
        r"Solving for pd, Initial residual = (\S+), Final residual = (\S+), No Iterations (\d+)")
    time_re = re.compile(r"^Time = (\S+)")
    exec_re = re.compile(r"^ExecutionTime = (\S+) s\s+ClockTime")

    with open(log_path) as f:
        for line in f:
            m = time_re.match(line)
            if m:
                if current is not None and "pd_iters_all" in current:
                    records.append(current)
                current = {"time": float(m.group(1)), "pd_iters_all": [],
                           "pd_initial_all": [], "pd_final_all": [],
                           "exec_time": None}
                continue
            if current is None:
                continue
            m = pd_re.search(line)
            if m:
                current["pd_initial_all"].append(float(m.group(1)))
                current["pd_final_all"].append(float(m.group(2)))
                current["pd_iters_all"].append(int(m.group(3)))
                continue
            m = exec_re.match(line)
            if m:
                current["exec_time"] = float(m.group(1))
    if current is not None and "pd_iters_all" in current:
        records.append(current)
    return records


def parse_solverInfo(si_path: Path) -> list[dict]:
    """solverInfo.dat has columns: Time T_solver T_init T_final T_iters T_conv pd_solver pd_init pd_final pd_iters pd_conv"""
    out = []
    with open(si_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            try:
                out.append({
                    "time": float(parts[0]),
                    "T_solver": parts[1],
                    "T_iters": int(parts[4]),
                    "pd_solver": parts[6],
                    "pd_initial": float(parts[7]),
                    "pd_final": float(parts[8]),
                    "pd_iters": int(parts[9]),
                    "pd_converged": parts[10] == "true",
                })
            except (ValueError, IndexError):
                continue
    return out


def main():
    print(f"Reading log:  {LOG}")
    log_records = parse_log(LOG)
    print(f"  → {len(log_records)} timesteps parsed from log")

    print(f"Reading solverInfo:  {SI}")
    si_records = parse_solverInfo(SI)
    print(f"  → {len(si_records)} timesteps parsed from solverInfo")

    # Compute per-step wall (delta of ExecutionTime)
    log_records = [r for r in log_records if r["exec_time"] is not None and r["pd_iters_all"]]
    print(f"  → {len(log_records)} timesteps with both ExecutionTime and pd info")

    # Walls
    walls = []  # per-step wall
    iters_sum = []  # sum of all corrector pd iter counts
    iters_first = []  # first corrector iter (the "main" pd solve)
    iters_max = []  # max of any corrector in a step
    pd_n_corrs = []  # how many pd calls per step (1-3)
    times = []
    prev_exec = log_records[0]["exec_time"]
    for r in log_records[1:]:
        dt = r["exec_time"] - prev_exec
        walls.append(dt)
        iters_sum.append(sum(r["pd_iters_all"]))
        iters_first.append(r["pd_iters_all"][0])
        iters_max.append(max(r["pd_iters_all"]))
        pd_n_corrs.append(len(r["pd_iters_all"]))
        times.append(r["time"])
        prev_exec = r["exec_time"]

    walls = np.array(walls)
    iters_sum = np.array(iters_sum)
    iters_first = np.array(iters_first)
    iters_max = np.array(iters_max)
    times = np.array(times)

    # Compute approximate ms per pd iter (upper bound — wall includes T/VOF/laser too)
    ms_per_pd_iter_upper = walls * 1000 / iters_sum

    # Cross-check with solverInfo (which has only ONE pd entry per timestep — usually the
    # last corrector or just one)
    si_iters = np.array([r["pd_iters"] for r in si_records])
    si_residuals_init = np.array([r["pd_initial"] for r in si_records])
    si_residuals_final = np.array([r["pd_final"] for r in si_records])

    summary = {
        "n_timesteps": len(walls),
        "time_range_s": [float(times.min()), float(times.max())],
        "wall_per_step_s": {
            "min": float(walls.min()), "max": float(walls.max()),
            "mean": float(walls.mean()), "median": float(np.median(walls)),
            "std": float(walls.std()),
        },
        "pd_iters_per_step_total": {
            "min": int(iters_sum.min()), "max": int(iters_sum.max()),
            "mean": float(iters_sum.mean()), "median": int(np.median(iters_sum)),
            "std": float(iters_sum.std()),
        },
        "pd_iters_first_corrector": {
            "min": int(iters_first.min()), "max": int(iters_first.max()),
            "mean": float(iters_first.mean()), "median": int(np.median(iters_first)),
        },
        "pd_iters_corrector_counts": {
            "1-corr": int(np.sum(pd_n_corrs == 1)),
            "2-corr": int(np.sum(pd_n_corrs == 2)),
            "3-corr": int(np.sum(pd_n_corrs == 3)),
        },
        "ms_per_pd_iter_upper_bound": {
            "min": float(ms_per_pd_iter_upper.min()),
            "max": float(ms_per_pd_iter_upper.max()),
            "mean": float(ms_per_pd_iter_upper.mean()),
            "median": float(np.median(ms_per_pd_iter_upper)),
            "note": "wall / iter — includes T solver + VOF + laser ray-trace; pd is one component"
        },
        "solverInfo_cross_check": {
            "n": len(si_records),
            "pd_iters_mean": float(si_iters.mean()),
            "pd_iters_max": int(si_iters.max()),
            "pd_iters_min": int(si_iters.min()),
            "residuals_initial_mean": float(si_residuals_init.mean()),
            "residuals_final_mean": float(si_residuals_final.mean()),
            "all_converged": all(r["pd_converged"] for r in si_records),
        }
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"  → {OUT_JSON}")

    # Print headline
    print(f"\n=== E01 headline (C006 settle) ===")
    print(f"  Sample size:    {len(walls)} timesteps (t={times.min():.2e} → {times.max():.2e} s)")
    print(f"  per-step wall:  mean {walls.mean():.1f}s  median {np.median(walls):.1f}s  range [{walls.min():.1f}, {walls.max():.1f}]s")
    print(f"  pd iter/step:   mean {iters_sum.mean():.0f}  median {int(np.median(iters_sum))}  range [{iters_sum.min()}, {iters_sum.max()}]")
    print(f"  ms/pd-iter UB:  mean {ms_per_pd_iter_upper.mean():.0f}ms  median {np.median(ms_per_pd_iter_upper):.0f}ms")
    print(f"  (UB because wall includes T solver + VOF + laser ray-trace, not just pd)")
    print()
    print(f"  Prior claim (predecessor session): wall = pd_iters × 110 ms (THIS IS A WALL ms/iter)")
    print(f"  → C006 verdict:")
    if 80 <= np.median(ms_per_pd_iter_upper) <= 150:
        print(f"    Mean ms/iter UB ({np.median(ms_per_pd_iter_upper):.0f}) is COMPATIBLE with 110ms estimate")
        print(f"    BUT: original number was arithmetic not measurement (build_solver_comparison_table.py:104)")
        print(f"    Verdict: ORIGINAL ARITHMETIC LUCKY-HIT (estimate happened to be close to actual)")
    else:
        print(f"    Mean ms/iter UB ({np.median(ms_per_pd_iter_upper):.0f}) is OUT OF the 80-150ms range")
        print(f"    → 110ms estimate was wrong by {np.median(ms_per_pd_iter_upper)/110:.2f}×")

    # ----- Plot -----
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    ax.plot(times*1e6, walls, marker=".", ms=3, linewidth=0.8, color="#1f77b4")
    ax.axhline(walls.mean(), color="red", linestyle="--", linewidth=1,
               label=f"mean = {walls.mean():.1f}s")
    ax.set_xlabel("simulation time (μs)")
    ax.set_ylabel("wall per timestep (s)")
    ax.set_title("E01 per-step wall time (300 timesteps, deltaT=1ns)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(times*1e6, iters_sum, marker=".", ms=3, linewidth=0.8, color="#d62728",
            label="total iter (sum of correctors)")
    ax.plot(times*1e6, iters_first, marker=".", ms=2, linewidth=0.6, color="#2ca02c",
            label="first corrector only")
    ax.axhline(iters_sum.mean(), color="red", linestyle="--", linewidth=1,
               label=f"mean total = {iters_sum.mean():.0f}")
    ax.set_xlabel("simulation time (μs)")
    ax.set_ylabel("pd PCG iter count")
    ax.set_title(f"E01 per-step pd iter (PISO 3 correctors)\n"
                  f"corrector mix: {summary['pd_iters_corrector_counts']}")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.hist(ms_per_pd_iter_upper, bins=40, color="#ff7f0e", edgecolor="black", alpha=0.7)
    ax.axvline(110, color="red", linestyle="--", linewidth=2,
               label="predecessor claim: 110 ms/iter")
    ax.axvline(np.median(ms_per_pd_iter_upper), color="blue", linestyle="--", linewidth=2,
               label=f"measured median: {np.median(ms_per_pd_iter_upper):.0f} ms/iter")
    ax.set_xlabel("wall ms per pd iter (upper bound)")
    ax.set_ylabel("count")
    ax.set_title("ms / pd-iter distribution (wall / pd_iters per step, upper bound)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(iters_sum, walls, alpha=0.4, s=12, color="#9467bd")
    # Fit a line through origin (the "ms per iter" interpretation)
    if len(iters_sum) > 5:
        slope, intercept = np.polyfit(iters_sum, walls, 1)
        x_fit = np.linspace(iters_sum.min(), iters_sum.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept,
                color="red", linewidth=1.5,
                label=f"linear: wall = {slope*1000:.0f}ms × iter + {intercept:.2f}s")
    ax.set_xlabel("total pd iters per step")
    ax.set_ylabel("wall per step (s)")
    ax.set_title("Wall vs pd-iter linearity (intercept = non-pd cost)")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(
        f"E01 Xeon laserMeltFoam — 300 timesteps t=1.20e-6 → 1.50e-6, BLAS-pin=1 thread\n"
        f"per-step wall mean {walls.mean():.1f}s, pd iter total mean {iters_sum.mean():.0f}, "
        f"ms/pd-iter UB median {np.median(ms_per_pd_iter_upper):.0f}",
        fontsize=11, y=0.995
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    print(f"  → {OUT_FIG}")


if __name__ == "__main__":
    main()
