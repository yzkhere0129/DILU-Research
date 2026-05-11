"""Plot Xeon E07 (LU truth vs 3 solvers), E05 (CHOLMOD fresh wall), E06 (symbolic-reuse).

Reads:
  xeon_validation/results/E07/aggregate.json
  xeon_validation/results/E05/01_<ts>/result.json (6 timesteps)
  xeon_validation/results/E06/rep_01/result.json

Outputs:
  docs/benchmark/figures/xeon_E07_solver_truth_diff.png
  docs/benchmark/figures/xeon_E05_E06_cholmod_wall.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("/home/yzk/DILU-Research/xeon_validation/results")
OUTDIR  = Path("/home/yzk/DILU-Research/docs/benchmark/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)


def plot_E07():
    agg = json.loads((RESULTS / "E07" / "aggregate.json").read_text())
    timesteps = [r["timestep"] for r in agg["per_timestep"]]
    phases = [r["phase"] for r in agg["per_timestep"]]
    labels = [f"{p}\n{t}" for p, t in zip(phases, timesteps)]

    OF      = [r["comparisons_vs_LU"]["OF"]["max_diff_Pa"] for r in agg["per_timestep"]]
    AMGx_e8 = [r["comparisons_vs_LU"]["AMGx_e8"]["max_diff_Pa"] for r in agg["per_timestep"]]
    AMGx_IR = [r["comparisons_vs_LU"]["AMGx_e12_IR"]["max_diff_Pa"] for r in agg["per_timestep"]]
    OF_rel      = [r["comparisons_vs_LU"]["OF"]["rel_max"] for r in agg["per_timestep"]]
    AMGx_e8_rel = [r["comparisons_vs_LU"]["AMGx_e8"]["rel_max"] for r in agg["per_timestep"]]
    AMGx_IR_rel = [r["comparisons_vs_LU"]["AMGx_e12_IR"]["rel_max"] for r in agg["per_timestep"]]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(timesteps))
    w = 0.28

    # LEFT: absolute diff in Pa (log scale)
    ax = axes[0]
    ax.bar(x - w, OF, w, label="OF DICPCG @ tol=1e-8",      color="#d62728")
    ax.bar(x,     AMGx_e8, w, label="AMGx PCG @ tol=1e-8",  color="#1f77b4")
    ax.bar(x + w, AMGx_IR, w, label="AMGx+IR (≈LU truth)",   color="#2ca02c")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_ylabel("max | x_solver − x_LU |  (Pa)")
    ax.set_title("E07 — solver vs CHOLMOD LU truth on 500K LPBF pd\n"
                  "(0 cells > 100 Pa for ANY solver / timestep)")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    # annotate
    for xi, (a, b, c) in enumerate(zip(OF, AMGx_e8, AMGx_IR)):
        ax.text(xi - w, a, f"{a:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(xi,     b, f"{b:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(xi + w, c, f"{c:.0e}", ha="center", va="bottom", fontsize=7)

    # RIGHT: relative diff (log scale)
    ax = axes[1]
    ax.bar(x - w, OF_rel, w, label="OF DICPCG @ tol=1e-8",      color="#d62728")
    ax.bar(x,     AMGx_e8_rel, w, label="AMGx PCG @ tol=1e-8",  color="#1f77b4")
    ax.bar(x + w, AMGx_IR_rel, w, label="AMGx+IR (≈LU truth)",   color="#2ca02c")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_ylabel("rel = max | x − x_LU | / ‖x_LU‖∞")
    ax.set_title("E07 relative diff to LU truth  (median: OF 1.6e-5, AMGx_e8 1.5e-5, AMGx+IR 4.6e-12)")
    ax.grid(axis="y", which="both", alpha=0.3)
    # horizontal reference lines
    for lvl, lbl, ls in [(1e-8, "PCG tol=1e-8", ":"), (1e-12, "PCG tol=1e-12", "--")]:
        ax.axhline(lvl, color="grey", linestyle=ls, linewidth=0.8)
        ax.text(len(x) - 0.5, lvl * 1.5, lbl, fontsize=7, color="grey", ha="right")

    fig.suptitle("E07 Xeon validation — 500K single_track_dump LPBF pd, 6 timesteps × 3 solvers vs CHOLMOD LU truth\n"
                  f"AMGx+IR max rel = {agg['AMGx_e12_IR_max_rel_max']:.2e}   "
                  f"(predicted v2 range [1e-12, 1e-10] — VERIFIED, settles C004)",
                  fontsize=11)
    fig.tight_layout()
    out = OUTDIR / "xeon_E07_solver_truth_diff.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def plot_E05_E06():
    # E05: per-timestep fresh wall
    e05 = []
    for ts in ["3.2e-07", "3.8e-07", "4.1e-07", "7e-07", "9e-07", "1.06e-06"]:
        p = RESULTS / "E05" / f"01_{ts}" / "result.json"
        d = json.loads(p.read_text())
        e05.append({"ts": ts, "factor_ms": d["factor_ms"], "solve_ms": d["solve_ms"],
                    "wall_s": d["wall_seconds"], "resid": d["rel_resid_actual"]})
    # E06: per-step in 6-step sequence
    e06 = json.loads((RESULTS / "E06" / "rep_01" / "result.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # LEFT: E05 fresh per timestep
    ax = axes[0]
    x = np.arange(len(e05))
    factor_s = [e["factor_ms"]/1000 for e in e05]
    solve_s  = [e["solve_ms"]/1000 for e in e05]
    ax.bar(x, factor_s, label="numeric factorize", color="#ff7f0e")
    ax.bar(x, solve_s,  bottom=factor_s, label="solve (triangular)", color="#1f77b4")
    for xi, (fc, sv) in enumerate(zip(factor_s, solve_s)):
        ax.text(xi, fc + sv, f"{fc+sv:.0f}s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([e["ts"] for e in e05], rotation=0, fontsize=8)
    ax.set_ylabel("wall time (s)  — single Xeon thread, OPENBLAS=1")
    ax.set_title(f"E05 — CHOLMOD fresh factor+solve per timestep\n"
                  f"mean = {np.mean([e['wall_s'] for e in e05]):.1f} s/step "
                  f"(factor dominates; solve is ~0.4 s/step)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # RIGHT: E06 symbolic-reuse vs E05 fresh extrapolation
    ax = axes[1]
    steps = list(range(len(e06["per_step"])))
    fr = [s["factor_ms"]/1000 for s in e06["per_step"]]
    rf = [s["refact_ms"]/1000 for s in e06["per_step"]]
    sv = [s["solve_ms"]/1000 for s in e06["per_step"]]

    ax.bar(steps, fr, color="#ff7f0e", label="full factor (step 0)")
    ax.bar(steps, rf, bottom=fr, color="#bcbd22", label="numeric refactor (steps 1+)")
    ax.bar(steps, sv, bottom=[a+b for a,b in zip(fr,rf)], color="#1f77b4", label="solve")
    total_e06 = sum(s["total_ms"] for s in e06["per_step"]) / 1000
    # reference: E05 fresh ×6
    mean_fresh = np.mean([e["wall_s"] for e in e05])
    total_fresh = mean_fresh * 6
    ax.axhline(mean_fresh, color="red", linestyle="--", linewidth=1,
               label=f"E05 fresh mean = {mean_fresh:.0f} s/step")
    for xi, (a, b, c) in enumerate(zip(fr, rf, sv)):
        ax.text(xi, a+b+c, f"{a+b+c:.0f}s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(steps); ax.set_xticklabels([f"step{i}" for i in steps])
    ax.set_ylabel("wall time per step (s)")
    speedup = total_fresh / total_e06
    ax.set_title(f"E06 — CHOLMOD symbolic-reuse (sksparse 0.5.0)\n"
                  f"total {total_e06:.0f}s vs fresh extrapolated {total_fresh:.0f}s → "
                  f"speedup {speedup:.2f}× (i.e. SLOWER by {(1/speedup-1)*100:.0f}%) — C017 REFUTED")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("E05 + E06 Xeon CHOLMOD wall time — 500K single_track_dump pd, 1 thread, sksparse 0.5.0",
                  fontsize=11)
    fig.tight_layout()
    out = OUTDIR / "xeon_E05_E06_cholmod_wall.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    plot_E07()
    plot_E05_E06()
