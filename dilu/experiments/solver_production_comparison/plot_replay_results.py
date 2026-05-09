"""Plot amortized replay results — wall-time per timestep + cumulative + bar chart.

Reads replay_results.json. Outputs:
  docs/benchmark/figures/replay_per_step.png    — line plot per-timestep wall
  docs/benchmark/figures/replay_cumulative.png  — total wall accumulating
  docs/benchmark/figures/replay_summary_bar.png — bar chart of total wall
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[3]
OUTDIR = REPO / "docs/benchmark/figures"


COLORS = {
    "amgx_fresh":           "tab:blue",
    "amgx_amortized":       "tab:cyan",
    "amgx_amortized_warm":  "tab:green",
    "lu_fresh_CHOLMOD":     "tab:red",
    "lu_fresh_SuperLU":     "tab:red",
    "lu_symbolic_reuse":    "tab:purple",
}

MARKERS = {
    "amgx_fresh":           "o",
    "amgx_amortized":       "s",
    "amgx_amortized_warm":  "D",
    "lu_fresh_CHOLMOD":     "^",
    "lu_fresh_SuperLU":     "^",
    "lu_symbolic_reuse":    "v",
}


def load(json_path: Path):
    return json.loads(json_path.read_text())


def main():
    json_path = REPO / "dilu/experiments/solver_production_comparison/replay_results.json"
    if not json_path.exists():
        raise SystemExit(f"missing {json_path} — run replay_amortized.py first")
    data = load(json_path)
    rows = data["per_step"]
    aggs = data["aggregates"]
    n_steps = data["n_timesteps"]

    # Group by mode
    modes = sorted(set(r["mode"] for r in rows))

    # ===== Fig 1: wall per step (line plot) =====
    fig, ax = plt.subplots(figsize=(13, 6))
    for mode in modes:
        sub = sorted([r for r in rows if r["mode"] == mode], key=lambda r: r["step"])
        if not sub: continue
        steps = [r["step"] for r in sub]
        walls = [r["total_ms"] for r in sub]
        c = COLORS.get(mode, "gray")
        m = MARKERS.get(mode, "o")
        ax.plot(steps, walls, marker=m, color=c, linewidth=1.5,
                 markersize=8, label=mode)
    ax.set_xlabel("PISO timestep index", fontsize=11)
    ax.set_ylabel("wall time per step (ms, log)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title(f"Per-timestep wall — replay of {n_steps} dumped pd matrices "
                  f"(500K cells, tol=1e-8)\n"
                  f"AMGx amortized: setup amortized 1× over {n_steps} steps",
                  fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    out = OUTDIR / "replay_per_step.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")

    # ===== Fig 2: cumulative wall =====
    fig, ax = plt.subplots(figsize=(13, 6))
    for mode in modes:
        sub = sorted([r for r in rows if r["mode"] == mode], key=lambda r: r["step"])
        if not sub: continue
        steps = [r["step"] for r in sub]
        walls = np.cumsum([r["total_ms"] for r in sub]) / 1000   # to seconds
        c = COLORS.get(mode, "gray")
        m = MARKERS.get(mode, "o")
        ax.plot(steps, walls, marker=m, color=c, linewidth=2,
                 markersize=8, label=f"{mode}  (total {walls[-1]:.1f}s)")
    ax.set_xlabel("PISO timestep index", fontsize=11)
    ax.set_ylabel("cumulative wall time (s)", fontsize=11)
    ax.set_title(f"Cumulative wall time — replay of {n_steps} dumped pd matrices "
                  f"(500K cells, tol=1e-8)",
                  fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    out = OUTDIR / "replay_cumulative.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")

    # ===== Fig 3: summary bar chart =====
    fig, ax = plt.subplots(figsize=(11, 6))
    mode_names = list(aggs.keys())
    totals = [aggs[m]["total_wall_s"] for m in mode_names]
    colors = [COLORS.get(m, "gray") for m in mode_names]
    bars = ax.bar(mode_names, totals, color=colors, alpha=0.85, edgecolor="black")
    for bar, t in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width()/2, t * 1.05,
                 f"{t:.2f}s\n({t*1000/n_steps:.0f} ms/step)",
                 ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(f"total wall time over {n_steps} timesteps (s, log)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title(f"Total wall — replay of {n_steps} dumped pd matrices  "
                  f"(500K cells, tol=1e-8)\n"
                  f"Lower is better.  amortized warm-start should win.",
                  fontsize=11)
    ax.grid(alpha=0.3, axis="y", which="both")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    out = OUTDIR / "replay_summary_bar.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")

    # Console summary
    print(f"\n{'='*80}")
    print(f"REPLAY SUMMARY — {n_steps} timesteps × {len(modes)} modes")
    print(f"{'='*80}")
    print(f"{'mode':<25s} {'total (s)':>10s} {'mean/step (ms)':>14s} "
          f"{'1st step':>10s} {'rest avg':>10s} {'speedup':>10s}")
    print("-"*82)
    base = max(aggs[m]["total_wall_s"] for m in mode_names)
    for m in mode_names:
        a = aggs[m]
        rest = a.get('rest_mean_ms')
        rest_s = f"{rest:.0f}ms" if rest is not None else "n/a"
        speedup = base / a["total_wall_s"]
        print(f"{m:<25s} {a['total_wall_s']:>10.2f} "
              f"{a['mean_per_step_ms']:>14.0f} "
              f"{a['first_step_ms']:>10.0f} {rest_s:>10s} "
              f"{speedup:>9.1f}×")


if __name__ == "__main__":
    main()
