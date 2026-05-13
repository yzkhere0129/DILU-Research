"""4-solver wall-time comparison bar chart on 6 single_track timesteps.

Reads dilu/amgx/bench/solver_comparison_500K.json.
Output: docs/benchmark/figures/solver_wall_compare_500K.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("/home/yzk/DILU-Research/dilu/amgx/bench/solver_comparison_500K.json")
OUT = Path("/home/yzk/DILU-Research/docs/benchmark/figures/solver_wall_compare_500K.png")

with open(SRC) as f:
    data = json.load(f)

labels = [r["label"] for r in data]
phases = [r["phase"] for r in data]

solvers = [
    ("OF DICPCG\n@ tol=1e-8",           "OF",          "#1f77b4"),
    ("AMGx PCG\n@ tol=1e-8",            "AMGx_e8",     "#ff7f0e"),
    ("AMGx PCG\n@ tol=1e-12 + 1 IR",    "AMGx_e12_IR", "#2ca02c"),
    ("CHOLMOD LU\n(direct, dev 16-th)", "LU_CHOLMOD",  "#d62728"),
]

# wall in seconds
wall_s = np.array([
    [r[key]["wall_ms"] / 1000.0 for r in data]
    for _, key, _ in solvers
])
iters = [
    [r[key]["iter"] for r in data]
    for _, key, _ in solvers
]
err_max = [
    [r[key]["err_max_Pa"] for r in data]
    for _, key, _ in solvers
]

fig, (ax_wall, ax_err) = plt.subplots(1, 2, figsize=(16, 6))

# ---- Panel 1: wall log-scale ----
n_solvers = len(solvers)
n_steps = len(labels)
x = np.arange(n_steps)
w = 0.2

for i, ((name, _, color), row, it_row) in enumerate(zip(solvers, wall_s, iters)):
    offset = (i - (n_solvers - 1) / 2) * w
    bars = ax_wall.bar(x + offset, row, w, label=name, color=color, edgecolor="black", linewidth=0.5)
    # annotate wall (s) on top
    for b, v, it in zip(bars, row, it_row):
        txt = f"{v:.1f}s\n({it})" if it != "direct" else f"{v:.1f}s\n(LU)"
        ax_wall.text(b.get_x() + b.get_width()/2, v * 1.05, txt,
                     ha="center", va="bottom", fontsize=7.5)

ax_wall.set_yscale("log")
ax_wall.set_xticks(x)
ax_wall.set_xticklabels([f"{l}\n({p})" for l, p in zip(labels, phases)], fontsize=8)
ax_wall.set_ylabel("Wall time (s) — log scale", fontsize=10)
ax_wall.set_title("Single-solve wall time on 500K LPBF pd matrix\n(top number = wall s, bottom = iter count)",
                  fontsize=11)
ax_wall.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax_wall.grid(True, axis="y", which="both", alpha=0.3)
ax_wall.set_ylim(0.3, 300)

# ---- Panel 2: err_max_Pa log-scale ----
# LU has err=0 (truth), set a floor for plotting
err_floor = 1e-7

for i, ((name, _, color), row) in enumerate(zip(solvers, err_max)):
    offset = (i - (n_solvers - 1) / 2) * w
    row_plot = [max(v, err_floor) for v in row]
    bars = ax_err.bar(x + offset, row_plot, w, label=name, color=color, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, row):
        if v == 0:
            txt = "0"
        elif v < 1e-3:
            txt = f"{v:.1e}"
        elif v < 1:
            txt = f"{v:.3g}"
        else:
            txt = f"{v:.1f}"
        ax_err.text(b.get_x() + b.get_width()/2, max(v, err_floor) * 1.4, txt,
                    ha="center", va="bottom", fontsize=7.5)

ax_err.set_yscale("log")
ax_err.set_xticks(x)
ax_err.set_xticklabels([f"{l}\n({p})" for l, p in zip(labels, phases)], fontsize=8)
ax_err.set_ylabel("max |Δ| vs CHOLMOD LU truth (Pa) — log scale", fontsize=10)
ax_err.set_title("Solution error vs CHOLMOD LU truth\n(LU plotted at floor 1e-7 Pa for visibility — actual = 0)",
                 fontsize=11)
ax_err.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
ax_err.grid(True, axis="y", which="both", alpha=0.3)
ax_err.set_ylim(err_floor * 0.5, 200)
# horizontal reference lines
ax_err.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
ax_err.text(n_steps - 0.5, 1.0, " 1 Pa", fontsize=7, va="center", color="gray")
ax_err.axhline(10.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
ax_err.text(n_steps - 0.5, 10.0, " 10 Pa", fontsize=7, va="center", color="gray")

fig.suptitle(
    "4-solver comparison on 500K LPBF pd matrices (single_track_dump, 6 timesteps)\n"
    "OF wall = log.run step-avg estimate (wall_estimated=true)  |  "
    "AMGx wall = dev RTX 3050 GPU  |  CHOLMOD wall = dev 16-thread BLAS  |  "
    "Xeon CHOLMOD single-thread = ~460s (lib limit)",
    fontsize=11, y=1.02
)
plt.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"→ {OUT}")

# Print summary stats
print("\nWall summary (s):")
print(f"  {'solver':<28} {'mean':>8} {'median':>8} {'min':>8} {'max':>8}")
for (name, _, _), row in zip(solvers, wall_s):
    n = name.replace("\n", " ")
    print(f"  {n:<28} {np.mean(row):>8.2f} {np.median(row):>8.2f} {np.min(row):>8.2f} {np.max(row):>8.2f}")

print("\nerr_max_Pa summary:")
for (name, _, _), row in zip(solvers, err_max):
    n = name.replace("\n", " ")
    vals = [v for v in row if v > 0]
    if not vals:
        print(f"  {n:<28} 0 (truth)")
    else:
        print(f"  {n:<28} mean={np.mean(vals):.3g} max={max(vals):.3g}")
