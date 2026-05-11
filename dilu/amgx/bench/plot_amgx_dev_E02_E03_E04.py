"""Plot AMGx E02/E03/E04 dev (RTX 3050) results: fresh vs amortized vs warm-start.

Reads:
  audit_overnight_20260509/xeon_validation/results/E02/01_<ts>/result.json (6)
  audit_overnight_20260509/xeon_validation/results/E03/rep_01/result.json
  audit_overnight_20260509/xeon_validation/results/E04/rep_01/result.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/home/yzk/DILU-Research/audit_overnight_20260509/xeon_validation/results")
OUTDIR = Path("/home/yzk/DILU-Research/docs/benchmark/figures")

# Load
e02 = []
for f in sorted(R.glob("E02/01_*/result.json")):
    d = json.load(open(f)); e02.append(d)
e03 = json.load(open(R / "E03/rep_01/result.json"))
e04 = json.load(open(R / "E04/rep_01/result.json"))

# Sort E02 by timestep order
phase_order = {"3.2e-07": 0, "3.8e-07": 1, "4.1e-07": 2,
               "7e-07": 3, "9e-07": 4, "1.06e-06": 5}
e02_sorted = sorted(e02, key=lambda d: phase_order.get(d["timestep"], 99))
ts_labels = [f"{d['phase']}\n{d['timestep']}" for d in e02_sorted]
e02_wall = [d["wall_seconds"] for d in e02_sorted]
e02_iter = [d.get("iter_count", 0) for d in e02_sorted]

e03_steps = e03["per_step"]
e04_steps = e04["per_step"]
e03_wall = [s["total_ms"]/1000 for s in e03_steps]
e04_wall = [s["total_ms"]/1000 for s in e04_steps]
e03_iter = [s["iter_count"] for s in e03_steps]
e04_iter = [s["iter_count"] for s in e04_steps]
e04_sim  = [s.get("similarity_to_prev") for s in e04_steps]

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# (1) Per-step wall comparison
ax = axes[0, 0]
x = np.arange(6); w = 0.27
ax.bar(x - w, e02_wall, w, color="#d62728", label=f"E02 fresh    (Σ={sum(e02_wall):.1f}s)")
ax.bar(x,     e03_wall, w, color="#ff7f0e", label=f"E03 amortized (Σ={sum(e03_wall):.1f}s)")
ax.bar(x + w, e04_wall, w, color="#2ca02c", label=f"E04 +warm-start (Σ={sum(e04_wall):.1f}s)")
ax.set_xticks(x); ax.set_xticklabels(ts_labels, fontsize=8, rotation=0)
ax.set_ylabel("wall time per step (s)")
ax.set_title(f"AMGx solve wall per timestep (RTX 3050, AMGx 2.5.0 + 1 IR)\n"
              f"E04 vs E02 total speedup = {sum(e02_wall)/sum(e04_wall):.2f}×   "
              f"E03 vs E02 = {sum(e02_wall)/sum(e03_wall):.2f}× (amortize alone gives ~nothing)")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
for xi, v in enumerate(e02_wall):
    ax.text(xi - w, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
for xi, v in enumerate(e03_wall):
    ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
for xi, v in enumerate(e04_wall):
    ax.text(xi + w, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color="green", fontweight="bold")

# (2) PCG iter count comparison
ax = axes[0, 1]
ax.bar(x - w, e02_iter, w, color="#d62728", label="E02 fresh")
ax.bar(x,     e03_iter, w, color="#ff7f0e", label="E03 amortized (no warm)")
ax.bar(x + w, e04_iter, w, color="#2ca02c", label="E04 warm-start")
ax.set_xticks(x); ax.set_xticklabels(ts_labels, fontsize=8, rotation=0)
ax.set_ylabel("PCG iteration count to tol=1e-8")
ax.set_title("Warm-start cuts PCG iter by 50-65%  (C016 VERIFIED)")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
for xi, (a, b, c) in enumerate(zip(e02_iter, e03_iter, e04_iter)):
    ax.text(xi - w, a, str(a), ha="center", va="bottom", fontsize=7)
    ax.text(xi,     b, str(b), ha="center", va="bottom", fontsize=7)
    ax.text(xi + w, c, str(c), ha="center", va="bottom", fontsize=7, color="green", fontweight="bold")

# (3) Setup/update/solve breakdown for E03
ax = axes[1, 0]
setup_ms = [s.get("setup_ms", 0) for s in e03_steps]
update_ms = [s.get("update_ms", 0) for s in e03_steps]
solve_ms = [s["solve_ms"] for s in e03_steps]
ax.bar(x, setup_ms, color="#1f77b4", label=f"setup ({sum(setup_ms):.0f}ms total — only step 0)")
ax.bar(x, update_ms, bottom=setup_ms, color="#9467bd", label=f"update ({sum(update_ms):.0f}ms total)")
ax.bar(x, solve_ms, bottom=[s+u for s,u in zip(setup_ms, update_ms)], color="#ff7f0e",
       label=f"solve ({sum(solve_ms):.0f}ms total)")
ax.set_xticks(x); ax.set_xticklabels([f"step{i}" for i in range(6)])
ax.set_ylabel("time per step (ms)")
ax.set_title("E03 amortized breakdown — setup tiny vs solve dominant\n"
              f"C015 setup 2-3s HYPOTHESIS REFUTED — actual setup is {setup_ms[0]} ms")
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", alpha=0.3)

# (4) E04 warm-start similarity + iter relationship
ax = axes[1, 1]
ax2 = ax.twinx()
steps = list(range(6))
bars = ax.bar(steps, e04_iter, color="#2ca02c", alpha=0.7, label="warm-start iter")
sims_clean = [s if s is not None else np.nan for s in e04_sim]
ax2.plot(steps, sims_clean, marker="o", color="#d62728", linewidth=2,
         label="similarity_to_prev = (x_prev·x_curr)/(‖x_prev‖·‖x_curr‖)")
ax.set_xticks(steps); ax.set_xticklabels([f"step{i}\n{e04_steps[i]['phase']}" for i in steps], fontsize=8)
ax.set_ylabel("PCG iter (green bars)", color="#2ca02c")
ax2.set_ylabel("similarity (red line, 0-1)", color="#d62728")
ax.set_title("E04 warm-start: high similarity → fewer iter\n"
              f"step0 cold iter={e04_iter[0]}; mean of step1+ iter = {np.mean(e04_iter[1:]):.0f} "
              f"(cold→warm reduction = {(1-np.mean(e04_iter[1:])/e04_iter[0])*100:.0f}%)")
ax.legend(loc="upper left", fontsize=8)
ax2.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "AMGx E02/E03/E04 on dev RTX 3050 — single_track_dump 500K pd, AMGx PCG @ tol=1e-8 + 1 IR\n"
    f"HEADLINE: warm-start gives {sum(e02_wall)/sum(e04_wall):.2f}× wall speedup (16.2s vs 26.4s for 6 steps)   "
    f"|  amortize-setup-only = {sum(e02_wall)/sum(e03_wall):.2f}× (no benefit, C009 REFUTED)",
    fontsize=11
)
fig.tight_layout()
out = OUTDIR / "amgx_dev_E02_E03_E04_warmstart.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"  → {out}")
print(f"\nKey numbers:")
print(f"  E02 fresh sum: {sum(e02_wall):.2f}s  iters: {e02_iter}")
print(f"  E03 amort sum: {sum(e03_wall):.2f}s  iters: {e03_iter}")
print(f"  E04 warm  sum: {sum(e04_wall):.2f}s  iters: {e04_iter}")
print(f"  warm-vs-fresh speedup: {sum(e02_wall)/sum(e04_wall):.3f}×")
print(f"  amort-vs-fresh speedup: {sum(e02_wall)/sum(e03_wall):.3f}×")
print(f"  iter reduction (warm step1+ vs cold step0): "
      f"{(1-np.mean(e04_iter[1:])/e04_iter[0])*100:.1f}%")
