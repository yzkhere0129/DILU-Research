"""Plot 5-rep variance for E05 + E06 CHOLMOD wall on Xeon.

For each E05 timestep: 5 wall-time replicates → bar + error bar (mean ± std).
For E06: 5 reps × 6 steps per rep → per-step error bars + total wall stats.

Also flags rep01 (single-process, no E09 concurrent) vs rep02-05 (concurrent with E09).
This is honest about the variance source revealed during the run.

Output: docs/benchmark/figures/xeon_E05_E06_variance.png
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

# E05 — load 5 reps × 6 timesteps
E05 = {}  # E05[ts] = [wall_s × 5 reps]
for d in sorted((R / "E05").iterdir()):
    if not d.is_dir(): continue
    rep = int(d.name.split("_")[0])  # 01_xxx → rep 1
    ts = d.name.split("_", 1)[1]
    rj = d / "result.json"
    if not rj.exists(): continue
    js = json.load(open(rj))
    E05.setdefault(ts, []).append((rep, js["wall_seconds"], js["factor_ms"], js["solve_ms"],
                                     js["rel_resid_actual"]))

# E06 — 5 reps, each with 6 per_step
E06 = []  # list of dicts
for rd in sorted((R / "E06").iterdir()):
    if not rd.is_dir(): continue
    js = json.load(open(rd / "result.json"))
    E06.append({"rep": js["rep"], "wall": js["wall_seconds"], "per_step": js["per_step"]})
E06.sort(key=lambda d: d["rep"])

# Order timesteps chronologically
TS_ORDER = ["3.2e-07", "3.8e-07", "4.1e-07", "7e-07", "9e-07", "1.06e-06"]
phase_order = {"3.2e-07": "melting", "3.8e-07": "melting", "4.1e-07": "melting",
                "7e-07": "evap_early", "9e-07": "evap", "1.06e-06": "evap_late"}

# Plot
fig, axes = plt.subplots(2, 2, figsize=(17, 12))

# (0,0): E05 wall per timestep with 5-rep error bars
ax = axes[0, 0]
x = np.arange(len(TS_ORDER))
means = []
stds = []
all_reps = []
for ts in TS_ORDER:
    walls = [w for rep, w, fm, sm, res in sorted(E05[ts])]
    means.append(np.mean(walls)); stds.append(np.std(walls)); all_reps.append(walls)
ax.bar(x, means, yerr=stds, capsize=5, color="#ff7f0e",
       error_kw={"elinewidth": 1.5, "ecolor": "black"})
# overlay individual reps as dots
for xi, walls in enumerate(all_reps):
    ax.scatter([xi]*len(walls), walls, s=18, color="black", alpha=0.7, zorder=3)
ax.set_xticks(x); ax.set_xticklabels([f"{phase_order[t]}\n{t}" for t in TS_ORDER], fontsize=8)
ax.set_ylabel("wall (s) per CHOLMOD fresh factor+solve")
overall_mean = np.mean([m for m in means])
overall_cv = np.mean(stds) / overall_mean * 100
ax.set_title(f"E05 — 5-rep CHOLMOD fresh wall per 500K timestep   "
              f"grand mean = {overall_mean:.1f} s, mean CV = {overall_cv:.1f}%")
ax.grid(axis="y", alpha=0.3)
for xi, (m, s) in enumerate(zip(means, stds)):
    ax.text(xi, m + s + 5, f"{m:.0f}±{s:.0f}", ha="center", va="bottom", fontsize=8)

# (0,1): E06 total wall per rep
ax = axes[0, 1]
rep_walls = [r["wall"] for r in E06]
# Annotate: rep01 was single-process; rep02-05 ran concurrently with E09
colors = ["#1f77b4"] + ["#d62728"]*4
labels = ["solo (rep01)"] + ["concurrent with E09 (rep02-05)"]*4
ax.bar(range(1, 6), rep_walls, color=colors)
for i, w in enumerate(rep_walls):
    ax.text(i+1, w + 30, f"{w:.0f}s", ha="center", va="bottom", fontsize=9)
ax.axhline(rep_walls[0], color="#1f77b4", linestyle="--", linewidth=1.2,
           label=f"rep01 = {rep_walls[0]:.0f}s (no concurrency)")
ax.axhline(np.mean(rep_walls[1:]), color="#d62728", linestyle="--", linewidth=1.2,
           label=f"rep02-05 mean = {np.mean(rep_walls[1:]):.0f}s (concurrent)")
ax.set_xlabel("rep"); ax.set_ylabel("E06 total wall (s) for 6-step sequence")
ax.set_title(f"E06 — 5-rep total wall, BLAS pin=1 thread still suffered cache contention\n"
              f"concurrent slowdown: {(np.mean(rep_walls[1:])/rep_walls[0] - 1)*100:.0f}% vs solo")
ax.set_xticks(range(1, 6))
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# (1,0): E06 per-step breakdown across reps
ax = axes[1, 0]
n_steps = 6
step_x = np.arange(n_steps)
for rep_d in E06:
    rep_id = rep_d["rep"]
    steps = rep_d["per_step"]
    totals = [s["total_ms"]/1000 for s in steps]
    color = "#1f77b4" if rep_id == 1 else "#d62728"
    alpha = 1.0 if rep_id == 1 else 0.55
    ax.plot(step_x, totals, marker="o", color=color, alpha=alpha,
             label=f"rep{rep_id:02d}{' (solo)' if rep_id == 1 else ''}")
ax.set_xticks(step_x)
ax.set_xticklabels(["step0\nfull factor"] + [f"step{i}\nrefactor" for i in range(1, 6)])
ax.set_ylabel("per-step wall (s)  [full_factor or symbolic_reuse_refactor + solve]")
ax.set_title("E06 per-step wall across 5 reps — refactor (steps 1-5) slower than full factor (step 0)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(axis="y", alpha=0.3)

# (1,1): summary numbers + headline ratios
ax = axes[1, 1]
ax.axis("off")
e05_solo = [E05[ts][0][1] for ts in TS_ORDER]  # rep01 walls
e05_solo_total = sum(e05_solo)
e06_solo_total = E06[0]["wall"]
e05_all_means = [np.mean([w for r,w,fm,sm,res in E05[ts]]) for ts in TS_ORDER]
e05_total_mean = sum(e05_all_means)
e06_total_mean = np.mean(rep_walls)

txt = f"""E05 + E06 5-rep summary (Xeon HR54WV2, BLAS pin=1 thread, sksparse 0.5.0)

E05 fresh factor (per 500K timestep, 5 reps):
  per-timestep mean wall:   {np.mean(e05_all_means):.1f} s  (mean of 6 timestep means)
  per-timestep CV:          {overall_cv:.1f}%   ← small, single-thread is reproducible
  range across timesteps:   {min(e05_all_means):.0f}–{max(e05_all_means):.0f} s
  6-timestep total mean:    {e05_total_mean:.0f} s
  rep01 (solo) total:       {e05_solo_total:.0f} s  ← cleanest baseline

E06 symbolic-reuse (5 reps × 6 steps):
  rep01 wall  (solo):       {E06[0]['wall']:.0f} s     ← baseline
  rep02-05 mean (concurrent E09 running):
                            {np.mean(rep_walls[1:]):.0f} s  ← +{(np.mean(rep_walls[1:])/E06[0]['wall']-1)*100:.0f}% from cache contention

C017 verdict locked using CLEAN rep01 data:
  E05 solo total ({e05_solo_total:.0f}s) vs E06 solo ({e06_solo_total:.0f}s)
  E06 / E05 ratio = {e06_solo_total/e05_solo_total:.3f}   (>1.0 means slower)
  → symbolic-reuse is {(e06_solo_total/e05_solo_total-1)*100:.0f}% SLOWER than 6 fresh factors
  → C017 REFUTED (was: 2-5× speedup; actual: 30% slowdown solo)

Honest scope note: rep02-05 ran concurrent with E09 → cache contention inflated
their walls. Use rep01-only for clean E05 vs E06 comparison."""
ax.text(0.0, 1.0, txt, transform=ax.transAxes, family="monospace",
         fontsize=9.5, verticalalignment="top")

fig.suptitle("E05 + E06 Xeon 5-rep variance — CHOLMOD fresh vs symbolic-reuse on 500K LPBF pd",
              fontsize=12, y=0.995)
fig.tight_layout()
out = OUTDIR / "xeon_E05_E06_variance.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"  → {out}")

# Numerical summary
print(f"\n=== E05 5-rep stats ===")
for ts in TS_ORDER:
    walls = [w for rep, w, fm, sm, res in sorted(E05[ts])]
    print(f"  {phase_order[ts]:<10s} {ts:<10s}  walls={[f'{w:.1f}' for w in walls]}  "
          f"mean={np.mean(walls):.2f}±{np.std(walls):.2f}s  CV={np.std(walls)/np.mean(walls)*100:.2f}%")
print(f"\n=== E06 5-rep total walls ===")
for r in E06:
    print(f"  rep{r['rep']:02d}: {r['wall']:.1f} s")
print(f"\nE05 rep01 total (solo, fresh ×6):  {e05_solo_total:.0f} s")
print(f"E06 rep01 total (solo, symbolic):   {e06_solo_total:.0f} s")
print(f"E06/E05 ratio (solo): {e06_solo_total/e05_solo_total:.3f}× (symbolic SLOWER)")
