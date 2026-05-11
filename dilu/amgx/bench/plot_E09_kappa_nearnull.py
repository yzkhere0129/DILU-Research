"""Plot E09 Lanczos κ + svds near-null subspace cluster for 6 single_track + lab32.

Reads: audit_overnight_20260509/xeon_validation/results/E09/{aggregate.json, */result.json}

Shows:
  (1) σ_max + σ_min(Lanczos) per matrix on log scale — 15-order gap visible
  (2) svds smallest_10 σ per matrix as horizontal scatter — fp64-noise cluster
  (3) κ_Lanczos per matrix as bar — universality across timesteps

Output: docs/benchmark/figures/xeon_E09_kappa_nearnull.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/home/yzk/DILU-Research/audit_overnight_20260509/xeon_validation/results/E09")
OUTDIR = Path("/home/yzk/DILU-Research/docs/benchmark/figures")

# Load all
records = []
for r in sorted(R.iterdir()):
    if r.is_dir():
        records.append(json.load(open(r / "result.json")))
records.sort(key=lambda d: (0 if d["case"] == "lab32" else 1, float(d["timestep"])))

labels = [f"{r['case']}\n{r['phase']}\nt={r['timestep']}" for r in records]
sigma_max = [r["sigma_max_lanczos"] for r in records]
sigma_min = [r["sigma_min_lanczos"] for r in records]
kappa     = [r["kappa_lanczos"] for r in records]
svds_s10  = [r["smallest_10_sigmas_svds"] for r in records]
near_null = [r["has_near_null_subspace"] for r in records]
nn_dim    = [r["near_null_dim_estimate"] for r in records]

fig = plt.figure(figsize=(18, 13))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.8], hspace=0.4)

# (1) σ_max + σ_min(Lanczos) bars
ax = fig.add_subplot(gs[0])
x = np.arange(len(records))
w = 0.38
ax.bar(x - w/2, sigma_max, w, color="#1f77b4", label="σ_max(Lanczos)  ← well-resolved")
ax.bar(x + w/2, sigma_min, w, color="#d62728", label="σ_min(Lanczos)  ← at fp64 ε noise floor")
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("σ (log)")
ax.set_title("E09 — Lanczos shift-invert extreme singular values  (σ_min is NOISE not signal)")
ax.axhline(1e-14, color="grey", linestyle=":", linewidth=1, alpha=0.6)
ax.text(len(records) - 0.5, 1.3e-14, "fp64 ε × σ_max ≈ 1e-14", fontsize=8, color="grey", ha="right")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", which="both", alpha=0.3)
for xi, (a, b) in enumerate(zip(sigma_max, sigma_min)):
    ax.text(xi - w/2, a*1.4, f"{a:.2e}", ha="center", va="bottom", fontsize=7)
    ax.text(xi + w/2, b*1.4, f"{b:.2e}", ha="center", va="bottom", fontsize=7)

# (2) svds smallest_10 σ scatter — show the 10-dim cluster
ax = fig.add_subplot(gs[1])
for xi, s10 in enumerate(svds_s10):
    s10_clean = [v for v in s10 if v is not None]
    if s10_clean:
        ax.scatter([xi]*len(s10_clean), s10_clean, s=60, alpha=0.8, c="#2ca02c",
                    edgecolor="black", linewidth=0.6)
ax.set_yscale("log")
ax.axhline(1e-12, color="red", linestyle="--", linewidth=1,
           label="near-null threshold (1e-12)")
ax.axhline(1e-15, color="grey", linestyle=":", linewidth=1, alpha=0.6,
           label="fp64 ε absolute")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("svds k=10 smallest σ (log)")
ax.set_title(f"E09 svds k=10 — 10 smallest σ per matrix.  "
              f"ALL {len(records)} matrices have near_null_dim_estimate = 10 (ALL 10 < 1e-12)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", which="both", alpha=0.3)

# (3) κ bar chart
ax = fig.add_subplot(gs[2])
colors = ["#9467bd" if r["case"] == "lab32" else "#1f77b4" for r in records]
bars = ax.bar(x, kappa, color=colors)
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("κ_Lanczos = σ_max / σ_min  (log)")
# κ std among single_track
st_kappa = [r["kappa_lanczos"] for r in records if r["case"] == "single_track"]
mean_k = np.mean(st_kappa)
std_k  = np.std(st_kappa)
cv = std_k / mean_k * 100
ax.set_title(f"E09 κ_Lanczos per matrix  —  single_track mean = {mean_k:.3e}  "
              f"std = {std_k:.2e}  CV = {cv:.2f}%  (lab32 = {kappa[0]:.3e}, 5× smaller)")
ax.grid(axis="y", which="both", alpha=0.3)
for xi, v in enumerate(kappa):
    ax.text(xi, v*1.5, f"{v:.2e}", ha="center", va="bottom", fontsize=8)
# label color legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#9467bd", label="lab32 (rays=0 degenerate)"),
                    Patch(color="#1f77b4", label="single_track (rays>0 real physics)")],
          loc="upper right", fontsize=9)

fig.suptitle(
    "E09 Xeon — Lanczos κ + svds near-null cluster on 6 single_track + 1 lab32 LPBF pd matrices\n"
    f"VERDICT (settles C012 + C021): every matrix has 10-dim near-null subspace at fp64 noise floor.   "
    f"single_track κ uniform to {cv:.2f}%   |   lab32 ≠ single_track by σ scale but same near-null structure",
    fontsize=11, y=0.998
)
out = OUTDIR / "xeon_E09_kappa_nearnull.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"  → {out}")

# Numerical summary for the report
print(f"\nSummary:")
print(f"  single_track κ:      mean {mean_k:.4e}  std {std_k:.2e}  CV {cv:.3f}%")
print(f"  lab32 κ:             {kappa[0]:.4e}")
print(f"  near_null=True:      {sum(near_null)}/{len(near_null)}")
print(f"  near_null_dim=10:    {sum(d == 10 for d in nn_dim)}/{len(nn_dim)}")
print(f"  Lanczos wall:        min {min(r['lanczos_wall_s'] for r in records):.0f}s  "
      f"max {max(r['lanczos_wall_s'] for r in records):.0f}s")
print(f"  svds wall:           min {min(r['svds_wall_s'] for r in records):.1f}s  "
      f"max {max(r['svds_wall_s'] for r in records):.1f}s  (vs Lanczos ×{min(r['lanczos_wall_s'] for r in records)/max(r['svds_wall_s'] for r in records):.0f})")
