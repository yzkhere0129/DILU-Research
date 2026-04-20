#!/usr/bin/env python3
"""Generate initial geometry plot for Barkhudarov Test 2 (3D version).

Configuration matches test2_obstacle.py paper specs:
  Domain: [0,1] x [0,1] x [0, 0.1]
  Obstacle: [0, 0.5] x [0, 0.5] x [0, 0.1] (full z-depth slab)
  Droplet: sphere R=0.19, center=(0.70, 0.70, 0.05)
  Velocity: (-1, -1, 0)

This is the REVIEW GATE — user must approve before running the simulation.
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── 3D Parameters (from paper) ─────────────────────────────────────
DOMAIN_XY = 1.0
NX, NY, NZ = 50, 50, 5
DX = DOMAIN_XY / NX  # 0.02
DOMAIN_Z = NZ * DX  # 0.1

OBS_X, OBS_Y, OBS_Z = 0.5, 0.5, DOMAIN_Z  # full z-depth
DROP_CX, DROP_CY, DROP_CZ = 0.70, 0.70, 0.05
DROP_R = 0.20
U_INF, V_INF, W_INF = -1.0, -1.0, 0.0

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Compute gaps ──────────────────────────────────────────────────
gap_x = DROP_CX - OBS_X - DROP_R
gap_y = DROP_CY - OBS_Y - DROP_R

# ── Plot: xy-slice at z-midplane ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))

ax.set_xlim(0, DOMAIN_XY)
ax.set_ylim(0, DOMAIN_XY)
ax.set_aspect("equal")

# Grid lines
for i in range(NX + 1):
    x = i * DOMAIN_XY / NX
    ax.axvline(x, color="lightgray", lw=0.2, alpha=0.5)
    ax.axhline(x, color="lightgray", lw=0.2, alpha=0.5)

# Obstacle (rectangle from origin)
obs = patches.Rectangle(
    (0, 0), OBS_X, OBS_Y,
    lw=2.5, ec="black", fc="#999999", alpha=0.9,
    label=f"Obstacle [{OBS_X}×{OBS_Y}]",
)
ax.add_patch(obs)

# Droplet (circle in xy at z-midplane)
theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(
    DROP_CX + DROP_R * np.cos(theta),
    DROP_CY + DROP_R * np.sin(theta),
    "r-", lw=2.5, label=f"Droplet (R={DROP_R})",
)
ax.fill(
    DROP_CX + DROP_R * np.cos(theta),
    DROP_CY + DROP_R * np.sin(theta),
    color="red", alpha=0.15,
)

# Center point
ax.plot(DROP_CX, DROP_CY, "ro", ms=8, label=f"Center ({DROP_CX},{DROP_CY})")

# Velocity arrow
ax.quiver(
    DROP_CX, DROP_CY,
    U_INF * 0.12, V_INF * 0.12,
    angles="xy", scale_units="xy", scale=1,
    color="blue", width=0.015, headwidth=4, headlength=5,
    label=f"Velocity ({U_INF},{V_INF})",
)

# Gap annotations
ax.annotate(
    "", xy=(OBS_X, DROP_CY), xytext=(DROP_CX - DROP_R, DROP_CY),
    arrowprops=dict(arrowstyle="<->", color="green", lw=1.5),
)
ax.text(
    (OBS_X + DROP_CX - DROP_R) / 2, DROP_CY + 0.03,
    f"Δx = {gap_x:.3f}", ha="center", va="bottom", fontsize=10,
    color="green", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="green", alpha=0.8),
)

ax.annotate(
    "", xy=(DROP_CX, OBS_Y), xytext=(DROP_CX, DROP_CY - DROP_R),
    arrowprops=dict(arrowstyle="<->", color="green", lw=1.5),
)
ax.text(
    DROP_CX + 0.03, (OBS_Y + DROP_CY - DROP_R) / 2,
    f"Δy = {gap_y:.3f}", ha="left", va="center", fontsize=10,
    color="green", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="green", alpha=0.8),
)

# Labels
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)
ax.set_title(
    "Barkhudarov Test 2: Initial Geometry (3D)\n"
    f"Droplet Impacting Obstacle — z-slice at z={DROP_CZ}\n"
    f"Grid: {NX}×{NY}×{NZ}, dx=dy=dz={DX:.3f}, "
    f"Domain: [{DOMAIN_XY}×{DOMAIN_XY}×{DOMAIN_Z:.2f}]",
    fontsize=13, fontweight="bold",
)

ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
ax.set_xticks(np.arange(0, DOMAIN_XY + 0.1, 0.1))
ax.set_yticks(np.arange(0, DOMAIN_XY + 0.1, 0.1))
ax.tick_params(labelsize=10)
ax.grid(True, alpha=0.2)

plt.tight_layout()
out_path = os.path.join(OUT, "test2_initial_geometry_3d.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")
print(f"\nConfiguration:")
print(f"  Domain: [{DOMAIN_XY}×{DOMAIN_XY}×{DOMAIN_Z:.2f}]")
print(f"  Grid: {NX}×{NY}×{NZ}, dx=dy=dz={DX:.3f}")
print(f"  Obstacle: (0,0,0) → ({OBS_X},{OBS_Y},{OBS_Z:.2f})")
print(f"  Droplet: R={DROP_R}, center=({DROP_CX},{DROP_CY},{DROP_CZ})")
print(f"  Velocity: ({U_INF},{V_INF},{W_INF})")
print(f"  Gap to obstacle: Δx={gap_x:.4f}, Δy={gap_y:.4f}")
print(f"\n  ** REVIEW GATE: Please inspect the plot before running test2. **")
