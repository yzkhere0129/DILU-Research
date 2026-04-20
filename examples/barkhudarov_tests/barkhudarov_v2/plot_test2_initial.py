#!/usr/bin/env python3
"""Generate initial geometry plot for Barkhudarov Test 2."""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Parameters (same as test2) ─────────────────────────────────────
DOMAIN = 1.0
OBS_X, OBS_Y = 0.5, 0.5
DROP_CX, DROP_CY = 0.70, 0.70
DROP_R = 0.19
U_INF, V_INF = -1.0, -1.0

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Plot ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))

# Domain
ax.set_xlim(0, DOMAIN)
ax.set_ylim(0, DOMAIN)
ax.set_aspect("equal")

# Grid lines (50×50)
NX = 50
for i in range(NX + 1):
    x = i * DOMAIN / NX
    ax.axvline(x, color="lightgray", lw=0.2, alpha=0.5)
    ax.axhline(x, color="lightgray", lw=0.2, alpha=0.5)

# Obstacle (rectangle from origin)
obs = patches.Rectangle(
    (0, 0),
    OBS_X,
    OBS_Y,
    lw=2.5,
    ec="black",
    fc="#999999",
    alpha=0.9,
    label=f"Obstacle [{OBS_X}×{OBS_Y}]",
)
ax.add_patch(obs)

# Droplet
theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(
    DROP_CX + DROP_R * np.cos(theta),
    DROP_CY + DROP_R * np.sin(theta),
    "r-",
    lw=2.5,
    label=f"Droplet (R={DROP_R})",
)

# Fill droplet
ax.fill(
    DROP_CX + DROP_R * np.cos(theta),
    DROP_CY + DROP_R * np.sin(theta),
    color="red",
    alpha=0.15,
)

# Center point
ax.plot(DROP_CX, DROP_CY, "ro", ms=8, label=f"Center ({DROP_CX},{DROP_CY})")

# Velocity arrow
ax.quiver(
    DROP_CX,
    DROP_CY,
    U_INF * 0.12,
    V_INF * 0.12,
    angles="xy",
    scale_units="xy",
    scale=1,
    color="blue",
    width=0.015,
    headwidth=4,
    headlength=5,
    label=f"Velocity ({U_INF},{V_INF})",
)

# Gap annotations
gap_x = DROP_CX - OBS_X - DROP_R
gap_y = DROP_CY - OBS_Y - DROP_R

# Horizontal gap line
ax.annotate(
    "",
    xy=(OBS_X, DROP_CY),
    xytext=(DROP_CX - DROP_R, DROP_CY),
    arrowprops=dict(arrowstyle="<->", color="green", lw=1.5),
)
ax.text(
    (OBS_X + DROP_CX - DROP_R) / 2,
    DROP_CY + 0.03,
    f"Δx = {gap_x:.3f}",
    ha="center",
    va="bottom",
    fontsize=10,
    color="green",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="green", alpha=0.8),
)

# Vertical gap line
ax.annotate(
    "",
    xy=(DROP_CX, OBS_Y),
    xytext=(DROP_CX, DROP_CY - DROP_R),
    arrowprops=dict(arrowstyle="<->", color="green", lw=1.5),
)
ax.text(
    DROP_CX + 0.03,
    (OBS_Y + DROP_CY - DROP_R) / 2,
    f"Δy = {gap_y:.3f}",
    ha="left",
    va="center",
    fontsize=10,
    color="green",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="green", alpha=0.8),
)

# Labels
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)
ax.set_title(
    "Barkhudarov Test 2: Initial Geometry\nDroplet Impacting Rectangular Obstacle",
    fontsize=14,
    fontweight="bold",
)

ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

# Minor ticks
ax.set_xticks(np.arange(0, DOMAIN + 0.1, 0.1))
ax.set_yticks(np.arange(0, DOMAIN + 0.1, 0.1))
ax.tick_params(labelsize=10)
ax.grid(True, alpha=0.2)

plt.tight_layout()
out_path = os.path.join(OUT, "test2_initial_geometry.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")
print(f"\nConfiguration:")
print(f"  Domain: [{DOMAIN}×{DOMAIN}]")
print(f"  Obstacle: (0,0) → ({OBS_X},{OBS_Y})")
print(f"  Droplet: R={DROP_R}, center=({DROP_CX},{DROP_CY})")
print(f"  Velocity: ({U_INF},{V_INF})")
print(f"  Gap to obstacle: Δx={gap_x:.4f}, Δy={gap_y:.4f}")
