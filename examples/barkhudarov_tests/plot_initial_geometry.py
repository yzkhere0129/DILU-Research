#!/usr/bin/env python3
"""Preview initial geometry for Test 2: droplet impacting obstacle."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Paper configuration ──────────────────────────────────────────
# Domain: [0,1]×[0,1]
# Obstacle: [0, 0.5]×[0, 0.5]  (bottom-left corner)
# Droplet: radius=0.12, center chosen so it's diagonal from obstacle,
#          not too small, not too far
# Velocity: u=-1, v=-1 (toward obstacle)

DOMAIN   = 1.0
OBS_X    = 0.5    # obstacle spans [0, 0.5] in x
OBS_Y    = 0.5    # obstacle spans [0, 0.5] in y
DROP_R   = 0.19   # radius → diameter = 0.38
DROP_CX  = 0.70   # droplet center x
DROP_CY  = 0.70   # droplet center y
NX       = 50     # grid cells (cell size = 0.02 = D/12 ≈ paper's D/10)

dx = DOMAIN / NX

# ── Build F field ─────────────────────────────────────────────────
x = np.linspace(dx/2, DOMAIN - dx/2, NX)
y = np.linspace(dx/2, DOMAIN - dx/2, NX)
X, Y = np.meshgrid(x, y, indexing="ij")

r = np.sqrt((X - DROP_CX)**2 + (Y - DROP_CY)**2)
half_diag = np.sqrt(2)*dx/2
F = np.where(r + half_diag <= DROP_R, 1.0,
    np.where(r - half_diag >= DROP_R, 0.0,
             np.clip((DROP_R - r + half_diag) / (2*half_diag), 0, 1)))

# Obstacle: F=0 inside
obs_mask = (X < OBS_X) & (Y < OBS_Y)
F[obs_mask] = 0.0

# ── Plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_facecolor("#F0F4F8")

# VOF field
cf = ax.contourf(X, Y, F, levels=np.linspace(0, 1, 21), cmap="Blues", alpha=0.85)
ax.contour(X, Y, F, levels=[0.5], colors="navy", linewidths=2.0, label="Droplet interface")

# Obstacle (grey filled)
obs_patch = patches.Rectangle((0, 0), OBS_X, OBS_Y,
                                linewidth=2, edgecolor="black",
                                facecolor="#888888", label="Obstacle (solid)")
ax.add_patch(obs_patch)

# Grid lines (light)
for xi in np.linspace(0, DOMAIN, NX+1):
    ax.axvline(xi, color="gray", lw=0.15, alpha=0.4)
    ax.axhline(xi, color="gray", lw=0.15, alpha=0.4)

# Velocity arrow
ax.annotate("", xy=(DROP_CX-0.15, DROP_CY-0.15), xytext=(DROP_CX, DROP_CY),
            arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
ax.text(DROP_CX-0.10, DROP_CY-0.17, "u=-1, v=-1", color="red", fontsize=11,
        ha="center", va="top")

# Annotations
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(DROP_CX + DROP_R*np.cos(theta), DROP_CY + DROP_R*np.sin(theta),
        'g--', lw=1.5, alpha=0.7, label=f"Exact circle R={DROP_R}")
ax.plot(DROP_CX, DROP_CY, 'r+', ms=10, mew=2)
ax.annotate(f"Drop center\n({DROP_CX},{DROP_CY})", xy=(DROP_CX, DROP_CY),
            xytext=(DROP_CX+0.05, DROP_CY+0.06), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black", lw=1))
ax.annotate("Obstacle\n0.5×0.5", xy=(0.25, 0.25), xytext=(0.05, 0.55),
            fontsize=10, color="white", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="white", lw=1.5))

ax.set_xlim(0, DOMAIN); ax.set_ylim(0, DOMAIN)
ax.set_aspect("equal")
ax.set_xlabel("x", fontsize=13); ax.set_ylabel("y", fontsize=13)
ax.set_title(f"Test 2: Initial Geometry — Droplet Impacting Obstacle\n"
             f"Grid {NX}×{NX} (dx={dx:.3f}), D={2*DROP_R:.2f}, D/dx={2*DROP_R/dx:.1f} cells/diameter",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper left", fontsize=10)
plt.colorbar(cf, ax=ax, shrink=0.8, label="F (volume fraction)")
plt.tight_layout()

out = os.path.join(OUT, "test2_initial_geometry.png")
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
print(f"Grid: {NX}×{NX}, dx={dx:.4f}")
print(f"Droplet: R={DROP_R}, D={2*DROP_R:.2f}, D/dx={2*DROP_R/dx:.1f} cells/diameter")
print(f"Obstacle: [{0},{OBS_X}] × [{0},{OBS_Y}]")
print(f"Velocity: u={-1}, v={-1} (toward obstacle corner)")
print(f"Gap droplet→obstacle surface: {min(DROP_CX-OBS_X, DROP_CY-OBS_Y) - DROP_R:.3f}")
