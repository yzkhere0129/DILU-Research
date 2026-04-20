#!/usr/bin/env python3
"""Summary visualization for the droplet_45deg PLIC advection result.

Produces a single 3-row figure:

    Row 1: phase evolution -- 6 snapshots of F with analytic position
    Row 2: trajectory      -- initial vs final contour + analytic circle
    Row 3: metrics         -- V drift, centroid error, perimeter curves

Run from this directory after ``run.py`` has written the snapshots.
"""

from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from case_config import CASE  # noqa: E402


VOF_CMAP = LinearSegmentedColormap.from_list(
    "vof", ["#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"], N=256
)


def _load_snapshots(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "snapshot_*.npz")))
    snaps = []
    for f in files:
        d = np.load(f)
        vf = d["volume_fraction"]
        nz_phys = int(d["nz"])
        if vf.ndim == 3 and vf.shape[2] > nz_phys:
            # Old snapshot with z halos: strip
            nh = (vf.shape[2] - nz_phys) // 2
            vf = vf[:, :, nh : nh + nz_phys]
        if vf.ndim == 3:
            vf_2d = vf[:, :, vf.shape[2] // 2]
        else:
            vf_2d = vf
        snaps.append((float(d["time"]), int(d["step"]), vf_2d))
    return snaps


def _centroid(vf_2d, dx, dy):
    nx, ny = vf_2d.shape
    x = np.linspace(dx / 2.0, 1.0 - dx / 2.0, nx)
    y = np.linspace(dy / 2.0, 1.0 - dy / 2.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    total = float(vf_2d.sum()) + 1e-30
    return float((vf_2d * X).sum() / total), float((vf_2d * Y).sum() / total)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "results")
    snaps = _load_snapshots(results_dir)
    if not snaps:
        print("No snapshots found in", results_dir)
        return 1

    dx = CASE["grid_dx"] if "grid_dx" in CASE else (CASE["domain"]["x"][1] - CASE["domain"]["x"][0]) / CASE["grid"]["nx"]
    dy = (CASE["domain"]["y"][1] - CASE["domain"]["y"][0]) / CASE["grid"]["ny"]
    dz = (CASE["domain"]["z"][1] - CASE["domain"]["z"][0]) / max(CASE["grid"]["nz"], 1)
    cell_area = dx * dy
    cx0, cy0 = CASE["droplet"]["center"]
    R = CASE["droplet"]["radius"]
    nx, ny = CASE["grid"]["nx"], CASE["grid"]["ny"]
    x_centers = np.linspace(dx / 2.0, 1.0 - dx / 2.0, nx)
    y_centers = np.linspace(dy / 2.0, 1.0 - dy / 2.0, ny)
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")

    # Time series
    times = np.array([s[0] for s in snaps])
    areas = np.array([float(s[2].sum() * cell_area) for s in snaps])
    centroids = np.array([_centroid(s[2], dx, dy) for s in snaps])

    # Pick 6 equally spaced snapshots for the phase evolution row
    n_snap = len(snaps)
    n_show = min(6, n_snap)
    idx = np.linspace(0, n_snap - 1, n_show, dtype=int)
    show = [snaps[i] for i in idx]

    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, n_show, figure=fig, height_ratios=[1.0, 1.2, 0.9], hspace=0.35, wspace=0.15)

    # ------------------ Row 1: phase evolution ------------------
    for k, (t, step, vf) in enumerate(show):
        ax = fig.add_subplot(gs[0, k])
        ax.contourf(X, Y, vf, levels=np.linspace(0, 1, 21), cmap=VOF_CMAP)
        ax.contour(X, Y, vf, levels=[0.5], colors="blue", linewidths=1.2)
        # Analytic circle at this time
        cx_a, cy_a = cx0 + t, cy0 + t
        theta = np.linspace(0, 2 * np.pi, 80)
        ax.plot(cx_a + R * np.cos(theta), cy_a + R * np.sin(theta),
                "g--", linewidth=1.2, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t:.3f} s (step {step})", fontsize=10)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.set_ylabel("y", fontsize=9)
        ax.set_xlabel("x", fontsize=9)

    # ------------------ Row 2: trajectory (initial vs final, zoom) ----------
    ax_traj = fig.add_subplot(gs[1, :])
    # Initial vs final contour + analytic circles
    vf0 = snaps[0][2]
    vf_end = snaps[-1][2]
    ax_traj.contour(X, Y, vf0, levels=[0.5], colors="blue", linewidths=2.5, label="_nolegend_")
    ax_traj.contour(X, Y, vf_end, levels=[0.5], colors="red", linewidths=2.5, label="_nolegend_")

    theta = np.linspace(0, 2 * np.pi, 120)
    cx_f, cy_f = CASE["analytic_final_center"]
    ax_traj.plot(cx0 + R * np.cos(theta), cy0 + R * np.sin(theta),
                 "b--", linewidth=1.5, alpha=0.7, label=f"analytic t=0: ({cx0}, {cy0})")
    ax_traj.plot(cx_f + R * np.cos(theta), cy_f + R * np.sin(theta),
                 "g--", linewidth=1.5, alpha=0.8, label=f"analytic t={times[-1]:.4f}: ({cx_f}, {cy_f})")
    # Trajectory line
    ax_traj.plot(centroids[:, 0], centroids[:, 1], "k.-",
                 markersize=6, linewidth=1.2, label="PLIC centroid path")
    ax_traj.plot([cx0, cx_f], [cy0, cy_f], "k:", linewidth=0.8, alpha=0.4,
                 label="45-deg direction")
    # Interior contours (initial, final) with labels
    ax_traj.text(cx0 - 0.04, cy0 - 0.08, "t=0\n(blue)", fontsize=9, ha="center", color="blue")
    ax_traj.text(cx_f + 0.04, cy_f + 0.08, f"t={times[-1]:.3f}\n(red)", fontsize=9, ha="center", color="red")

    ax_traj.set_xlim(0.05, 0.75)
    ax_traj.set_ylim(0.05, 0.75)
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("x (m)", fontsize=11)
    ax_traj.set_ylabel("y (m)", fontsize=11)
    ax_traj.set_title("Droplet trajectory: initial (blue) vs final (red) vs analytic (green dashed)",
                      fontsize=12, fontweight="bold")
    ax_traj.grid(True, alpha=0.3, ls="--")
    ax_traj.legend(fontsize=9, loc="upper left")

    # ------------------ Row 3: metrics -------------------------------------
    # Panel a: area drift
    ax_area = fig.add_subplot(gs[2, 0:2])
    area0 = areas[0]
    drift = (areas - area0) / area0
    ax_area.plot(times, drift, "ro-", markersize=5, linewidth=1.0)
    ax_area.axhline(0, color="gray", ls="--", alpha=0.5)
    ax_area.set_xlabel("Time (s)")
    ax_area.set_ylabel("(A - A0) / A0")
    ax_area.set_title("(a) 2D area drift", fontsize=11, fontweight="bold")
    ax_area.grid(True, alpha=0.3, ls="--")
    ax_area.ticklabel_format(style="sci", axis="y", scilimits=(-6, -6))

    # Panel b: centroid error along 45-deg
    ax_cent = fig.add_subplot(gs[2, 2:4])
    exp_cx = cx0 + times
    exp_cy = cy0 + times
    err_cx = centroids[:, 0] - exp_cx
    err_cy = centroids[:, 1] - exp_cy
    ax_cent.plot(times, err_cx / dx, "b.-", markersize=6, label="cx err / dx")
    ax_cent.plot(times, err_cy / dx, "r.-", markersize=6, label="cy err / dx")
    ax_cent.axhline(0, color="gray", ls="--", alpha=0.5)
    ax_cent.set_xlabel("Time (s)")
    ax_cent.set_ylabel("centroid error (cells)")
    ax_cent.set_title("(b) Centroid error vs 45-deg trajectory", fontsize=11, fontweight="bold")
    ax_cent.legend(fontsize=9, loc="best")
    ax_cent.grid(True, alpha=0.3, ls="--")

    # Panel c: perimeter
    ax_per = fig.add_subplot(gs[2, 4:])
    perimeters = []
    for _, _, vf in snaps:
        vf_np = np.asarray(vf)
        h = ((vf_np[:, :-1] - 0.5) * (vf_np[:, 1:] - 0.5)) < 0
        v = ((vf_np[:-1, :] - 0.5) * (vf_np[1:, :] - 0.5)) < 0
        perimeters.append(float(h.sum() * dy + v.sum() * dx))
    perimeters = np.array(perimeters)
    ax_per.plot(times, perimeters, "gs-", markersize=5, linewidth=1.0)
    P_analytic = 2 * math.pi * R
    ax_per.axhline(P_analytic, color="gray", ls="--", alpha=0.5, label=f"analytic 2πR = {P_analytic:.4f}")
    ax_per.set_xlabel("Time (s)")
    ax_per.set_ylabel("perimeter (m)")
    ax_per.set_title("(c) Interface perimeter (marching-squares)", fontsize=11, fontweight="bold")
    ax_per.legend(fontsize=9)
    ax_per.grid(True, alpha=0.3, ls="--")

    fig.suptitle(
        f"Eulerian PLIC droplet_45deg — Stage 2 result  "
        f"(100x100, CFL=0.5, {len(snaps)} snapshots)",
        fontsize=14, fontweight="bold", y=0.995,
    )

    out_path = os.path.join(results_dir, "summary.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")

    # Print gate summary
    print(f"\nGate summary")
    print(f"  area drift (final)       : {drift[-1]:+.3e}  (PASS if |.| < 1e-3)")
    print(f"  cx error (final) / dx    : {err_cx[-1] / dx:+.4f}  (PASS if |.| < 2)")
    print(f"  cy error (final) / dx    : {err_cy[-1] / dx:+.4f}  (PASS if |.| < 2)")
    print(f"  final centroid: ({centroids[-1, 0]:.5f}, {centroids[-1, 1]:.5f})")
    print(f"  analytic final: ({cx0 + times[-1]:.5f}, {cy0 + times[-1]:.5f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
