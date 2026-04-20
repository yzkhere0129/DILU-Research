#!/usr/bin/env python3
"""Render a Zalesak-3D overview plot for the OpenFOAM interIsoFoam result.

Mirror of `export_for_paraview.py::plot_overview` but reads the OF extraction
(`results/zalesak_3d_openfoam.npz`) so the two figures can be placed
side-by-side for direct comparison.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "results")


def plot_overview(F_ini, F_fin, dx, out_path, L1_rel, V_err, title_suffix):
    N = F_ini.shape[0]
    x = np.linspace(dx/2, 1 - dx/2, N)
    X, Y = np.meshgrid(x, x, indexing="ij")

    z_idx = int(0.75 / dx)
    x_idx = N // 2

    XY_XLIM = (0.30, 0.70); XY_YLIM = (0.30, 0.70)
    YZ_YLIM = (0.30, 0.70); YZ_ZLIM = (0.55, 0.95)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))

    def render(ax, Xp, Yp, data, title, xlim, ylim, xlab, ylab):
        ax.pcolormesh(Xp, Yp, (data > 0.01).astype(float),
                      cmap="Blues", vmin=0, vmax=1, shading="auto")
        ax.contour(Xp, Yp, data, levels=[0.5], colors="k", linewidths=1.2)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=11)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel(xlab, fontsize=10); ax.set_ylabel(ylab, fontsize=10)
        ax.grid(True, alpha=0.15)

    render(axes[0, 0], X, Y, F_ini[:, :, z_idx],
           "Initial  z = 0.75  (xy plane at sphere center)",
           XY_XLIM, XY_YLIM, "x", "y")
    render(axes[1, 0], X, Y, F_fin[:, :, z_idx],
           "Final  z = 0.75  (xy plane, after one rotation)",
           XY_XLIM, XY_YLIM, "x", "y")

    render(axes[0, 1], X, Y, F_ini[x_idx, :, :],
           "Initial  x = 0.50  (yz rotation plane)",
           YZ_YLIM, YZ_ZLIM, "y", "z")
    render(axes[1, 1], X, Y, F_fin[x_idx, :, :],
           "Final  x = 0.50  (yz rotation plane)",
           YZ_YLIM, YZ_ZLIM, "y", "z")

    overlay_cfg = [
        (X, Y, F_ini[:, :, z_idx], F_fin[:, :, z_idx],
         f"Overlay z = 0.75  (xy plane)\nL1 error = {L1_rel:.2f}%",
         XY_XLIM, XY_YLIM, "x", "y"),
        (X, Y, F_ini[x_idx, :, :], F_fin[x_idx, :, :],
         f"Overlay x = 0.50  (yz rotation plane)\nV error = {V_err:.4f}%",
         YZ_YLIM, YZ_ZLIM, "y", "z"),
    ]
    for row, (Xp, Yp, ini_s, fin_s, title, xlim, ylim, xlab, ylab) in enumerate(overlay_cfg):
        ax = axes[row, 2]
        ax.contour(Xp, Yp, ini_s, levels=[0.5], colors="b", linewidths=2.0)
        ax.contour(Xp, Yp, fin_s, levels=[0.5], colors="r", linewidths=2.0)
        ax.set_aspect("equal"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel(xlab, fontsize=10); ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(title, fontsize=11); ax.grid(True, alpha=0.3)
        ax.legend(
            [Line2D([0], [0], color="b", lw=2), Line2D([0], [0], color="r", lw=2)],
            ["Initial (t=0)", "Final (t=2π)"],
            loc="upper right", fontsize=9,
        )

    fig.suptitle(
        f"Zalesak 3D Slotted Sphere — {title_suffix}\n"
        f"Grid 128³ (2.1M cells)   Domain [0,1]³   Rotation axis = x at (y, z) = (0.5, 0.5)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)")


def main():
    data = np.load(os.path.join(RESULTS, "zalesak_3d_openfoam.npz"))
    F_ini = data["F_init"]
    F_fin = data["F_final"]
    dx = float(data["DX"])

    L1 = float(np.sum(np.abs(F_fin - F_ini)) * dx ** 3)
    V0 = float(np.sum(F_ini) * dx ** 3)
    V1 = float(np.sum(F_fin) * dx ** 3)
    L1_rel = L1 / V0 * 100
    V_err = abs(V1 - V0) / V0 * 100

    print(f"OpenFOAM interIsoFoam — V0={V0:.6e}  L1={L1_rel:.2f}%  V_err={V_err:.4f}%")

    plot_overview(
        F_ini, F_fin, dx,
        os.path.join(RESULTS, "zalesak_3d_openfoam_overview.png"),
        L1_rel, V_err,
        title_suffix="OpenFOAM interIsoFoam",
    )


if __name__ == "__main__":
    main()
