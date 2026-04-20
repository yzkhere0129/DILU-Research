#!/usr/bin/env python3
"""Export Zalesak 3D result to ParaView (VTI format) + higher-quality plots.

Writes:
  - zalesak_3d_initial.vti  (initial F field, structured ImageData)
  - zalesak_3d_final.vti    (final F field after one full rotation)
  - zalesak_3d_overview.png (2x3 high-resolution slice panels)
  - zalesak_3d_isosurface.png (3D F=0.5 isosurface render via matplotlib)

The .vti files use VTK XML ImageData format, natively supported by
ParaView. No external VTK library required — we write the XML directly.
"""
from __future__ import annotations

import base64
import os
import struct
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "results")


def write_vti(path, F, dx, name="F"):
    """Write a VTK XML ImageData file with one scalar cell-centered field.

    ParaView opens .vti directly. The field is stored cell-centered at
    positions (i+0.5)*dx, (j+0.5)*dx, (k+0.5)*dx for i,j,k in 0..N-1.
    """
    N = F.shape[0]
    assert F.shape == (N, N, N), "F must be cube"

    # VTI uses POINT data on (N+1)^3 grid OR CELL data on N^3 grid.
    # We store as CellData so indexing matches physical cells directly.
    raw = F.astype(np.float32).tobytes(order="F")   # Fortran order: x fastest
    compressed = zlib.compress(raw, level=6)
    n_blocks = 1
    header = struct.pack("<IIII", n_blocks, len(raw), len(raw), len(compressed))
    encoded = base64.b64encode(header + compressed).decode("ascii")

    # Whole extent uses POINTS: (0..N) * 3 = (N+1)^3 points → N^3 cells
    extent = f"0 {N} 0 {N} 0 {N}"

    xml = f'''<?xml version="1.0"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="{dx} {dx} {dx}">
    <Piece Extent="{extent}">
      <CellData Scalars="{name}">
        <DataArray type="Float32" Name="{name}" format="binary">
          {encoded}
        </DataArray>
      </CellData>
    </Piece>
  </ImageData>
</VTKFile>
'''
    with open(path, "w") as f:
        f.write(xml)
    print(f"Wrote {path}  ({os.path.getsize(path)/1024:.1f} KB, N={N}, dx={dx})")


def plot_overview(F_ini, F_fin, dx, out_path, L1_rel, V_err):
    """High-resolution 2×3 panel: xy / yz slices + overlays."""
    from matplotlib.lines import Line2D
    N = F_ini.shape[0]
    x = np.linspace(dx / 2, 1 - dx / 2, N)
    X, Y = np.meshgrid(x, x, indexing="ij")

    # Sphere center (0.5, 0.5, 0.75), R=0.15, slot in z-direction
    # → xy cross-section at z=0.75 shows a circle centered at (0.5, 0.5) with a small square slot hole
    # → yz cross-section at x=0.50 shows a circle centered at (y=0.5, z=0.75) with a slot from z=0.65 to z=0.90
    z_idx = int(0.75 / dx)
    x_idx = N // 2

    # Per-plane limits (tight around the sphere)
    XY_XLIM = (0.30, 0.70); XY_YLIM = (0.30, 0.70)   # sphere at (0.5, 0.5)
    YZ_YLIM = (0.30, 0.70); YZ_ZLIM = (0.55, 0.95)   # sphere at (y=0.5, z=0.75)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))

    def render(ax, Xp, Yp, data, title, xlim, ylim, xlab, ylab):
        ax.pcolormesh(Xp, Yp, (data > 0.01).astype(float),
                      cmap="Blues", vmin=0, vmax=1, shading="auto")
        ax.contour(Xp, Yp, data, levels=[0.5], colors="k", linewidths=1.2)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel(xlab, fontsize=10); ax.set_ylabel(ylab, fontsize=10)
        ax.grid(True, alpha=0.15)

    # Column 0: xy slices at z=0.75 (sphere center plane)
    # X axis = x, Y axis = y
    render(axes[0, 0], X, Y, F_ini[:, :, z_idx],
           "Initial  z = 0.75  (xy plane at sphere center)",
           XY_XLIM, XY_YLIM, "x", "y")
    render(axes[1, 0], X, Y, F_fin[:, :, z_idx],
           "Final  z = 0.75  (xy plane, after one rotation)",
           XY_XLIM, XY_YLIM, "x", "y")

    # Column 1: yz slices at x=0.50 (the rotation plane)
    # X axis = y, Y axis = z
    render(axes[0, 1], X, Y, F_ini[x_idx, :, :],
           "Initial  x = 0.50  (yz rotation plane)",
           YZ_YLIM, YZ_ZLIM, "y", "z")
    render(axes[1, 1], X, Y, F_fin[x_idx, :, :],
           "Final  x = 0.50  (yz rotation plane)",
           YZ_YLIM, YZ_ZLIM, "y", "z")

    # Column 2: contour overlays
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
        ax.set_aspect("equal")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel(xlab, fontsize=10); ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(
            [Line2D([0], [0], color="b", lw=2), Line2D([0], [0], color="r", lw=2)],
            ["Initial (t=0)", "Final (t=2π)"],
            loc="upper right", fontsize=9,
        )

    fig.suptitle(
        f"Zalesak 3D Slotted Sphere — Eulerian PLIC VOF on JAX\n"
        f"Grid 128³ (2.1M cells)   Domain [0,1]³   Rotation axis = x at (y, z) = (0.5, 0.5)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)")


def plot_isosurface(F_fin, dx, out_path):
    """3D isosurface render (F=0.5) via matplotlib."""
    from skimage import measure

    N = F_fin.shape[0]
    try:
        verts, faces, normals, _ = measure.marching_cubes(F_fin, level=0.5, spacing=(dx, dx, dx))
    except ValueError:
        print("skimage marching_cubes failed (likely no F=0.5 crossing)")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                    lw=0.0, alpha=0.75, color="#4a90d9",
                    edgecolor="#2868a8", shade=True)

    # Rotation axis (x-axis at y=z=0.5)
    ax.plot([0, 1], [0.5, 0.5], [0.5, 0.5], "r--", lw=1, label="Rotation axis (x)")

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Zalesak 3D Slotted Sphere — F=0.5 isosurface after one full rotation",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)")


def main():
    data = np.load(os.path.join(RESULTS, "zalesak_3d_final.npz"))
    F_ini = data["F_init"]
    F_fin = data["F_final"]
    N = int(data["N"])
    dx = float(data["DX"])

    L1 = float(np.sum(np.abs(F_fin - F_ini)) * dx ** 3)
    V0 = float(np.sum(F_ini) * dx ** 3)
    V1 = float(np.sum(F_fin) * dx ** 3)
    L1_rel = L1 / V0 * 100
    V_err = abs(V1 - V0) / V0 * 100

    print(f"Dataset: N={N}, dx={dx}, V0={V0:.6e}, L1={L1_rel:.2f}%, V_err={V_err:.4f}%")

    # 1. VTI files for ParaView
    write_vti(os.path.join(RESULTS, "zalesak_3d_initial.vti"), F_ini, dx, name="F_initial")
    write_vti(os.path.join(RESULTS, "zalesak_3d_final.vti"), F_fin, dx, name="F_final")

    # 2. High-quality 2×3 overview panel
    plot_overview(F_ini, F_fin, dx,
                  os.path.join(RESULTS, "zalesak_3d_overview.png"),
                  L1_rel, V_err)

    # 3. 3D isosurface render
    try:
        plot_isosurface(F_fin, dx,
                        os.path.join(RESULTS, "zalesak_3d_isosurface.png"))
    except ImportError:
        print("skimage not available, skipping isosurface")


if __name__ == "__main__":
    main()
