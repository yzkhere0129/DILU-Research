"""2D slices through lab32 dump to reveal melt pool / vapor structure.

3D scatter doesn't work well when 99% of cells are uniform background.
This script uses imshow on structured-mesh slices:

  Top row:    z = 100, 112, 124, 136 μm  — XY (top-down) views through melt
              pool depth (powder surface ≈ z=96μm)
  Bottom row: y = mid_y XZ side view, melting + evaporation
              (shows the laser scan track in y-direction)

Per phase one PNG: lab32_slices_<phase>_pd.png

Usage:
    python -m dilu.amgx.bench.plot_lab32_slices \\
        --npz dilu/amgx/bench/lab32_plot_melting_pd_corr0_3.8e-07.npz \\
        --mesh-shape 50,200,50 --dx 4e-6 --phase melting

    python -m dilu.amgx.bench.plot_lab32_slices \\
        --npz dilu/amgx/bench/lab32_plot_evaporation_pd_corr0_7e-07.npz \\
        --mesh-shape 50,200,50 --dx 4e-6 --phase evaporation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

OUTDIR = Path("/home/yzk/DILU-Research/docs/benchmark/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--mesh-shape", required=True, help="nx,ny,nz")
    ap.add_argument("--dx", type=float, required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    nx, ny, nz = [int(v) for v in args.mesh_shape.split(",")]
    dx = args.dx

    z = np.load(Path(args.npz), allow_pickle=True)
    meta = json.loads(str(z["meta"][0]))
    x_truth = z["x_truth"]
    x_OF    = z["x_OF"]
    n = int(z["n"][0])
    if nx * ny * nz != n:
        raise SystemExit(f"mesh_shape {nx}*{ny}*{nz} != N={n}")

    # Reshape to 3D structured grid: cell_id = i + j*nx + k*nx*ny
    field = x_truth.reshape((nz, ny, nx))   # [k, j, i]
    bg = float(np.median(x_truth))
    print(f"N={n}  bg(median)={bg:.3e} Pa  range [{x_truth.min():.3e}, "
          f"{x_truth.max():.3e}]")

    # Color: diverging around bg (1 atm = 1.01e5 Pa)
    pd_lo = float(np.percentile(x_truth, 0.1))
    pd_hi = float(np.percentile(x_truth, 99.9))
    norm = TwoSlopeNorm(vcenter=bg, vmin=pd_lo, vmax=pd_hi)
    cmap = "RdBu_r"

    # ======= Layout: 2 rows × 4 cols =======
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.30)

    # ---- Row 1: 4 horizontal slices across the melt-pool depth ----
    z_targets_um = [96, 108, 120, 132]    # μm
    for col, z_um in enumerate(z_targets_um):
        k = int(round(z_um / (dx * 1e6)))
        if k >= nz: k = nz - 1
        slc = field[k]  # [j, i]
        ax = fig.add_subplot(gs[0, col])
        # imshow expects [y, x] = slc[j, i] mapped to (j-axis, i-axis)
        # We want x horizontal, y vertical — so transpose: [i, j] → imshow with origin=lower
        # Actually plot extent: i ∈ [0, nx*dx], j ∈ [0, ny*dx]. j is the long laser-scan axis.
        # Use slc directly (rows=j, cols=i) with extent (i_min, i_max, j_min, j_max).
        im = ax.imshow(slc, origin="lower", aspect="equal",
                        extent=(0, nx*dx*1e6, 0, ny*dx*1e6),
                        norm=norm, cmap=cmap, interpolation="nearest")
        ax.set_xlabel("x (μm)", fontsize=9)
        if col == 0:
            ax.set_ylabel("y (μm) — laser scan axis", fontsize=9)
        ax.set_title(f"z = {z_um} μm  (k={k})\n"
                      f"slice mean={slc.mean():.2e}, "
                      f"std={slc.std():.2e}, max={slc.max():.2e}",
                      fontsize=9)
        ax.tick_params(labelsize=8)

    # ---- Row 2: side views (XZ) at 4 different j positions ----
    j_targets = [25, 50, 100, 150]
    for col, j_target in enumerate(j_targets):
        if j_target >= ny: j_target = ny - 1
        slc = field[:, j_target, :]  # [k, i]
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(slc, origin="lower", aspect="auto",
                        extent=(0, nx*dx*1e6, 0, nz*dx*1e6),
                        norm=norm, cmap=cmap, interpolation="nearest")
        ax.set_xlabel("x (μm)", fontsize=9)
        if col == 0:
            ax.set_ylabel("z (μm) — depth", fontsize=9)
        ax.axhline(96, color="white", linestyle=":", alpha=0.5,
                    linewidth=1)  # nominal powder surface
        ax.set_title(f"y = {j_target*dx*1e6:.0f} μm  (j={j_target})\n"
                      f"slice mean={slc.mean():.2e}, max={slc.max():.2e}",
                      fontsize=9)
        ax.tick_params(labelsize=8)

    # Single shared colorbar
    cb = fig.colorbar(im, ax=fig.axes, fraction=0.015, pad=0.02,
                       shrink=0.7, orientation='vertical')
    cb.set_label(f"pd (Pa)  centered on median={bg:.2e}", fontsize=10)

    # Title
    fig.suptitle(
        f"Lab Xeon 32-rank dump — pd slices  ({args.phase}, t={meta['time']})\n"
        f"truth = AMGx tol=1e-12 (iter={meta['amgx_truth']['iters']}, "
        f"rel_resid={meta['amgx_truth']['rel_resid_actual']:.1e})  |  "
        f"OF dump consistency: ‖A·x_OF-b‖/‖b‖ = "
        f"{meta['rel_OF_consistency']:.1e}\n"
        f"Top: horizontal slices at 4 depths  |  Bottom: 4 side views at "
        f"different y positions (white dashed = nominal powder surface z≈96μm)",
        fontsize=11, y=0.995)

    out_name = args.out_name or f"lab32_slices_{args.phase}_pd"
    out = OUTDIR / f"{out_name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
