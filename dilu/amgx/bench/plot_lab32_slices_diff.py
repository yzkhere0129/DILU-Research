"""2D slice visualization of |x_OF - x_AMGx_e8| through lab32 dump.

Same XY/XZ slice layout as plot_lab32_slices.py but the displayed field
is the cell-wise solver disagreement (in Pa), log-color so the localized
hotspots near the powder/gas interface pop out of the 1 Pa noise floor.

Usage:
    python -m dilu.amgx.bench.plot_lab32_slices_diff \\
        --npz dilu/amgx/bench/lab32_plot_melting_pd_corr0_3.8e-07.npz \\
        --mesh-shape 50,200,50 --dx 4e-6 --phase melting

    python -m dilu.amgx.bench.plot_lab32_slices_diff \\
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
from matplotlib.colors import LogNorm

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
    x_OF    = z["x_OF"]
    x_AMGx  = z["x_AMGx_e8"]
    diff    = np.abs(x_OF - x_AMGx)
    n = int(z["n"][0])
    if nx * ny * nz != n:
        raise SystemExit(f"mesh_shape {nx}*{ny}*{nz} != N={n}")

    # Reshape to [k, j, i]
    field = diff.reshape((nz, ny, nx))

    # Log color: vmin = 1 Pa (below = treated as ~no diff), vmax = data max
    pos_only = diff[diff > 0]
    if pos_only.size == 0:
        raise SystemExit("All diffs are zero — nothing to plot")
    vmax = float(diff.max())
    vmin = 1.0   # below 1 Pa treated as numerical zero
    norm = LogNorm(vmin=vmin, vmax=max(vmax, 100.0))
    cmap = "magma"

    n_above_1Pa   = int((diff > 1.0).sum())
    n_above_100Pa = int((diff > 100.0).sum())
    n_above_1kPa  = int((diff > 1000.0).sum())
    denom = max(float(np.abs(x_OF).max()), 1e-300)

    print(f"phase={args.phase}  N={n}")
    print(f"  |OF-AMGx|: max={diff.max():.3e} Pa "
          f"(rel={diff.max()/denom:.3e}), median={np.median(diff):.3e}")
    print(f"  cells above 1Pa / 100Pa / 1kPa: "
          f"{n_above_1Pa:,} / {n_above_100Pa:,} / {n_above_1kPa:,}")

    # ======= 2 rows × 4 cols =======
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.30)

    # ---- Row 1: 4 horizontal slices XY ----
    z_targets_um = [96, 108, 120, 132]
    for col, z_um in enumerate(z_targets_um):
        k = int(round(z_um / (dx * 1e6)))
        if k >= nz: k = nz - 1
        slc = field[k]    # [j, i]
        ax = fig.add_subplot(gs[0, col])
        # Use small floor to avoid log(0)
        slc_plot = np.maximum(slc, vmin * 0.5)
        im = ax.imshow(slc_plot, origin="lower", aspect="equal",
                        extent=(0, nx*dx*1e6, 0, ny*dx*1e6),
                        norm=norm, cmap=cmap, interpolation="nearest")
        ax.set_xlabel("x (μm)", fontsize=9)
        if col == 0:
            ax.set_ylabel("y (μm) — laser scan axis", fontsize=9)
        n_active = int((slc > 1.0).sum())
        ax.set_title(f"z = {z_um} μm  (k={k})\n"
                      f"|diff| max in slice = {slc.max():.2e} Pa\n"
                      f"{n_active} cells > 1 Pa",
                      fontsize=9)
        ax.tick_params(labelsize=8)

    # ---- Row 2: side views XZ at 4 j positions ----
    j_targets = [25, 50, 100, 150]
    for col, j_target in enumerate(j_targets):
        if j_target >= ny: j_target = ny - 1
        slc = field[:, j_target, :]    # [k, i]
        ax = fig.add_subplot(gs[1, col])
        slc_plot = np.maximum(slc, vmin * 0.5)
        im = ax.imshow(slc_plot, origin="lower", aspect="auto",
                        extent=(0, nx*dx*1e6, 0, nz*dx*1e6),
                        norm=norm, cmap=cmap, interpolation="nearest")
        ax.axhline(96, color="cyan", linestyle=":", alpha=0.6,
                    linewidth=1)  # nominal powder surface
        ax.set_xlabel("x (μm)", fontsize=9)
        if col == 0:
            ax.set_ylabel("z (μm) — depth", fontsize=9)
        n_active = int((slc > 1.0).sum())
        ax.set_title(f"y = {j_target*dx*1e6:.0f} μm  (j={j_target})\n"
                      f"|diff| max = {slc.max():.2e} Pa, "
                      f"{n_active} cells > 1 Pa",
                      fontsize=9)
        ax.tick_params(labelsize=8)

    # Shared colorbar
    cb = fig.colorbar(im, ax=fig.axes, fraction=0.015, pad=0.02,
                       shrink=0.7, orientation='vertical')
    cb.set_label(f"|x_OF - x_AMGx_e8|  (Pa, log)", fontsize=10)

    fig.suptitle(
        f"Lab Xeon 32-rank dump pd — OF DICPCG vs AMGx (both @ tol=1e-8)\n"
        f"|x_OF - x_AMGx_e8| spatial slices  ({args.phase}, t={meta['time']})\n"
        f"max diff = {diff.max():.2e} Pa (rel = {diff.max()/denom:.2e}),  "
        f"{n_above_100Pa} cells > 100 Pa,  "
        f"{n_above_1kPa} cells > 1 kPa  (of {n:,} total)\n"
        f"Top: horizontal slices at 4 depths  |  Bottom: side views at 4 y "
        f"positions (cyan dotted = nominal powder surface z≈96μm)",
        fontsize=11, y=0.998)

    out_name = args.out_name or f"lab32_slices_{args.phase}_OF_vs_AMGx_diff"
    out = OUTDIR / f"{out_name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
