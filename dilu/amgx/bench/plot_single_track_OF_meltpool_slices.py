"""OF physical melt-pool slices on the SAME coordinate grid as the diff slices.

For each timestep, render x_OF (T-field, from OF DICPCG @ tol=1e-8) on the same
slice grid as plot_single_track_solver_diff_slices.py:
  Row 1: xy slices at z=96, 108, 120, 132 μm (top-down)
  Row 2: xz slices at y=100, 300, 500, 620 μm (side views)

With white iso-contours at:
  T = 1000 K  (HAZ)
  T = 1900 K  (melt boundary)
  T = 3000 K  (vapor / keyhole boundary)

Companion plot to single_track_<phase>_<t>_OF_vs_*_diff_slices.png — shows
where the melt pool sits in the same coordinate frame as where solver
disagreement hot-spots live.

Output: docs/benchmark/figures/single_track_<phase>_<t>_OF_meltpool_slices.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

NPZ_DIR = Path("/home/yzk/DILU-Research/dilu/amgx/bench")
OUTDIR  = Path("/home/yzk/DILU-Research/docs/benchmark/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Same as diff script — must MATCH for direct overlay
Z_SLICES_UM = [96, 108, 120, 132]
Y_SLICES_UM = [100, 300, 500, 620]
POWDER_SURFACE_UM = 100.0

# T_corr0 (temperature) npz files
NPZ_BY_LABEL = {
    ("melt", "3.2e-07"):       "single_melting_T_corr0_3.2e-07.npz",
    ("melt", "3.8e-07"):       "single_melting_T_corr0_3.8e-07.npz",
    ("melt", "4.1e-07"):       "single_melting_T_corr0_4.1e-07.npz",
    ("evap_early", "7e-07"):   "single_evap_early_T_corr0_7e-07.npz",
    ("evap", "9e-07"):         "single_evap_T_corr0_9e-07.npz",
    ("evap_late", "1.06e-06"): "single_evap_late_T_corr0_1.06e-06.npz",
}

T_AMBIENT = 298.0   # K
T_HAZ = 1000.0
T_MELT = 1900.0
T_VAP = 3000.0


def load_case(phase: str, t: str):
    fn = NPZ_BY_LABEL[(phase, t)]
    d = np.load(NPZ_DIR / fn, allow_pickle=True)
    return {
        "x_OF":   d["x_OF"],
        "i": d["i"], "j": d["j"], "k": d["k"],
        "nx": int(d["nx"][0]), "ny": int(d["ny"][0]), "nz": int(d["nz"][0]),
        "dx_um": float(d["dx"][0]) * 1e6,
        "phase": phase, "t": t,
    }


def to_volume(flat, case):
    vol = np.zeros((case["nz"], case["ny"], case["nx"]))
    vol[case["k"], case["j"], case["i"]] = flat
    return vol


def add_iso_contours(ax, X, Y, slc, levels, labels):
    valid = [(lv, lb) for lv, lb in zip(levels, labels) if slc.min() < lv < slc.max()]
    if valid:
        lv_only = [lv for lv, _ in valid]
        cs = ax.contour(X, Y, slc, levels=lv_only, colors="white",
                         linewidths=1.0, alpha=0.9)
        fmt = {lv: lb for lv, lb in valid}
        ax.clabel(cs, inline=True, fontsize=6, fmt=fmt)


def main(phase: str, t: str):
    case = load_case(phase, t)
    dx = case["dx_um"]
    nx, ny, nz = case["nx"], case["ny"], case["nz"]
    vol = to_volume(case["x_OF"], case)

    T_max = float(vol.max())
    vmin, vmax = T_AMBIENT, max(T_max, T_HAZ * 1.1)

    # aspect="equal" + 4:1 height ratio so XY (200×800) shows its real 1:4 shape
    fig, axes = plt.subplots(2, 4, figsize=(14, 16),
                              gridspec_kw={"height_ratios": [4, 1]})
    cmap = "hot"
    norm = Normalize(vmin=vmin, vmax=vmax)

    iso_levels = [T_HAZ, T_MELT, T_VAP]
    iso_labels = ["HAZ 1000K", "MELT 1900K", "VAP 3000K"]

    # ---- Row 1: xy slices at 4 z-depths ----
    for col, z_um in enumerate(Z_SLICES_UM):
        ax = axes[0, col]
        k = int(round(z_um / dx))
        slc = vol[k, :, :]
        x_cc = (np.arange(nx) + 0.5) * dx
        y_cc = (np.arange(ny) + 0.5) * dx
        X, Y = np.meshgrid(x_cc, y_cc)
        ax.pcolormesh(X, Y, slc, norm=norm, cmap=cmap,
                       shading="auto", rasterized=True)
        add_iso_contours(ax, X, Y, slc, iso_levels, iso_labels)
        n_melt = int(np.sum(slc > T_MELT))
        ax.set_title(f"z = {z_um} μm  (k={k})\n"
                      f"T_max = {slc.max():.0f} K  ({n_melt} cells > {T_MELT:.0f}K)",
                      fontsize=8.5)
        ax.set_xlabel("x (μm)", fontsize=8)
        if col == 0: ax.set_ylabel("y (μm) — laser scan axis", fontsize=8)
        ax.set_xlim(0, nx*dx); ax.set_ylim(0, ny*dx)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")

    # ---- Row 2: xz slices at 4 y-positions ----
    for col, y_um in enumerate(Y_SLICES_UM):
        ax = axes[1, col]
        j = int(round(y_um / dx))
        slc = vol[:, j, :]
        x_cc = (np.arange(nx) + 0.5) * dx
        z_cc = (np.arange(nz) + 0.5) * dx
        X, Z = np.meshgrid(x_cc, z_cc)
        ax.pcolormesh(X, Z, slc, norm=norm, cmap=cmap,
                       shading="auto", rasterized=True)
        add_iso_contours(ax, X, Z, slc, iso_levels, iso_labels)
        ax.axhline(POWDER_SURFACE_UM, color="cyan", linestyle=":", linewidth=1, alpha=0.7)
        n_melt = int(np.sum(slc > T_MELT))
        ax.set_title(f"y = {y_um} μm  (j={j})\n"
                      f"T_max = {slc.max():.0f} K  ({n_melt} cells > {T_MELT:.0f}K)",
                      fontsize=8.5)
        ax.set_xlabel("x (μm)", fontsize=8)
        if col == 0: ax.set_ylabel("z (μm) — depth", fontsize=8)
        ax.set_xlim(0, nx*dx); ax.set_ylim(0, nz*dx)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                         ax=axes.ravel().tolist(), shrink=0.7, pad=0.02,
                         fraction=0.022)
    cbar.set_label(f"T from OF (K) — white iso: HAZ 1000K, MELT 1900K, VAP 3000K",
                    fontsize=10)

    n_total_melt = int(np.sum(vol > T_MELT))
    n_total_vap = int(np.sum(vol > T_VAP))
    fig.suptitle(
        f"single_track_dump T_corr0 (OF DICPCG @ tol=1e-8) — physical melt pool\n"
        f"SAME coordinate frame as diff-slice plots: rows = horizontal (xy at z=96/108/120/132) "
        f"+ side (xz at y=100/300/500/620)\n"
        f"{phase}, t={t}, T_max = {T_max:.0f} K   |   total {n_total_melt:,} cells > MELT, "
        f"{n_total_vap:,} > VAP   |   cyan dotted = substrate-powder interface z≈{POWDER_SURFACE_UM:.0f}μm",
        fontsize=10, y=0.995
    )
    out = OUTDIR / f"single_track_{phase}_{t}_OF_meltpool_slices.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}  T_max={T_max:.0f}K, melt_cells={n_total_melt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="melt")
    ap.add_argument("--t", default="4.1e-07")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        for (ph, t) in NPZ_BY_LABEL:
            main(ph, t)
    else:
        main(args.phase, args.t)
