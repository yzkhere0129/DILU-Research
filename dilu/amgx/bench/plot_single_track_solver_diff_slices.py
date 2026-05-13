"""Spatial-slice plots of |x_OF - x_AMGx_e8| on single_track_dump data, in the
same 2-row 4-column style as the legacy lab32 diff figure.

For each timestep:
  Row 1: xy slices at 4 z-depths (top-down) — shows surface footprint of diff
  Row 2: xz slices at 4 y-positions (side views) — shows depth profile

Log color scale, capped at 10 Pa (vmax requested by user).

Output: docs/benchmark/figures/single_track_<phase>_<t>_OF_vs_AMGx_diff_slices.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

NPZ_DIR = Path("/home/yzk/DILU-Research/dilu/amgx/bench")
OUTDIR  = Path("/home/yzk/DILU-Research/docs/benchmark/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

NPZ_BY_LABEL = {
    ("melt", "3.2e-07"):       "single_melting_pd_corr0_3.2e-07.npz",
    ("melt", "3.8e-07"):       "single_melting_pd_corr0_3.8e-07.npz",
    ("melt", "4.1e-07"):       "single_melting_pd_corr0_4.1e-07.npz",
    ("evap_early", "7e-07"):   "single_evap_early_pd_corr0_7e-07.npz",
    ("evap", "9e-07"):         "single_evap_pd_corr0_9e-07.npz",
    ("evap_late", "1.06e-06"): "single_evap_late_pd_corr0_1.06e-06.npz",
}

# Z-depth slices (μm) — span the powder bed slab z=100-144
Z_SLICES_UM = [96, 108, 120, 132]
# Y-position side-view slices (μm)
Y_SLICES_UM = [100, 300, 500, 620]

# Log color scale: vmin=1e-12 Pa, vmax=1e1 Pa (user request 2026-05-13)
# 13 orders of magnitude covers everything from AMGx+IR vs LU (~1e-12 Pa quantum)
# to OF vs AMGx PCG (~10 Pa max).  Unified scale lets all 3 solver pairs be
# compared on the same colorbar.
VMIN = 1e-12
VMAX = 1e1  # 10 Pa

# Powder surface (substrate top) for cyan reference line in side views
POWDER_SURFACE_UM = 100.0


def load_case(phase: str, t: str):
    fn = NPZ_BY_LABEL[(phase, t)]
    d = np.load(NPZ_DIR / fn)
    return {
        "x_OF":      d["x_OF"],
        "x_AMGx_e8": d["x_AMGx_e8"],
        "x_truth":   d["x_truth"],
        "i": d["i"], "j": d["j"], "k": d["k"],
        "nx": int(d["nx"][0]), "ny": int(d["ny"][0]), "nz": int(d["nz"][0]),
        "dx_um": float(d["dx"][0]) * 1e6,
        "phase": phase, "t": t,
    }


def to_volume(flat, case):
    vol = np.zeros((case["nz"], case["ny"], case["nx"]))
    vol[case["k"], case["j"], case["i"]] = flat
    return vol


def main(phase: str, t: str, pair: str = "OF_vs_AMGx_e8"):
    case = load_case(phase, t)
    dx = case["dx_um"]
    nx, ny, nz = case["nx"], case["ny"], case["nz"]
    N = case["x_OF"].size

    if pair == "OF_vs_AMGx_e8":
        a_name, b_name = "x_OF (DICPCG @ 1e-8)", "x_AMGx_e8 (PCG @ 1e-8)"
        a, b = case["x_OF"], case["x_AMGx_e8"]
        title_pair = "OF DICPCG vs AMGx (both @ tol=1e-8)"
        field_name = "|x_OF - x_AMGx_e8|"
    elif pair == "OF_vs_truth":
        a_name, b_name = "x_OF (DICPCG @ 1e-8)", "x_truth (AMGx+IR ≈ LU)"
        a, b = case["x_OF"], case["x_truth"]
        title_pair = "OF DICPCG @ tol=1e-8 vs LU truth"
        field_name = "|x_OF - x_truth|"
    elif pair == "AMGx_e8_vs_truth":
        a_name, b_name = "x_AMGx_e8 (PCG @ 1e-8)", "x_truth (AMGx+IR ≈ LU)"
        a, b = case["x_AMGx_e8"], case["x_truth"]
        title_pair = "AMGx PCG @ tol=1e-8 vs LU truth"
        field_name = "|x_AMGx_e8 - x_truth|"
    else:
        raise ValueError(f"unknown pair {pair}")

    diff_flat = np.abs(a - b)
    diff_vol = to_volume(diff_flat, case)
    x_max = float(np.max(np.abs(b)))  # scale relative to one of the solutions

    # Global stats
    max_diff = float(diff_flat.max())
    rel_max = max_diff / max(x_max, 1e-300)
    cells_gt_1Pa = int(np.sum(diff_flat > 1.0))
    cells_gt_10Pa = int(np.sum(diff_flat > 10.0))
    cells_gt_100Pa = int(np.sum(diff_flat > 100.0))
    cells_gt_1kPa = int(np.sum(diff_flat > 1000.0))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    cmap = "magma"
    norm = LogNorm(vmin=VMIN, vmax=VMAX)

    # ---- Row 1: xy slices at 4 z-depths ----
    for col, z_um in enumerate(Z_SLICES_UM):
        ax = axes[0, col]
        k = int(round(z_um / dx))
        slc = diff_vol[k, :, :]   # shape (ny, nx)
        # Clip below VMIN to VMIN for log scale display (no negatives anyway since |·|)
        # Clip below VMIN so log-scale doesn't blow up on exact zeros
        slc_disp = np.maximum(slc, VMIN)

        x_cc = (np.arange(nx) + 0.5) * dx
        y_cc = (np.arange(ny) + 0.5) * dx
        X, Y = np.meshgrid(x_cc, y_cc)
        im = ax.pcolormesh(X, Y, slc_disp, norm=norm, cmap=cmap,
                            shading="auto", rasterized=True)
        n_gt1 = int(np.sum(slc > 1.0))
        n_med = int(np.sum(slc > 1e-6))
        ax.set_title(f"z = {z_um} μm  (k={k})\n"
                      f"|diff| max in slice = {slc.max():.2e} Pa\n"
                      f"{n_gt1} cells > 1 Pa, {n_med} > 1e-6 Pa", fontsize=8.5)
        ax.set_xlabel("x (μm)", fontsize=8)
        if col == 0: ax.set_ylabel("y (μm) — laser scan axis", fontsize=8)
        ax.set_xlim(0, nx*dx); ax.set_ylim(0, ny*dx)
        ax.tick_params(labelsize=7)
        ax.set_aspect("auto")

    # ---- Row 2: xz side-view slices at 4 y-positions ----
    for col, y_um in enumerate(Y_SLICES_UM):
        ax = axes[1, col]
        j = int(round(y_um / dx))
        slc = diff_vol[:, j, :]   # shape (nz, nx)
        slc_disp = np.maximum(slc, VMIN)

        x_cc = (np.arange(nx) + 0.5) * dx
        z_cc = (np.arange(nz) + 0.5) * dx
        X, Z = np.meshgrid(x_cc, z_cc)
        ax.pcolormesh(X, Z, slc_disp, norm=norm, cmap=cmap,
                       shading="auto", rasterized=True)
        # cyan reference at powder/substrate interface
        ax.axhline(POWDER_SURFACE_UM, color="cyan", linestyle=":", linewidth=1, alpha=0.7)
        n_gt1 = int(np.sum(slc > 1.0))
        n_med = int(np.sum(slc > 1e-6))
        ax.set_title(f"y = {y_um} μm  (j={j})\n"
                      f"|diff| max = {slc.max():.2e} Pa, {n_gt1} > 1 Pa, {n_med} > 1e-6 Pa", fontsize=8.5)
        ax.set_xlabel("x (μm)", fontsize=8)
        if col == 0: ax.set_ylabel("z (μm) — depth", fontsize=8)
        ax.set_xlim(0, nx*dx); ax.set_ylim(0, nz*dx)
        ax.tick_params(labelsize=7)
        ax.set_aspect("auto")

    # Shared colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                         ax=axes.ravel().tolist(), shrink=0.7, pad=0.02,
                         fraction=0.022)
    cbar.set_label(f"{field_name}  (Pa, log scale, range {VMIN:.0e} ... {VMAX:.0e} Pa)",
                    fontsize=10)

    fig.suptitle(
        f"single_track_dump pd_corr0 — {title_pair}\n"
        f"{field_name} spatial slices  ({phase}, t={t})\n"
        f"max diff = {max_diff:.2e} Pa (rel = {rel_max:.2e}), "
        f"{cells_gt_1Pa} cells > 1 Pa, {cells_gt_10Pa} > 10 Pa, "
        f"{cells_gt_100Pa} > 100 Pa, {cells_gt_1kPa} > 1 kPa  (of {N:,} total)\n"
        f"Top: horizontal slices at 4 depths   |   "
        f"Bottom: side views at 4 y positions (cyan dotted = nominal substrate-powder interface z≈{POWDER_SURFACE_UM:.0f}μm)",
        fontsize=10, y=0.995
    )
    out = OUTDIR / f"single_track_{phase}_{t}_{pair}_diff_slices.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}  (max|Δ|={max_diff:.2e}Pa, rel={rel_max:.2e}, >1Pa={cells_gt_1Pa}, >10Pa={cells_gt_10Pa})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="melt")
    ap.add_argument("--t", default="4.1e-07")
    ap.add_argument("--pair", default="OF_vs_AMGx_e8",
                    choices=["OF_vs_AMGx_e8", "OF_vs_truth", "AMGx_e8_vs_truth"])
    ap.add_argument("--all", action="store_true",
                    help="generate all 6 timesteps × all 3 pairs")
    args = ap.parse_args()

    if args.all:
        for (ph, t) in NPZ_BY_LABEL:
            for pair in ["OF_vs_AMGx_e8", "OF_vs_truth", "AMGx_e8_vs_truth"]:
                main(ph, t, pair)
    else:
        main(args.phase, args.t, args.pair)
