"""Three-solver T-field melt pool — uses TEMPERATURE not pressure.

Pressure (pd) does NOT carve out the melt pool shape — pd is dominated by the
powder-bed slab structure and gas background.  TEMPERATURE does:
    T > T_melt ≈ 1900 K   →   liquid metal = the melt pool
    T > T_vap  ≈ 3000 K   →   vapor / keyhole core

For one timestep, render 3 solvers (OF, AMGx_e8, truth ≈ LU) + diff:
  TOP row: xy slice at z = peak-T (top-down on melt pool surface)
  MID row: xz slice at y = peak-T (scan-perp cross section — classic tear-drop)
  BOT row: yz slice at x = peak-T (along-scan profile — laser track)

Iso-contours overlaid at T = 1000 K (HAZ), 1900 K (melt boundary), 3000 K (vapor boundary).

Output: docs/benchmark/figures/single_track_T_<phase>_<t>_solver_meltpool.png
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

NPZ_BY_LABEL = {
    ("melt", "3.2e-07"):   "single_melting_T_corr0_3.2e-07.npz",
    ("melt", "3.8e-07"):   "single_melting_T_corr0_3.8e-07.npz",
    ("melt", "4.1e-07"):   "single_melting_T_corr0_4.1e-07.npz",
    ("evap_early", "7e-07"):     "single_evap_early_T_corr0_7e-07.npz",
    ("evap", "9e-07"):     "single_evap_T_corr0_9e-07.npz",
    ("evap_late", "1.06e-06"):   "single_evap_late_T_corr0_1.06e-06.npz",
}

# Physical thresholds for typical metals (Ti-6Al-4V-like)
T_HAZ  = 1000.0  # heat-affected zone
T_MELT = 1900.0  # solidus / melt pool boundary
T_VAP  = 3000.0  # vaporization threshold

# Crop ±extents in μm around the peak T cell
HALF_X, HALF_Y, HALF_Z = 80.0, 100.0, 60.0


def load_case(phase: str, t: str):
    fn = NPZ_BY_LABEL[(phase, t)]
    d = np.load(NPZ_DIR / fn, allow_pickle=True)
    return {
        "x_OF":      d["x_OF"],
        "x_AMGx_e8": d["x_AMGx_e8"],
        "x_truth":   d["x_truth"],
        "i": d["i"], "j": d["j"], "k": d["k"],
        "nx": int(d["nx"][0]), "ny": int(d["ny"][0]), "nz": int(d["nz"][0]),
        "dx_um": float(d["dx"][0]) * 1e6,
        "phase": phase, "t": t,
    }


def to_volume(x_flat, case):
    vol = np.zeros((case["nz"], case["ny"], case["nx"]), dtype=np.float64)
    vol[case["k"], case["j"], case["i"]] = x_flat
    return vol


def peak_indices(vol):
    cid = int(np.argmax(vol))
    k, j, i = np.unravel_index(cid, vol.shape)
    return int(i), int(j), int(k)


def crop_box(case, peak_ijk):
    dx = case["dx_um"]
    i_p, j_p, k_p = peak_ijk
    di = int(np.ceil(HALF_X / dx))
    dj = int(np.ceil(HALF_Y / dx))
    dk = int(np.ceil(HALF_Z / dx))
    i_lo = max(0, i_p - di); i_hi = min(case["nx"] - 1, i_p + di) + 1
    j_lo = max(0, j_p - dj); j_hi = min(case["ny"] - 1, j_p + dj) + 1
    k_lo = max(0, k_p - dk); k_hi = min(case["nz"] - 1, k_p + dk) + 1
    return {
        "i_lo": i_lo, "i_hi": i_hi,
        "j_lo": j_lo, "j_hi": j_hi,
        "k_lo": k_lo, "k_hi": k_hi,
        "peak_xyz_um": (i_p * dx, j_p * dx, k_p * dx),
        "peak_ijk": peak_ijk,
        "dx_um": dx,
    }


def render_slice(ax, vol_full, crop, axis, vmin, vmax, cmap, *,
                   iso_levels=None, iso_labels=None):
    dx = crop["dx_um"]
    i_p, j_p, k_p = crop["peak_ijk"]
    i_lo, i_hi = crop["i_lo"], crop["i_hi"]
    j_lo, j_hi = crop["j_lo"], crop["j_hi"]
    k_lo, k_hi = crop["k_lo"], crop["k_hi"]

    if axis == "xy":
        slc = vol_full[k_p, j_lo:j_hi, i_lo:i_hi]
        u_cc = (np.arange(i_lo, i_hi) + 0.5) * dx
        v_cc = (np.arange(j_lo, j_hi) + 0.5) * dx
        u_label, v_label = "x (μm)", "y (μm)"
        u_peak, v_peak = i_p*dx + dx/2, j_p*dx + dx/2
        title = f"xy slice at z = {k_p*dx:.0f} μm  (top-down on melt pool surface)"
    elif axis == "xz":
        slc = vol_full[k_lo:k_hi, j_p, i_lo:i_hi]
        u_cc = (np.arange(i_lo, i_hi) + 0.5) * dx
        v_cc = (np.arange(k_lo, k_hi) + 0.5) * dx
        u_label, v_label = "x (μm)", "z (μm)"
        u_peak, v_peak = i_p*dx + dx/2, k_p*dx + dx/2
        title = f"xz slice at y = {j_p*dx:.0f} μm  (scan-perp — classic tear-drop)"
    else:
        slc = vol_full[k_lo:k_hi, j_lo:j_hi, i_p]
        u_cc = (np.arange(j_lo, j_hi) + 0.5) * dx
        v_cc = (np.arange(k_lo, k_hi) + 0.5) * dx
        u_label, v_label = "y (μm)", "z (μm)"
        u_peak, v_peak = j_p*dx + dx/2, k_p*dx + dx/2
        title = f"yz slice at x = {i_p*dx:.0f} μm  (along-scan profile)"

    U, V = np.meshgrid(u_cc, v_cc)
    cf = ax.contourf(U, V, slc, levels=30, cmap=cmap, vmin=vmin, vmax=vmax,
                      extend="both")
    if iso_levels is not None:
        # Filter iso levels to those in the data range
        valid = [lv for lv in iso_levels if slc.min() < lv < slc.max()]
        if valid:
            cs = ax.contour(U, V, slc, levels=valid, colors="white",
                             linewidths=1.3, alpha=0.95)
            if iso_labels:
                fmt = {lv: lbl for lv, lbl in zip(iso_levels, iso_labels) if lv in valid}
                ax.clabel(cs, inline=True, fontsize=7, fmt=fmt)
    ax.plot([u_peak], [v_peak], marker="*", c="white", markersize=14,
             markeredgewidth=1.2, markeredgecolor="black",
             label=f"peak T")
    ax.set_xlabel(u_label, fontsize=8)
    ax.set_ylabel(v_label, fontsize=8)
    ax.set_title(title, fontsize=8.5)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
    return cf


def main(phase: str, t: str):
    case = load_case(phase, t)
    vol_OF    = to_volume(case["x_OF"],      case)
    vol_AMGx  = to_volume(case["x_AMGx_e8"], case)
    vol_truth = to_volume(case["x_truth"],   case)
    vol_diff  = vol_OF - vol_truth

    peak_ijk = peak_indices(vol_truth)
    crop = crop_box(case, peak_ijk)
    T_peak = float(vol_truth[peak_ijk[2], peak_ijk[1], peak_ijk[0]])
    print(f"\ncase: phase={phase} t={t}  T_max={T_peak:.0f} K  peak@"
          f"({crop['peak_xyz_um'][0]:.0f},{crop['peak_xyz_um'][1]:.0f},"
          f"{crop['peak_xyz_um'][2]:.0f}) μm")

    diff_max = float(np.abs(case["x_OF"] - case["x_truth"]).max())
    diff_rel = diff_max / max(np.abs(case["x_truth"]).max(), 1e-300)
    print(f"  T_OF vs T_truth: max|Δ|={diff_max:.2e} K  rel={diff_rel:.2e}")

    # Color scale: from ambient (298 K) up to peak
    vmin = 298.0
    vmax = T_peak
    print(f"  color scale: {vmin:.0f} → {vmax:.0f} K  (HAZ {T_HAZ}, melt {T_MELT}, vap {T_VAP})")
    iso_levels = [T_HAZ, T_MELT, T_VAP]
    iso_labels = ["HAZ 1000K", "MELT 1900K", "VAP 3000K"]

    # Diff scale
    diff_box_max = max(diff_max, 1e-6)
    dvmin, dvmax = -diff_box_max, +diff_box_max

    fig = plt.figure(figsize=(20, 15))

    panels = [
        ("T_OF\n(OF DICPCG @ tol=1e-8)",         vol_OF,    "hot",      vmin, vmax, False),
        ("T_AMGx_e8\n(AMGx PCG @ tol=1e-8)",     vol_AMGx,  "hot",      vmin, vmax, False),
        ("T_truth\n(AMGx+IR ≈ LU)",              vol_truth, "hot",      vmin, vmax, False),
        (f"T_OF − T_truth\n(max|Δ|={diff_max:.2e}K, rel={diff_rel:.2e})",
         vol_diff, "coolwarm", dvmin, dvmax, True),
    ]

    sm_main = None; sm_diff = None
    for col, (label, vol_, cmap, vmn, vmx, is_diff) in enumerate(panels):
        ax = fig.add_subplot(3, 4, col + 1)
        render_slice(ax, vol_, crop, "xy", vmn, vmx, cmap,
                      iso_levels=None if is_diff else iso_levels,
                      iso_labels=None if is_diff else iso_labels)
        if col == 0:
            ax.text(-0.18, 0.5, "TOP-DOWN", transform=ax.transAxes,
                    rotation=90, ha="center", va="center", fontsize=11,
                    fontweight="bold")
        ax.text(0.5, 1.22, label, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

        ax = fig.add_subplot(3, 4, col + 5)
        render_slice(ax, vol_, crop, "xz", vmn, vmx, cmap,
                      iso_levels=None if is_diff else iso_levels,
                      iso_labels=None if is_diff else iso_labels)
        if col == 0:
            ax.text(-0.18, 0.5, "SCAN-PERP\n(xz cross)", transform=ax.transAxes,
                    rotation=90, ha="center", va="center", fontsize=11,
                    fontweight="bold")

        ax = fig.add_subplot(3, 4, col + 9)
        render_slice(ax, vol_, crop, "yz", vmn, vmx, cmap,
                      iso_levels=None if is_diff else iso_levels,
                      iso_labels=None if is_diff else iso_labels)
        if col == 0:
            ax.text(-0.18, 0.5, "ALONG-SCAN\n(yz cross)", transform=ax.transAxes,
                    rotation=90, ha="center", va="center", fontsize=11,
                    fontweight="bold")

        sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmn, vmax=vmx), cmap=cmap)
        if is_diff: sm_diff = sm
        else:       sm_main = sm

    solver_axes = [fig.axes[i] for i in range(12) if (i % 4) != 3]
    diff_axes   = [fig.axes[i] for i in range(12) if (i % 4) == 3]
    cbar1 = fig.colorbar(sm_main, ax=solver_axes, shrink=0.55,
                          fraction=0.018, pad=0.02)
    cbar1.set_label("Temperature T (K)  — white iso: 1000/1900/3000 K", fontsize=9)
    cbar2 = fig.colorbar(sm_diff, ax=diff_axes, shrink=0.55,
                          fraction=0.025, pad=0.04)
    cbar2.set_label("ΔT = T_OF − T_truth (K)", fontsize=9)

    fig.suptitle(
        f"single_track_dump T_corr0 — {phase} phase, t = {t} s   "
        f"(N=50×200×50=500 K, dx={case['dx_um']}μm)\n"
        f"peak T = {T_peak:.0f} K @ ({crop['peak_xyz_um'][0]:.0f},"
        f"{crop['peak_xyz_um'][1]:.0f},{crop['peak_xyz_um'][2]:.0f}) μm    "
        f"|    crop ±{HALF_X:.0f}×{HALF_Y:.0f}×{HALF_Z:.0f} μm    |    "
        f"white iso-contours: HAZ 1000K, MELT 1900K, VAP 3000K",
        fontsize=10, y=0.995)
    out = OUTDIR / f"single_track_T_{phase}_{t}_solver_meltpool.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="melt")
    ap.add_argument("--t", default="4.1e-07")
    args = ap.parse_args()
    main(args.phase, args.t)
