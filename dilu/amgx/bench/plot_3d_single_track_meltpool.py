"""Three-solver 3D pressure cloud on single_track_dump (real melt pool case).

NEW (v2): Use marching-cubes iso-surfaces + smooth contourf slices to give a
clean "melt pool" feel instead of pixel-art.

For one chosen timestep, render side-by-side:
  col 1: x_OF        (OF DICPCG @ tol=1e-8)
  col 2: x_AMGx_e8   (AMGx PCG  @ tol=1e-8)
  col 3: x_truth     (AMGx+IR ≈ LU,    ‖A·x-b‖/‖b‖ ≈ 1e-15)
  col 4: |x_OF - x_truth|   (gap between OF's converged solution and high-precision truth)

Each column: TOP = 3D iso-surface (two nested shells at p=800kPa + p=1.1MPa),
             BOT = xz slice through laser focus rendered with contourf (smooth).

Mesh metadata (i, j, k, dx, nx, ny, nz) read straight from the npz.

Output: docs/benchmark/figures/single_track_<phase>_<t>_solver_meltpool.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from skimage import measure
from scipy.ndimage import gaussian_filter

NPZ_DIR = Path("/home/yzk/DILU-Research/dilu/amgx/bench")
OUTDIR  = Path("/home/yzk/DILU-Research/docs/benchmark/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

NPZ_BY_LABEL = {
    ("melt", "3.2e-07"):  "single_melting_pd_corr0_3.2e-07.npz",
    ("melt", "3.8e-07"):  "single_melting_pd_corr0_3.8e-07.npz",
    ("melt", "4.1e-07"):  "single_melting_pd_corr0_4.1e-07.npz",
    ("evap_early", "7e-07"):    "single_evap_early_pd_corr0_7e-07.npz",
    ("evap", "9e-07"):    "single_evap_pd_corr0_9e-07.npz",
    ("evap_late", "1.06e-06"):  "single_evap_late_pd_corr0_1.06e-06.npz",
}

# Crop ±extents in μm around the |p| peak.  5× the v3 tight crop, gets capped at domain bounds.
HALF_X, HALF_Y, HALF_Z = 200.0, 300.0, 150.0


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


def to_volume(x_flat, case):
    """Pack flat (i + j*nx + k*nx*ny) ordering into (nz, ny, nx) array."""
    vol = np.zeros((case["nz"], case["ny"], case["nx"]), dtype=np.float64)
    vol[case["k"], case["j"], case["i"]] = x_flat
    return vol


def peak_indices(vol):
    cid = int(np.argmax(np.abs(vol)))
    k, j, i = np.unravel_index(cid, vol.shape)
    return int(i), int(j), int(k)


def crop_box(case, peak_ijk):
    dx = case["dx_um"]
    i_p, j_p, k_p = peak_ijk
    di = int(np.ceil(HALF_X / dx)); dj = int(np.ceil(HALF_Y / dx)); dk = int(np.ceil(HALF_Z / dx))
    i_lo, i_hi = max(0, i_p - di), min(case["nx"] - 1, i_p + di) + 1
    j_lo, j_hi = max(0, j_p - dj), min(case["ny"] - 1, j_p + dj) + 1
    k_lo, k_hi = max(0, k_p - dk), min(case["nz"] - 1, k_p + dk) + 1
    return {
        "i_lo": i_lo, "i_hi": i_hi,
        "j_lo": j_lo, "j_hi": j_hi,
        "k_lo": k_lo, "k_hi": k_hi,
        "box_um": ((i_lo*dx, i_hi*dx), (j_lo*dx, j_hi*dx), (k_lo*dx, k_hi*dx)),
        "peak_xyz_um": (i_p * dx, j_p * dx, k_p * dx),
        "peak_ijk": peak_ijk,
        "dx_um": dx,
    }


def crop_vol(vol, crop):
    return vol[crop["k_lo"]:crop["k_hi"],
               crop["j_lo"]:crop["j_hi"],
               crop["i_lo"]:crop["i_hi"]]


def draw_isosurface(ax, vol_crop, crop, level, color, alpha=0.4,
                      smooth_sigma=0.6):
    """Marching cubes at `level`, drawn as a translucent shaded mesh."""
    dx = crop["dx_um"]
    # mild smoothing so the iso-surface looks like a clean blob, not voxelated
    vol_s = gaussian_filter(vol_crop, sigma=smooth_sigma)
    if vol_s.min() < level < vol_s.max():
        try:
            verts, faces, normals, _ = measure.marching_cubes(
                vol_s, level=level, spacing=(dx, dx, dx))
            # verts columns are (z, y, x); translate to global μm coords
            verts[:, 0] += crop["k_lo"] * dx
            verts[:, 1] += crop["j_lo"] * dx
            verts[:, 2] += crop["i_lo"] * dx
            # reorder to (x, y, z) for matplotlib
            verts_xyz = np.column_stack([verts[:, 2], verts[:, 1], verts[:, 0]])
            mesh = Poly3DCollection(verts_xyz[faces], facecolor=color,
                                     alpha=alpha, linewidth=0, edgecolor='none')
            mesh.set_facecolor((*color, alpha))
            ax.add_collection3d(mesh)
            return verts.shape[0], faces.shape[0]
        except (RuntimeError, ValueError):
            pass
    return 0, 0


def setup_3d(ax, crop, title):
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = crop["box_um"]
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_zlim(zlo, zhi)
    ax.set_xlabel("x (μm)", fontsize=8)
    ax.set_ylabel("y (μm)", fontsize=8)
    ax.set_zlabel("z (μm)", fontsize=8)
    ax.set_box_aspect((xhi - xlo, yhi - ylo, zhi - zlo))
    ax.view_init(elev=18, azim=-58)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)


def render_iso_panel(ax, vol_full, crop, levels_colors, title):
    """3D plot: nested iso-surfaces.  levels_colors = [(level_Pa, rgb, alpha), ...]"""
    vol_crop_ = crop_vol(vol_full, crop)
    n_total_verts = 0
    for level, rgb, alpha in levels_colors:
        nv, _ = draw_isosurface(ax, vol_crop_, crop, level, rgb, alpha=alpha)
        n_total_verts += nv
    # mark peak
    px, py, pz = crop["peak_xyz_um"]
    ax.scatter([px], [py], [pz], c="black", s=60, marker="*", depthshade=False,
                edgecolor="white", linewidth=0.8)
    setup_3d(ax, crop, title)


def render_slice(ax, vol_full, crop, axis, vmin, vmax, cmap, *,
                   iso_levels=None):
    """Smooth contourf of a slice through the |p| peak.
    axis = 'xy' (top-down at z=peak), 'xz' (side at y=peak), 'yz' (front at x=peak).
    """
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
        title = f"xy slice at z = {k_p*dx:.0f} μm  (top-down view of melt pool)"
    elif axis == "xz":
        slc = vol_full[k_lo:k_hi, j_p, i_lo:i_hi]
        u_cc = (np.arange(i_lo, i_hi) + 0.5) * dx
        v_cc = (np.arange(k_lo, k_hi) + 0.5) * dx
        u_label, v_label = "x (μm)", "z (μm)"
        u_peak, v_peak = i_p*dx + dx/2, k_p*dx + dx/2
        title = f"xz slice at y = {j_p*dx:.0f} μm  (scan-perp cross-section)"
    else:  # yz
        slc = vol_full[k_lo:k_hi, j_lo:j_hi, i_p]
        u_cc = (np.arange(j_lo, j_hi) + 0.5) * dx
        v_cc = (np.arange(k_lo, k_hi) + 0.5) * dx
        u_label, v_label = "y (μm)", "z (μm)"
        u_peak, v_peak = j_p*dx + dx/2, k_p*dx + dx/2
        title = f"yz slice at x = {i_p*dx:.0f} μm  (along laser path)"

    U, V = np.meshgrid(u_cc, v_cc)
    cf = ax.contourf(U, V, slc, levels=24, cmap=cmap, vmin=vmin, vmax=vmax,
                      extend="both")
    if iso_levels is not None:
        cs = ax.contour(U, V, slc, levels=iso_levels, colors="white",
                         linewidths=0.7, alpha=0.85)
    ax.plot([u_peak], [v_peak], marker="*", c="white",
             markersize=16, markeredgewidth=1.4, markeredgecolor="black",
             label="peak |p|")
    ax.set_xlabel(u_label, fontsize=8)
    ax.set_ylabel(v_label, fontsize=8)
    ax.set_title(title, fontsize=8.5)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
    return cf


def main(phase: str, t: str):
    case = load_case(phase, t)
    print(f"\ncase: phase={phase} t={t}  mesh=({case['nx']},{case['ny']},{case['nz']}) "
          f"dx={case['dx_um']}μm  N={case['x_truth'].size}")

    vol_OF    = to_volume(case["x_OF"],      case)
    vol_AMGx  = to_volume(case["x_AMGx_e8"], case)
    vol_truth = to_volume(case["x_truth"],   case)
    vol_diff  = vol_OF - vol_truth

    # Use truth peak as the reference
    peak_ijk = peak_indices(vol_truth)
    crop = crop_box(case, peak_ijk)
    print(f"  peak @ ({crop['peak_xyz_um'][0]:.0f}, {crop['peak_xyz_um'][1]:.0f}, "
          f"{crop['peak_xyz_um'][2]:.0f}) μm   crop box: "
          f"{crop['box_um'][0][1]-crop['box_um'][0][0]:.0f}×"
          f"{crop['box_um'][1][1]-crop['box_um'][1][0]:.0f}×"
          f"{crop['box_um'][2][1]-crop['box_um'][2][0]:.0f} μm")
    p_peak = float(vol_truth[peak_ijk[2], peak_ijk[1], peak_ijk[0]])
    print(f"  peak pressure: {p_peak:.3e} Pa")

    diff_max = float(np.abs(case["x_OF"] - case["x_truth"]).max())
    diff_rel = diff_max / max(np.abs(case["x_truth"]).max(), 1e-300)
    diff_amgx_max = float(np.abs(case["x_AMGx_e8"] - case["x_truth"]).max())
    diff_amgx_rel = diff_amgx_max / max(np.abs(case["x_truth"]).max(), 1e-300)
    print(f"  x_OF      vs x_truth: max|diff|={diff_max:.2e} Pa  rel={diff_rel:.2e}")
    print(f"  x_AMGx_e8 vs x_truth: max|diff|={diff_amgx_max:.2e} Pa  rel={diff_amgx_rel:.2e}")

    # Iso levels (Pa): outer shell, inner core
    # Pick relative to peak so it works across timesteps
    iso_outer = 0.55 * p_peak  # outer shell
    iso_inner = 0.82 * p_peak  # inner core
    print(f"  iso levels: outer={iso_outer:.2e} Pa  inner={iso_inner:.2e} Pa")

    # Shared color scale for slice colors: percentile-based
    crop_idx = ((case["i"] >= crop["i_lo"]) & (case["i"] < crop["i_hi"]) &
                (case["j"] >= crop["j_lo"]) & (case["j"] < crop["j_hi"]) &
                (case["k"] >= crop["k_lo"]) & (case["k"] < crop["k_hi"]))
    all_p_box = np.concatenate([case["x_OF"][crop_idx],
                                 case["x_AMGx_e8"][crop_idx],
                                 case["x_truth"][crop_idx]])
    vmin = float(np.percentile(all_p_box, 5))
    vmax = float(np.percentile(all_p_box, 99.5))
    print(f"  slice color scale: vmin={vmin:.2e}  vmax={vmax:.2e} Pa")

    # Diff color scale: symmetric
    diff_in_box = vol_diff.flat[crop_idx]  # not quite right; rebuild from vol
    # Better: from flat array
    diff_flat = case["x_OF"] - case["x_truth"]
    d_box = diff_flat[crop_idx]
    d_max = max(float(np.percentile(np.abs(d_box), 99.5)), 1e-3)
    dvmin, dvmax = -d_max, d_max

    fig = plt.figure(figsize=(20, 15))

    panels = [
        ("x_OF\n(OF DICPCG @ tol=1e-8)",      vol_OF,    "plasma",   vmin, vmax, False),
        ("x_AMGx_e8\n(AMGx PCG @ tol=1e-8)",  vol_AMGx,  "plasma",   vmin, vmax, False),
        (f"x_truth\n(AMGx+IR ≈ LU,  ‖Ax-b‖/‖b‖ ~1e-15)", vol_truth, "plasma", vmin, vmax, False),
        (f"x_OF − x_truth\n(global max|Δ|={diff_max:.2e} Pa, rel={diff_rel:.2e})",
         vol_diff, "coolwarm", dvmin, dvmax, True),
    ]

    sm_main = None; sm_diff = None
    iso_levels = [iso_outer, iso_inner]

    for col, (label, vol_, cmap, vmn, vmx, is_diff) in enumerate(panels):
        # Row 1: xy top-down
        ax = fig.add_subplot(3, 4, col + 1)
        render_slice(ax, vol_, crop, "xy", vmn, vmx, cmap,
                      iso_levels=None if is_diff else iso_levels)
        if col == 0:
            ax.text(-0.18, 0.5, "TOP-DOWN", transform=ax.transAxes,
                    rotation=90, ha="center", va="center", fontsize=11,
                    fontweight="bold")
        ax.text(0.5, 1.20, label, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

        # Row 2: xz cross section perpendicular to scan
        ax = fig.add_subplot(3, 4, col + 5)
        render_slice(ax, vol_, crop, "xz", vmn, vmx, cmap,
                      iso_levels=None if is_diff else iso_levels)
        if col == 0:
            ax.text(-0.18, 0.5, "SCAN-PERP\n(xz cross)", transform=ax.transAxes,
                    rotation=90, ha="center", va="center", fontsize=11,
                    fontweight="bold")

        # Row 3: yz along laser scan
        ax = fig.add_subplot(3, 4, col + 9)
        render_slice(ax, vol_, crop, "yz", vmn, vmx, cmap,
                      iso_levels=None if is_diff else iso_levels)
        if col == 0:
            ax.text(-0.18, 0.5, "ALONG-SCAN\n(yz cross)", transform=ax.transAxes,
                    rotation=90, ha="center", va="center", fontsize=11,
                    fontweight="bold")

        sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmn, vmax=vmx), cmap=cmap)
        if is_diff:
            sm_diff = sm
        else:
            sm_main = sm

    # Color bars
    main_axes = [ax for ax in fig.axes if ax.get_xlabel() in ("x (μm)", "y (μm)")]
    # Group the 3 solver columns vs the diff column for colorbars
    solver_axes = [fig.axes[i] for i in range(12) if (i % 4) != 3]
    diff_axes   = [fig.axes[i] for i in range(12) if (i % 4) == 3]
    cbar1 = fig.colorbar(sm_main, ax=solver_axes, shrink=0.6,
                          fraction=0.018, pad=0.02)
    cbar1.set_label("pressure pd (Pa) — white iso-contours at outer + inner shell",
                     fontsize=9)
    cbar2 = fig.colorbar(sm_diff, ax=diff_axes, shrink=0.6,
                          fraction=0.025, pad=0.04)
    cbar2.set_label("Δp = x_OF − x_truth (Pa)", fontsize=9)

    fig.suptitle(
        f"single_track_dump pd_corr0 — {phase} phase, t = {t} s   "
        f"(N=50×200×50=500 K, dx={case['dx_um']}μm)\n"
        f"laser focus @ ({crop['peak_xyz_um'][0]:.0f},{crop['peak_xyz_um'][1]:.0f},"
        f"{crop['peak_xyz_um'][2]:.0f}) μm,  peak p={p_peak/1e6:.2f} MPa,  "
        f"crop ±{HALF_X:.0f}×{HALF_Y:.0f}×{HALF_Z:.0f} μm   |   "
        f"3 orthogonal smooth contour slices per solver, white isobars at outer={iso_outer/1e6:.2f}MPa "
        f"and inner={iso_inner/1e6:.2f}MPa shell",
        fontsize=10, y=0.995)
    out = OUTDIR / f"single_track_{phase}_{t}_solver_meltpool.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="melt")
    ap.add_argument("--t", default="4.1e-07")
    args = ap.parse_args()
    main(args.phase, args.t)
