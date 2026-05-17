"""Physical pd field spatial slices on single_track_dump data.

Same 8-panel layout as diff plots:
  Row 1: xy slices at 4 z-depths (top-down)
  Row 2: xz slices at 4 y-positions (side views)

Uses aspect="equal" so XY slices show their true 1:4 (x:y) shape, not
the compressed square form used in earlier figures.

Source field: pd_corr0 PISO pressure-correction (Pa), from the OF / AMGx
truth solution. Diverging colormap centered at 0 (negative = below ambient).

Output: docs/benchmark/figures/single_track_<phase>_<t>_pd_field_slices.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, TwoSlopeNorm

NPZ_DIR = Path("/home/yzk/DILU-Research/dilu/amgx/bench")
LU_DIR  = Path("/home/yzk/DILU-Research/audit_overnight_20260509/xeon_validation/results/E12_LU_single_track")
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
# Companion T-field npz files (same case, T_corr0 system) — used for melt-pool
# iso-contour overlay so we can SEE where the melt sits on top of the pd plot.
T_NPZ_BY_LABEL = {
    ("melt", "3.2e-07"):       "single_melting_T_corr0_3.2e-07.npz",
    ("melt", "3.8e-07"):       "single_melting_T_corr0_3.8e-07.npz",
    ("melt", "4.1e-07"):       "single_melting_T_corr0_4.1e-07.npz",
    ("evap_early", "7e-07"):   "single_evap_early_T_corr0_7e-07.npz",
    ("evap", "9e-07"):         "single_evap_T_corr0_9e-07.npz",
    ("evap_late", "1.06e-06"): "single_evap_late_T_corr0_1.06e-06.npz",
}
T_MELT = 1900.0   # K — solid/liquid threshold for Ti-6Al-4V (per audit doc)
T_VAP  = 3000.0   # K — vapor onset

Z_SLICES_UM = [96, 108, 120, 132]
Y_SLICES_UM = [100, 300, 500, 620]
POWDER_SURFACE_UM = 100.0


def load_case(phase: str, t: str, source: str):
    fn = NPZ_BY_LABEL[(phase, t)]
    d = np.load(NPZ_DIR / fn)
    if source == "x_truth":
        x = d["x_truth"]   # AMGx@1e-12+IR (ε-machine truth)
        src_label = "x_truth = AMGx PCG @ tol=1e-12 + 1 IR (ε-machine)"
    elif source == "x_LU":
        # CHOLMOD direct LU solution from Xeon
        lu_path = LU_DIR / f"x_LU_{t}.npz"
        if not lu_path.exists():
            raise FileNotFoundError(f"x_LU not yet computed for t={t}: {lu_path}")
        x = np.load(lu_path)["x_LU"]
        src_label = "x_LU = CHOLMOD direct (ε-machine)"
    elif source == "x_OF":
        x = d["x_OF"]
        src_label = "x_OF = OF DICPCG @ tol=1e-8"
    elif source == "x_AMGx_e8":
        x = d["x_AMGx_e8"]
        src_label = "x_AMGx_e8 = AMGx PCG @ tol=1e-8"
    else:
        raise ValueError(f"unknown source {source}")
    return {
        "x": x, "src_label": src_label,
        "i": d["i"], "j": d["j"], "k": d["k"],
        "nx": int(d["nx"][0]), "ny": int(d["ny"][0]), "nz": int(d["nz"][0]),
        "dx_um": float(d["dx"][0]) * 1e6,
        "phase": phase, "t": t,
    }


def to_volume(flat, case):
    vol = np.zeros((case["nz"], case["ny"], case["nx"]))
    vol[case["k"], case["j"], case["i"]] = flat
    return vol


def load_T_volume(phase, t, case):
    """Load companion T_corr0 field as (nz, ny, nx) volume. Returns None if absent."""
    key = (phase, t)
    if key not in T_NPZ_BY_LABEL:
        return None
    p = NPZ_DIR / T_NPZ_BY_LABEL[key]
    if not p.exists():
        return None
    d = np.load(p)
    T_flat = d["x_OF"]   # OF-solved T (K); ε-machine truth not relevant for iso overlay
    return to_volume(T_flat, case)


def main(phase: str, t: str, source: str, view: str = "diverging"):
    global args_view
    args_view = view
    case = load_case(phase, t, source)
    dx = case["dx_um"]
    nx, ny, nz = case["nx"], case["ny"], case["nz"]
    x_flat = case["x"]
    N = x_flat.size
    T_vol = load_T_volume(phase, t, case)  # for melt iso overlay
    has_T = T_vol is not None
    if has_T:
        t_max_global = float(T_vol.max())

    x_min = float(x_flat.min())
    x_max = float(x_flat.max())

    if args_view == "abs_log":
        # |pd| log scale — best for SEEING the spatial structure of
        # melt-zone pressure peaks against the 1e5-Pa bulk baseline.
        x_show = np.abs(x_flat)
        vmin = max(float(np.percentile(x_show, 5)), 1e-6)
        vmax = float(np.percentile(x_show, 99.9))
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = "magma"
        vol = to_volume(x_show, case)
    elif args_view == "grad":
        # |grad(pd)| log scale — highlights interfaces and melt-zone boundaries
        # where pd jumps sharply (vapor recoil, surface tension, density jump).
        # Better than |pd| at exposing the melt-pool footprint because pd
        # itself has a large global baseline that masks local structure.
        vol_pd = to_volume(x_flat, case)
        dx_m = dx * 1e-6  # μm → m
        gz_v, gy_v, gx_v = np.gradient(vol_pd, dx_m, edge_order=2)
        grad_mag = np.sqrt(gx_v**2 + gy_v**2 + gz_v**2)  # shape (nz, ny, nx)
        # Convert to Pa/μm for friendlier numbers
        grad_mag = grad_mag * 1e-6
        flat_grad = grad_mag.reshape(-1)
        vmin = max(float(np.percentile(flat_grad, 5)), 1e-6)
        vmax = float(np.percentile(flat_grad, 99.9))
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = "inferno"
        vol = grad_mag  # already (nz, ny, nx)
    else:  # "diverging"
        vlim = float(np.percentile(np.abs(x_flat), 95))
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=-vlim, vmax=vlim)
        cmap = "RdBu_r"
        vol = to_volume(x_flat, case)

    fig, axes = plt.subplots(2, 4, figsize=(14, 16),
                              gridspec_kw={"height_ratios": [4, 1]})

    # ---- Row 1: xy slices at 4 z-depths ----
    for col, z_um in enumerate(Z_SLICES_UM):
        ax = axes[0, col]
        k = int(round(z_um / dx))
        slc = vol[k, :, :]
        x_cc = (np.arange(nx) + 0.5) * dx
        y_cc = (np.arange(ny) + 0.5) * dx
        X, Y = np.meshgrid(x_cc, y_cc)
        ax.pcolormesh(X, Y, slc, norm=norm, cmap=cmap, shading="auto", rasterized=True)
        # Overlay melt iso-contours from companion T field
        n_melt_slice = 0
        if has_T:
            T_slc = T_vol[k, :, :]
            n_melt_slice = int(np.sum(T_slc > T_MELT))
            for level, color, lbl in [(T_MELT, "cyan", "MELT"), (T_VAP, "magenta", "VAP")]:
                if T_slc.min() < level < T_slc.max():
                    cs = ax.contour(X, Y, T_slc, levels=[level], colors=color,
                                     linewidths=1.6, alpha=0.95)
        unit = "Pa/μm" if args_view == "grad" else "Pa"
        melt_str = f"  ({n_melt_slice} cells >MELT)" if has_T else ""
        ax.set_title(f"z = {z_um} μm  (k={k}){melt_str}\n"
                      f"max = {slc.max():.2e} {unit}\n"
                      f"min = {slc.min():.2e} {unit}", fontsize=8.5)
        ax.set_xlabel("x (μm)", fontsize=8)
        if col == 0: ax.set_ylabel("y (μm) — laser scan axis", fontsize=8)
        ax.set_xlim(0, nx*dx); ax.set_ylim(0, ny*dx)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")

    # ---- Row 2: xz side-view slices at 4 y-positions ----
    for col, y_um in enumerate(Y_SLICES_UM):
        ax = axes[1, col]
        j = int(round(y_um / dx))
        slc = vol[:, j, :]
        x_cc = (np.arange(nx) + 0.5) * dx
        z_cc = (np.arange(nz) + 0.5) * dx
        X, Z = np.meshgrid(x_cc, z_cc)
        ax.pcolormesh(X, Z, slc, norm=norm, cmap=cmap, shading="auto", rasterized=True)
        # Overlay melt + vapor iso-contours
        n_melt_slice = 0
        if has_T:
            T_slc = T_vol[:, j, :]
            n_melt_slice = int(np.sum(T_slc > T_MELT))
            for level, color in [(T_MELT, "cyan"), (T_VAP, "magenta")]:
                if T_slc.min() < level < T_slc.max():
                    ax.contour(X, Z, T_slc, levels=[level], colors=color,
                                linewidths=1.6, alpha=0.95)
        ax.axhline(POWDER_SURFACE_UM, color="white", linestyle=":", linewidth=0.8, alpha=0.5)
        unit = "Pa/μm" if args_view == "grad" else "Pa"
        melt_str = f"  ({n_melt_slice} >MELT)" if has_T else ""
        ax.set_title(f"y = {y_um} μm  (j={j}){melt_str}\n"
                      f"range [{slc.min():.1e}, {slc.max():.1e}] {unit}", fontsize=8.5)
        ax.set_xlabel("x (μm)", fontsize=8)
        if col == 0: ax.set_ylabel("z (μm) — depth", fontsize=8)
        ax.set_xlim(0, nx*dx); ax.set_ylim(0, nz*dx)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                         ax=axes.ravel().tolist(), shrink=0.7, pad=0.02,
                         fraction=0.022)
    if args_view == "abs_log":
        cbar.set_label(f"|pd_corr0| (Pa, log scale)", fontsize=10)
    elif args_view == "grad":
        cbar.set_label(f"|∇pd_corr0| (Pa/μm, log scale) — large = interface / melt boundary",
                        fontsize=10)
    else:
        cbar.set_label(f"pd_corr0 (Pa, linear ±{vlim:.0e}; saturated at 95% |pd|; "
                        f"true peaks reach ±{max(abs(x_min),abs(x_max)):.1e})",
                        fontsize=10)

    overlay_str = ""
    if has_T:
        n_total_melt = int(np.sum(T_vol > T_MELT))
        overlay_str = (f"   |   T overlay: cyan = MELT 1900K, magenta = VAP 3000K  "
                       f"(T_max={t_max_global:.0f}K, {n_total_melt:,} cells > MELT)")
    fig.suptitle(
        f"single_track_dump pd_corr0 field — {case['src_label']}\n"
        f"global range: [{x_min:.2e}, {x_max:.2e}] Pa  (phase={phase}, t={t}){overlay_str}\n"
        f"Top: xy at 4 depths   |   Bottom: xz at 4 y-positions",
        fontsize=10, y=0.995
    )
    out = OUTDIR / f"single_track_{phase}_{t}_pd_field_{source}_{view}_slices.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}  range [{x_min:.2e}, {x_max:.2e}] Pa")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="evap_late")
    ap.add_argument("--t",     default="1.06e-06")
    ap.add_argument("--source", default="x_truth",
                     choices=["x_truth", "x_LU", "x_OF", "x_AMGx_e8"])
    ap.add_argument("--view", default="grad",
                     choices=["abs_log", "diverging", "grad"],
                     help="grad: |∇pd| log-scale — exposes melt-pool / interface boundaries [default]; "
                          "abs_log: |pd| log-scale; "
                          "diverging: signed RdBu_r")
    ap.add_argument("--all", action="store_true",
                     help="generate all 6 timesteps for the chosen source")
    args = ap.parse_args()

    if args.all:
        for (ph, t) in NPZ_BY_LABEL:
            main(ph, t, args.source, args.view)
    else:
        main(args.phase, args.t, args.source, args.view)
