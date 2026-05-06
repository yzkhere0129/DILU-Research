"""TRUE 3D visualization of OUR OWN LPBF_crosscheck dumps (NOT senior's data).

Source data: /home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_crosscheck/
Mesh: 80×320×80 = 2,048,000 cells, dx=2.5μm, domain 200×800×200 μm
6 timesteps, each has T_corr0 + pd_corr0/1/2 dumps.

These dumps ARE consistent (A·x_final ≈ b within 1e-8) because our matrixDumper
adds boundary_internalCoeffs into the diagonal explicitly before writing.

Two figures produced:
  1. amgx_3d_lpbf_pressure.png — 3D scatter of x_final (pressure, pd_corr0) at 3 timesteps
  2. amgx_3d_lpbf_temperature.png — same for T (temperature, T_corr0)
  3. amgx_3d_lpbf_error.png — 3D scatter of |x_ours - x_final| using our PCG-DIC

3D rendering uses alpha = (|val - background| / max_dev)^γ so the gas-phase
background fades out and the laser-affected zone pops.  Sub-samples 1/n cells
to keep matplotlib responsive at 2M points.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)
from matplotlib.colors import Normalize, LogNorm

LPBF = Path("/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_crosscheck/postProcessing/matrices")
OUTDIR = Path("/home/yzk/DILU-Research/docs/benchmark/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

NX, NY, NZ = 80, 320, 80
N = NX * NY * NZ
DX = 2.5e-6  # 2.5 μm


def load_field(timestep: str, eq: str) -> np.ndarray:
    """Load x_final.mm for given timestep and equation."""
    p = LPBF / timestep / eq / "x_final.mm"
    return sio.mmread(str(p)).flatten()


def cell_xyz_indices(N: int = N):
    """Return (i, j, k) arrays of length N matching cellID ordering i+j*NX+k*NX*NY."""
    cid = np.arange(N, dtype=np.int64)
    i = cid % NX
    j = (cid // NX) % NY
    k = cid // (NX * NY)
    return i, j, k


def subsample_by_deviation(values, threshold_frac=0.05, max_points=50000):
    """Return cell indices to plot: keep cells with |val - background| above
    `threshold_frac × max_dev`, then thin to at most max_points."""
    bkg = float(np.median(values))
    dev = np.abs(values - bkg)
    max_dev = float(dev.max()) if dev.max() > 0 else 1.0
    mask = dev >= threshold_frac * max_dev
    idx = np.where(mask)[0]
    if idx.size > max_points:
        rng = np.random.default_rng(0xC0FFEE)
        idx = rng.choice(idx, size=max_points, replace=False)
    return idx, bkg, max_dev


def plot_3d_scatter(ax, values, label, *, cmap="inferno",
                     threshold_frac=0.30, max_points=40000, vlog=False,
                     vmin=None, vmax=None, marker_size=10,
                     elev=22, azim=-55,
                     z_window=None):
    """Render `values` as a 3D scatter — only cells with deviation >
    `threshold_frac × max_dev` AND within optional z_window=(zmin_um,zmax_um).
    Wider markers / denser sampling let small structures still be visible.

    Color = value, alpha = (dev / max_dev) ** 0.55.
    """
    i_all, j_all, k_all = cell_xyz_indices()
    base_mask = np.ones(values.size, dtype=bool)
    if z_window is not None:
        zmin_um, zmax_um = z_window
        kmin = int(np.floor(zmin_um / (DX*1e6)))
        kmax = int(np.ceil(zmax_um / (DX*1e6)))
        base_mask &= (k_all >= kmin) & (k_all <= kmax)

    bkg = float(np.median(values[base_mask]))
    dev = np.abs(values - bkg)
    max_dev = float(dev[base_mask].max()) if base_mask.any() else 1.0
    mask = base_mask & (dev >= threshold_frac * max_dev)
    idx = np.where(mask)[0]
    if idx.size > max_points:
        rng = np.random.default_rng(0xC0FFEE)
        idx = rng.choice(idx, size=max_points, replace=False)

    xs = i_all[idx] * DX * 1e6
    ys = j_all[idx] * DX * 1e6
    zs = k_all[idx] * DX * 1e6
    vs = values[idx]

    if vmin is None:
        vmin = float(np.percentile(values[base_mask], 1))
    if vmax is None:
        vmax = float(np.percentile(values[base_mask], 99.0))

    if vlog:
        v_pos = np.maximum(np.abs(vs), 1e-300)
        vmin_eff = max(float(np.abs(values[base_mask]).min()), 1e-30)
        vmax_eff = max(float(np.abs(values).max()), vmin_eff * 10)
        norm = LogNorm(vmin=vmin_eff, vmax=vmax_eff)
        colors = plt.get_cmap(cmap)(norm(v_pos))
        log_v = np.log10(v_pos)
        log_min = np.log10(vmin_eff); log_max = np.log10(vmax_eff)
        alpha = (log_v - log_min) / max(log_max - log_min, 1e-12)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors = plt.get_cmap(cmap)(norm(vs))
        dev_local = np.abs(vs - bkg)
        alpha = (dev_local / max_dev) ** 0.55

    colors[:, 3] = np.clip(alpha, 0.15, 0.95)

    ax.scatter(xs, ys, zs, c=colors, s=marker_size,
               marker="o", linewidth=0, depthshade=True)
    ax.set_xlabel("x (μm)", fontsize=8)
    ax.set_ylabel("y (μm)", fontsize=8)
    ax.set_zlabel("z (μm)", fontsize=8)
    ax.set_xlim(0, NX * DX * 1e6)
    ax.set_ylim(0, NY * DX * 1e6)
    if z_window is not None:
        ax.set_zlim(z_window)
    else:
        ax.set_zlim(0, NZ * DX * 1e6)
    # Box aspect: physical proportions, but emphasize y (the laser scan length)
    if z_window is not None:
        zspan = max(z_window[1] - z_window[0], 1.0)
        ax.set_box_aspect((NX*DX*1e6, NY*DX*1e6, zspan))
    else:
        ax.set_box_aspect((NX, NY, NZ))
    ax.set_title(label, fontsize=9)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=7)
    print(f"    drew {idx.size} of {values.size} cells (threshold={threshold_frac:.2f})")
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def fig_pressure():
    """Pressure (pd_corr0) at 3 timesteps — TRUE 3D scatter, 2 view angles."""
    timesteps = ["1.417633672e-09", "3.537365257e-09", "8.80470235e-09"]
    print(f"\n=== Figure 1: pressure pd_corr0 — 3 timesteps × 2 views ===")
    fields = [load_field(ts, "pd_corr0") for ts in timesteps]
    # Use absolute pressure scale that highlights structure
    vmin = float(np.percentile(np.concatenate(fields), 5))
    vmax = float(np.percentile(np.concatenate(fields), 99.0))
    print(f"  shared color range: [{vmin:.3e}, {vmax:.3e}] Pa")

    # The pressure structure is a thin slab at z≈100-145 μm (powder/melt zone);
    # zoom into z=80-160 μm so the 3D scatter actually shows volumetric texture
    # and isn't compressed flat against a 200μm-tall box.
    Z_WIN = (80.0, 160.0)
    fig = plt.figure(figsize=(20, 9))
    sm = None
    # Row 1: oblique view; Row 2: top-down (so the 800μm scan length pops)
    for col, (ts, x) in enumerate(zip(timesteps, fields), 1):
        title = (f"t = {float(ts)*1e9:.2f} ns   "
                  f"p_range [{x.min():.2e}, {x.max():.2e}] Pa")
        ax = fig.add_subplot(2, 3, col, projection='3d')
        sm = plot_3d_scatter(ax, x, title,
                              cmap="inferno", threshold_frac=0.05,
                              max_points=40000, marker_size=8,
                              vmin=vmin, vmax=vmax,
                              elev=22, azim=-55,
                              z_window=Z_WIN)
        ax2 = fig.add_subplot(2, 3, col + 3, projection='3d')
        sm = plot_3d_scatter(ax2, x, "(top-down view of melt zone)",
                              cmap="inferno", threshold_frac=0.05,
                              max_points=40000, marker_size=8,
                              vmin=vmin, vmax=vmax,
                              elev=85, azim=-90,
                              z_window=Z_WIN)
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.55,
                         fraction=0.022, pad=0.04)
    cbar.set_label("pressure pd (Pa)", fontsize=10)
    fig.suptitle(
        "LPBF_crosscheck pd_corr0 — 80×320×80 = 2.05 M cells, 200×800×200 μm domain\n"
        "(only cells with deviation > 30% of max-deviation shown; alpha ∝ deviation^0.55)",
        fontsize=11, y=0.995)
    out = OUTDIR / "amgx_3d_lpbf_pressure.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def fig_temperature():
    """T_corr0 across all 6 timesteps in this crosscheck case is uniform 298K
    (laser power was insufficient to heat the bulk in the integrated time
    span, t_end=8.8 ns).  Skipping the figure — it would be a single solid
    color and uninformative."""
    print(f"\n=== Figure 2: temperature SKIPPED ===")
    print("  T = 298 K everywhere across all 6 timesteps in this case.")
    print("  Would render as a uniform color, no spatial information.")


def fig_error_field():
    """Solver error field with proper truth reference.

    SuperLU truth on 2M cells OOMs the 9.7GB dev box, so we use a near-machine
    -precision proxy: PCG at tol=1e-13 (warm-started from a tol=1e-10 run).
    Verified: ‖A·x_tight - b‖/‖b‖ = 7.7e-14 on this case.

    Three error fields plotted:
      (a) |x_loose - x_tight|  ← actual solver iteration error
      (b) |x_loose - x_OF|     ← gap to OF's tol=1e-8 reference (mostly OF's noise)
    """
    from dilu.openfoam_cpu.python.ldu import csr_to_ldu
    from dilu.openfoam_cpu.python import pcg

    ts = "3.537365257e-09"
    print(f"\n=== Figure 3: solver error vs LU-proxy truth, t={ts} ===")
    p = LPBF / ts / "pd_corr0"
    A = sio.mmread(str(p / "A.mm")).tocsr()
    b = sio.mmread(str(p / "b.mm")).flatten()
    x_final = sio.mmread(str(p / "x_final.mm")).flatten()
    x0 = sio.mmread(str(p / "x0.mm")).flatten()
    print(f"  N={A.shape[0]}, nnz={A.nnz}")
    ldu = csr_to_ldu(A)

    # Loose: standard tol=1e-10
    import time
    t0 = time.time()
    res_loose = pcg.solve(ldu, b, x0.copy(),
                           tolerance=1e-10, min_iter=1, max_iter=300)
    print(f"  loose (tol=1e-10): iter={res_loose.n_iterations}, "
          f"wall={time.time()-t0:.1f}s, rN={res_loose.final_residual:.2e}")

    # Tight = near-truth proxy: warm-start from loose, push to 1e-13
    t0 = time.time()
    res_tight = pcg.solve(ldu, b, res_loose.x.copy(),
                           tolerance=1e-13, min_iter=1, max_iter=500)
    print(f"  tight (tol=1e-13): iter={res_tight.n_iterations}, "
          f"wall={time.time()-t0:.1f}s, rN={res_tight.final_residual:.2e}")
    actual_resid = float(np.linalg.norm(A @ res_tight.x - b)
                          / max(np.linalg.norm(b), 1e-300))
    print(f"  truth proxy ‖A·x_tight - b‖/‖b‖ = {actual_resid:.2e}")

    # Use tight as truth from here on
    res = res_loose                  # what we report wall/iter for
    wall = res_loose.n_iterations    # iteration count
    err = np.abs(res_loose.x - res_tight.x)   # actual solver error
    err_vs_OF = np.abs(res_loose.x - x_final)  # apparent "error" vs OF
    denom_truth = max(float(np.abs(res_tight.x).max()), 1e-300)

    print(f"  rel(loose, tight  ← proxy truth) = {err.max()/denom_truth:.3e}")
    print(f"  rel(loose, OF x_final)            = {err_vs_OF.max()/denom_truth:.3e}")
    x_final = res_tight.x   # use tight as plotted reference (renaming for downstream code)
    denom = denom_truth

    Z_WIN = (80.0, 160.0)
    fig = plt.figure(figsize=(20, 11))

    # LEFT col: pressure x_tight (our near-truth proxy, machine-ε precision)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    sm1 = plot_3d_scatter(ax1, x_final,                         # x_final = x_tight per rename
                           f"x_tight (PCG @ tol=1e-13, ‖A·x-b‖/‖b‖={actual_resid:.1e})\n"
                           f"oblique view — pressure pd (Pa)",
                           cmap="inferno", threshold_frac=0.05,
                           max_points=40000, marker_size=8,
                           elev=22, azim=-55, z_window=Z_WIN)
    cb1 = fig.colorbar(sm1, ax=ax1, shrink=0.7, fraction=0.04, pad=0.05)
    cb1.set_label("p (Pa)", fontsize=9)

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    plot_3d_scatter(ax3, x_final, "x_tight  top-down view of melt zone",
                     cmap="inferno", threshold_frac=0.05,
                     max_points=40000, marker_size=8,
                     elev=85, azim=-90, z_window=Z_WIN)

    # RIGHT col: actual solver error |x_loose - x_tight|
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    sm2 = plot_3d_scatter(ax2, err,
                           f"|x_PCG_loose - x_PCG_tight|  oblique  (log color)\n"
                           f"max abs = {err.max():.2e} Pa,  rel = {err.max()/denom:.2e}",
                           cmap="viridis", threshold_frac=0.0,
                           max_points=40000, marker_size=8, vlog=True,
                           elev=22, azim=-55, z_window=Z_WIN)
    cb2 = fig.colorbar(sm2, ax=ax2, shrink=0.7, fraction=0.04, pad=0.05)
    cb2.set_label("|error| (Pa)", fontsize=9)

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    plot_3d_scatter(ax4, err, "|x_loose - x_tight|  top-down  (log)",
                     cmap="viridis", threshold_frac=0.0,
                     max_points=40000, marker_size=8, vlog=True,
                     elev=85, azim=-90, z_window=Z_WIN)

    fig.suptitle(
        f"LPBF_crosscheck pd_corr0 @ t={float(ts)*1e9:.2f} ns — solver error vs near-truth (PCG@1e-13)\n"
        f"loose iter={res_loose.n_iterations}, tight iter={res_tight.n_iterations}, "
        f"max solver rel-err = {err.max()/denom:.2e}   "
        f"(SuperLU OOMs at 2M cells on this 9.7GB box; PCG@1e-13 used as ε-machine proxy)",
        fontsize=11, y=0.995)
    out = OUTDIR / "amgx_3d_lpbf_error.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def main():
    print(f"Source dataset: OUR OWN LPBF_crosscheck dumps (NOT senior's data)")
    print(f"  path: {LPBF}")
    print(f"  mesh: {NX}×{NY}×{NZ} = {N} cells, dx={DX*1e6:.1f}μm")
    fig_pressure()
    fig_temperature()
    fig_error_field()
    print(f"\nAll figures in: {OUTDIR}")


if __name__ == "__main__":
    main()
