"""3D pressure cloud — TIGHT crop around laser/melt-pool peak with orthogonal slices.

For each of 3 timesteps:
  - col 1: 3D scatter, ALL cells in crop box, top 50% of |p| only (so spike pops)
  - col 2: yz slice at x = peak (front face of recoil column)
  - col 3: xz slice at y = peak (side face, melt-pool cross section)
  - col 4: xy slice at z = peak (top-down on melt-pool surface)

Output: docs/benchmark/figures/amgx_3d_lpbf_pressure_zoom.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import Normalize

LPBF = Path("/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_crosscheck/postProcessing/matrices")
OUTDIR = Path("/home/yzk/DILU-Research/docs/benchmark/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

NX, NY, NZ = 80, 320, 80
N = NX * NY * NZ
DX = 2.5e-6  # 2.5 μm

# Crop window half-extents in μm around the global |p| peak
HALF_X, HALF_Y, HALF_Z = 30.0, 40.0, 30.0

TIMESTEPS = ["1.417633672e-09", "3.537365257e-09", "8.80470235e-09"]


def cell_xyz(cid):
    i = cid % NX
    j = (cid // NX) % NY
    k = cid // (NX * NY)
    return i, j, k


def load_p(ts):
    return sio.mmread(str(LPBF / ts / "pd_corr0/x_final.mm")).flatten()


def crop_around_peak(values):
    cid_peak = int(np.argmax(np.abs(values)))
    i_p, j_p, k_p = cell_xyz(cid_peak)

    di = int(np.ceil(HALF_X / (DX * 1e6)))
    dj = int(np.ceil(HALF_Y / (DX * 1e6)))
    dk = int(np.ceil(HALF_Z / (DX * 1e6)))
    i_lo, i_hi = max(0, i_p - di), min(NX - 1, i_p + di)
    j_lo, j_hi = max(0, j_p - dj), min(NY - 1, j_p + dj)
    k_lo, k_hi = max(0, k_p - dk), min(NZ - 1, k_p + dk)

    cid_all = np.arange(N, dtype=np.int64)
    i_a, j_a, k_a = cell_xyz(cid_all)
    mask = ((i_a >= i_lo) & (i_a <= i_hi) &
            (j_a >= j_lo) & (j_a <= j_hi) &
            (k_a >= k_lo) & (k_a <= k_hi))
    idx = np.where(mask)[0]
    return {
        "idx": idx,
        "i": i_a[idx], "j": j_a[idx], "k": k_a[idx],
        "p": values[idx],
        "peak_ijk": (i_p, j_p, k_p),
        "peak_xyz_um": (i_p * DX * 1e6, j_p * DX * 1e6, k_p * DX * 1e6),
        "box_um": ((i_lo*DX*1e6, i_hi*DX*1e6),
                    (j_lo*DX*1e6, j_hi*DX*1e6),
                    (k_lo*DX*1e6, k_hi*DX*1e6)),
    }


def slice_volume(p_full, axis, fixed_idx, halfu=None, halfv=None,
                 peak_i=None, peak_j=None, peak_k=None):
    """Extract 2D slice from the full N-vector.
    axis='x': yz plane at i=fixed_idx;   u=y, v=z
    axis='y': xz plane at j=fixed_idx;   u=x, v=z
    axis='z': xy plane at k=fixed_idx;   u=x, v=y
    halfu/halfv: optional half-window in μm to crop around peak's u/v indices.
    Returns (slc, extent_um).
    """
    cid_all = np.arange(N, dtype=np.int64)
    i_a, j_a, k_a = cell_xyz(cid_all)

    if axis == 'x':
        sel = (i_a == fixed_idx)
        u, v = j_a[sel], k_a[sel]
        Nu, Nv = NY, NZ
        u_peak, v_peak = peak_j, peak_k
        u_label, v_label = "y (μm)", "z (μm)"
    elif axis == 'y':
        sel = (j_a == fixed_idx)
        u, v = i_a[sel], k_a[sel]
        Nu, Nv = NX, NZ
        u_peak, v_peak = peak_i, peak_k
        u_label, v_label = "x (μm)", "z (μm)"
    else:  # 'z'
        sel = (k_a == fixed_idx)
        u, v = i_a[sel], j_a[sel]
        Nu, Nv = NX, NY
        u_peak, v_peak = peak_i, peak_j
        u_label, v_label = "x (μm)", "y (μm)"

    slc = np.full((Nv, Nu), np.nan)
    slc[v, u] = p_full[sel]

    if halfu is not None:
        du = int(np.ceil(halfu / (DX*1e6)))
        u_lo, u_hi = max(0, u_peak - du), min(Nu, u_peak + du + 1)
    else:
        u_lo, u_hi = 0, Nu
    if halfv is not None:
        dv = int(np.ceil(halfv / (DX*1e6)))
        v_lo, v_hi = max(0, v_peak - dv), min(Nv, v_peak + dv + 1)
    else:
        v_lo, v_hi = 0, Nv
    slc_c = slc[v_lo:v_hi, u_lo:u_hi]
    extent = (u_lo*DX*1e6, u_hi*DX*1e6, v_lo*DX*1e6, v_hi*DX*1e6)
    return slc_c, extent, u_label, v_label


def render_3d(ax, crop, vmin, vmax, *, cmap="inferno", elev=22, azim=-55,
              top_frac=0.5):
    """Plot only top-`top_frac` of |p - median| cells in crop, to highlight spike."""
    vs = crop["p"]
    med = float(np.median(vs))
    dev = np.abs(vs - med)
    cutoff = np.quantile(dev, 1 - top_frac)
    mask = dev >= cutoff
    xs = crop["i"][mask] * DX * 1e6
    ys = crop["j"][mask] * DX * 1e6
    zs = crop["k"][mask] * DX * 1e6
    vs_p = vs[mask]

    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = plt.get_cmap(cmap)(norm(vs_p))
    max_dev = float(dev[mask].max()) if mask.any() else 1.0
    a = ((dev[mask] / max_dev) ** 0.6)
    colors[:, 3] = np.clip(a, 0.10, 0.95)

    ms = 18
    ax.scatter(xs, ys, zs, c=colors, s=ms, marker="o",
               linewidth=0, depthshade=True)
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = crop["box_um"]
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_zlim(zlo, zhi)
    ax.set_xlabel("x (μm)", fontsize=8)
    ax.set_ylabel("y (μm)", fontsize=8)
    ax.set_zlabel("z (μm)", fontsize=8)
    ax.set_box_aspect((xhi - xlo, yhi - ylo, zhi - zlo))
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=7)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap), int(mask.sum())


def render_slice(ax, slc, extent, vmin, vmax, *, u_label, v_label,
                  title, cmap="inferno", peak_uv=None):
    im = ax.imshow(slc, origin="lower", extent=extent, aspect="equal",
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if peak_uv is not None:
        ax.plot([peak_uv[0]], [peak_uv[1]], marker="x", c="cyan",
                markersize=12, markeredgewidth=2, label="peak |p|")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
    ax.set_xlabel(u_label, fontsize=8)
    ax.set_ylabel(v_label, fontsize=8)
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=7)
    return im


def main():
    fields = [load_p(ts) for ts in TIMESTEPS]
    crops = [crop_around_peak(x) for x in fields]

    # Color scale: clip top 25% of the crop so the slab + spike both render bright.
    all_p = np.concatenate([c["p"] for c in crops])
    vmin = float(np.percentile(all_p, 75))   # ≈ slab top
    vmax = float(np.percentile(all_p, 99.9))  # near peak (avoid single-cell outlier saturating)
    print(f"  color scale: vmin (75th %) = {vmin:.3e} Pa, vmax (99.9th %) = {vmax:.3e} Pa "
          f"(bulk median = {float(np.median(all_p)):.3e} Pa, true max = {float(all_p.max()):.3e} Pa)")

    fig = plt.figure(figsize=(20, 13))
    sm_3d = None
    for row, (ts, p_full, crop) in enumerate(zip(TIMESTEPS, fields, crops)):
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = crop["box_um"]
        i_p, j_p, k_p = crop["peak_ijk"]

        # col 1: 3D scatter
        ax = fig.add_subplot(3, 4, row*4 + 1, projection='3d')
        sm_3d, n_plotted = render_3d(ax, crop, vmin, vmax,
                                       cmap="inferno", elev=22, azim=-55,
                                       top_frac=0.10)
        ax.set_title(
            f"t = {float(ts)*1e9:.2f} ns   3D scatter (top 10% by |p-med|)\n"
            f"box {xhi-xlo:.0f}×{yhi-ylo:.0f}×{zhi-zlo:.0f} μm, "
            f"{n_plotted}/{crop['idx'].size} cells",
            fontsize=8.5)

        # col 2: yz slice at i=peak_i
        slc, ext, ul, vl = slice_volume(p_full, 'x', i_p,
                                          halfu=HALF_Y, halfv=HALF_Z,
                                          peak_i=i_p, peak_j=j_p, peak_k=k_p)
        ax = fig.add_subplot(3, 4, row*4 + 2)
        render_slice(ax, slc, ext, vmin, vmax, u_label=ul, v_label=vl,
                      title=f"yz slice at x = {i_p*DX*1e6:.1f} μm",
                      peak_uv=(j_p*DX*1e6, k_p*DX*1e6))

        # col 3: xz slice at j=peak_j  (melt-pool cross-section)
        slc, ext, ul, vl = slice_volume(p_full, 'y', j_p,
                                          halfu=HALF_X, halfv=HALF_Z,
                                          peak_i=i_p, peak_j=j_p, peak_k=k_p)
        ax = fig.add_subplot(3, 4, row*4 + 3)
        render_slice(ax, slc, ext, vmin, vmax, u_label=ul, v_label=vl,
                      title=f"xz slice at y = {j_p*DX*1e6:.1f} μm (cross-section)",
                      peak_uv=(i_p*DX*1e6, k_p*DX*1e6))

        # col 4: xy slice at k=peak_k  (top-down)
        slc, ext, ul, vl = slice_volume(p_full, 'z', k_p,
                                          halfu=HALF_X, halfv=HALF_Y,
                                          peak_i=i_p, peak_j=j_p, peak_k=k_p)
        ax = fig.add_subplot(3, 4, row*4 + 4)
        render_slice(ax, slc, ext, vmin, vmax, u_label=ul, v_label=vl,
                      title=f"xy slice at z = {k_p*DX*1e6:.1f} μm (top-down)",
                      peak_uv=(i_p*DX*1e6, j_p*DX*1e6))

    cbar = fig.colorbar(sm_3d, ax=fig.axes, shrink=0.55,
                         fraction=0.018, pad=0.04)
    cbar.set_label("pressure pd (Pa)", fontsize=9)
    fig.suptitle(
        "LPBF_crosscheck pd_corr0 — tight crop around laser peak, 3 timesteps × (3D scatter + 3 orthogonal slices)\n"
        f"crop: ±{HALF_X:.0f}×{HALF_Y:.0f}×{HALF_Z:.0f} μm around |p| peak.  "
        "color scale clipped from bulk median to peak so 2.3 MPa recoil spike pops",
        fontsize=11, y=0.995)
    out = OUTDIR / "amgx_3d_lpbf_pressure_zoom.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

    for ts, crop in zip(TIMESTEPS, crops):
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = crop["box_um"]
        print(f"  t={float(ts)*1e9:5.2f}ns  peak@({crop['peak_xyz_um'][0]:.1f},"
              f"{crop['peak_xyz_um'][1]:.1f},{crop['peak_xyz_um'][2]:.1f})μm "
              f"p∈[{crop['p'].min():.2e},{crop['p'].max():.2e}]Pa")


if __name__ == "__main__":
    main()
