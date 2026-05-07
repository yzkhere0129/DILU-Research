"""3D point-cloud of |x_OF - x_AMGx_e8| for lab32_dump pd matrices.

Direct disagreement between OF DICPCG @ tol=1e-8 and AMGx @ tol=1e-8 on the
same lab Xeon 32-rank dump. Shows where two solvers diverge given the same
A, b, and nominal tolerance.

The diff field is strongly localized (only ~25-30 cells > 100 Pa out of 500K),
so we plot:
  Col 1: 3D scatter of cells with |diff| > 1 Pa (oblique)
  Col 2: top-down 3D scatter (same cells)
  Col 3: histogram of diff distribution

Per phase one row → 2 phases × 3 cols figure.

Usage:
    python -m dilu.amgx.bench.plot_lab32_OF_vs_AMGx \\
        --melting     dilu/amgx/bench/lab32_plot_melting_pd_corr0_3.8e-07.npz \\
        --evaporation dilu/amgx/bench/lab32_plot_evaporation_pd_corr0_7e-07.npz \\
        --mesh-shape 50,200,50 --dx 4e-6
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


def load_phase(npz_path: Path, mesh: tuple) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["meta"][0]))
    n = int(z["n"][0])
    nx, ny, nz, dx = mesh
    if nx * ny * nz != n:
        raise SystemExit(f"mesh {nx}*{ny}*{nz} != N={n}")
    cid = np.arange(n, dtype=np.int64)
    i = (cid % nx).astype(np.int32)
    j = ((cid // nx) % ny).astype(np.int32)
    k = (cid // (nx * ny)).astype(np.int32)
    return dict(
        x_OF=z["x_OF"], x_AMGx_e8=z["x_AMGx_e8"], x_truth=z["x_truth"],
        b=z["b"], i=i, j=j, k=k, nx=nx, ny=ny, nz=nz, dx=dx, n=n, meta=meta,
    )


def scatter3d_diff(ax, d, *, title, threshold_Pa, max_points=20000,
                     elev=22, azim=-55, vmin=None, vmax=None):
    diff = np.abs(d["x_OF"] - d["x_AMGx_e8"])
    nx, ny, nz, dx = d["nx"], d["ny"], d["nz"], d["dx"]

    active = diff > threshold_Pa
    idx = np.where(active)[0]
    n_active = idx.size
    if idx.size > max_points:
        # Keep the largest-diff cells preferentially
        order = np.argsort(-diff[idx])[:max_points]
        idx = idx[order]
    if idx.size == 0:
        ax.text2D(0.5, 0.5,
                   f"All cells |x_OF - x_AMGx_e8| < {threshold_Pa} Pa\n"
                   f"max diff = {diff.max():.2e} Pa",
                   transform=ax.transAxes, ha="center", va="center",
                   fontsize=11)
        ax.set_title(title, fontsize=10)
        return None

    xs = d["i"][idx] * dx * 1e6
    ys = d["j"][idx] * dx * 1e6
    zs = d["k"][idx] * dx * 1e6
    vs = diff[idx]

    if vmin is None: vmin = max(threshold_Pa, vs.min())
    if vmax is None: vmax = vs.max()
    norm = LogNorm(vmin=vmin, vmax=max(vmax, 10*vmin))
    cmap = "magma"
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(vs)

    # Big markers: all shown cells are "hotspots", make them prominent
    ax.scatter(xs, ys, zs, c=vs, norm=norm, cmap=cmap,
                s=200, marker="o", linewidth=0.8, edgecolor="black",
                alpha=0.92, depthshade=True)
    nx_um = nx * dx * 1e6; ny_um = ny * dx * 1e6; nz_um = nz * dx * 1e6
    ax.set_xlim(0, nx_um); ax.set_ylim(0, ny_um); ax.set_zlim(0, nz_um)
    ax.set_box_aspect((nx_um, ny_um, nz_um))
    ax.set_xlabel("x (μm)", fontsize=8)
    ax.set_ylabel("y (μm)", fontsize=8)
    ax.set_zlabel("z (μm)", fontsize=8)
    ax.set_title(f"{title}\n"
                  f"showing {idx.size} cells with |OF-AMGx| > {threshold_Pa} Pa "
                  f"(of {n_active} total)",
                  fontsize=9)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=7)
    return sm


def render_phase_row(fig, gs_row, d, *, phase_label):
    diff = np.abs(d["x_OF"] - d["x_AMGx_e8"])
    n_total = d["n"]
    denom = max(float(np.abs(d["x_OF"]).max()), 1e-300)
    rel = diff / denom
    threshold_Pa = 100.0  # show cells where they diverge by ≥ 100 Pa
    n_above_1Pa     = int((diff > 1.0).sum())
    n_above_100Pa   = int((diff > 100.0).sum())
    n_above_1kPa    = int((diff > 1000.0).sum())

    # Col 1: 3D oblique
    ax1 = fig.add_subplot(gs_row[0], projection='3d')
    sm = scatter3d_diff(ax1, d, threshold_Pa=threshold_Pa,
                          title=f"[{phase_label}]  |x_OF - x_AMGx_e8|  oblique",
                          elev=22, azim=-55)

    # Col 2: top-down 3D
    ax2 = fig.add_subplot(gs_row[1], projection='3d')
    sm2 = scatter3d_diff(ax2, d, threshold_Pa=threshold_Pa,
                           title=f"[{phase_label}]  top-down (XY plane)",
                           elev=85, azim=-90)

    # Shared colorbar (use sm from col 1; if None, use col 2)
    if sm is not None:
        cb = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.55, fraction=0.025,
                           pad=0.05, orientation='vertical')
        cb.set_label("|x_OF - x_AMGx_e8|  (Pa, log)", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    # Col 3: histogram of all diffs
    ax3 = fig.add_subplot(gs_row[2])
    nonzero = diff[diff > 1e-8]
    if nonzero.size > 0:
        bins = np.logspace(np.log10(nonzero.min()),
                            np.log10(max(nonzero.max(), nonzero.min()*10)), 60)
        ax3.hist(nonzero, bins=bins, color="tab:purple", alpha=0.7,
                  edgecolor="black", linewidth=0.3)
        ax3.set_xscale("log")
        ax3.axvline(1.0,    color="green", linestyle=":", alpha=0.7,
                     label=f"1 Pa ({n_above_1Pa:,} cells)")
        ax3.axvline(100.0,  color="orange", linestyle=":", alpha=0.7,
                     label=f"100 Pa ({n_above_100Pa:,} cells)")
        ax3.axvline(1000.0, color="red", linestyle=":", alpha=0.7,
                     label=f"1 kPa ({n_above_1kPa:,} cells)")
        ax3.set_xlabel("|x_OF - x_AMGx_e8|  (Pa)", fontsize=9)
        ax3.set_ylabel("# cells", fontsize=9)
        ax3.set_title(f"[{phase_label}]  histogram\n"
                       f"max = {diff.max():.2e} Pa "
                       f"(rel = {rel.max():.2e}),  "
                       f"median = {np.median(diff):.2e} Pa",
                       fontsize=9)
        ax3.legend(fontsize=8, loc="upper left")
        ax3.grid(alpha=0.3, which="both")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--melting", required=True)
    ap.add_argument("--evaporation", required=True)
    ap.add_argument("--mesh-shape", required=True)
    ap.add_argument("--dx", type=float, required=True)
    ap.add_argument("--out-name", default="lab32_OF_vs_AMGx_e8_pd")
    args = ap.parse_args()

    nx, ny, nz = [int(v) for v in args.mesh_shape.split(",")]
    mesh = (nx, ny, nz, args.dx)

    d_melt = load_phase(Path(args.melting), mesh)
    d_evap = load_phase(Path(args.evaporation), mesh)

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28,
                            width_ratios=[1, 1, 1.2])

    render_phase_row(fig, [gs[0, 0], gs[0, 1], gs[0, 2]],
                       d_melt, phase_label="melting (380 ns)")
    render_phase_row(fig, [gs[1, 0], gs[1, 1], gs[1, 2]],
                       d_evap, phase_label="evaporation (700 ns)")

    fig.suptitle(
        "Lab Xeon 32-rank dump pd — OF DICPCG vs AMGx (both @ tol=1e-8)\n"
        "|x_OF - x_AMGx_e8| spatial distribution  "
        f"(N=500K cells, mesh {nx}×{ny}×{nz}, dx={args.dx*1e6:.1f}μm)",
        fontsize=12, y=0.995)

    out = OUTDIR / f"{args.out_name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
