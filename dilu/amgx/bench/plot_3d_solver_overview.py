"""3D overview: physical pd field + error distributions for 6 timesteps × 3 solvers.

Two figures:

  Fig A — 3D physical pd field at 6 timesteps (pd 解本身)
          ⤷ shows melt pool / vapor structure spatially
          single panel per timestep, oblique view

  Fig B — 3D error |x_solver - x_LU| at 6 timesteps × {OF, AMGx_e8, AMGx_e12+IR}
          ⤷ shows where errors concentrate (laser focus / alpha interface)
          6 rows × 3 cols = 18 panels
          Linear color, vmax = max overall, only show cells > 1 Pa filtered

Truth: AMGx tol=1e-12 + 1 IR (verified vs CHOLMOD LU to rel 1e-11 max).

Usage:
    python -m dilu.amgx.bench.plot_3d_solver_overview
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

REPO = Path(__file__).resolve().parents[3]
OUTDIR = REPO / "docs/benchmark/figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

NX, NY, NZ = 50, 200, 50
DX = 4e-6  # 4 μm

TIMESTEPS = [
    ("3.2e-07",  "melting",     "320 ns"),
    ("3.8e-07",  "melting",     "380 ns"),
    ("4.1e-07",  "melting",     "410 ns"),
    ("7e-07",    "evap_early",  "700 ns"),
    ("9e-07",    "evap",        "900 ns"),
    ("1.06e-06", "evap_late",   "1060 ns"),
]


def load_data():
    out = []
    for t, phase, label in TIMESTEPS:
        p = REPO / "dilu/amgx/bench" / f"single_{phase}_pd_corr0_{t}.npz"
        z = np.load(p, allow_pickle=True)
        out.append(dict(t=t, phase=phase, label=label,
                        x_OF=z["x_OF"], x_AMGx_e8=z["x_AMGx_e8"],
                        x_truth=z["x_truth"], b=z["b"],
                        meta=json.loads(str(z["meta"][0]))))
    return out


def cell_xyz():
    cid = np.arange(NX*NY*NZ, dtype=np.int64)
    i = (cid % NX).astype(np.int32)
    j = ((cid // NX) % NY).astype(np.int32)
    k = (cid // (NX*NY)).astype(np.int32)
    return i, j, k


def figure_A_physical(data):
    """6 panel 3D scatter showing pd field."""
    i, j, k = cell_xyz()
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25)

    # Determine global vmin/vmax for consistent color across panels
    all_x = np.concatenate([d["x_truth"] for d in data])
    pd_lo = float(np.percentile(all_x, 1))
    pd_hi = float(np.percentile(all_x, 99))
    bg = float(np.median(all_x))
    norm = TwoSlopeNorm(vcenter=bg, vmin=pd_lo, vmax=pd_hi)
    cmap = "RdBu_r"

    rng = np.random.default_rng(0xC0FFEE)
    max_pts = 30000
    val_range = max(np.abs(all_x.max() - all_x.min()), 1.0)

    for n, d in enumerate(data):
        row, col = n // 3, n % 3
        ax = fig.add_subplot(gs[row, col], projection='3d')
        x = d["x_truth"]
        # Filter: only plot cells with deviation > 5% of range
        active = np.abs(x - bg) > 0.05 * val_range
        idx = np.where(active)[0]
        if idx.size > max_pts:
            idx = rng.choice(idx, size=max_pts, replace=False)
        xs = i[idx] * DX * 1e6
        ys = j[idx] * DX * 1e6
        zs = k[idx] * DX * 1e6
        vs = x[idx]
        colors = plt.get_cmap(cmap)(norm(vs))
        # Alpha: more deviation = more visible
        alpha = np.clip(np.abs(vs - bg) / val_range, 0.2, 0.95)
        colors[:, 3] = alpha
        ax.scatter(xs, ys, zs, c=colors, s=8, marker='o',
                    linewidth=0, depthshade=True)
        ax.set_xlim(0, NX*DX*1e6); ax.set_ylim(0, NY*DX*1e6); ax.set_zlim(0, NZ*DX*1e6)
        ax.set_box_aspect((NX*DX*1e6, NY*DX*1e6, NZ*DX*1e6))
        ax.set_xlabel("x (μm)", fontsize=8)
        ax.set_ylabel("y (μm)", fontsize=8)
        ax.set_zlabel("z (μm)", fontsize=8)
        ax.set_title(f"t = {d['label']} ({d['phase']})\n"
                      f"pd range [{x.min():.2e}, {x.max():.2e}] Pa  "
                      f"({idx.size} active cells)",
                      fontsize=9)
        ax.view_init(elev=22, azim=-55)
        ax.tick_params(labelsize=7)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=fig.axes, fraction=0.018, pad=0.02,
                       shrink=0.7, orientation='vertical')
    cb.set_label(f"pd (Pa)  diverging around median = {bg:.2e}", fontsize=10)
    fig.suptitle(
        "Physical pd field — 6 timesteps  (single-track 500K, 300W laser)\n"
        "color centered at median (~1 atm); only cells with |pd - bg| > 5% of "
        "range shown (laser focus + alpha interface)",
        fontsize=12, y=0.998)
    out = OUTDIR / "scientific_3d_physical_pd_6timesteps.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


def figure_B_errors(data):
    """6 rows × 3 cols error 3D scatter for OF, AMGx_e8, AMGx_e12+IR."""
    i, j, k = cell_xyz()

    # AMGx_e12+IR uses LU as ref (we don't have x_LU array locally for single_track,
    # but verified that AMGx_e12+IR vs LU rel < 1e-11 → for plotting use 0 / annotate)
    # OF and AMGx_e8 use AMGx_e12+IR as truth (effective LU at 1e-11 grade)

    # Determine overall vmax for OF and AMGx_e8 errors
    all_OF = np.concatenate([np.abs(d["x_OF"] - d["x_truth"]) for d in data])
    all_e8 = np.concatenate([np.abs(d["x_AMGx_e8"] - d["x_truth"]) for d in data])
    vmax = float(max(all_OF.max(), all_e8.max()))

    fig = plt.figure(figsize=(18, 30))
    gs = fig.add_gridspec(6, 3, hspace=0.45, wspace=0.30)

    rng = np.random.default_rng(0xC0FFEE)
    max_pts = 8000

    for row, d in enumerate(data):
        x_truth = d["x_truth"]
        for col, (name, x_solver, label_long) in enumerate([
            ("OF",       d["x_OF"],      "OF DICPCG (tol=1e-8)"),
            ("AMGx_e8",  d["x_AMGx_e8"], "AMGx PCG (tol=1e-8)"),
            ("AMGx_e12", d["x_truth"],   "AMGx PCG + 1 IR (tol=1e-12)"),
        ]):
            ax = fig.add_subplot(gs[row, col], projection='3d')
            # Error: AMGx_e12 case is "self-truth" → cosmic-tiny < 1e-11 (annotation only)
            err = np.abs(x_solver - x_truth)

            if name == "AMGx_e12":
                # Show annotation: max error vs LU (from lab Xeon hardcoded)
                LU_E12_MAX = {"3.2e-07":1.250e-5,"3.8e-07":1.019e-5,"4.1e-07":2.180e-6,
                              "7e-07":1.449e-5,"9e-07":1.569e-6,"1.06e-06":1.129e-5}
                e12_max = LU_E12_MAX.get(d["t"], 0)
                ax.text2D(0.5, 0.5,
                           f"AMGx tol=1e-12 + 1 IR\n"
                           f"max |x - x_LU| = {e12_max:.2e} Pa\n"
                           f"rel ≤ 1.13e-11\n\n"
                           f"(verified vs CHOLMOD\non lab Xeon)",
                           transform=ax.transAxes, ha="center", va="center",
                           fontsize=9.5, color="0.4")
                ax.set_xlim(0, NX*DX*1e6); ax.set_ylim(0, NY*DX*1e6); ax.set_zlim(0, NZ*DX*1e6)
                ax.set_box_aspect((NX*DX*1e6, NY*DX*1e6, NZ*DX*1e6))
                ax.view_init(elev=22, azim=-55)
                ax.set_title(f"{label_long}\nt = {d['label']}", fontsize=9)
                ax.tick_params(labelsize=7)
                continue

            # Filter cells with err > 1 Pa (skip noise floor)
            active = err > 1.0
            idx = np.where(active)[0]
            n_active = idx.size
            if idx.size > max_pts:
                # Keep top-error cells preferentially
                order = np.argsort(-err[idx])[:max_pts]
                idx = idx[order]

            if idx.size == 0:
                ax.text2D(0.5, 0.5, f"All cells |err| < 1 Pa\nmax = {err.max():.2e} Pa",
                           transform=ax.transAxes, ha="center", va="center",
                           fontsize=10, color="0.4")
            else:
                xs = i[idx] * DX * 1e6
                ys = j[idx] * DX * 1e6
                zs = k[idx] * DX * 1e6
                vs = err[idx]
                norm = Normalize(vmin=1.0, vmax=vmax)
                colors = plt.get_cmap("magma_r")(norm(vs))
                alpha = np.clip(vs / vmax, 0.3, 0.95)
                colors[:, 3] = alpha
                ax.scatter(xs, ys, zs, c=colors, s=20, marker='o',
                            linewidth=0.3, edgecolor="black",
                            depthshade=True)

            ax.set_xlim(0, NX*DX*1e6); ax.set_ylim(0, NY*DX*1e6); ax.set_zlim(0, NZ*DX*1e6)
            ax.set_box_aspect((NX*DX*1e6, NY*DX*1e6, NZ*DX*1e6))
            ax.set_xlabel("x (μm)", fontsize=8)
            ax.set_ylabel("y (μm)", fontsize=8)
            ax.set_zlabel("z (μm)", fontsize=8)
            ax.set_title(f"{label_long}\nt = {d['label']}\n"
                          f"max |err| = {err.max():.1f} Pa, "
                          f"{n_active} cells > 1 Pa",
                          fontsize=8.5)
            ax.view_init(elev=22, azim=-55)
            ax.tick_params(labelsize=7)

    # Shared colorbar (for OF and AMGx_e8 columns)
    norm = Normalize(vmin=1.0, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="magma_r")
    sm.set_array([])
    cb = fig.colorbar(sm, ax=fig.axes, fraction=0.012, pad=0.02,
                       shrink=0.6, orientation='vertical')
    cb.set_label(f"|x_solver - x_truth|  (Pa, linear; cells > 1 Pa shown)",
                  fontsize=10)
    fig.suptitle(
        "Error spatial distribution — 6 timesteps × 3 solvers\n"
        "Truth = AMGx tol=1e-12 + 1 IR (verified vs CHOLMOD LU to rel 1e-11)\n"
        "Cells with |err| < 1 Pa filtered out  (= numerical noise floor)",
        fontsize=12, y=0.997)
    out = OUTDIR / "scientific_3d_errors_6timesteps_3solvers.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


def main():
    data = load_data()
    print(f"Loaded {len(data)} timesteps")
    print()
    print("Generating Fig A: physical pd field ...")
    figure_A_physical(data)
    print()
    print("Generating Fig B: error distributions ...")
    figure_B_errors(data)


if __name__ == "__main__":
    main()
