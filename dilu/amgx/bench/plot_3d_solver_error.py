"""3D spatial error fields: OF tol=1e-8 vs AMGx tol=1e-8 vs AMGx tol=1e-12.

Data source: LPBF_crosscheck pd_corr0 t=3.5e-9 (2M cells, 80×320×80 mesh,
domain 200×800×200 μm), our own clean serial dump.

Truth = AMGx tol=1e-12 (rel_resid 1.7e-12 ≈ machine ε proxy on dev RTX 3050).
This avoids needing scipy SuperLU which OOMs on 2M cells.

Three error fields plotted in 3D scatter (oblique + top-down):
  |x_OF      - x_truth|  ← OF DICPCG @ tol=1e-8 truncation noise
  |x_AMGx_e8 - x_truth|  ← AMGx @ tol=1e-8 noise
  |x_AMGx_e12- x_truth|  ← essentially 0 (truth itself)

Wall comparison sourced from:
  - OF 32-rank Xeon historical (49.57 ms / pd_corr0)
  - AMGx 5060 RTX from tol_sweep_results.json
  (NOT dev RTX 3050 numbers, which are 20× slower)

Output: docs/benchmark/figures/amgx_3d_solver_error.png
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")

from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUTDIR = Path("/home/yzk/DILU-Research/docs/benchmark/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)
DUMP = Path("/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_crosscheck/postProcessing/matrices/3.537365257e-09/pd_corr0")

NX, NY, NZ = 80, 320, 80
DX = 2.5e-6  # 2.5 μm cell


def cell_xyz_indices(N=NX*NY*NZ):
    cid = np.arange(N, dtype=np.int64)
    return cid % NX, (cid // NX) % NY, cid // (NX * NY)


def scatter3d_error(ax, err, *, title, max_points=40000, threshold_pa=0.0,
                       elev=22, azim=-55, z_window=(80, 160),
                       cmap="hot", vmin=None, vmax=None):
    """3D scatter with log colormap. Threshold filters tiny errors out for clarity."""
    i, j, k = cell_xyz_indices()
    # Z-window crop (melt zone)
    kmin = int(z_window[0] / (DX * 1e6))
    kmax = int(z_window[1] / (DX * 1e6))
    z_mask = (k >= kmin) & (k <= kmax)
    err_mask = err >= max(threshold_pa, 1e-30)
    full_mask = z_mask & err_mask
    idx = np.where(full_mask)[0]
    if idx.size == 0:
        # Nothing to plot at this threshold — relax it
        idx = np.where(z_mask)[0]
    if idx.size > max_points:
        rng = np.random.default_rng(0xC0FFEE)
        idx = rng.choice(idx, size=max_points, replace=False)

    xs = i[idx] * DX * 1e6
    ys = j[idx] * DX * 1e6
    zs = k[idx] * DX * 1e6
    es = err[idx]
    es_pos = np.maximum(es, 1e-30)

    if vmin is None:
        vmin = max(float(es_pos[es_pos > 1e-30].min()) if (es_pos > 1e-30).any() else 1e-15, 1e-15)
    if vmax is None:
        vmax = max(float(err.max()), vmin * 100)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    colors = plt.get_cmap(cmap)(norm(es_pos))

    log_e = np.log10(es_pos)
    log_min, log_max = np.log10(vmin), np.log10(vmax)
    alpha = (log_e - log_min) / max(log_max - log_min, 1e-12)
    colors[:, 3] = np.clip(alpha, 0.20, 0.95)

    ax.scatter(xs, ys, zs, c=colors, s=8, marker="o",
                linewidth=0, depthshade=True)
    ax.set_xlabel("x (μm)", fontsize=8); ax.set_ylabel("y (μm)", fontsize=8)
    ax.set_zlabel("z (μm)", fontsize=8)
    ax.set_xlim(0, NX*DX*1e6); ax.set_ylim(0, NY*DX*1e6)
    ax.set_zlim(z_window)
    zspan = z_window[1] - z_window[0]
    ax.set_box_aspect((NX*DX*1e6, NY*DX*1e6, zspan))
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=7)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def main():
    print(f"Loading dump + AMGx outputs ...", flush=True)
    A = sio.mmread(str(DUMP / "A.mm")).tocsr()
    b = sio.mmread(str(DUMP / "b.mm")).flatten()
    x_OF = sio.mmread(str(DUMP / "x_final.mm")).flatten()

    # Load AMGx results saved earlier
    x_AMGx_e8  = np.load("/tmp/x_AMGx_e8_LPBF_crosscheck.npy")
    x_AMGx_e12 = np.load("/tmp/x_AMGx_e12_LPBF_crosscheck.npy")

    # Truth = AMGx tol=1e-12 (rel_resid 1.7e-12, machine ε proxy)
    x_truth = x_AMGx_e12

    err_OF      = np.abs(x_OF        - x_truth)
    err_AMGx_e8 = np.abs(x_AMGx_e8   - x_truth)

    denom = max(float(np.abs(x_truth).max()), 1e-300)
    print(f"\n  x_truth (AMGx tol=1e-12) range:  [{x_truth.min():.3e}, {x_truth.max():.3e}]  {denom:.2e}")
    print(f"  |x_OF - x_truth|:        max {err_OF.max():.3e} Pa,  rel {err_OF.max()/denom:.3e}")
    print(f"  |x_AMGx_e8 - x_truth|:   max {err_AMGx_e8.max():.3e} Pa,  rel {err_AMGx_e8.max()/denom:.3e}")
    print(f"  |x_truth - x_truth|:     0  (by construction)")

    # Common log color range
    err_max = max(err_OF.max(), err_AMGx_e8.max())
    err_min = max(min(err_OF[err_OF > 0].min() if (err_OF > 0).any() else 1e-6,
                       err_AMGx_e8[err_AMGx_e8 > 0].min() if (err_AMGx_e8 > 0).any() else 1e-6),
                   1e-6)
    print(f"  shared log color range: {err_min:.2e} – {err_max:.2e} Pa")

    # ====== Figure: 3 columns × 2 views ======
    fig = plt.figure(figsize=(20, 11))

    # Row 1: oblique view
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    sm1 = scatter3d_error(ax1, err_OF,
                           title=f"|x_OF - x_truth|  (OF DICPCG, tol=1e-8)\n"
                                 f"max abs = {err_OF.max():.2e} Pa,  rel = {err_OF.max()/denom:.2e}",
                           cmap="hot", vmin=err_min, vmax=err_max,
                           elev=22, azim=-55)

    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    sm2 = scatter3d_error(ax2, err_AMGx_e8,
                           title=f"|x_AMGx - x_truth|  (AMGx, tol=1e-8)\n"
                                 f"max abs = {err_AMGx_e8.max():.2e} Pa,  rel = {err_AMGx_e8.max()/denom:.2e}",
                           cmap="viridis", vmin=err_min, vmax=err_max,
                           elev=22, azim=-55)

    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    sm3 = scatter3d_error(ax3, np.abs(x_truth - x_truth) + 1e-15,  # ≈0
                           title=f"|x_AMGx - x_truth|  (AMGx, tol=1e-12)\n"
                                 f"≈ machine ε  (truth itself, rel_resid 1.7e-12)",
                           cmap="Greens", vmin=err_min, vmax=err_max,
                           elev=22, azim=-55)

    # Row 2: top-down view
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    scatter3d_error(ax4, err_OF, title="OF DICPCG  top-down",
                     cmap="hot", vmin=err_min, vmax=err_max,
                     elev=85, azim=-90)
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    scatter3d_error(ax5, err_AMGx_e8, title="AMGx tol=1e-8  top-down",
                     cmap="viridis", vmin=err_min, vmax=err_max,
                     elev=85, azim=-90)
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    scatter3d_error(ax6, np.abs(x_truth - x_truth) + 1e-15,
                     title="AMGx tol=1e-12  top-down  (≈0)",
                     cmap="Greens", vmin=err_min, vmax=err_max,
                     elev=85, azim=-90)

    cbar1 = fig.colorbar(sm1, ax=[ax1, ax4], shrink=0.6, fraction=0.025)
    cbar1.set_label("|err| (Pa)", fontsize=9)
    cbar2 = fig.colorbar(sm2, ax=[ax2, ax5], shrink=0.6, fraction=0.025)
    cbar2.set_label("|err| (Pa)", fontsize=9)

    fig.suptitle(
        "Per-cell solver error — LPBF_crosscheck pd_corr0 (2 M cells, 200×800×200 μm)\n"
        "Truth = AMGx tol=1e-12 (rel_resid 1.7e-12, machine ε proxy)\n"
        "Shows only z=80–160 μm (melt zone), errors threshold filtered to be clear",
        fontsize=12, y=0.998)
    fig.tight_layout()
    out = OUTDIR / "amgx_3d_solver_error.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
