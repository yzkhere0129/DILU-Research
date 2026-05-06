"""TRUE 3D pressure + error figure with LU direct-solve as truth.

Source: dumper_pipeline_test (16×32×16 = 8192 cells, real physics +
LU-feasible-on-9.7GB-dev-box).

Compares three solutions side-by-side in 3D scatter:
  x_LU       — SuperLU direct solve (algebraic-exact, machine ε truth)
  x_PCG      — our C++ PCG-DIC at tol=1e-12
  x_OF       — OpenFOAM's x_final at tol=1e-8

Plus error fields (log scale):
  |x_PCG - x_LU|  — actual solver error (small, ~1e-9)
  |x_OF  - x_LU|  — OF's truncation noise (1e-5 ~ 1e-7)

Output: docs/benchmark/figures/amgx_3d_LU_truth_<ts>_<eq>.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse.linalg import splu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

from dilu.openfoam_cpu.python.ldu import csr_to_ldu
from dilu.openfoam_cpu.python import pcg


CASE = Path("/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/dumper_pipeline_test")
OUTDIR = Path("/home/yzk/DILU-Research/docs/benchmark/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

NX, NY, NZ = 16, 32, 16
DX = 2.5e-6
N = NX * NY * NZ


def cell_xyz_indices(N=N):
    cid = np.arange(N, dtype=np.int64)
    return cid % NX, (cid // NX) % NY, cid // (NX * NY)


def subsample(values, threshold_frac=0.05, max_points=8000, base_mask=None):
    bkg = float(np.median(values))
    dev = np.abs(values - bkg)
    md = float(dev.max()) if dev.max() > 0 else 1.0
    mask = dev >= threshold_frac * md
    if base_mask is not None:
        mask &= base_mask
    idx = np.where(mask)[0]
    if idx.size > max_points:
        rng = np.random.default_rng(0xC0FFEE)
        idx = rng.choice(idx, size=max_points, replace=False)
    return idx, bkg, md


def scatter3d(ax, values, title, *, cmap="inferno",
                 vmin=None, vmax=None, vlog=False,
                 marker_size=22, threshold_frac=0.05, alpha_floor=0.18,
                 elev=22, azim=-55):
    i, j, k = cell_xyz_indices()
    idx, bkg, md = subsample(values, threshold_frac=threshold_frac)
    xs = i[idx] * DX * 1e6
    ys = j[idx] * DX * 1e6
    zs = k[idx] * DX * 1e6
    vs = values[idx]
    if vmin is None:
        vmin = float(np.percentile(values, 1))
    if vmax is None:
        vmax = float(np.percentile(values, 99))

    if vlog:
        vp = np.maximum(np.abs(vs), 1e-300)
        vmin_eff = max(float(np.abs(values[values != 0]).min())
                       if (values != 0).any() else 1e-30, 1e-30)
        vmax_eff = max(float(np.abs(values).max()), vmin_eff * 10)
        norm = LogNorm(vmin=vmin_eff, vmax=vmax_eff)
        colors = plt.get_cmap(cmap)(norm(vp))
        log_v = np.log10(vp)
        log_min, log_max = np.log10(vmin_eff), np.log10(vmax_eff)
        alpha = (log_v - log_min) / max(log_max - log_min, 1e-12)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors = plt.get_cmap(cmap)(norm(vs))
        alpha = (np.abs(vs - bkg) / md) ** 0.55

    colors[:, 3] = np.clip(alpha, alpha_floor, 0.95)
    ax.scatter(xs, ys, zs, c=colors, s=marker_size,
                marker="o", linewidth=0, depthshade=True)
    ax.set_xlabel("x (μm)", fontsize=8); ax.set_ylabel("y (μm)", fontsize=8)
    ax.set_zlabel("z (μm)", fontsize=8)
    ax.set_xlim(0, NX*DX*1e6); ax.set_ylim(0, NY*DX*1e6); ax.set_zlim(0, NZ*DX*1e6)
    ax.set_box_aspect((NX, NY, NZ))
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=7)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def make_figure(ts: str, eq: str = "pd_corr0"):
    p = CASE / "postProcessing" / "matrices" / ts / eq
    A = sio.mmread(str(p / "A.mm")).tocsr()
    b = sio.mmread(str(p / "b.mm")).flatten()
    x_OF = sio.mmread(str(p / "x_final.mm")).flatten()
    x0 = sio.mmread(str(p / "x0.mm")).flatten()
    n = A.shape[0]
    print(f"=== {ts} / {eq}: N={n} ===")
    assert n == N, f"mesh mismatch ({N} != {n})"

    # LU direct solve (machine-precision truth)
    t0 = time.time()
    lu = splu(A.tocsc(), permc_spec="COLAMD")
    x_LU = lu.solve(b)
    t_LU = time.time() - t0
    rN_LU = float(np.linalg.norm(A @ x_LU - b) / max(np.linalg.norm(b), 1e-300))
    print(f"  LU  : wall={t_LU*1e3:.0f}ms, ‖A·x-b‖/‖b‖={rN_LU:.2e}")

    # Our PCG @ tol=1e-12
    ldu = csr_to_ldu(A)
    t0 = time.time()
    res = pcg.solve(ldu, b, x0.copy(), tolerance=1e-12,
                    min_iter=1, max_iter=500)
    t_PCG = time.time() - t0
    print(f"  PCG : iter={res.n_iterations}, wall={t_PCG*1e3:.0f}ms, "
          f"rN={res.final_residual:.2e}")

    # Errors
    denom = max(float(np.abs(x_LU).max()), 1e-300)
    err_PCG = np.abs(res.x - x_LU)
    err_OF = np.abs(x_OF - x_LU)
    rel_PCG = err_PCG.max() / denom
    rel_OF  = err_OF.max() / denom
    print(f"  rel(PCG, LU) = {rel_PCG:.3e}    ← real solver error")
    print(f"  rel(OF,  LU) = {rel_OF:.3e}    ← OF tol noise")

    # ----- Figure ------
    fig = plt.figure(figsize=(20, 11.5))

    # Pressure scale shared across x_LU, x_PCG, x_OF (basically identical)
    vmin_p = float(np.percentile(x_LU, 1))
    vmax_p = float(np.percentile(x_LU, 99))

    # Top row: LU truth (oblique + top-down)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    sm1 = scatter3d(ax1, x_LU,
                      f"x_LU (truth, machine ε)\n"
                      f"range [{x_LU.min():.2e}, {x_LU.max():.2e}] Pa",
                      cmap="inferno", vmin=vmin_p, vmax=vmax_p)
    cb = plt.colorbar(sm1, ax=ax1, fraction=0.04, pad=0.05)
    cb.set_label("p (Pa)", fontsize=8)

    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    scatter3d(ax2, x_LU, "x_LU  top-down view",
                cmap="inferno", vmin=vmin_p, vmax=vmax_p,
                elev=85, azim=-90)

    # PCG vs LU error (oblique) — log color
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    sm3 = scatter3d(ax3, err_PCG,
                      f"|x_PCG - x_LU| (actual solver error)\n"
                      f"max abs={err_PCG.max():.2e},  rel={rel_PCG:.2e}",
                      cmap="viridis", threshold_frac=0,
                      vlog=True, alpha_floor=0.30)
    cb = plt.colorbar(sm3, ax=ax3, fraction=0.04, pad=0.05)
    cb.set_label("|err| (Pa)", fontsize=8)

    # Bottom row: x_OF + |x_OF - x_LU|  + summary
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    scatter3d(ax4, x_OF,
                f"x_OF (OpenFOAM tol=1e-8)\n"
                f"range [{x_OF.min():.2e}, {x_OF.max():.2e}] Pa",
                cmap="cividis", vmin=vmin_p, vmax=vmax_p)

    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    sm5 = scatter3d(ax5, err_OF,
                      f"|x_OF - x_LU| (OF truncation)\n"
                      f"max abs={err_OF.max():.2e},  rel={rel_OF:.2e}",
                      cmap="hot", threshold_frac=0,
                      vlog=True, alpha_floor=0.30)
    cb = plt.colorbar(sm5, ax=ax5, fraction=0.04, pad=0.05)
    cb.set_label("|err| (Pa)", fontsize=8)

    ax6 = fig.add_subplot(2, 3, 6); ax6.axis("off")
    summary = f"""
Bundle: dumper_pipeline_test
        timestep = {ts},  eq = {eq}
N = {n}  (mesh {NX}×{NY}×{NZ})

PRECISION (each run on identical (A, b)):

   LU (truth) ‖A·x_LU - b‖₂/‖b‖₂  =  {rN_LU:.2e}
   PCG        rN_final              =  {res.final_residual:.2e}
              iterations             =  {res.n_iterations}

ERROR vs LU TRUTH (max-norm, relative):

   rel(PCG, LU)  =  {rel_PCG:.2e}      ← our solver
   rel(OF,  LU)  =  {rel_OF:.2e}      ← OF's noise

   ratio:   OF noise / our error  =  {rel_OF/max(rel_PCG,1e-300):.0f} ×

INTERPRETATION
   ▸ Our PCG matches LU truth to ~{rel_PCG:.0e}.
   ▸ OF's reference x_final differs from truth by {rel_OF:.0e},
     ~{rel_OF/max(rel_PCG,1e-300):.0f}× LARGER than our solver error.
   ▸ "OF noise" is OF's own tol=1e-8 cutoff, not our problem.
"""
    ax6.text(0.0, 1.0, summary, fontfamily="monospace",
             fontsize=9, verticalalignment="top")

    fig.suptitle(
        f"True 3D precision: our PCG vs LU truth vs OF reference\n"
        f"dumper_pipeline_test {ts}/{eq}, mesh {NX}×{NY}×{NZ}",
        fontsize=12, y=0.995)
    out = OUTDIR / f"amgx_3d_LU_truth_{ts}_{eq}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", default="3.115041853e-11,8.92992e-12",
                     help="comma-separated")
    ap.add_argument("--eq", default="pd_corr0")
    args = ap.parse_args()
    for ts in args.timesteps.split(","):
        make_figure(ts.strip(), args.eq)


if __name__ == "__main__":
    main()
