"""3D pressure-field evolution + per-cell error field.

Two figures:

(1) `amgx_field3d_evap_evolution.png` — 5 evaporation time steps
    side-by-side, mid-x slice (y-z plane). Shows recoil pressure pit
    forming and migrating during 700 ns – 1.06 μs.

(2) `amgx_field3d_initial_error.png` — initial-period bundle 5/1.
    Runs our C++ PCG to convergence, plots per-cell |x_ours - x_xref|
    on 3 orthogonal slices in log scale. Confirms the residual is
    spatially uniform (= null-space drift, not localized error).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from dilu.amgx.bench.senior_data_loader import set_dataset, load_step
from dilu.openfoam_cpu.python.ldu import csr_to_ldu
from dilu.openfoam_cpu.python import pcg

OUTDIR = Path("/home/yzk/DILU-Research/docs/benchmark/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)
NX = NY = NZ = 80
DX = 2.5e-6


def to_grid(values):
    return values.reshape((NZ, NY, NX)).transpose(2, 1, 0)


def axis_edges(n):
    return (np.arange(n + 1) - 0.5) * DX * 1e6


def fig_evap_evolution():
    set_dataset("evaporation")
    # Time axis: pick 5 representative steps (start → recoil deepest → late)
    picks = [(84, 1), (89, 1), (95, 1), (99, 1), (102, 1)]
    bundles = [load_step(s, c) for s, c in picks]
    bundles = [b for b in bundles if b is not None]

    if not bundles:
        print("no evap bundles, skipping evolution fig")
        return

    # Shared color scale across panels — use the global max for consistency
    vmax = max(float(b.x_ref.max()) for b in bundles)
    vmin = min(float(b.x_ref.min()) for b in bundles)

    fig, axes = plt.subplots(1, len(bundles),
                             figsize=(3.5 * len(bundles) + 1, 4.0))
    e0 = axis_edges(NY)
    e1 = axis_edges(NZ)

    for ax, b in zip(axes, bundles):
        g = to_grid(b.x_ref)
        slab = g[NX // 2, :, :]   # mid-x slice (y-z plane)
        pcm = ax.pcolormesh(e0, e1, slab.T,
                             cmap="inferno",
                             vmin=vmin, vmax=vmax, shading="auto")
        ax.set_aspect("equal")
        ax.set_title(f"step {b.step}/{b.corr}\n"
                      f"max p = {b.x_ref.max():.2e} Pa", fontsize=9)
        ax.set_xlabel("y (μm)", fontsize=8)
        ax.set_ylabel("z (μm)", fontsize=8)

    cbar = fig.colorbar(pcm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("pressure (Pa)", fontsize=9)
    fig.suptitle("Evaporation phase — pressure x_xref evolution (mid-x slice)",
                 fontsize=12, y=1.02)
    out = OUTDIR / "amgx_field3d_evap_evolution.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def fig_initial_error():
    set_dataset("initial")
    # Pick step 5/1 — has slight non-uniformity from initial heat input
    bundle = load_step(5, 1)
    if bundle is None:
        print("initial 5/1 missing")
        return

    print(f"  initial 5/1: x_xref ∈ "
          f"[{bundle.x_ref.min():.3e}, {bundle.x_ref.max():.3e}]")

    ldu = csr_to_ldu(bundle.A)
    print("  running PCG (C++ DIC, tol=1e-12) ...")
    res = pcg.solve(ldu, bundle.b, np.zeros_like(bundle.b),
                     tolerance=1e-12, min_iter=1, max_iter=500)
    print(f"  iter={res.n_iterations}, rN={res.final_residual:.2e}")

    err = np.abs(res.x - bundle.x_ref)
    denom = max(float(np.abs(bundle.x_ref).max()), 1e-300)
    print(f"  max abs err = {err.max():.3e},   "
          f"max rel = {err.max() / denom:.3e}")

    # Most error is null-space drift (constant offset) from setReference
    # difference. Subtract median to see structured residual:
    delta = res.x - bundle.x_ref
    delta_centered = delta - np.median(delta)
    print(f"  median offset (null-space drift) = {np.median(delta):.3e}")
    print(f"  centered |Δ|.max = {np.abs(delta_centered).max():.3e}")

    g_xref = to_grid(bundle.x_ref)
    g_err  = to_grid(err)
    g_dc   = to_grid(np.abs(delta_centered))

    fig, axes = plt.subplots(3, 3, figsize=(13, 11.5))
    e_y = axis_edges(NY); e_z = axis_edges(NZ)
    e_x = axis_edges(NX)

    # Row 0 — x_xref (small variation around 1.01e5)
    vmin, vmax = float(g_xref.min()), float(g_xref.max())
    extent_pairs = [
        (axes[0, 0], g_xref[NX // 2, :, :], e_y, e_z, "y (μm)", "z (μm)", "mid-x"),
        (axes[0, 1], g_xref[:, NY // 2, :], e_x, e_z, "x (μm)", "z (μm)", "mid-y"),
        (axes[0, 2], g_xref[:, :, NZ // 2], e_x, e_y, "x (μm)", "y (μm)", "mid-z"),
    ]
    for ax, slab, ex, ey, xl, yl, lbl in extent_pairs:
        pcm = ax.pcolormesh(ex, ey, slab.T,
                             cmap="viridis", vmin=vmin, vmax=vmax,
                             shading="auto")
        ax.set_title(f"x_xref ({lbl})  {vmin:.3e} … {vmax:.3e} Pa",
                      fontsize=9)
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
        ax.set_aspect("equal")
        plt.colorbar(pcm, ax=ax, fraction=0.045, pad=0.02)

    # Row 1 — raw error (mostly uniform = null-space drift)
    emin = max(float(g_err.min()), 1e-20)
    emax = max(float(g_err.max()), emin * 10)
    rows1 = [
        (axes[1, 0], g_err[NX // 2, :, :], e_y, e_z, "y (μm)", "z (μm)", "mid-x"),
        (axes[1, 1], g_err[:, NY // 2, :], e_x, e_z, "x (μm)", "z (μm)", "mid-y"),
        (axes[1, 2], g_err[:, :, NZ // 2], e_x, e_y, "x (μm)", "y (μm)", "mid-z"),
    ]
    for ax, slab, ex, ey, xl, yl, lbl in rows1:
        pcm = ax.pcolormesh(ex, ey, slab.T, cmap="hot",
                             norm=LogNorm(vmin=emin, vmax=emax),
                             shading="auto")
        ax.set_title(f"|x_PCG - x_xref| ({lbl})", fontsize=9)
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
        ax.set_aspect("equal")
        plt.colorbar(pcm, ax=ax, fraction=0.045, pad=0.02)

    # Row 2 — centered (null-space drift removed) → reveals structured residual
    dmin = max(float(g_dc[g_dc > 0].min()) if (g_dc > 0).any() else 1e-20, 1e-20)
    dmax = max(float(g_dc.max()), dmin * 10)
    rows2 = [
        (axes[2, 0], g_dc[NX // 2, :, :], e_y, e_z, "y (μm)", "z (μm)", "mid-x"),
        (axes[2, 1], g_dc[:, NY // 2, :], e_x, e_z, "x (μm)", "z (μm)", "mid-y"),
        (axes[2, 2], g_dc[:, :, NZ // 2], e_x, e_y, "x (μm)", "y (μm)", "mid-z"),
    ]
    for ax, slab, ex, ey, xl, yl, lbl in rows2:
        pcm = ax.pcolormesh(ex, ey, slab.T, cmap="cividis",
                             norm=LogNorm(vmin=dmin, vmax=dmax),
                             shading="auto")
        ax.set_title(f"|Δ - median(Δ)| ({lbl}) — null-space-corrected error",
                      fontsize=9)
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
        ax.set_aspect("equal")
        plt.colorbar(pcm, ax=ax, fraction=0.045, pad=0.02)

    fig.suptitle(
        f"Initial period 5/1 — our PCG (C++ DIC) vs OF xref\n"
        f"iter={res.n_iterations}, rN={res.final_residual:.2e}, "
        f"max rel err = {err.max()/denom:.2e} "
        f"(median offset = {np.median(delta):.2e} = null-space drift)",
        fontsize=11, y=0.998)
    fig.tight_layout()
    out = OUTDIR / "amgx_field3d_initial_error.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out}")


def main():
    fig_evap_evolution()
    fig_initial_error()
    print()
    print(f"Figures in: {OUTDIR}")


if __name__ == "__main__":
    main()
