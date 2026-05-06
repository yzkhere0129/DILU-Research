"""3D spatial visualization of AMGx error and physical pressure field.

For a chosen senior pd bundle (step, corr) on the structured 80×80×80
mesh (dx = 2.5 μm), produce one figure with:
  Top row    : x_xref (OpenFOAM reference pressure)  — 3 orthogonal slices
  Middle row : x_AMGx (our solver output)             — same slices
  Bottom row : log10(|x_AMGx - x_xref|)               — same slices

Slices: mid-x (y-z plane), mid-y (x-z plane), mid-z (x-y plane).

Output: docs/benchmark/figures/amgx_field3d_<dataset>_<step>_<corr>.png

For the *initial* dataset, x_AMGx is a re-solve (AMGx is converged to
1e-15, valid xref proxy). For the *evaporation* dataset we only plot
x_xref — bare (A, b) is not in column space of A so AMGx cannot reach
the OF solution from x0=0 (out-of-scope data-format issue, not a solver
bug; tracked in test_pcg_cpp_perf.py docstring).
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from dilu.amgx.bench.senior_data_loader import set_dataset, load_step

OUTDIR = Path("/home/yzk/DILU-Research/docs/benchmark/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Mesh
NX = NY = NZ = 80
DX = 2.5e-6  # 2.5 μm


def to_grid(values: np.ndarray) -> np.ndarray:
    """Flat (512000,) → (NX, NY, NZ).  cellID = i + j*NX + k*NX*NY (i fastest)."""
    assert values.size == NX * NY * NZ
    return values.reshape((NZ, NY, NX)).transpose(2, 1, 0)


def solve_amgx(A, b, tol=1e-12, n_refine=1):
    """Run AMGx with iterative refinement; returns x and final residual."""
    from dilu.amgx.python import amgx_solve_with_refinement
    res = amgx_solve_with_refinement(
        A, b, np.zeros_like(b),
        eq_kind="pd", tol=tol, n_refine=n_refine, max_iters=500)
    return np.asarray(res["x"]), float(res["rel_residual"])


def add_slice(ax, grid, axis, idx, *, dx=DX, log=False, cmap="viridis",
              vmin=None, vmax=None, title=""):
    """Pcolormesh of one orthogonal slice of `grid` (shape (NX,NY,NZ))."""
    if   axis == 0: data = grid[idx, :, :]; xlbl, ylbl = "y (μm)", "z (μm)"
    elif axis == 1: data = grid[:, idx, :]; xlbl, ylbl = "x (μm)", "z (μm)"
    elif axis == 2: data = grid[:, :, idx]; xlbl, ylbl = "x (μm)", "y (μm)"
    else: raise ValueError(axis)

    n0, n1 = data.shape
    e0 = (np.arange(n0 + 1) - 0.5) * dx * 1e6
    e1 = (np.arange(n1 + 1) - 0.5) * dx * 1e6

    norm = None
    if log:
        d = np.maximum(np.abs(data), 1e-300)
        norm = LogNorm(vmin=vmin or d[d > 0].min(), vmax=vmax or d.max())
        data = d
    pcm = ax.pcolormesh(e0, e1, data.T,
                        cmap=cmap, norm=norm,
                        vmin=None if log else vmin,
                        vmax=None if log else vmax,
                        shading="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlbl, fontsize=8)
    ax.set_ylabel(ylbl, fontsize=8)
    ax.set_aspect("equal")
    cb = plt.colorbar(pcm, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=7)


def plot_one(dataset: str, step: int, corr: int, run_amgx: bool = True):
    set_dataset(dataset)
    bundle = load_step(step, corr)
    if bundle is None:
        print(f"  bundle {dataset}/{step}/{corr} not found, skipping")
        return

    print(f"[{dataset} {step}/{corr}]  N={bundle.n}, "
          f"x_xref ∈ [{bundle.x_ref.min():.3e}, {bundle.x_ref.max():.3e}]")

    g_xref = to_grid(bundle.x_ref)

    have_amgx = False
    if run_amgx:
        try:
            x_amgx, rel_resid = solve_amgx(bundle.A, bundle.b, tol=1e-12, n_refine=1)
            err = np.abs(x_amgx - bundle.x_ref)
            denom = max(float(np.abs(bundle.x_ref).max()), 1e-300)
            print(f"  AMGx: rel_resid={rel_resid:.2e}, "
                  f"max |x_amgx - x_xref|/||x_xref||∞ = {err.max()/denom:.3e}")
            g_amgx = to_grid(x_amgx)
            g_err = to_grid(err)
            have_amgx = True
        except Exception as e:
            print(f"  AMGx solve failed: {e}")

    n_rows = 3 if have_amgx else 1
    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(13, 4.0 * n_rows + 0.5),
                             squeeze=False)

    # Row 0: x_xref
    vmin, vmax = float(g_xref.min()), float(g_xref.max())
    add_slice(axes[0, 0], g_xref, axis=0, idx=NX // 2,
              vmin=vmin, vmax=vmax,
              title=f"x_xref slice mid-x  (range {vmin:.3e}…{vmax:.3e})")
    add_slice(axes[0, 1], g_xref, axis=1, idx=NY // 2,
              vmin=vmin, vmax=vmax,
              title="x_xref slice mid-y")
    add_slice(axes[0, 2], g_xref, axis=2, idx=NZ // 2,
              vmin=vmin, vmax=vmax,
              title="x_xref slice mid-z")

    if have_amgx:
        add_slice(axes[1, 0], g_amgx, axis=0, idx=NX // 2,
                  vmin=vmin, vmax=vmax, title="x_AMGx slice mid-x")
        add_slice(axes[1, 1], g_amgx, axis=1, idx=NY // 2,
                  vmin=vmin, vmax=vmax, title="x_AMGx slice mid-y")
        add_slice(axes[1, 2], g_amgx, axis=2, idx=NZ // 2,
                  vmin=vmin, vmax=vmax, title="x_AMGx slice mid-z")

        # Row 2: log10 error
        emin = max(float(g_err[g_err > 0].min()) if (g_err > 0).any() else 1e-20,
                   1e-20)
        emax = max(float(g_err.max()), emin * 10)
        add_slice(axes[2, 0], g_err, axis=0, idx=NX // 2, log=True,
                  vmin=emin, vmax=emax, cmap="hot",
                  title=f"|x_AMGx - x_xref| mid-x (log; max {emax:.2e})")
        add_slice(axes[2, 1], g_err, axis=1, idx=NY // 2, log=True,
                  vmin=emin, vmax=emax, cmap="hot",
                  title="|x_AMGx - x_xref| mid-y (log)")
        add_slice(axes[2, 2], g_err, axis=2, idx=NZ // 2, log=True,
                  vmin=emin, vmax=emax, cmap="hot",
                  title="|x_AMGx - x_xref| mid-z (log)")

    fig.suptitle(
        f"3D spatial field — {dataset} step={step} corr={corr}  "
        f"(80³ = 512K cells, dx=2.5μm)",
        fontsize=12, y=0.995)
    fig.tight_layout()
    out = OUTDIR / f"amgx_field3d_{dataset}_{step:03d}_{corr}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=(
        "initial:5:1,initial:10:1,initial:11:1,"
        "evaporation:84:1,evaporation:95:1,evaporation:102:1"
    ))
    ap.add_argument("--no-amgx", action="store_true",
                    help="skip AMGx re-solve (only plot x_xref)")
    args = ap.parse_args()

    for spec in args.cases.split(","):
        ds, step, corr = spec.split(":")
        plot_one(ds.strip(), int(step), int(corr),
                 run_amgx=not args.no_amgx and ds.strip() == "initial")
        # For evaporation we know AMGx can't converge from x0=0 (b·1 ≠ 0),
        # so skip the re-solve and only show x_xref structure.

    print()
    print(f"All figures in: {OUTDIR}")


if __name__ == "__main__":
    main()
