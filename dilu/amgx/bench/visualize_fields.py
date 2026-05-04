"""Visualize physical fields (T, p, alpha, |U|) to verify whether our
matrix dump set covers any actual melt-pool physics.

Two data sources:
  1. Senior's pd CSVs in DICPCG_Benchmark_Data — has solution (pressure)
     reshaped to 80×80×80 grid, slice plot.
  2. Our LaserbeamFoam VTK time dirs — has T, U, alpha.metal, p
     parsed with phase_detector, plot mid-plane slices.

Output: side-by-side figure showing fields at multiple times.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "/home/yzk/DILU-Research")

from dilu.amgx.bench.senior_data_loader import load_step, MESH_NX, MESH_NY, MESH_NZ
from dilu.amgx.bench.phase_detector import parse_internal_field


# ---------------------------------------------------------------------------
# Senior pd field viz (assume structured 80³)
# ---------------------------------------------------------------------------

def _reshape(values: np.ndarray, nx, ny, nz) -> np.ndarray:
    """OF blockMesh writes cells in (i,j,k) loop order with k slowest.
    cellID = i + j*NX + k*NX*NY."""
    return values.reshape((nz, ny, nx))


def plot_senior_pd_evolution(out_png: Path):
    """Plot pd field mid-z slice across multiple PISO step/correctors."""
    sweep = [(1, 1), (1, 3), (5, 1), (5, 3), (10, 1), (10, 3), (11, 3)]
    n = len(sweep)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))

    for i, (step, corr) in enumerate(sweep):
        b = load_step(step, corr)
        if b is None:
            for ax in axes[:, i]: ax.set_axis_off()
            continue
        x_grid = _reshape(b.x_ref, MESH_NX, MESH_NY, MESH_NZ)
        b_grid = _reshape(b.b,     MESH_NX, MESH_NY, MESH_NZ)
        midy = MESH_NY // 2  # mid-y reveals the z-layered pressure plateau

        ax = axes[0, i]
        # Plot deviation from baseline (101000 Pa) so the 1-7 Pa variation
        # (5 discrete plateaus, sit on z-layers) shows up.
        slc = x_grid[:, midy, :] - 1.01e5
        spread = slc.max() - slc.min()
        im = ax.imshow(slc, origin="lower", cmap="viridis", aspect="equal")
        ax.set_title(f"step{step}/corr{corr}\nx_ref - 101000 (mid-y, z↑)\n"
                     f"deviation [{slc.min():.2f}, {slc.max():.2f}] Pa",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, label="Pa")

        ax = axes[1, i]
        b_slc = b_grid[:, midy, :]
        b_max = max(abs(b_slc.min()), abs(b_slc.max()), 1e-30)
        im = ax.imshow(b_slc, origin="lower", cmap="RdBu_r",
                       vmin=-b_max, vmax=b_max, aspect="equal")
        ax.set_title(f"b (RHS, mid-y)\nrange [{b_slc.min():.2e}, {b_slc.max():.2e}]",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"Senior's pd PISO evolution (Initial_Period, mid-z slice of 80×80×80 mesh)\n"
        f"x_ref values stay essentially uniform 1.01e5 Pa across all 11 steps "
        f"→ no flow yet, no melt pool, pure cold-start",
        fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


# ---------------------------------------------------------------------------
# Our LPBF VTK case viz (T, alpha)
# ---------------------------------------------------------------------------

def list_time_dirs_with_T(case_dir: Path) -> list[tuple[float, Path]]:
    out = []
    for d in case_dir.iterdir():
        if not d.is_dir(): continue
        try: t = float(d.name)
        except ValueError: continue
        if (d / "T").exists(): out.append((t, d))
    return sorted(out)


def infer_grid_shape(case_dir: Path) -> tuple[int, int, int] | None:
    """Try to read blockMeshDict to extract mesh dimensions."""
    bm = case_dir / "system" / "blockMeshDict"
    if not bm.exists(): return None
    text = bm.read_text()
    # Find pattern like "(NX NY NZ)" inside hex...simpleGrading line
    import re
    m = re.search(r"hex\s*\([^)]+\)\s*\((\d+)\s+(\d+)\s+(\d+)\)", text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def plot_our_lpbf_evolution(case_dir: Path, out_png: Path):
    """Plot T and alpha.metal mid-y slices across time."""
    times_dirs = list_time_dirs_with_T(case_dir)
    if not times_dirs:
        print(f"  (no time dirs with T in {case_dir})")
        return
    grid = infer_grid_shape(case_dir)
    if grid is None:
        print(f"  could not infer grid shape from {case_dir}/system/blockMeshDict")
        return

    NX, NY, NZ = grid
    print(f"  grid: {NX}×{NY}×{NZ} = {NX*NY*NZ}")

    # pick up to 6 timesteps log-uniformly
    pick = []
    if len(times_dirs) <= 6:
        pick = times_dirs
    else:
        idx = np.linspace(0, len(times_dirs) - 1, 6, dtype=int)
        pick = [times_dirs[i] for i in idx]

    n = len(pick)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, (t, d) in enumerate(pick):
        T = parse_internal_field(d / "T")
        alpha = parse_internal_field(d / "alpha.metal")
        if T is None or alpha is None or T.size != NX*NY*NZ:
            for ax in axes[:, i]: ax.set_axis_off()
            continue
        T_grid = T.reshape((NZ, NY, NX))
        a_grid = alpha.reshape((NZ, NY, NX))
        midy = NY // 2

        ax = axes[0, i]
        im = ax.imshow(T_grid[:, midy, :], origin="lower", cmap="hot",
                       vmin=298, vmax=max(T.max(), 300), aspect="equal")
        ax.set_title(f"t={t:.2e}s\nT (mid-y)\nmax={T.max():.1f} K",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, i]
        im = ax.imshow(a_grid[:, midy, :], origin="lower", cmap="Greys",
                       vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"alpha.metal (mid-y)\nmin={alpha.min():.3f}",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"Our case {case_dir.name}: T + alpha.metal (mid-y slice, {NX}×{NY}×{NZ} mesh)\n"
        f"endTime={pick[-1][0]:.2e}s → if T flat at 298K, no melt onset yet",
        fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    out_dir = Path("/home/yzk/DILU-Research/dilu/amgx/bench")

    print("Plotting senior's pd PISO evolution ...")
    plot_senior_pd_evolution(out_dir / "field_viz_senior_pd.png")

    print("\nPlotting our LPBF_crosscheck T + alpha evolution ...")
    plot_our_lpbf_evolution(
        Path("/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_crosscheck"),
        out_dir / "field_viz_ours_lpbf_crosscheck.png")

    print("\nPlotting our dumper_pipeline_test T + alpha evolution ...")
    plot_our_lpbf_evolution(
        Path("/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/dumper_pipeline_test"),
        out_dir / "field_viz_ours_pipeline.png")
