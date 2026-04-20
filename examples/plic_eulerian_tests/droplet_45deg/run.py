#!/usr/bin/env python3
"""Run the 45-degree droplet PLIC advection benchmark (Stage 2).

Stage 2 is now live: this script runs the 2-D Strang-split PLIC
advection loop on the droplet, saves snapshots at the configured
``save_dt`` intervals, and writes a numpy ``.npz`` per snapshot to
``results/``. The post-processing is handled by ``extract_metrics.py``
and ``plot_results.py`` in the same directory.

Pipeline per time step:

    halo update (symmetry) -> Youngs normal -> Regula Falsi intercept
    -> sweep flux (x/2) -> apply -> halo
    -> same three steps in y (full dt)
    -> same three steps in x (final x/2 to complete the Strang split)

At ``t = t_end`` (≈ 0.3536 s) the droplet's analytic center has
translated from (0.2, 0.2) to (0.5536, 0.5536) — 5 diameters along
the 45-degree flow direction. Expected gate metrics:

    |dV / V0|   < 0.5%   (volume conservation)
    F_min/max   within [-1e-6, 1 + 1e-6]
    center err  < 2 * dx
"""

from __future__ import annotations

import sys
import os
import time as timer

# --- Path setup ---------------------------------------------------------
# Allow "python run.py" from this directory to import both the sibling
# modules (case_config, init_droplet) AND the repo's in-tree jax_laseram
# package without requiring an editable install. The plic-research branch
# lives in its own checkout (``JAX-LaserAM-plic-research``) which may not
# match the system ``pip install -e``ed copy.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_SRC)

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from case_config import CASE
from init_droplet import build_grid, create_droplet_vof

# Stage 1 PLIC primitives
from jax_laseram.vof.plic.normal_youngs import compute_youngs_normal_3d
from jax_laseram.vof.plic.intercept_solver import solve_intercept
from jax_laseram.vof.plic.volume_formula import volume_below_plane_3d  # noqa: F401

# Stage 2 PLIC time stepping + diagnostics
from jax_laseram.vof.plic.strang_sweep import plic_time_step_2d_strang
from jax_laseram.vof.plic.diagnostics import compute_stats, format_stats

# Reuse the initial-condition helper's symmetry halo so the time loop
# uses exactly the same BCs that built the initial field.
from init_droplet import _apply_symmetry_bcs


def _interior_slices(grid):
    """Interior slicing for x, y, z (handles both nz=1 and nz>=3 layouts)."""
    nh = grid.nh
    sx = slice(nh, nh + grid.nx)
    sy = slice(nh, nh + grid.ny)
    if grid.nz > 1:
        sz = slice(nh, nh + grid.nz)
    else:
        sz = slice(0, 1)
    return sx, sy, sz


def stage1_smoke_test(F, grid):
    """Call Stage-1 primitives on the initial field and print stats.

    This is NOT a timestep. It only demonstrates that:
      (a) Youngs normals can be computed on the initial droplet;
      (b) the intercept solver converges for every interface cell;
      (c) shapes are compatible with this case's grid / halo layout.

    All outputs are plain Python floats / ints; no files are written.
    """
    print("-" * 60)
    print("Stage 1 PLIC smoke test")
    print("-" * 60, flush=True)

    dx, dy, dz = grid.dx, grid.dy, grid.dz
    ix, iy, iz = _interior_slices(grid)

    # --- 1) Volume / perimeter sanity ---
    # Sum over the full interior (all z layers) — for the quasi-2D
    # nz=3 layout each layer contains the same droplet cross-section.
    F_interior_full = np.array(F[ix, iy, iz])
    V0 = float(F_interior_full.sum() * dx * dy * dz)
    V0_analytic = float(CASE["analytic_V0"])
    print(
        f"  V0 (numeric)  = {V0:.10f} m^3",
        flush=True,
    )
    print(
        f"  V0 (analytic) = {V0_analytic:.10f} m^3  "
        f"(rel err = {(V0 - V0_analytic) / V0_analytic * 100:+.4f} %)",
        flush=True,
    )

    # --- 2) Youngs normal on the full (halo-included) field ---
    nx_n, ny_n, nz_n = compute_youngs_normal_3d(F, dx, dy, dz)

    # Interface mask: where the normal magnitude is non-trivial.
    mag = jnp.sqrt(nx_n * nx_n + ny_n * ny_n + nz_n * nz_n)
    is_interface = mag > 1.0e-6
    n_iface = int(is_interface.sum())
    print(f"  interface cells (|n|>1e-6) : {n_iface}", flush=True)

    if n_iface > 0:
        mean_mag = float(jnp.where(is_interface, mag, 0.0).sum() / n_iface)
        print(f"  mean |n| over interface    : {mean_mag:.6f}  (expect ~1.0)", flush=True)

        # For a circular droplet the normal should be radial — so |nx|
        # and |ny| should be roughly comparable (both ~0.7 on average).
        abs_nx = jnp.where(is_interface, jnp.abs(nx_n), 0.0)
        abs_ny = jnp.where(is_interface, jnp.abs(ny_n), 0.0)
        mean_abs_nx = float(abs_nx.sum() / n_iface)
        mean_abs_ny = float(abs_ny.sum() / n_iface)
        print(
            f"  mean |nx| = {mean_abs_nx:.4f}, mean |ny| = {mean_abs_ny:.4f}  "
            f"(circular droplet: expect ~0.6 for both)",
            flush=True,
        )

    # --- 3) Plane intercept solve on the full field ---
    C = solve_intercept(nx_n, ny_n, nz_n, F, dx, dy, dz)
    C_iface = jnp.where(is_interface, C, 0.0)
    C_min = float(C_iface.min())
    C_max = float(C_iface.max())
    C_half_max = 0.5 * (abs(dx) + abs(dy) + abs(dz))
    print(
        f"  intercept C range on interface : "
        f"[{C_min:+.6e}, {C_max:+.6e}]  "
        f"(|C| <= {C_half_max:.6e} expected)",
        flush=True,
    )

    print("Stage 1 smoke test OK.", flush=True)


def main():
    # --- Setup ---
    print("=" * 60)
    print(f"CASE: {CASE['name']}")
    print(f"  description: {CASE['description']}")
    print(f"  reference  : {CASE['reference']}")
    print("=" * 60, flush=True)

    grid = build_grid()
    print(
        f"Grid: {grid.nx} x {grid.ny} x {grid.nz}  "
        f"(nh={grid.nh}, dx={grid.dx:.4f}, dy={grid.dy:.4f}, dz={grid.dz:.4f})",
        flush=True,
    )

    t0 = timer.time()
    F = create_droplet_vof(grid)
    print(f"Initialized droplet VOF in {timer.time() - t0:.3f}s", flush=True)

    # --- Stage 1 smoke test (kept as a sanity check before the time loop) ---
    stage1_smoke_test(F, grid)

    # --- Stage 2 Strang-split time loop --------------------------------
    u = float(CASE["velocity"]["u"])
    v = float(CASE["velocity"]["v"])
    w = float(CASE["velocity"]["w"])
    assert w == 0.0, "quasi-2D droplet benchmark: w must be zero"

    dx, dy, dz = grid.dx, grid.dy, grid.dz
    cell_vol = dx * dy * dz

    cfl = float(CASE["time"]["cfl"])
    t_end = float(CASE["time"]["end"])
    save_dt = float(CASE["time"]["save_dt"])

    # CFL: max face speed / min dx. For u = v = 1, dx = 0.01 this gives
    # dt = 0.5 * 0.01 / 1 = 5e-3; 71 steps to reach 0.3536.
    max_vel = max(abs(u), abs(v))
    dt_cfl = cfl * min(dx, dy) / max_vel
    n_steps = int(np.ceil(t_end / dt_cfl))
    dt = t_end / n_steps  # exact fit
    # Save roughly every ``save_dt`` seconds, but at an integer step.
    save_every = max(1, int(round(save_dt / dt)))

    print("-" * 60)
    print("Stage 2 Strang-split advection loop")
    print("-" * 60, flush=True)
    print(
        f"  u, v, w       : {u}, {v}, {w}\n"
        f"  t_end, save_dt: {t_end}, {save_dt}\n"
        f"  n_steps       : {n_steps}  (dt = {dt:.6e}, CFL = {max_vel * dt / min(dx, dy):.3f})\n"
        f"  save_every    : {save_every} steps",
        flush=True,
    )

    # Halo update closure for this case: symmetry on all sides.
    has_z_halo = grid.nz > 1

    def halo_fn(F_arr):
        return _apply_symmetry_bcs(F_arr, grid.nh, has_z_halo)

    # JIT-compile the full Strang step so XLA can fuse all elementwise
    # ops into a handful of GPU kernels. Without this, each jnp.where /
    # jnp.abs / arithmetic op dispatches a separate kernel launch, and
    # the 15-iteration fori_loop in the intercept solver alone would
    # produce ~1800 individual launches whose overhead dominates runtime.
    # Measured speedup on RTX 3050: 55x (2232 ms -> 40 ms per step).
    @jax.jit
    def jitted_step(F_arr):
        return plic_time_step_2d_strang(
            F_arr, u, v, dt, dx, dy, dz, halo_fn
        )

    # Initial metrics + first snapshot.
    out_dir = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    stats0 = compute_stats(F, grid.nh, cell_vol)
    v0 = stats0.volume
    print(f"  step 0 t=0.000000 {format_stats(stats0, v_ref=v0)}", flush=True)
    _save_snapshot(out_dir, 0, 0.0, F, grid)

    # Warmup JIT compile (first call traces + compiles the XLA graph).
    print("  JIT compiling full Strang step...", end=" ", flush=True)
    t_jit0 = timer.time()
    F_warmup = jitted_step(F)
    jax.block_until_ready(F_warmup)
    print(f"done in {timer.time() - t_jit0:.2f}s", flush=True)

    t = 0.0
    t_wall0 = timer.time()
    next_save_step = save_every

    for step in range(1, n_steps + 1):
        F = jitted_step(F)
        t = step * dt

        if step == next_save_step or step == n_steps:
            stats = compute_stats(F, grid.nh, cell_vol)
            print(
                f"  step {step:3d} t={t:.6f} {format_stats(stats, v_ref=v0)}  "
                f"wall={timer.time() - t_wall0:.1f}s",
                flush=True,
            )
            _save_snapshot(out_dir, step, t, F, grid)
            next_save_step += save_every

    wall = timer.time() - t_wall0
    print(f"\nDone. {n_steps} steps in {wall:.1f}s ({wall / n_steps * 1000:.1f} ms/step)", flush=True)

    # --- Post-run summary --------------------------------------------------
    stats_end = compute_stats(F, grid.nh, cell_vol)
    vol_drift = (stats_end.volume - v0) / v0
    print(
        f"\nGate metrics:\n"
        f"  Volume drift |dV/V0|          : {abs(vol_drift):.3e}  (gate < 5.0e-3)\n"
        f"  Final F range                 : [{stats_end.f_min:+.3e}, {stats_end.f_max:+.3e}]\n"
        f"  Out-of-bound cells (below/above): {stats_end.n_below} / {stats_end.n_above}",
        flush=True,
    )
    return 0


def _save_snapshot(out_dir: str, step: int, t: float, F, grid) -> None:
    """Persist a snapshot as an .npz file for post-processing.

    Strips ALL halos (including z halos on quasi-2D layouts) so the
    saved array has shape ``(nx, ny, nz)`` and downstream scripts can
    treat it as the physical interior field without any halo
    bookkeeping.
    """
    fname = os.path.join(out_dir, f"snapshot_{step:04d}.npz")
    nh = grid.nh
    has_z_halo = grid.nz > 1
    z_slice = slice(nh, nh + grid.nz) if has_z_halo else slice(0, 1)
    F_interior = np.asarray(
        F[nh : nh + grid.nx, nh : nh + grid.ny, z_slice]
    )
    np.savez_compressed(
        fname,
        step=int(step),
        time=float(t),
        volume_fraction=F_interior.astype(np.float32),
        dx=float(grid.dx),
        dy=float(grid.dy),
        dz=float(grid.dz),
        nx=int(grid.nx),
        ny=int(grid.ny),
        nz=int(grid.nz),
    )


if __name__ == "__main__":
    sys.exit(main())
