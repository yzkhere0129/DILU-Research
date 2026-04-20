#!/usr/bin/env python3
"""Run the Rayleigh-Taylor PLIC benchmark skeleton.

Status
------
This script exercises the Stage-1 PLIC primitives on the sinusoidal RT
initial interface and early-returns. Running the real RT evolution
requires the incompressible Navier-Stokes coupling planned for Stage 4
(pressure projection + gravity + density-weighted momentum). See
``/home/yzk/.claude/plans/vast-napping-scott.md``.

What this script DOES
---------------------
1. Builds the grid (128 x 512 x 1) and the sub-grid-sampled alpha field.
2. Computes Youngs normals on the initial interface and verifies the
   ``|n_y|`` component dominates (interface is nearly horizontal).
3. Computes the intercept C for every interface cell.
4. Prints diagnostic statistics and exits.
"""

from __future__ import annotations

import os
import sys
import time as timer

# --- Path setup ---------------------------------------------------------
# See droplet_45deg/run.py for rationale.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_SRC)

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from case_config import CASE
from init_alpha import build_grid, create_rt_vof

# Stage 1 PLIC primitives — these are the only imports allowed.
from jax_laseram.vof.plic.normal_youngs import compute_youngs_normal_3d
from jax_laseram.vof.plic.intercept_solver import solve_intercept
from jax_laseram.vof.plic.volume_formula import volume_below_plane_3d  # noqa: F401

# TODO(Stage 4): uncomment once the NS + PLIC coupling exists.
# from jax_laseram.vof.plic.geometric_flux import sweep_flux_x, sweep_flux_y
# from jax_laseram.vof.plic.strang_sweep import strang_split_step
# from jax_laseram.pressure.projection import project_incompressible


def stage1_smoke_test(F, grid):
    """Stage-1 PLIC primitives on the initial RT interface."""
    print("-" * 60)
    print("Stage 1 PLIC smoke test")
    print("-" * 60, flush=True)

    dx, dy, dz = grid.dx, grid.dy, grid.dz
    nh = grid.nh

    # --- Mass / volume sanity (sum over all interior z layers) ---
    iz = slice(nh, nh + grid.nz) if grid.nz > 1 else slice(0, 1)
    F_interior = F[nh : nh + grid.nx, nh : nh + grid.ny, iz]
    V_total = float(F_interior.sum() * dx * dy * dz)
    dV = dx * dy * dz
    rho_heavy = CASE["rt"]["rho_heavy"]
    rho_light = CASE["rt"]["rho_light"]
    mass = float(
        (F_interior * rho_heavy + (1.0 - F_interior) * rho_light).sum() * dV
    )
    print(f"  total alpha volume : {V_total:.6f} m^3", flush=True)
    print(f"  initial mass       : {mass:.6f} kg", flush=True)

    # --- Youngs normal ---
    nx_n, ny_n, nz_n = compute_youngs_normal_3d(F, dx, dy, dz)
    mag = jnp.sqrt(nx_n * nx_n + ny_n * ny_n + nz_n * nz_n)
    is_interface = mag > 1.0e-6
    n_iface = int(is_interface.sum())
    print(f"  interface cells (|n|>1e-6) : {n_iface}", flush=True)

    if n_iface > 0:
        abs_nx = jnp.where(is_interface, jnp.abs(nx_n), 0.0)
        abs_ny = jnp.where(is_interface, jnp.abs(ny_n), 0.0)
        mean_abs_nx = float(abs_nx.sum() / n_iface)
        mean_abs_ny = float(abs_ny.sum() / n_iface)
        max_abs_ny = float(abs_ny.max())
        print(
            f"  mean |nx| = {mean_abs_nx:.4f}, mean |ny| = {mean_abs_ny:.4f}  "
            f"(RT: expect |ny| >> |nx|)",
            flush=True,
        )
        print(f"  max |ny|  = {max_abs_ny:.4f}  (expect ~1.0)", flush=True)
        assert mean_abs_ny > mean_abs_nx, (
            "RT interface should be nearly horizontal; "
            "|ny| must exceed |nx| on average."
        )

    # --- Intercept solver ---
    C = solve_intercept(nx_n, ny_n, nz_n, F, dx, dy, dz)
    C_iface = jnp.where(is_interface, C, 0.0)
    C_half_max = 0.5 * (abs(dx) + abs(dy) + abs(dz))
    print(
        f"  intercept C range   : "
        f"[{float(C_iface.min()):+.6e}, {float(C_iface.max()):+.6e}]  "
        f"(|C| <= {C_half_max:.6e} expected)",
        flush=True,
    )
    print("Stage 1 smoke test OK.", flush=True)


def main():
    print("=" * 60)
    print(f"CASE: {CASE['name']}")
    print(f"  description: {CASE['description']}")
    print(f"  reference  : {CASE['reference']}")
    print("=" * 60, flush=True)

    grid = build_grid()
    print(
        f"Grid: {grid.nx} x {grid.ny} x {grid.nz}  "
        f"(nh={grid.nh}, dx={grid.dx:.6f}, dy={grid.dy:.6f}, dz={grid.dz:.6f})",
        flush=True,
    )

    t0 = timer.time()
    F = create_rt_vof(grid)
    print(f"Initialized RT alpha in {timer.time() - t0:.3f}s", flush=True)
    _ = np  # keep numpy import alive for optional downstream use

    stage1_smoke_test(F, grid)

    # TODO(Stage 4): full RT evolution requires NS coupling.
    #   - density rho(F) = rho_heavy*F + rho_light*(1-F)
    #   - momentum advection + viscous stress + gravity
    #   - incompressible pressure projection
    #   - PLIC advection of F using the projected velocity
    #
    # Stage 2 alone cannot drive RT because PLIC advection needs a
    # divergence-free velocity field that responds to buoyancy.
    print("", flush=True)
    print("RT requires Stage 4 NS coupling "
          "(projection + gravity) - no time loop run.", flush=True)
    print(
        "  Required modules: jax_laseram.vof.plic.{geometric_flux, strang_sweep}, "
        "jax_laseram.pressure.projection",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
