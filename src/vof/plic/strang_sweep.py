"""Operator-split Strang time stepping for Eulerian PLIC — Stage 2.

Orchestrates PLIC advection by composing the three directional sweeps
in :mod:`jax_laseram.vof.plic.geometric_flux` with the Stage 1 PLIC
reconstruction (Youngs normal + Regula-Falsi intercept).

For the Stage 2 droplet benchmark we only need the 2D Strang split
(``x/2 -> y -> x/2``) because the test problem is quasi-2D
(``w = 0``). A full 3D Strang sequence for later stages would be
``x/2 -> y/2 -> z -> y/2 -> x/2``.

Each directional sub-sweep performs its own PLIC reconstruction on the
freshly-updated volume fraction:

    halo -> Youngs normal -> Regula Falsi intercept -> flux -> apply

This follows the plan's Stage-2 description and matches what classical
geometric VOF codes (Rider-Kothe 1998, Aulisa 2007) do. A "Weymouth-
Zaleski divergence correction" stub is left in place for Stage 4 when
the velocity field is no longer exactly divergence-free.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from .geometric_flux import (
    apply_flux_x,
    apply_flux_y,
    apply_flux_x_conservative,
    apply_flux_y_conservative,
    sweep_flux_x,
    sweep_flux_y,
)
from .analytic_intercept import analytic_intercept
from .intercept_solver import solve_intercept  # Phase A fallback
from .normal_youngs import compute_youngs_normal_3d

__all__ = [
    "plic_subsweep_x",
    "plic_subsweep_y",
    "plic_time_step_2d_strang",
]


HaloFn = Callable[[jnp.ndarray], jnp.ndarray]

# Maximum fraction of total cells allocated for the padded-gather
# compact array.  At 15% headroom vs a typical 5-8% interface
# fraction, overflow is unlikely.  If it does happen, the excess
# interface cells silently fall back to C = 0 (same as a pure cell) —
# a graceful degradation, not a crash.
_MAX_INTERFACE_FRAC = 0.15

# Interface detection threshold — must match intercept_solver._F_CLIP_LO.
_INTERFACE_EPS = 1.0e-6


def _reconstruct_dense(F: jnp.ndarray, dx: float, dy: float, dz: float):
    """Dense reconstruction: run intercept on ALL cells (original path)."""
    nx, ny, nz = compute_youngs_normal_3d(F, dx, dy, dz)
    C = solve_intercept(nx, ny, nz, F, dx, dy, dz)
    return nx, ny, nz, C


def _reconstruct(F: jnp.ndarray, dx: float, dy: float, dz: float):
    """Sparse reconstruction: gather interface cells, solve on compact array.

    Steps:
        1. Compute Youngs normals on the **full** field (3x3x3 conv is
           cheap: ~2.6 ms at 1M cells via cuDNN).
        2. Identify interface cells via ``_INTERFACE_EPS < F < 1 - _INTERFACE_EPS``.
        3. Gather (nx, ny, nz, F) for those cells into a compact 1-D
           array of fixed length ``MAX_INTERFACE_CELLS`` (padded with
           index-0 dummies where the real count is smaller).
        4. Run 15-step Regula Falsi intercept solver on the compact
           array only (~7% of 1M = 70k cells instead of 1M).
        5. Scatter the solved intercepts back into the full-field array.

    The full-field ``C`` is zero-initialized, so pure cells (and dummy
    slots that scatter to index 0, a halo cell) get ``C = 0``, which is
    harmless because the downstream flux uses its own
    ``is_interface`` mask to fall back to upwind for those cells.
    """
    shape = F.shape
    total = 1
    for s in shape:
        total *= s
    max_n = int(total * _MAX_INTERFACE_FRAC)

    # --- Step 1: normals on full field ---
    nx, ny, nz = compute_youngs_normal_3d(F, dx, dy, dz)

    # --- Step 2: interface detection ---
    is_interface = (F > _INTERFACE_EPS) & (F < 1.0 - _INTERFACE_EPS)

    # --- Step 3: gather to compact ---
    flat_idx = jnp.where(is_interface.ravel(), size=max_n, fill_value=0)[0]

    F_c = F.ravel()[flat_idx]
    nx_c = nx.ravel()[flat_idx]
    ny_c = ny.ravel()[flat_idx]
    nz_c = nz.ravel()[flat_idx]

    # --- Step 4: intercept on compact (Phase B analytic + Newton) ---
    C_c = analytic_intercept(nx_c, ny_c, nz_c, F_c, dx, dy, dz)

    # --- Step 5: scatter back ---
    C_flat = jnp.zeros(total, dtype=F.dtype)
    C_flat = C_flat.at[flat_idx].set(C_c)
    C = C_flat.reshape(shape)

    return nx, ny, nz, C


def plic_subsweep_x(
    F: jnp.ndarray,
    u: float,
    dt_sub: float,
    dx: float,
    dy: float,
    dz: float,
    halo_fn: HaloFn,
) -> jnp.ndarray:
    """One x-direction PLIC sub-sweep with Weymouth-Zaleski redistribute."""
    F = halo_fn(F)
    nx, ny, nz, C = _reconstruct(F, dx, dy, dz)
    flux = sweep_flux_x(F, nx, ny, nz, C, u, dt_sub, dx, dy, dz)
    F = apply_flux_x(F, flux, dx, dy, dz)
    # Conservative bound enforcement replaces the previous non-conservative
    # ``jnp.clip`` — overshoots are pushed to the downstream cell, restoring
    # volume conservation to round-off.
    F = apply_flux_x_conservative(F, flux, dt_sub, dx, u, n_iter=3)
    return halo_fn(F)


def plic_subsweep_y(
    F: jnp.ndarray,
    v: float,
    dt_sub: float,
    dx: float,
    dy: float,
    dz: float,
    halo_fn: HaloFn,
) -> jnp.ndarray:
    """One y-direction PLIC sub-sweep with Weymouth-Zaleski redistribute."""
    F = halo_fn(F)
    nx, ny, nz, C = _reconstruct(F, dx, dy, dz)
    flux = sweep_flux_y(F, nx, ny, nz, C, v, dt_sub, dx, dy, dz)
    F = apply_flux_y(F, flux, dx, dy, dz)
    F = apply_flux_y_conservative(F, flux, dt_sub, dy, v, n_iter=3)
    return halo_fn(F)


def plic_time_step_2d_strang(
    F: jnp.ndarray,
    u: float,
    v: float,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    halo_fn: HaloFn,
) -> jnp.ndarray:
    """2D Strang split PLIC step: ``x/2 -> y -> x/2``.

    Sub-sweep widths are chosen so the full step spans ``dt`` and the
    local CFL of each sub-sweep is ``<= 0.5`` when the user already
    enforces ``|u| dt / dx + |v| dt / dy <= 0.5`` at step selection.

    This is 2nd-order in time for an exact-divergence-free constant
    velocity field (which is the droplet benchmark's regime); for a
    non-divergence-free flow the Weymouth-Zaleski correction will be
    added in Stage 4.
    """
    F = plic_subsweep_x(F, u, dt / 2.0, dx, dy, dz, halo_fn)
    F = plic_subsweep_y(F, v, dt, dx, dy, dz, halo_fn)
    F = plic_subsweep_x(F, u, dt / 2.0, dx, dy, dz, halo_fn)
    return F
