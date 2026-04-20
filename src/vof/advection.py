"""Operator-split VOF advection — STRICT CONSERVATION, NO CLIP.

F is the conservative transport field. It may have O(10⁻⁶) overshoots.
Physics modules use F_prop = clip(F, 0, 1) — the "double-field" strategy.

Strang splitting: x/2 → y → x/2.
"""

import jax.numpy as jnp
from ..data_types import GridInfo, Array
from .thinc import compute_flux_volume, compute_youngs_gradient


def _apply_vof_bcs(F: Array, nh: int) -> Array:
    F = F.at[:nh, :, :].set(F[nh:nh+1, :, :])
    F = F.at[-nh:, :, :].set(F[-nh-1:-nh, :, :])
    F = F.at[:, :nh, :].set(F[:, nh:nh+1, :])
    F = F.at[:, -nh:, :].set(F[:, -nh-1:-nh, :])
    return F


def _sweep_x(F, u_face, grid, dt, beta, dFdx, dFdy):
    nh = grid.nh; nx, ny = grid.nx, grid.ny
    F = _apply_vof_bcs(F, nh)
    dFdx, dFdy = compute_youngs_gradient(F, grid.dx, grid.dy)
    flux_vol = compute_flux_volume(
        F[:-1,:,:], F[1:,:,:], u_face, dt, grid.dx,
        beta, axis=0, dFdx_full=dFdx, dFdy_full=dFdy, dy=grid.dy)
    flux_vol = flux_vol.at[nh-1,:,:].set(0.0)
    flux_vol = flux_vol.at[nh+nx-1,:,:].set(0.0)
    flux_R = flux_vol[nh:nh+nx, nh:nh+ny, :]
    flux_L = flux_vol[nh-1:nh+nx-1, nh:nh+ny, :]
    # STRICT conservation: NO clip
    F = F.at[nh:nh+nx, nh:nh+ny, :].set(F[nh:nh+nx, nh:nh+ny, :] - (flux_R - flux_L))
    return F


def _sweep_y(F, v_face, grid, dt, beta, dFdx, dFdy):
    nh = grid.nh; nx, ny = grid.nx, grid.ny
    F = _apply_vof_bcs(F, nh)
    dFdx, dFdy = compute_youngs_gradient(F, grid.dx, grid.dy)
    flux_vol = compute_flux_volume(
        F[:,:-1,:], F[:,1:,:], v_face, dt, grid.dy,
        beta, axis=1, dFdx_full=dFdx, dFdy_full=dFdy, dy=grid.dx)
    flux_vol = flux_vol.at[:,nh-1,:].set(0.0)
    flux_vol = flux_vol.at[:,nh+ny-1,:].set(0.0)
    flux_T = flux_vol[nh:nh+nx, nh:nh+ny, :]
    flux_B = flux_vol[nh:nh+nx, nh-1:nh+ny-1, :]
    F = F.at[nh:nh+nx, nh:nh+ny, :].set(F[nh:nh+nx, nh:nh+ny, :] - (flux_T - flux_B))
    return F


def advect_vof_xy(F, u_face, v_face, grid, dt, beta=2.0):
    d = compute_youngs_gradient(F, grid.dx, grid.dy)
    F = _sweep_x(F, u_face, grid, dt*0.5, beta, *d)
    d = compute_youngs_gradient(F, grid.dx, grid.dy)
    F = _sweep_y(F, v_face, grid, dt, beta, *d)
    d = compute_youngs_gradient(F, grid.dx, grid.dy)
    F = _sweep_x(F, u_face, grid, dt*0.5, beta, *d)
    return _apply_vof_bcs(F, grid.nh)


def advect_vof_yx(F, u_face, v_face, grid, dt, beta=2.0):
    d = compute_youngs_gradient(F, grid.dx, grid.dy)
    F = _sweep_y(F, v_face, grid, dt*0.5, beta, *d)
    d = compute_youngs_gradient(F, grid.dx, grid.dy)
    F = _sweep_x(F, u_face, grid, dt, beta, *d)
    d = compute_youngs_gradient(F, grid.dx, grid.dy)
    F = _sweep_y(F, v_face, grid, dt*0.5, beta, *d)
    return _apply_vof_bcs(F, grid.nh)
