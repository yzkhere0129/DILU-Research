"""Eulerian donor-region PLIC flux — Stage 2.

Computes the volume of "fluid 1" (F = 1 phase) that crosses each cell
face per time step, using the Scardovelli-Zaleski analytic cube/plane
intersection on the axis-aligned *sweep box* (the portion of the donor
cell that is swept across the face in time :math:`\\Delta t`).

This is the Eulerian analogue of the Lagrangian hex/box clipping used
in ``jax_laseram.vof.lagrangian_3d``, but *much* simpler: because the
advection frame is axis-aligned, the sweep box is a rectangular
sub-region of the donor cell, and its volume-below-plane is directly
``volume_below_plane_3d`` with rescaled coefficients — no polygon
clipping, no vmap + scan explosion.

API
---
All routines take the halo-padded VOF and its PLIC reconstruction
``(F, nx, ny, nz, C)`` with shape ``(Nx, Ny, Nz)`` and return face
fluxes with shape reduced by one along the sweep axis::

    flux_x[i, j, k] = signed volume of F across face (i+1/2, j, k),
                      positive = rightward (``+x``) direction.

The scalar velocity arguments ``u``, ``v``, ``w`` may be signed; the
routines handle upwind donor selection internally via branchless
``jnp.where``. Cell-centered storage is assumed: a positive ``u``
advects fluid in the cell at index ``i`` across the face ``i+1/2``
into cell ``i+1``.

Phase-A variant
---------------
Stage 2 of the research plan ships a *scalar-velocity* variant that
takes a constant ``u``, ``v``, ``w``. This is enough for the
``droplet_45deg`` benchmark (``u = v = 1``) and keeps the code concise
while we sanity-check the end-to-end pipeline. A face-velocity variant
for variable flow fields (Stage 4, NS coupling) will arrive later and
reuse the same ``_sub_box_flux_kernel`` helper.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .volume_formula import volume_below_plane_3d
# Re-export the conservative bound-preserving variants (drop-in
# replacements for ``jnp.clip(F, 0, 1)`` after ``apply_flux_*``) from
# ``conservative_bounds`` — single source of truth.  The signature
# matches the test-suite API contract:
#   apply_flux_x_conservative(F, flux, dt, dx, u_face, n_iter=3)
# where ``flux``, ``dt``, ``dx`` are accepted for forward-compatibility
# and ``u_face`` (scalar or face array) is used to pick the downstream
# neighbour during redistribution.
from .conservative_bounds import (  # noqa: F401  (re-export)
    redistribute_bounds_x,
    redistribute_bounds_y,
    redistribute_bounds_z,
    apply_flux_x_conservative,
    apply_flux_y_conservative,
    apply_flux_z_conservative,
)

__all__ = [
    "sweep_flux_x",
    "sweep_flux_y",
    "sweep_flux_z",
    "apply_flux_x",
    "apply_flux_y",
    "apply_flux_z",
    "apply_flux_x_conservative",
    "apply_flux_y_conservative",
    "apply_flux_z_conservative",
]

# Interface / pure-cell threshold — matches the Stage 1 convention in
# ``normal_youngs`` and ``intercept_solver``.
_INTERFACE_EPS = 1.0e-6


def _sub_box_flux_kernel(
    F_donor: jnp.ndarray,
    n_axis: jnp.ndarray,
    n_other1: jnp.ndarray,
    n_other2: jnp.ndarray,
    C_donor: jnp.ndarray,
    sub_box_center_axis: jnp.ndarray,
    abs_sub_box_axis: jnp.ndarray,
    dx_other1: float,
    dx_other2: float,
) -> jnp.ndarray:
    """Return the *fraction* of the sub-box that lies below the PLIC plane.

    The sub-box is aligned with the cell axes and occupies:

    - along the sweep axis: a band of width ``abs_sub_box_axis``
      centred at ``sub_box_center_axis`` (in donor-cell-local coords)
    - along the two transverse axes: the full cell width

    ``n_axis``, ``n_other1``, ``n_other2`` are the components of the
    interface normal in the donor cell along the sweep axis and the two
    transverse axes respectively. ``C_donor`` is the plane intercept in
    physical cell-centered coordinates (the output of
    ``solve_intercept``). Pure donor cells (``F_donor`` near 0 or 1) are
    replaced by their analytical upwind value after the PLIC fraction
    is computed.
    """
    # Plane coefficients on the sub-box (Scardovelli-Zaleski convention):
    a_sub = n_axis * abs_sub_box_axis          # scaled to sub-box width
    b_sub = n_other1 * dx_other1
    c_sub = n_other2 * dx_other2
    # Plane intercept relative to the *sub-box* centre.
    C_sub = C_donor - n_axis * sub_box_center_axis

    V_frac_plic = volume_below_plane_3d(C_sub, a_sub, b_sub, c_sub)

    is_interface = (F_donor > _INTERFACE_EPS) & (F_donor < 1.0 - _INTERFACE_EPS)
    # Upwind fallback: pure cells donate exactly ``F_donor`` of the
    # sub-box. This trivially handles F=0 (no flux) and F=1 (full
    # sub-box flows out).
    V_frac = jnp.where(is_interface, V_frac_plic, F_donor)
    return V_frac


# ---------------------------------------------------------------------------
# x sweep
# ---------------------------------------------------------------------------


def sweep_flux_x(
    F: jnp.ndarray,
    nx: jnp.ndarray,
    ny: jnp.ndarray,
    nz: jnp.ndarray,
    C: jnp.ndarray,
    u: float,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
) -> jnp.ndarray:
    """Flux of F across x-faces for a scalar x-velocity ``u``.

    Parameters
    ----------
    F, nx, ny, nz, C : jnp.ndarray, shape ``(Nx, Ny, Nz)``
        Halo-padded fields. Normals should be unit vectors on interface
        cells and zero on pure cells; intercepts are from
        ``solve_intercept`` (cell-centered convention).
    u : float
        Scalar x-velocity (signed). Typical CFL for Stage 2 is
        ``|u| * dt / dx <= 0.5`` but the formula tolerates ``<= 1``.
    dt, dx, dy, dz : float
        Time step and cell sizes.

    Returns
    -------
    flux : jnp.ndarray, shape ``(Nx-1, Ny, Nz)``
        Signed volume of F crossing each x-face rightward in time dt.
    """
    abs_udt = jnp.abs(u) * dt
    sign_u = jnp.sign(u)

    F_left = F[:-1]
    F_right = F[1:]
    nx_left = nx[:-1]
    nx_right = nx[1:]
    ny_left = ny[:-1]
    ny_right = ny[1:]
    nz_left = nz[:-1]
    nz_right = nz[1:]
    C_left = C[:-1]
    C_right = C[1:]

    # Donor (upwind) cell depends on the sign of u.
    F_donor = jnp.where(u >= 0, F_left, F_right)
    nx_d = jnp.where(u >= 0, nx_left, nx_right)
    ny_d = jnp.where(u >= 0, ny_left, ny_right)
    nz_d = jnp.where(u >= 0, nz_left, nz_right)
    C_d = jnp.where(u >= 0, C_left, C_right)

    # Sub-box center along x in donor-local coordinates:
    #   u >= 0: right edge of donor,  center_x = +dx/2 - u*dt/2
    #   u  < 0: left  edge of donor,  center_x = -dx/2 - u*dt/2
    # Both cases collapse to  sign(u)*dx/2 - u*dt/2.
    center_x = sign_u * (dx / 2.0) - (u * dt) / 2.0

    V_frac = _sub_box_flux_kernel(
        F_donor=F_donor,
        n_axis=nx_d,
        n_other1=ny_d,
        n_other2=nz_d,
        C_donor=C_d,
        sub_box_center_axis=center_x,
        abs_sub_box_axis=abs_udt,
        dx_other1=dy,
        dx_other2=dz,
    )
    flux_mag = V_frac * abs_udt * dy * dz
    return sign_u * flux_mag  # signed rightward volume


def apply_flux_x(
    F: jnp.ndarray, flux_x: jnp.ndarray, dx: float, dy: float, dz: float
) -> jnp.ndarray:
    """Apply x-face fluxes conservatively to interior cells.

    ``flux_x[i, j, k]`` is the signed volume crossing face ``i+1/2``,
    so cell ``i`` loses ``flux_x[i]`` and cell ``i+1`` gains it.
    Boundary cells (indices 0 and -1) are left untouched — the caller
    is expected to refresh them via a halo update after each sweep.
    """
    cell_vol = dx * dy * dz
    # dF[i] = (flux_in_left - flux_out_right) / cell_vol
    #       = (flux_x[i-1] - flux_x[i]) / cell_vol    for interior i
    dF_interior = (flux_x[:-1] - flux_x[1:]) / cell_vol  # (Nx-2, Ny, Nz)
    return F.at[1:-1].add(dF_interior)


# ---------------------------------------------------------------------------
# y sweep
# ---------------------------------------------------------------------------


def sweep_flux_y(
    F: jnp.ndarray,
    nx: jnp.ndarray,
    ny: jnp.ndarray,
    nz: jnp.ndarray,
    C: jnp.ndarray,
    v: float,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
) -> jnp.ndarray:
    """Flux of F across y-faces, mirror of :func:`sweep_flux_x`."""
    abs_vdt = jnp.abs(v) * dt
    sign_v = jnp.sign(v)

    F_lo = F[:, :-1]
    F_hi = F[:, 1:]
    nx_lo = nx[:, :-1]
    nx_hi = nx[:, 1:]
    ny_lo = ny[:, :-1]
    ny_hi = ny[:, 1:]
    nz_lo = nz[:, :-1]
    nz_hi = nz[:, 1:]
    C_lo = C[:, :-1]
    C_hi = C[:, 1:]

    F_donor = jnp.where(v >= 0, F_lo, F_hi)
    nx_d = jnp.where(v >= 0, nx_lo, nx_hi)
    ny_d = jnp.where(v >= 0, ny_lo, ny_hi)
    nz_d = jnp.where(v >= 0, nz_lo, nz_hi)
    C_d = jnp.where(v >= 0, C_lo, C_hi)

    center_y = sign_v * (dy / 2.0) - (v * dt) / 2.0

    V_frac = _sub_box_flux_kernel(
        F_donor=F_donor,
        n_axis=ny_d,
        n_other1=nx_d,
        n_other2=nz_d,
        C_donor=C_d,
        sub_box_center_axis=center_y,
        abs_sub_box_axis=abs_vdt,
        dx_other1=dx,
        dx_other2=dz,
    )
    flux_mag = V_frac * abs_vdt * dx * dz
    return sign_v * flux_mag


def apply_flux_y(
    F: jnp.ndarray, flux_y: jnp.ndarray, dx: float, dy: float, dz: float
) -> jnp.ndarray:
    """Apply y-face fluxes conservatively (mirror of apply_flux_x)."""
    cell_vol = dx * dy * dz
    dF_interior = (flux_y[:, :-1] - flux_y[:, 1:]) / cell_vol
    return F.at[:, 1:-1].add(dF_interior)


# ---------------------------------------------------------------------------
# z sweep
# ---------------------------------------------------------------------------


def sweep_flux_z(
    F: jnp.ndarray,
    nx: jnp.ndarray,
    ny: jnp.ndarray,
    nz: jnp.ndarray,
    C: jnp.ndarray,
    w: float,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
) -> jnp.ndarray:
    """Flux of F across z-faces, mirror of :func:`sweep_flux_x`."""
    abs_wdt = jnp.abs(w) * dt
    sign_w = jnp.sign(w)

    F_lo = F[:, :, :-1]
    F_hi = F[:, :, 1:]
    nx_lo = nx[:, :, :-1]
    nx_hi = nx[:, :, 1:]
    ny_lo = ny[:, :, :-1]
    ny_hi = ny[:, :, 1:]
    nz_lo = nz[:, :, :-1]
    nz_hi = nz[:, :, 1:]
    C_lo = C[:, :, :-1]
    C_hi = C[:, :, 1:]

    F_donor = jnp.where(w >= 0, F_lo, F_hi)
    nx_d = jnp.where(w >= 0, nx_lo, nx_hi)
    ny_d = jnp.where(w >= 0, ny_lo, ny_hi)
    nz_d = jnp.where(w >= 0, nz_lo, nz_hi)
    C_d = jnp.where(w >= 0, C_lo, C_hi)

    center_z = sign_w * (dz / 2.0) - (w * dt) / 2.0

    V_frac = _sub_box_flux_kernel(
        F_donor=F_donor,
        n_axis=nz_d,
        n_other1=nx_d,
        n_other2=ny_d,
        C_donor=C_d,
        sub_box_center_axis=center_z,
        abs_sub_box_axis=abs_wdt,
        dx_other1=dx,
        dx_other2=dy,
    )
    flux_mag = V_frac * abs_wdt * dx * dy
    return sign_w * flux_mag


def apply_flux_z(
    F: jnp.ndarray, flux_z: jnp.ndarray, dx: float, dy: float, dz: float
) -> jnp.ndarray:
    """Apply z-face fluxes conservatively (mirror of apply_flux_x)."""
    cell_vol = dx * dy * dz
    dF_interior = (flux_z[:, :, :-1] - flux_z[:, :, 1:]) / cell_vol
    return F.at[:, :, 1:-1].add(dF_interior)


# ---------------------------------------------------------------------------
# Conservative bound-preserving variants (Weymouth-Zaleski redistribute)
# ---------------------------------------------------------------------------
# See ``conservative_bounds.py``.  ``apply_flux_{x,y,z}_conservative`` are
# re-exported above — they are *pure redistribute* (i.e. the caller should
# have already called ``apply_flux_{x,y,z}``; the wrapper only enforces
# ``F in [0, 1]`` conservatively).
