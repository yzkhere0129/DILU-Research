"""Initial volume-fraction field for the 45-degree droplet benchmark.

Provides a single function ``create_droplet_vof(grid)`` that returns a
3-D VOF array with halo cells, shape ``(Nx+2*nh, Ny+2*nh, 1)``.

The core ``F_interior`` formula (tanh smoothing with width 1.5*dx) is
copied VERBATIM from
``examples/droplet_advection_45deg/run_vof_only.py`` lines 27-54
(``create_droplet`` function). Boundary filling is a plain
zero-gradient copy (symmetry) rather than the reference's
``_apply_vof_bcs`` call — the two produce identical interior arrays
because the droplet is located at (0.2, 0.2) and never touches any
domain boundary at t=0.
"""

from __future__ import annotations

import jax.numpy as jnp

from jax_laseram.grid import create_grid

from case_config import CASE


def build_grid():
    """Build the GridInfo for this case from CASE dict."""
    g = CASE["grid"]
    d = CASE["domain"]
    return create_grid(
        nx=g["nx"],
        ny=g["ny"],
        nz=g["nz"],
        x_range=d["x"],
        y_range=d["y"],
        z_range=d["z"],
        nh=g["nh"],
    )


def _apply_symmetry_bcs(F: jnp.ndarray, nh: int, has_z_halo: bool) -> jnp.ndarray:
    """Zero-gradient / symmetry fill of halo cells.

    Mirrors the behaviour of ``jax_laseram.vof.advection._apply_vof_bcs``
    (reference ``run_vof_only.py``:14-19) without introducing an
    advection module dependency. Extended to also fill z halos when the
    grid is 3-D (``nz >= 2``), which is required by this case's quasi-2D
    layout (``nz=3``) so the PLIC Youngs 3x3x3 stencil sees a valid
    z-homogeneous field at every layer.
    """
    F = F.at[:nh, :, :].set(F[nh : nh + 1, :, :])
    F = F.at[-nh:, :, :].set(F[-nh - 1 : -nh, :, :])
    F = F.at[:, :nh, :].set(F[:, nh : nh + 1, :])
    F = F.at[:, -nh:, :].set(F[:, -nh - 1 : -nh, :])
    if has_z_halo:
        F = F.at[:, :, :nh].set(F[:, :, nh : nh + 1])
        F = F.at[:, :, -nh:].set(F[:, :, -nh - 1 : -nh])
    return F


def create_droplet_vof(grid) -> jnp.ndarray:
    """Create the initial circular-droplet VOF field.

    Parameters
    ----------
    grid : jax_laseram.data_types.GridInfo
        Grid object with ``nx, ny, nz, nh, dx, dy`` attributes.

    Returns
    -------
    F : jnp.ndarray
        Volume fraction, shape ``(nx+2*nh, ny+2*nh, max(nz, 1) )``,
        dtype matches JAX default (float64 when x64 is enabled).
    """
    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy

    # Interior cell centers — matches run_vof_only.py:36-37 exactly.
    x = jnp.linspace(dx / 2.0, 1.0 - dx / 2.0, nx)
    y = jnp.linspace(dy / 2.0, 1.0 - dy / 2.0, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    cx, cy = CASE["droplet"]["center"]
    radius = CASE["droplet"]["radius"]
    eps_cells = CASE["droplet"]["transition_width_cells"]

    # Smooth tanh transition (run_vof_only.py:43-46).
    r = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    eps = eps_cells * dx
    F_interior = 0.5 * (1.0 - jnp.tanh((r - radius) / eps))
    F_interior = jnp.clip(F_interior, 0.0, 1.0)

    # Embed into full array with ghost cells (run_vof_only.py:49-52).
    # Quasi-2D: replicate the droplet across all nz interior layers;
    # symmetry BCs then fill the z halos so the 3x3x3 stencil is valid
    # everywhere.
    if nz > 1:
        nz_total = nz + 2 * nh
    else:
        nz_total = 1
    has_z_halo = nz > 1
    shape = (nx + 2 * nh, ny + 2 * nh, nz_total)
    F = jnp.zeros(shape)
    # Broadcast over the nz interior layers.
    F_slab = jnp.broadcast_to(F_interior[:, :, None], (nx, ny, nz))
    z_slice = slice(nh, nh + nz) if has_z_halo else slice(0, 1)
    F = F.at[nh : nh + nx, nh : nh + ny, z_slice].set(F_slab)
    F = _apply_symmetry_bcs(F, nh, has_z_halo)

    return F
