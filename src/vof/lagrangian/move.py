"""Step 2: Lagrangian Face Move — JAX-Native.

Computes the displacement of all cell faces under a velocity field,
following Barkhudarov (2004) Eq. 6-7 with second-order correction.

Each face moves by:
  dx_face = u * dt / (1 + 0.5 * du/dx * dt)

where du/dx is the centered difference of face velocities.

Vertex positions are computed as the average of adjacent face displacements.
"""

import jax.numpy as jnp
from ...data_types import GridInfo, Array


def lagrangian_move_faces(u_face: Array, v_face: Array, grid: GridInfo, dt: float):
    """Compute deformed vertex positions after Lagrangian face move.

    Args:
        u_face: x-face velocities, shape (Ntot_x-1, Ntot_y, 1)
                u_face[i,j] = velocity at face between cells i and i+1
        v_face: y-face velocities, shape (Ntot_x, Ntot_y-1, 1)
                v_face[i,j] = velocity at face between cells j and j+1
        grid: GridInfo
        dt: timestep

    Returns:
        x_verts: deformed x-coordinates, shape (Ntot_x+1, Ntot_y+1, 1)
        y_verts: deformed y-coordinates, shape (Ntot_x+1, Ntot_y+1, 1)
    """
    nh = grid.nh
    dx, dy = grid.dx, grid.dy
    Ntot_x = grid.nx + 2 * nh
    Ntot_y = grid.ny + 2 * nh

    # ---- x-face displacements ----
    # du/dx via centered diff on adjacent faces (interior only)
    dudx = jnp.zeros_like(u_face)
    dudx = dudx.at[1:-1, :, :].set((u_face[2:, :, :] - u_face[:-2, :, :]) / (2.0 * dx))

    # Eq. 6: dx_face = u * dt / (1 + 0.5 * du/dx * dt)
    # Boundary faces: use velocity directly (free-slip, du/dx=0 at boundary)
    dx_face = u_face * dt / (1.0 + 0.5 * dudx * dt)

    # ---- y-face displacements ----
    dvdy = jnp.zeros_like(v_face)
    dvdy = dvdy.at[:, 1:-1, :].set((v_face[:, 2:, :] - v_face[:, :-2, :]) / (2.0 * dy))

    dy_face = v_face * dt / (1.0 + 0.5 * dvdy * dt)

    # ---- Vertex positions ----
    # Vertices: Ntot_x+1 vertices at spacing dx, from x_range[0]-nh*dx
    # linspace(start, stop, N) gives N-1 intervals
    # We need Ntot_x intervals of size dx → Ntot_x+1 points
    x_orig = grid.x_range[0] - nh * dx + jnp.arange(Ntot_x + 1) * dx
    y_orig = grid.y_range[0] - nh * dy + jnp.arange(Ntot_y + 1) * dy
    x_grid, y_grid = jnp.meshgrid(x_orig, y_orig, indexing="ij")
    x_verts_0 = x_grid[:, :, None]
    y_verts_0 = y_grid[:, :, None]

    # ---- Vertex displacements ----
    # Build displacement at all face positions (Ntot_x+1 entries)
    # dx_all[i] = displacement of x-face at position i
    #   i=0: left boundary → face 0 displacement
    #   i=1..Ntot_x-1: interior → face i-1 displacement
    #   i=Ntot_x: right boundary → face N-2 displacement
    # Then vertex i = 0.5*(dx_all[i] + dx_all[i+1])
    # dx_all[i] = face displacement at x = x_range[0] + i*dx
    #   i=0: left boundary face = dx_face[0]
    #   i=1..Ntot_x-1: interior faces = dx_face[0..Ntot_x-2]
    #   i=Ntot_x: right boundary face = dx_face[Ntot_x-2]
    dx_all = jnp.zeros((Ntot_x + 1, Ntot_y, 1))
    dx_all = dx_all.at[0, :, :].set(dx_face[0, :, :])
    dx_all = dx_all.at[1:-1, :, :].set(dx_face)  # all dx_face entries
    dx_all = dx_all.at[-1, :, :].set(dx_face[-1, :, :])
    dx_verts_x = 0.5 * (dx_all[:-1, :, :] + dx_all[1:, :, :])  # (Ntot_x, Ntot_y, 1)

    # Extend in y: vertex j=Ntot_y copies face j=Ntot_y-1
    dx_verts = jnp.pad(dx_verts_x, ((0, 0), (0, 1), (0, 0)))
    dx_verts = dx_verts.at[:, -1, :].set(dx_verts_x[:, -1, :])
    # Extend in x: vertex i=Ntot_x copies face i=Ntot_x-1
    dx_verts = jnp.pad(dx_verts, ((0, 1), (0, 0), (0, 0)))
    dx_verts = dx_verts.at[-1, :, :].set(dx_verts[-2, :, :])

    # y-displacement at vertices (in y-direction)
    dy_all = jnp.zeros((Ntot_x, Ntot_y + 1, 1))
    dy_all = dy_all.at[:, 0, :].set(dy_face[:, 0, :])
    dy_all = dy_all.at[:, 1:-1, :].set(dy_face)
    dy_all = dy_all.at[:, -1, :].set(dy_face[:, -1, :])
    dy_verts_y = 0.5 * (dy_all[:, :-1, :] + dy_all[:, 1:, :])

    dy_verts = jnp.pad(dy_verts_y, ((0, 1), (0, 0), (0, 0)))
    dy_verts = dy_verts.at[-1, :, :].set(dy_verts[-2, :, :])
    dy_verts = jnp.pad(dy_verts, ((0, 0), (0, 1), (0, 0)))
    dy_verts = dy_verts.at[:, -1, :].set(dy_verts[:, -2, :])

    x_verts = x_verts_0 + dx_verts
    y_verts = y_verts_0 + dy_verts

    return x_verts, y_verts
