"""3D Lagrangian VOF advection module (Barkhudarov 2004, 3D extension).

Pipeline:
  Step 1: PLIC reconstruction on ORIGINAL grid → 3D normals + intercepts
  Step 2: Move cell faces → deformed hexahedron vertex positions
  Step 3: Hex-based polyhedron clipping overlay onto Eulerian grid
  Step 4: Obstacle BC enforcement (vertex clamping + F zeroing)
"""

import jax
import jax.numpy as jnp

from .reconstruction_3d import (
    compute_plic_normals_3d,
    compute_intercept_C_3d,
    gl_nodes_5,
    gl_weights_5,
)
from .move_3d import (
    lagrangian_move_faces_3d,
    build_deformed_hexahedra,
    TET_INDICES,
    decompose_hex_to_tets,
)
from .overlay_3d import (
    overlay_lagrangian_3d,
    _hex_box_volume,
    _poly_volume,
    _init_hex_poly,
    _clip_poly_by_plane,
    _clip_poly_by_box,
)
from .overlay_native import overlay_lagrangian_3d_native, is_native_available
from ...data_types import GridInfo, Array


def advect_vof_lagrangian_3d(
    F: Array,
    u_face: Array,
    v_face: Array,
    w_face: Array,
    grid: GridInfo,
    dt: float,
    obs_x_max: float = float("-inf"),
    obs_y_max: float = float("-inf"),
    obs_z_max: float = float("-inf"),
) -> Array:
    """Single-step 3D Lagrangian VOF advection (Barkhudarov 2004).

    Pipeline:
      1. PLIC on ORIGINAL grid → normals + intercepts
      2. Move cell faces → deformed vertex positions (Eq. 6-7)
      3. Clamp vertices at obstacle + domain boundaries
      4. Build donor hexahedra, overlay onto Eulerian acceptor grid
      5. Zero F inside obstacle cells, clip to [0, 1]

    Args:
        F: volume fraction, shape (Ntot_x, Ntot_y, Ntot_z, 1)
        u_face: x-face velocities, shape (Ntot_x-1, Ntot_y, Ntot_z)
        v_face: y-face velocities, shape (Ntot_x, Ntot_y-1, Ntot_z)
        w_face: z-face velocities, shape (Ntot_x, Ntot_y, Ntot_z-1)
        grid: GridInfo
        dt: timestep
        obs_x_max: x-max of rectangular solid obstacle from origin. Default -inf = no obstacle.
        obs_y_max: y-max of rectangular solid obstacle from origin.
        obs_z_max: z-max of rectangular solid obstacle from origin.

    Returns:
        F_out: updated volume fraction, same shape as F.
    """
    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    # Step 1: PLIC reconstruction
    nx_f, ny_f, nz_f = compute_plic_normals_3d(F, dx, dy, dz)

    # Gradient-based interface gating (same logic as 2D)
    raw_gx = jnp.zeros_like(F)
    raw_gy = jnp.zeros_like(F)
    raw_gz = jnp.zeros_like(F)
    raw_gx = raw_gx.at[1:-1, :, :, :].set(
        (F[2:, :, :, :] - F[:-2, :, :, :]) / (2.0 * dx)
    )
    raw_gy = raw_gy.at[:, 1:-1, :, :].set(
        (F[:, 2:, :, :] - F[:, :-2, :, :]) / (2.0 * dy)
    )
    raw_gz = raw_gz.at[:, :, 1:-1, :].set(
        (F[:, :, 2:, :] - F[:, :, :-2, :]) / (2.0 * dz)
    )
    raw_grad = jnp.sqrt(raw_gx**2 + raw_gy**2 + raw_gz**2)
    interface_mask = raw_grad > 0.5
    nx_f = jnp.where(interface_mask, nx_f, 0.0)
    ny_f = jnp.where(interface_mask, ny_f, 0.0)
    nz_f = jnp.where(interface_mask, nz_f, 0.0)

    C_f = compute_intercept_C_3d(F, nx_f, ny_f, nz_f, dx, dy, dz)

    # Step 2: Move cell faces → deformed vertices
    x_verts, y_verts, z_verts = lagrangian_move_faces_3d(
        u_face, v_face, w_face, grid, dt
    )

    # Step 3: Clamp deformed vertices at obstacle + domain boundaries
    # Domain boundary clamp prevents fluid from leaking through walls.
    # This is a first-order approximation: fluid at the boundary cell that
    # would have moved past the wall is "stuck" at the wall instead.
    has_obstacle = (obs_x_max > float("-inf") or
                    obs_y_max > float("-inf") or
                    obs_z_max > float("-inf"))

    xv = jnp.clip(x_verts, grid.x_range[0], grid.x_range[1])
    yv = jnp.clip(y_verts, grid.y_range[0], grid.y_range[1])
    zv = jnp.clip(z_verts, grid.z_range[0], grid.z_range[1])

    if has_obstacle:
        # Clamp vertices at obstacle boundary (solid wall).
        # Pattern from 2D: vertices ON the obstacle face that deformed inside
        # get pushed back. Uses 3-condition check + float32-appropriate eps.
        Ntot_x = grid.nx + 2 * nh
        Ntot_y = grid.ny + 2 * nh
        Ntot_z = grid.nz + 2 * nh
        x0_grid = grid.x_range[0] - nh * dx + jnp.arange(Ntot_x + 1) * dx
        y0_grid = grid.y_range[0] - nh * dy + jnp.arange(Ntot_y + 1) * dy
        z0_grid = grid.z_range[0] - nh * dz + jnp.arange(Ntot_z + 1) * dz
        xv0, yv0, zv0 = jnp.meshgrid(x0_grid, y0_grid, z0_grid, indexing="ij")
        eps = 1e-6  # float32-appropriate tolerance

        # x-face clamp: vertices ON obstacle x-face that moved inside
        push_x = ((xv0 >= obs_x_max - eps) & (xv < obs_x_max) &
                  (yv < obs_y_max) & (zv < obs_z_max))
        xv = jnp.where(push_x, obs_x_max, xv)
        # y-face clamp — use <= for xv because x-push already set xv=obs_x_max
        # for corner vertices, and we still need to push y at the corner.
        push_y = ((yv0 >= obs_y_max - eps) & (yv < obs_y_max) &
                  (xv <= obs_x_max + eps) & (zv < obs_z_max))
        yv = jnp.where(push_y, obs_y_max, yv)
        # z-face clamp — use <= for both x and y (may have been clamped)
        push_z = ((zv0 >= obs_z_max - eps) & (zv < obs_z_max) &
                  (xv <= obs_x_max + eps) & (yv <= obs_y_max + eps))
        zv = jnp.where(push_z, obs_z_max, zv)

    x_verts, y_verts, z_verts = xv, yv, zv

    # Step 4: Overlay (no obs_mask — fluid may deposit in obstacle cells)
    if is_native_available():
        F_new = overlay_lagrangian_3d_native(
            F, x_verts, y_verts, z_verts, grid,
            nx_f=nx_f, ny_f=ny_f, nz_f=nz_f, C_f=C_f,
            obs_mask=None,
        )
    else:
        F_new = overlay_lagrangian_3d(
            F, x_verts, y_verts, z_verts, grid,
            nx_f=nx_f, ny_f=ny_f, nz_f=nz_f, C_f=C_f,
        )

    # Step 5: Redistribute obstacle fluid to adjacent fluid cells,
    # then zero all obstacle cells.  This preserves volume that would
    # otherwise be lost by simply zeroing obstacle cells.
    if has_obstacle:
        obs_ix = int(round((obs_x_max - grid.x_range[0]) / dx))
        obs_iy = int(round((obs_y_max - grid.y_range[0]) / dy))
        obs_iz = int(round((obs_z_max - grid.z_range[0]) / dz))
        # Clamp to valid range
        obs_ix = min(obs_ix, nx)
        obs_iy = min(obs_iy, ny)
        obs_iz = min(obs_iz, nz)

        # Push x-wall boundary fluid (last obstacle column) → first fluid column
        if 0 < obs_ix < nx:
            F_new = F_new.at[obs_ix, :obs_iy, :obs_iz].add(
                F_new[obs_ix - 1, :obs_iy, :obs_iz])
        # Push y-wall boundary fluid → first fluid row
        if 0 < obs_iy < ny:
            F_new = F_new.at[:obs_ix, obs_iy, :obs_iz].add(
                F_new[:obs_ix, obs_iy - 1, :obs_iz])

        # Build mask and zero all obstacle cells
        x_cc = grid.x_range[0] + (jnp.arange(nx) + 0.5) * dx
        y_cc = grid.y_range[0] + (jnp.arange(ny) + 0.5) * dy
        z_cc = grid.z_range[0] + (jnp.arange(nz) + 0.5) * dz
        obs_mask_3d = ((x_cc[:, None, None] < obs_x_max) &
                       (y_cc[None, :, None] < obs_y_max) &
                       (z_cc[None, None, :] < obs_z_max))
        F_new = jnp.where(obs_mask_3d, 0.0, F_new)

    # Clip to [0, 1] for numerical stability.
    # The overlay preserves total volume exactly (Σ transfer = F), but
    # fluid wrapping around obstacles can produce F > 1 in corner cells.
    # Clipping discards this overfill — track it externally as
    # Barkhudarov's "positive volume error" for correct reporting.
    F_new = jnp.clip(F_new, 0.0, 1.0)

    # Pack back with halos (nearest-neighbor extrapolation)
    F_out = jnp.zeros_like(F)
    F_out = F_out.at[nh : nh + nx, nh : nh + ny, nh : nh + nz, 0].set(F_new)

    # Extrapolate halos
    F_out = F_out.at[:nh, :, :, :].set(F_out[nh : nh + 1, :, :, :])
    F_out = F_out.at[-nh:, :, :, :].set(F_out[-nh - 1 : -nh, :, :, :])
    F_out = F_out.at[:, :nh, :, :].set(F_out[:, nh : nh + 1, :, :])
    F_out = F_out.at[:, -nh:, :, :].set(F_out[:, -nh - 1 : -nh, :, :])
    F_out = F_out.at[:, :, :nh, :].set(F_out[:, :, nh : nh + 1, :])
    F_out = F_out.at[:, :, -nh:, :].set(F_out[:, :, -nh - 1 : -nh, :])

    return F_out
