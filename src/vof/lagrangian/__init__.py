"""Lagrangian VOF advection module (Barkhudarov 2004).

Pipeline:
  Step 1: PLIC reconstruction on ORIGINAL grid → normals + intercepts
  Step 2: Move cell faces → deformed vertex positions
  Step 3: Build donor polygons on DEFORMED quads, overlay onto Eulerian grid
"""

import jax
import jax.numpy as jnp
from .reconstruction import compute_plic_normals, compute_intercept_C
from .move import lagrangian_move_faces
from .overlay_v2 import _convex_poly_rect_area, _clip_quad_by_plane, _quad_area


def advect_vof_lagrangian(F, u_face, v_face, grid, dt,
                          obs_x_max=-jnp.inf, obs_y_max=-jnp.inf):
    """Single-step Lagrangian VOF advection (Barkhudarov 2004).

    Pipeline:
      1. PLIC on ORIGINAL grid → normals + intercepts
      2. Move cell faces → deformed vertex positions (Eq. 6-7)
      3. Build donor polygons on deformed quads (PLIC for interface)
      4. Overlay donor polygons onto Eulerian acceptor grid (Eq. 8)
      5. Zero F in obstacle cells, cap at F=1 (discard overfill)

    Args:
        obs_x_max, obs_y_max: rectangular solid obstacle from (0,0) to
            (obs_x_max, obs_y_max). Deformed vertices are clamped at
            the boundary; F is zeroed inside obstacle after overlay.
            Default -inf = no obstacle.
    """
    nh = grid.nh
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    # Step 1: PLIC reconstruction on original grid
    nx_f, ny_f = compute_plic_normals(F, dx, dy)
    C_f = compute_intercept_C(F, nx_f, ny_f, dx, dy)

    # Gradient-based interface gating
    raw_gx = jnp.zeros_like(F)
    raw_gy = jnp.zeros_like(F)
    raw_gx = raw_gx.at[1:-1, :, :].set(
        (F[2:, :, :] - F[:-2, :, :]) / (2.0 * dx))
    raw_gy = raw_gy.at[:, 1:-1, :].set(
        (F[:, 2:, :] - F[:, :-2, :]) / (2.0 * dy))
    raw_grad = jnp.sqrt(raw_gx**2 + raw_gy**2)
    interface_mask = raw_grad > 0.5
    nx_f = jnp.where(interface_mask, nx_f, 0.0)
    ny_f = jnp.where(interface_mask, ny_f, 0.0)

    # Step 2: Move cell faces → deformed vertex positions
    x_verts, y_verts = lagrangian_move_faces(u_face, v_face, grid, dt)

    # Clamp deformed vertices at obstacle + domain boundary
    x0_grid = grid.x_range[0] - nh * dx + jnp.arange(nx + 2*nh + 1) * dx
    y0_grid = grid.y_range[0] - nh * dy + jnp.arange(ny + 2*nh + 1) * dy
    xv0, yv0 = jnp.meshgrid(x0_grid, y0_grid, indexing='ij')
    xv = x_verts[:, :, 0]
    yv = y_verts[:, :, 0]
    push_from_right = (xv0 >= obs_x_max - 1e-10) & (xv < obs_x_max) & (yv < obs_y_max)
    xv = jnp.where(push_from_right, obs_x_max, xv)
    push_from_top = (yv0 >= obs_y_max - 1e-10) & (yv < obs_y_max) & (xv < obs_x_max)
    yv = jnp.where(push_from_top, obs_y_max, yv)
    xv = jnp.maximum(xv, grid.x_range[0])
    yv = jnp.maximum(yv, grid.y_range[0])
    x_verts = xv[:, :, None]
    y_verts = yv[:, :, None]

    # Interior slices
    F_int = F[nh:nh+nx, nh:nh+ny, 0]
    nx_int = nx_f[nh:nh+nx, nh:nh+ny, 0]
    ny_int = ny_f[nh:nh+nx, nh:nh+ny, 0]
    C_int = C_f[nh:nh+nx, nh:nh+ny, 0]

    # Step 3: Build deformed quads and donor polygons
    ii = jnp.arange(nx)[:, None] + nh
    jj = jnp.arange(ny)[None, :] + nh
    deformed_quads = jnp.stack([
        jnp.stack([x_verts[ii, jj, 0], y_verts[ii, jj, 0]], -1),
        jnp.stack([x_verts[ii+1, jj, 0], y_verts[ii+1, jj, 0]], -1),
        jnp.stack([x_verts[ii+1, jj+1, 0], y_verts[ii+1, jj+1, 0]], -1),
        jnp.stack([x_verts[ii, jj+1, 0], y_verts[ii, jj+1, 0]], -1),
    ], axis=2)  # (nx, ny, 4, 2)

    def _build_donor(dq, nx_p, ny_p, C_p, F_val):
        eps = 1e-6
        mag = jnp.sqrt(nx_p**2 + ny_p**2 + 1e-30)
        has_normal = mag > 1e-6
        is_empty = F_val < eps
        is_full = F_val > 1.0 - eps
        is_interface = ~is_empty & ~is_full & has_normal

        n_hat = jnp.array([nx_p / mag, ny_p / mag])
        xc = 0.25 * jnp.sum(dq[:, 0])
        yc = 0.25 * jnp.sum(dq[:, 1])
        p0 = jnp.array([xc + C_p * nx_p / mag, yc + C_p * ny_p / mag])
        plic_poly, plic_nv = _clip_quad_by_plane(dq, n_hat, p0)

        full_poly = jnp.zeros((5, 2)).at[:4].set(dq)
        full_nv = jnp.int32(4)
        empty_poly = jnp.zeros((5, 2))
        empty_nv = jnp.int32(0)

        poly = jnp.where(is_empty, empty_poly,
               jnp.where(is_interface, plic_poly, full_poly))
        nv = jnp.where(is_empty, empty_nv,
             jnp.where(is_interface, plic_nv, full_nv))
        return poly, nv

    _build_vmap = jax.vmap(jax.vmap(_build_donor))
    donor_polys, donor_nv = _build_vmap(
        deformed_quads, nx_int, ny_int, C_int, F_int)

    # Polygon areas (shoelace)
    def _poly_area(v, n):
        idx = jnp.arange(5)
        nxt = jnp.where(idx < n - 1, idx + 1, 0)
        nxt = jnp.where(idx >= n, 0, nxt)
        return 0.5 * jnp.abs(jnp.sum(jnp.where(
            idx < n, v[idx, 0]*v[nxt, 1] - v[nxt, 0]*v[idx, 1], 0.)))
    _area_vmap = jax.vmap(jax.vmap(_poly_area))
    donor_area = _area_vmap(donor_polys, donor_nv)

    _overlap_vmap = jax.vmap(jax.vmap(_convex_poly_rect_area))

    # Acceptor grid
    x_acc = jnp.linspace(grid.x_range[0], grid.x_range[1], nx + 1)
    y_acc = jnp.linspace(grid.y_range[0], grid.y_range[1], ny + 1)

    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]

    # Obstacle cell mask
    x_cc = grid.x_range[0] + (jnp.arange(nx) + 0.5) * dx
    y_cc = grid.y_range[0] + (jnp.arange(ny) + 0.5) * dy
    obs_mask = (x_cc[:, None] < obs_x_max) & (y_cc[None, :] < obs_y_max)

    def _offset_transfer(di, dj):
        poly_pad = jnp.pad(donor_polys, ((1,1),(1,1),(0,0),(0,0)))
        nv_pad = jnp.pad(donor_nv, ((1,1),(1,1)))
        F_pad = jnp.pad(F_int, ((1,1),(1,1)))
        area_pad = jnp.pad(donor_area, ((1,1),(1,1)))

        d_poly = poly_pad[1-di:nx+1-di, 1-dj:ny+1-dj]
        d_nv = nv_pad[1-di:nx+1-di, 1-dj:ny+1-dj]
        d_F = F_pad[1-di:nx+1-di, 1-dj:ny+1-dj]
        d_area = jnp.maximum(area_pad[1-di:nx+1-di, 1-dj:ny+1-dj], 1e-30)

        ai, aj = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
        ar = jnp.stack([
            jnp.stack([x_acc[ai], y_acc[aj]], -1),
            jnp.stack([x_acc[ai+1], y_acc[aj]], -1),
            jnp.stack([x_acc[ai+1], y_acc[aj+1]], -1),
            jnp.stack([x_acc[ai], y_acc[aj+1]], -1),
        ], axis=2)

        overlap = _overlap_vmap(d_poly, d_nv, ar)
        return d_F * overlap / d_area

    F_new = jnp.zeros((nx, ny))
    for di, dj in offsets:
        F_new = F_new + _offset_transfer(di, dj)

    # Zero F inside obstacle, cap at [0, 1] (Barkhudarov: discard overfill)
    F_new = jnp.where(obs_mask, 0.0, F_new)
    F_new = jnp.clip(F_new, 0.0, 1.0)

    # Pack back with halos
    F_out = jnp.zeros_like(F)
    F_out = F_out.at[nh:nh+nx, nh:nh+ny, 0].set(F_new)
    F_out = F_out.at[:nh,:,:].set(F_out[nh:nh+1,:,:])
    F_out = F_out.at[-nh:,:,:].set(F_out[-nh-1:-nh,:,:])
    F_out = F_out.at[:,:nh,:].set(F_out[:,nh:nh+1,:])
    F_out = F_out.at[:,-nh:,:].set(F_out[:,-nh-1:-nh,:])
    return F_out
