"""Sparse Lagrangian VOF overlay — skip full/empty cells entirely.

Architecture:
  Full cells (F≈1): transfer = geometric overlap / cell_vol (CHEAP, no clipping)
    For uniform velocity + small CFL: overlap ≈ cell_vol for self, ≈ CFL for neighbors.
    This is just dF * vol(hex ∩ box) / cell_vol — computed with the FAST S-H engine.

  Empty cells (F≈0): transfer = 0 (FREE)

  Interface cells (0<F<1): transfer via PLIC clip (EXPENSIVE but rare)
    Only ~1-5% of cells. Uses S-H polyhedron clip (the proven fast engine).

Key insight: the S-H engine at 0.19s/step for 1750 cells is ALREADY fast.
The optimization is to NOT run it on cells that don't need it.
For 500K cells with 1% interface: 5K cells need S-H, rest are free.
"""

import jax
import jax.numpy as jnp

from .overlay_3d import (
    _init_hex_poly, _clip_poly_by_plane, _clip_poly_by_box, _poly_volume,
    _hex_box_volume, MAX_F, MAX_FV,
)


def overlay_lagrangian_3d_sparse(
    F, x_verts, y_verts, z_verts, grid,
    nx_f=None, ny_f=None, nz_f=None, C_f=None,
):
    """Sparse overlay: S-H engine on interface cells only, skip full/empty.

    Without PLIC: identical to overlay_3d (uses _hex_box_volume for all).
    With PLIC: gather interface cells → PLIC S-H clip → scatter back.
    """
    from .move_3d import build_deformed_hexahedra

    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    cell_vol = dx * dy * dz

    hex_v = build_deformed_hexahedra(x_verts, y_verts, z_verts, grid)
    x_acc = jnp.linspace(grid.x_range[0], grid.x_range[1], nx + 1)
    y_acc = jnp.linspace(grid.y_range[0], grid.y_range[1], ny + 1)
    z_acc = jnp.linspace(grid.z_range[0], grid.z_range[1], nz + 1)
    F_int = F[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]

    use_plic = nx_f is not None

    # Pre-pad
    hp = jnp.pad(hex_v, ((1,1),(1,1),(1,1),(0,0),(0,0)))
    Fp = jnp.pad(F_int, ((1,1),(1,1),(1,1)))
    ai, aj, ak = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), jnp.arange(nz), indexing="ij")
    bmin = jnp.stack([x_acc[ai], y_acc[aj], z_acc[ak]], axis=-1)
    bmax = jnp.stack([x_acc[ai+1], y_acc[aj+1], z_acc[ak+1]], axis=-1)

    offsets = jnp.array(
        [[di,dj,dk] for di in (-1,0,1) for dj in (-1,0,1) for dk in (-1,0,1)],
        dtype=jnp.int32)

    # Use the PROVEN FAST S-H engine for geometric overlap
    _hex_vol_vvv = jax.vmap(jax.vmap(jax.vmap(_hex_box_volume)))

    if use_plic:
        nx_int = nx_f[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]
        ny_int = ny_f[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]
        nz_int = nz_f[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]
        C_int = C_f[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]
        hex_cx = jnp.mean(hex_v[..., 0], axis=-1)
        hex_cy = jnp.mean(hex_v[..., 1], axis=-1)
        hex_cz = jnp.mean(hex_v[..., 2], axis=-1)
        plic_d_int = C_int + nx_int*hex_cx + ny_int*hex_cy + nz_int*hex_cz
        is_full = F_int > 1.0 - 1e-6
        is_empty = F_int < 1e-6
        plic_d_int = jnp.where(is_full, 1e6, jnp.where(is_empty, -1e6, plic_d_int))
        plic_n_int = jnp.stack([nx_int, ny_int, nz_int], axis=-1)
        plic_n_int = jnp.where((~is_full & ~is_empty)[...,None], plic_n_int, jnp.array([0.,0.,1.]))
        pn_pad = jnp.pad(plic_n_int, ((1,1),(1,1),(1,1),(0,0)))
        pd_pad = jnp.pad(plic_d_int, ((1,1),(1,1),(1,1)))

        # PLIC single-cell: S-H clip hex by PLIC then by box
        def _plic_single(hex8, pn, pd, F_val, bmin_, bmax_):
            poly = _init_hex_poly(hex8)
            fluid_poly = _clip_poly_by_plane(poly, pn, pd)
            fluid_vol = _poly_volume(fluid_poly)
            clipped = _clip_poly_by_box(fluid_poly, bmin_, bmax_)
            clip_vol = _poly_volume(clipped)
            return F_val * clip_vol / jnp.maximum(fluid_vol, jnp.float32(1e-20))

        # PLIC vmap — applied to gathered interface cells only
        _plic_vmap = jax.vmap(_plic_single)

    def _scan_body(F_acc, offset):
        di, dj, dk = offset[0], offset[1], offset[2]
        dh = jax.lax.dynamic_slice(hp, (1-di,1-dj,1-dk,0,0), (nx,ny,nz,8,3))
        dF = jax.lax.dynamic_slice(Fp, (1-di,1-dj,1-dk), (nx,ny,nz))

        # ── Geometric overlap for ALL cells (fast S-H engine) ──
        overlap = _hex_vol_vvv(dh, bmin, bmax)
        F_geom = dF * overlap / cell_vol

        if not use_plic:
            return F_acc + F_geom, None

        # ── PLIC correction for interface donor cells ──
        d_pn = jax.lax.dynamic_slice(pn_pad, (1-di,1-dj,1-dk,0), (nx,ny,nz,3))
        d_pd = jax.lax.dynamic_slice(pd_pad, (1-di,1-dj,1-dk), (nx,ny,nz))
        is_iface = (dF > 1e-6) & (dF < 1.0 - 1e-6)

        # ── GATHER interface cells into fixed buffer ──
        N_total = nx * ny * nz
        MAX_IFACE = max(int(N_total * 0.2), 64)

        dh_flat = dh.reshape(N_total, 8, 3)
        dF_flat = dF.ravel()
        dpn_flat = d_pn.reshape(N_total, 3)
        dpd_flat = d_pd.ravel()
        bmin_flat = bmin.reshape(N_total, 3)
        bmax_flat = bmax.reshape(N_total, 3)
        iface_flat = is_iface.ravel()

        positions = jnp.cumsum(iface_flat.astype(jnp.int32)) - 1
        safe_pos = jnp.where(iface_flat & (positions < MAX_IFACE), positions, MAX_IFACE)

        # Buffers with POOL+1 junk slot
        buf_hex = jnp.zeros((MAX_IFACE+1, 8, 3), jnp.float32).at[safe_pos].set(dh_flat)
        buf_F = jnp.zeros(MAX_IFACE+1, jnp.float32).at[safe_pos].set(dF_flat)
        buf_pn = jnp.broadcast_to(jnp.array([0.,0.,1.]), (MAX_IFACE+1, 3)).copy()
        buf_pn = buf_pn.at[safe_pos].set(dpn_flat)
        buf_pd = jnp.full(MAX_IFACE+1, -1e6, jnp.float32).at[safe_pos].set(dpd_flat)
        buf_bmin = jnp.zeros((MAX_IFACE+1, 3), jnp.float32).at[safe_pos].set(bmin_flat)
        buf_bmax = jnp.ones((MAX_IFACE+1, 3), jnp.float32).at[safe_pos].set(bmax_flat)
        buf_valid = jnp.zeros(MAX_IFACE+1, bool).at[safe_pos].set(iface_flat & (positions < MAX_IFACE))
        buf_orig_idx = jnp.zeros(MAX_IFACE+1, jnp.int32).at[safe_pos].set(
            jnp.where(iface_flat, jnp.arange(N_total), 0))

        # Trim junk slot
        buf_hex = buf_hex[:MAX_IFACE]
        buf_F = buf_F[:MAX_IFACE]
        buf_pn = buf_pn[:MAX_IFACE]
        buf_pd = buf_pd[:MAX_IFACE]
        buf_bmin = buf_bmin[:MAX_IFACE]
        buf_bmax = buf_bmax[:MAX_IFACE]
        buf_valid = buf_valid[:MAX_IFACE]
        buf_orig_idx = buf_orig_idx[:MAX_IFACE]

        # ── COMPUTE: PLIC on MAX_IFACE buffer only ──
        plic_result = _plic_vmap(buf_hex, buf_pn, buf_pd, buf_F, buf_bmin, buf_bmax)
        plic_result = jnp.where(buf_valid, plic_result, 0.0)

        # ── SCATTER: replace geometric values at interface cells ──
        F_geom_flat = F_geom.ravel()
        F_combined_flat = F_geom_flat.at[buf_orig_idx].set(
            jnp.where(buf_valid, plic_result, F_geom_flat[buf_orig_idx]))
        F_combined = F_combined_flat.reshape(nx, ny, nz)

        return F_acc + F_combined, None

    F_new, _ = jax.lax.scan(_scan_body, jnp.zeros((nx, ny, nz)), offsets)
    return jnp.clip(F_new, 0.0, 1.0)
