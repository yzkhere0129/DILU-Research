"""Batched Lagrangian VOF overlay — O(BATCH) memory, scales to millions.

V3: eliminated reshape to (n_batches, batch_size, ...) intermediate.
All batch indexing via dynamic_slice on flat arrays — zero extra copies.
"""

import jax
import jax.numpy as jnp

from .overlay_3d import _hex_box_volume, _hex_plic_box_volume

BATCH_SIZE = 4096


def overlay_lagrangian_3d_batched(
    F, x_verts, y_verts, z_verts, grid,
    nx_f=None, ny_f=None, nz_f=None, C_f=None,
    obs_mask=None,
    batch_size=BATCH_SIZE,
):
    from .move_3d import build_deformed_hexahedra

    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    cell_vol = dx * dy * dz
    N_total = nx * ny * nz
    n_batches = (N_total + batch_size - 1) // batch_size
    N_padded = n_batches * batch_size
    pad_n = N_padded - N_total

    hex_v = build_deformed_hexahedra(x_verts, y_verts, z_verts, grid)
    x_acc = jnp.linspace(grid.x_range[0], grid.x_range[1], nx + 1)
    y_acc = jnp.linspace(grid.y_range[0], grid.y_range[1], ny + 1)
    z_acc = jnp.linspace(grid.z_range[0], grid.z_range[1], nz + 1)
    F_int = F[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]

    # Barkhudarov Eq. 8: dV = V_old / V_new compensates for hex compression.
    # V_old = cell_vol (undeformed), V_new = deformed hex volume.
    # For uniform advection: V_new ≈ V_old → dV ≈ 1 (no effect).
    # Near obstacles: V_new < V_old → dV > 1 (amplifies transfer to compensate).
    # Compute V_new from hex vertices using the cross-product formula:
    #   V = |det([v1-v0, v3-v0, v4-v0])| (exact for parallelepiped)
    e1 = hex_v[..., 1, :] - hex_v[..., 0, :]  # (nx, ny, nz, 3)
    e2 = hex_v[..., 3, :] - hex_v[..., 0, :]
    e3 = hex_v[..., 4, :] - hex_v[..., 0, :]
    V_new = jnp.abs(
        e1[..., 0] * (e2[..., 1] * e3[..., 2] - e2[..., 2] * e3[..., 1])
        - e1[..., 1] * (e2[..., 0] * e3[..., 2] - e2[..., 2] * e3[..., 0])
        + e1[..., 2] * (e2[..., 0] * e3[..., 1] - e2[..., 1] * e3[..., 0])
    )  # (nx, ny, nz)
    dV = cell_vol / jnp.maximum(V_new, jnp.float32(1e-30))
    # Cap dV to prevent extreme amplification from degenerate hexes
    dV = jnp.minimum(dV, jnp.float32(4.0))
    dV_int = dV  # (nx, ny, nz) — per-donor volume ratio

    hp = jnp.pad(hex_v, ((1,1),(1,1),(1,1),(0,0),(0,0)))
    Fp = jnp.pad(F_int, ((1,1),(1,1),(1,1)))
    dVp = jnp.pad(dV_int, ((1,1),(1,1),(1,1)), constant_values=1.0)

    # Acceptor bounds — flat + padded ONCE (static, reused every offset)
    ai, aj, ak = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), jnp.arange(nz), indexing="ij")
    bmin_f = jnp.pad(
        jnp.stack([x_acc[ai], y_acc[aj], z_acc[ak]], -1).reshape(N_total, 3),
        ((0, pad_n), (0, 0)))
    bmax_f = jnp.pad(
        jnp.stack([x_acc[ai+1], y_acc[aj+1], z_acc[ak+1]], -1).reshape(N_total, 3),
        ((0, pad_n), (0, 0)), constant_values=1.0)

    offsets = jnp.array(
        [[di,dj,dk] for di in (-1,0,1) for dj in (-1,0,1) for dk in (-1,0,1)],
        dtype=jnp.int32)

    # Obstacle mask for acceptor cells: if provided, transfers to obstacle
    # cells are zeroed so fluid is NOT deposited inside the obstacle.
    # Without this, the overlay deposits fluid into obstacle cells,
    # which then gets zeroed by F=0 enforcement → volume loss.
    if obs_mask is not None:
        # obs_mask: (nx, ny, nz) bool, True = inside obstacle
        obs_f = jnp.pad((~obs_mask).ravel().astype(jnp.float32), ((0, pad_n),))
    else:
        obs_f = None

    use_plic = nx_f is not None

    if use_plic:
        # Build PLIC clip plane parameters (same logic as overlay_3d.py)
        nx_int = nx_f[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]
        ny_int = ny_f[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]
        nz_int = nz_f[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]
        C_int = C_f[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]

        # C is CENTER-RELATIVE (see _volume_below_3d): d_global = C + n·centroid
        hex_cx = jnp.mean(hex_v[..., 0], axis=-1)
        hex_cy = jnp.mean(hex_v[..., 1], axis=-1)
        hex_cz = jnp.mean(hex_v[..., 2], axis=-1)

        plic_d_int = C_int + nx_int * hex_cx + ny_int * hex_cy + nz_int * hex_cz

        is_full = F_int > 1.0 - 1e-6
        is_empty = F_int < 1e-6
        plic_d_int = jnp.where(is_full, 1e6, jnp.where(is_empty, -1e6, plic_d_int))

        plic_n_int = jnp.stack([nx_int, ny_int, nz_int], axis=-1)
        has_normal = ~is_full & ~is_empty
        plic_n_int = jnp.where(has_normal[..., None], plic_n_int,
                               jnp.array([0.0, 0.0, 1.0]))

        pn_pad = jnp.pad(plic_n_int, ((1,1),(1,1),(1,1),(0,0)))
        pd_pad = jnp.pad(plic_d_int, ((1,1),(1,1),(1,1)))

        # Flat + pad PLIC data
        pn_f = jnp.pad(plic_n_int.reshape(N_total, 3), ((0, pad_n), (0, 0)))
        pd_f = jnp.pad(plic_d_int.ravel(), ((0, pad_n),))

        _clip_batch = jax.vmap(_hex_plic_box_volume)
    else:
        _clip_batch = jax.vmap(_hex_box_volume)

    def _offset_body(F_acc, offset):
        di, dj, dk = offset[0], offset[1], offset[2]
        dh = jax.lax.dynamic_slice(hp, (1-di,1-dj,1-dk,0,0), (nx,ny,nz,8,3))
        dF = jax.lax.dynamic_slice(Fp, (1-di,1-dj,1-dk), (nx,ny,nz))
        d_dV = jax.lax.dynamic_slice(dVp, (1-di,1-dj,1-dk), (nx,ny,nz))

        # Flat + pad (pad_n is tiny: < batch_size)
        dh_f = jnp.pad(dh.reshape(N_total, 8, 3), ((0, pad_n),(0,0),(0,0)))
        dF_f = jnp.pad(dF.ravel(), ((0, pad_n),))
        dV_f = jnp.pad(d_dV.ravel(), ((0, pad_n),), constant_values=1.0)

        if use_plic:
            d_pn = jax.lax.dynamic_slice(pn_pad, (1-di,1-dj,1-dk,0), (nx,ny,nz,3))
            d_pd = jax.lax.dynamic_slice(pd_pad, (1-di,1-dj,1-dk), (nx,ny,nz))
            d_pn_f = jnp.pad(d_pn.reshape(N_total, 3), ((0, pad_n), (0, 0)))
            d_pd_f = jnp.pad(d_pd.ravel(), ((0, pad_n),))

        def _batch_body(F_acc, bidx):
            s = bidx * batch_size
            hb = jax.lax.dynamic_slice(dh_f, (s,0,0), (batch_size,8,3))
            fb = jax.lax.dynamic_slice(dF_f, (s,), (batch_size,))
            lo = jax.lax.dynamic_slice(bmin_f, (s,0), (batch_size,3))
            hi = jax.lax.dynamic_slice(bmax_f, (s,0), (batch_size,3))

            if use_plic:
                pnb = jax.lax.dynamic_slice(d_pn_f, (s,0), (batch_size,3))
                pdb = jax.lax.dynamic_slice(d_pd_f, (s,), (batch_size,))
                overlap, fluid_vol = _clip_batch(hb, pnb, pdb, fb, lo, hi)
                # overlap/fluid_vol = fraction of donor's fluid going to
                # this acceptor.  Summing over all acceptors gives 1.0,
                # so total deposited = F (exact volume conservation).
                # NOTE: dV = V_old/V_new is NOT needed here because the
                # PLIC normalization already handles hex compression.
                transfers = fb * overlap / jnp.maximum(fluid_vol, jnp.float32(1e-20))
            else:
                vols = _clip_batch(hb, lo, hi)
                transfers = fb * vols / cell_vol

            # Zero transfers to obstacle cells (solid wall: no fluid deposited inside)
            if obs_f is not None:
                acc_mask = jax.lax.dynamic_slice(obs_f, (s,), (batch_size,))
                transfers = transfers * acc_mask

            cur = jax.lax.dynamic_slice(F_acc, (s,), (batch_size,))
            F_acc = jax.lax.dynamic_update_slice(F_acc, cur + transfers, (s,))
            return F_acc, None

        F_acc, _ = jax.lax.scan(_batch_body, F_acc, jnp.arange(n_batches))
        return F_acc, None

    F_acc, _ = jax.lax.scan(_offset_body, jnp.zeros(N_padded), offsets)

    # No clip here — let __init__.py handle clip after redistribution.
    # Clipping inside overlay discards overfill BEFORE redistribution
    # can recover it, causing irreversible volume loss.
    return F_acc[:N_total].reshape(nx, ny, nz)
