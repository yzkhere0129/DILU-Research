"""Production Lagrangian VOF overlay — geometric gather-scatter for AM scale.

Target: 1M+ cell grids on 8GB GPU.

Key insight: for CFL < 1, 98%+ of cell-offset pairs are trivially
inside or outside the acceptor box. Only ~2% need actual S-H clipping.

Architecture per offset:
  1. CLASSIFY: 8-vertex box test → inside/outside/partial  (cheap: 24 comparisons/cell)
  2. FAST PATH inside: contribution = dF * hex_vol / cell_vol   (1 det per cell)
  3. FAST PATH outside: contribution = 0                         (free)
  4. GATHER partial cells into fixed MAX_PARTIAL buffer
  5. S-H CLIP on buffer only (expensive but rare)
  6. SCATTER results back

Memory: MAX_PARTIAL ≪ N_total → huge memory saving for the S-H intermediates.
"""

import jax
import jax.numpy as jnp

from .overlay_3d import _hex_box_volume, _poly_volume, _init_hex_poly


def _hex_vol_fast(hex_verts):
    """Volume of hex via 6-tet decomposition. Pure arithmetic."""
    from .move_3d import TET_INDICES
    tets = hex_verts[TET_INDICES]  # (6, 4, 3)
    def _tv(t):
        a = t[1]-t[0]; b = t[2]-t[0]; c = t[3]-t[0]
        return jnp.abs(a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0])) / 6.0
    return jnp.sum(jax.vmap(_tv)(tets))


def _classify_hex_box(hex_verts, box_min, box_max):
    """Classify hex-box relationship: 0=outside, 1=partial, 2=inside.

    Uses 8-vertex bounding test. ~24 comparisons per cell (CHEAP).
    Returns: scalar int (0, 1, or 2).
    """
    eps = jnp.float32(1e-8)
    inside = (
        (hex_verts[:, 0] >= box_min[0] - eps) & (hex_verts[:, 0] <= box_max[0] + eps) &
        (hex_verts[:, 1] >= box_min[1] - eps) & (hex_verts[:, 1] <= box_max[1] + eps) &
        (hex_verts[:, 2] >= box_min[2] - eps) & (hex_verts[:, 2] <= box_max[2] + eps)
    )
    all_inside = jnp.all(inside)
    all_outside = (
        jnp.all(hex_verts[:, 0] < box_min[0] - eps) |
        jnp.all(hex_verts[:, 0] > box_max[0] + eps) |
        jnp.all(hex_verts[:, 1] < box_min[1] - eps) |
        jnp.all(hex_verts[:, 1] > box_max[1] + eps) |
        jnp.all(hex_verts[:, 2] < box_min[2] - eps) |
        jnp.all(hex_verts[:, 2] > box_max[2] + eps)
    )
    return jnp.where(all_outside, 0, jnp.where(all_inside, 2, 1))


# Vectorized over full grid
_classify_vvv = jax.vmap(jax.vmap(jax.vmap(_classify_hex_box)))
_hexvol_vvv = jax.vmap(jax.vmap(jax.vmap(_hex_vol_fast)))


def overlay_lagrangian_3d_fast(
    F, x_verts, y_verts, z_verts, grid,
    nx_f=None, ny_f=None, nz_f=None, C_f=None,
    max_partial_frac=0.5,
):
    """Production overlay: classify → gather partial → S-H clip → scatter.

    Args:
        max_partial_frac: fraction of total cells for partial buffer.
            0.15 = 15%. For CFL=0.3, actual partial is ~2-5% per offset.
            Increase if hex deformation is large (high CFL or shear).
    """
    from .move_3d import build_deformed_hexahedra
    from .overlay_3d import _clip_poly_by_plane, _clip_poly_by_box

    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    cell_vol = dx * dy * dz
    N_total = nx * ny * nz
    MAX_PARTIAL = max(int(N_total * max_partial_frac), 64)

    hex_v = build_deformed_hexahedra(x_verts, y_verts, z_verts, grid)
    x_acc = jnp.linspace(grid.x_range[0], grid.x_range[1], nx + 1)
    y_acc = jnp.linspace(grid.y_range[0], grid.y_range[1], ny + 1)
    z_acc = jnp.linspace(grid.z_range[0], grid.z_range[1], nz + 1)
    F_int = F[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]

    hp = jnp.pad(hex_v, ((1,1),(1,1),(1,1),(0,0),(0,0)))
    Fp = jnp.pad(F_int, ((1,1),(1,1),(1,1)))
    ai, aj, ak = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), jnp.arange(nz), indexing="ij")
    bmin_grid = jnp.stack([x_acc[ai], y_acc[aj], z_acc[ak]], axis=-1)      # (Nx,Ny,Nz,3)
    bmax_grid = jnp.stack([x_acc[ai+1], y_acc[aj+1], z_acc[ak+1]], axis=-1)

    # Flatten for gather indexing
    bmin_flat = bmin_grid.reshape(N_total, 3)
    bmax_flat = bmax_grid.reshape(N_total, 3)

    offsets = jnp.array(
        [[di,dj,dk] for di in (-1,0,1) for dj in (-1,0,1) for dk in (-1,0,1)],
        dtype=jnp.int32)

    # PLIC setup
    use_plic = nx_f is not None
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

    # ── Per-cell S-H clip function (for partial cells) ──
    def _clip_single(hex8, bmin_, bmax_):
        """S-H clip hex by box → volume. The proven fast engine."""
        return _hex_box_volume(hex8, bmin_, bmax_)

    def _plic_clip_single(hex8, pn, pd, F_val, bmin_, bmax_):
        """PLIC clip: hex → PLIC → box → volume transfer."""
        poly = _init_hex_poly(hex8)
        fluid_poly = _clip_poly_by_plane(poly, pn, pd)
        fluid_vol = _poly_volume(fluid_poly)
        clipped = _clip_poly_by_box(fluid_poly, bmin_, bmax_)
        clip_vol = _poly_volume(clipped)
        return F_val * clip_vol / jnp.maximum(fluid_vol, jnp.float32(1e-20))

    _clip_partial_vmap = jax.vmap(_clip_single)
    if use_plic:
        _plic_partial_vmap = jax.vmap(_plic_clip_single)

    def _scan_body(F_acc, offset):
        di, dj, dk = offset[0], offset[1], offset[2]
        dh = jax.lax.dynamic_slice(hp, (1-di,1-dj,1-dk,0,0), (nx,ny,nz,8,3))
        dF = jax.lax.dynamic_slice(Fp, (1-di,1-dj,1-dk), (nx,ny,nz))

        # ═══════════════════════════════════════════
        # Step 1: CLASSIFY — skip empty donors and outside hexes
        # ═══════════════════════════════════════════
        classes = _classify_vvv(dh, bmin_grid, bmax_grid)  # 0=outside, 1=partial, 2=inside
        has_fluid = dF > 1e-8  # non-empty donor
        is_outside = (classes == 0)

        # Cells that need S-H clip: has fluid AND not completely outside
        needs_clip = has_fluid & ~is_outside  # (Nx,Ny,Nz)

        # ═══════════════════════════════════════════
        # Step 2: (no separate "inside" fast path — CFL>0 means hex always extends)
        # ═══════════════════════════════════════════

        # ═══════════════════════════════════════════
        # Step 3: GATHER cells that need clipping
        # ═══════════════════════════════════════════
        partial_flat = needs_clip.ravel()  # (N_total,)
        dh_flat = dh.reshape(N_total, 8, 3)
        dF_flat = dF.ravel()

        positions = jnp.cumsum(partial_flat.astype(jnp.int32)) - 1
        safe_pos = jnp.where(partial_flat & (positions < MAX_PARTIAL), positions, MAX_PARTIAL)

        # Dummy hex: unit cube at origin (S-H on it gives cell_vol, harmless)
        dummy_hex = jnp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                                [0,0,1],[1,0,1],[1,1,1],[0,1,1.]], jnp.float32) * dx

        buf_hex = jnp.broadcast_to(dummy_hex, (MAX_PARTIAL+1, 8, 3)).copy()
        buf_hex = buf_hex.at[safe_pos].set(dh_flat)
        buf_bmin = jnp.zeros((MAX_PARTIAL+1, 3), jnp.float32).at[safe_pos].set(bmin_flat)
        buf_bmax = (jnp.ones((MAX_PARTIAL+1, 3), jnp.float32) * dx).at[safe_pos].set(bmax_flat)
        buf_dF = jnp.zeros(MAX_PARTIAL+1, jnp.float32).at[safe_pos].set(dF_flat)
        buf_valid = jnp.zeros(MAX_PARTIAL+1, bool).at[safe_pos].set(
            partial_flat & (positions < MAX_PARTIAL))
        buf_orig_idx = jnp.zeros(MAX_PARTIAL+1, jnp.int32).at[safe_pos].set(
            jnp.where(partial_flat, jnp.arange(N_total), 0))

        # Trim junk slot
        buf_hex = buf_hex[:MAX_PARTIAL]
        buf_bmin = buf_bmin[:MAX_PARTIAL]
        buf_bmax = buf_bmax[:MAX_PARTIAL]
        buf_dF = buf_dF[:MAX_PARTIAL]
        buf_valid = buf_valid[:MAX_PARTIAL]
        buf_orig_idx = buf_orig_idx[:MAX_PARTIAL]

        # ═══════════════════════════════════════════
        # Step 4: S-H CLIP on partial buffer only
        # ═══════════════════════════════════════════
        partial_vols = _clip_partial_vmap(buf_hex, buf_bmin, buf_bmax)  # (MAX_PARTIAL,)
        partial_transfer = buf_dF * partial_vols / cell_vol
        partial_transfer = jnp.where(buf_valid, partial_transfer, 0.0)

        # ═══════════════════════════════════════════
        # Step 5: SCATTER partial results back (junk-slot protection)
        # ═══════════════════════════════════════════
        # Invalid entries must NOT overwrite valid results at position 0.
        # Route invalid writes to junk slot at index N_total.
        F_partial_ext = jnp.zeros(N_total + 1)
        safe_scatter_idx = jnp.where(buf_valid, buf_orig_idx, N_total)
        F_partial_ext = F_partial_ext.at[safe_scatter_idx].set(partial_transfer)
        F_partial_flat = F_partial_ext[:N_total]  # discard junk slot
        F_partial = F_partial_flat.reshape(nx, ny, nz)

        # ═══════════════════════════════════════════
        # Step 6: partial IS the full contribution (no separate inside path)
        # ═══════════════════════════════════════════
        contrib = F_partial

        # ═══════════════════════════════════════════
        # Step 7: PLIC correction (if enabled)
        # ═══════════════════════════════════════════
        if use_plic:
            d_pn = jax.lax.dynamic_slice(pn_pad, (1-di,1-dj,1-dk,0), (nx,ny,nz,3))
            d_pd = jax.lax.dynamic_slice(pd_pad, (1-di,1-dj,1-dk), (nx,ny,nz))
            is_iface = (dF > 1e-6) & (dF < 1.0 - 1e-6)

            # Gather interface cells
            iface_flat = is_iface.ravel()
            MAX_IFACE = max(int(N_total * 0.1), 64)

            dpn_flat = d_pn.reshape(N_total, 3)
            dpd_flat = d_pd.ravel()

            ipos = jnp.cumsum(iface_flat.astype(jnp.int32)) - 1
            isafe = jnp.where(iface_flat & (ipos < MAX_IFACE), ipos, MAX_IFACE)

            ibuf_hex = jnp.broadcast_to(dummy_hex, (MAX_IFACE+1, 8, 3)).copy()
            ibuf_hex = ibuf_hex.at[isafe].set(dh_flat)
            ibuf_pn = jnp.broadcast_to(jnp.array([0.,0.,1.]), (MAX_IFACE+1, 3)).copy()
            ibuf_pn = ibuf_pn.at[isafe].set(dpn_flat)
            ibuf_pd = jnp.full(MAX_IFACE+1, -1e6, jnp.float32).at[isafe].set(dpd_flat)
            ibuf_F = jnp.zeros(MAX_IFACE+1, jnp.float32).at[isafe].set(dF_flat)
            ibuf_bmin = jnp.zeros((MAX_IFACE+1, 3), jnp.float32).at[isafe].set(bmin_flat)
            ibuf_bmax = (jnp.ones((MAX_IFACE+1, 3), jnp.float32)*dx).at[isafe].set(bmax_flat)
            ibuf_valid = jnp.zeros(MAX_IFACE+1, bool).at[isafe].set(
                iface_flat & (ipos < MAX_IFACE))
            ibuf_idx = jnp.zeros(MAX_IFACE+1, jnp.int32).at[isafe].set(
                jnp.where(iface_flat, jnp.arange(N_total), 0))

            # Trim
            ibuf_hex=ibuf_hex[:MAX_IFACE]; ibuf_pn=ibuf_pn[:MAX_IFACE]
            ibuf_pd=ibuf_pd[:MAX_IFACE]; ibuf_F=ibuf_F[:MAX_IFACE]
            ibuf_bmin=ibuf_bmin[:MAX_IFACE]; ibuf_bmax=ibuf_bmax[:MAX_IFACE]
            ibuf_valid=ibuf_valid[:MAX_IFACE]; ibuf_idx=ibuf_idx[:MAX_IFACE]

            plic_result = _plic_partial_vmap(ibuf_hex, ibuf_pn, ibuf_pd, ibuf_F, ibuf_bmin, ibuf_bmax)
            plic_result = jnp.where(ibuf_valid, plic_result, 0.0)

            # Scatter PLIC, replacing geometric values (junk-slot protection)
            contrib_ext = jnp.concatenate([contrib.ravel(), jnp.zeros(1)])
            safe_plic_idx = jnp.where(ibuf_valid, ibuf_idx, N_total)
            contrib_ext = contrib_ext.at[safe_plic_idx].set(plic_result)
            contrib = contrib_ext[:N_total].reshape(nx, ny, nz)

        return F_acc + contrib, None

    F_new, _ = jax.lax.scan(_scan_body, jnp.zeros((nx, ny, nz)), offsets)
    return jnp.clip(F_new, 0.0, 1.0)
