"""Step 3+4: Eulerian Overlay with PLIC-in-Overlay (Barkhudarov proper).

For each donor cell:
  1. Construct fluid polygon by clipping cell rectangle with PLIC half-plane
  2. Displace fluid polygon vertices (already done via deformed donor quad)
  3. Clip displaced fluid polygon against acceptor rectangle
  4. Overlap area IS the fluid volume
"""

import jax
import jax.numpy as jnp
from ...data_types import GridInfo, Array

MAX_V = 8  # max vertices for all polygons


def shoelace_area(verts, valid):
    max_v = verts.shape[0]
    idx = jnp.arange(max_v)
    nv = jnp.sum(valid).astype(jnp.int32)
    nxt = jnp.where(idx < nv - 1, idx + 1, 0)
    nxt = jnp.where(idx == nv - 1, 0, nxt)
    nxt = jnp.where(idx >= nv, 0, nxt)
    cross = verts[idx, 0] * verts[nxt, 1] - verts[nxt, 0] * verts[idx, 1]
    cross = jnp.where(valid & valid[nxt], cross, 0.0)
    return 0.5 * jnp.abs(jnp.sum(cross))


def _clip_edge(poly, n_poly, en, ep, max_v=MAX_V):
    """Clip polygon against half-plane n·(p-p0) <= 0."""
    idx = jnp.arange(max_v)
    nxt = jnp.where(idx < n_poly - 1, idx + 1, 0)
    nxt = jnp.where(idx == n_poly - 1, 0, nxt)

    p, q = poly, poly[nxt]
    dp = jnp.sum((p - ep) * en[None, :], 1)
    dq = jnp.sum((q - ep) * en[None, :], 1)
    p_in = (dp <= 0) & (idx < n_poly)
    q_in = (dq <= 0) & (nxt < n_poly)

    d = q - p
    denom = jnp.where(jnp.abs(jnp.sum(d * en, 1)) < 1e-30, 1e-30, jnp.sum(d * en, 1))
    t_val = jnp.clip(jnp.sum((ep - p) * en, 1) / denom, 0, 1)
    inter = p + t_val[:, None] * d

    n_out = jnp.where(
        idx >= n_poly,
        0,
        jnp.where(
            p_in & q_in, 1, jnp.where(p_in & ~q_in, 1, jnp.where(~p_in & q_in, 2, 0))
        ),
    )
    cum = jnp.cumsum(n_out)
    pos0 = cum - n_out

    v1 = jnp.where(
        p_in[:, None] & q_in[:, None],
        q,
        jnp.where(
            p_in[:, None] & ~q_in[:, None],
            inter,
            jnp.where(~p_in[:, None] & q_in[:, None], inter, jnp.zeros(2)),
        ),
    )
    has_v2 = ~p_in & q_in & (idx < n_poly)

    out = jnp.zeros((max_v, 2))
    out_m = jnp.zeros(max_v, dtype=bool)
    pos_s = jnp.clip(pos0, 0, max_v - 1)
    act = (n_out > 0) & (idx < n_poly)

    def body(i, st):
        o, m = st
        o = jax.lax.cond(act[i], lambda: o.at[pos_s[i]].set(v1[i]), lambda: o)
        m = jax.lax.cond(act[i], lambda: m.at[pos_s[i]].set(True), lambda: m)
        p2 = jnp.clip(pos_s[i] + 1, 0, max_v - 1)
        o = jax.lax.cond(has_v2[i], lambda: o.at[p2].set(q[i]), lambda: o)
        m = jax.lax.cond(has_v2[i], lambda: m.at[p2].set(True), lambda: m)
        return o, m

    out, out_m = jax.lax.fori_loop(0, max_v, body, (out, out_m))
    return out, jnp.sum(out_m).astype(jnp.int32)


def _clip_scan(poly, n_init, normals, points, max_v=MAX_V):
    """Clip polygon against multiple edges via scan."""

    def step(c, inp):
        p, nv = c
        en, ep = inp
        p2, nv2 = _clip_edge(p, nv, en, ep, max_v)
        return (p2, nv2), None

    (pf, nvf), _ = jax.lax.scan(step, (poly, n_init), (normals, points))
    valid = jnp.arange(max_v) < nvf
    return pf, valid, nvf, shoelace_area(pf, valid)


# Acceptor rectangle edges (outward normals)
_ACC_NORMS = jnp.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])


def _clip_against_acceptor(poly, nv, acc_rect, max_v=MAX_V):
    """Clip a polygon against an acceptor rectangle."""
    pts = jnp.stack([acc_rect[0], acc_rect[1], acc_rect[2], acc_rect[0]])
    _, _, _, area = _clip_scan(poly, nv, _ACC_NORMS, pts, max_v)
    return area


def _quad_area(q):
    """Area of quadrilateral (4, 2)."""
    return 0.5 * jnp.abs(
        (q[0, 0] * q[1, 1] - q[1, 0] * q[0, 1])
        + (q[1, 0] * q[2, 1] - q[2, 0] * q[1, 1])
        + (q[2, 0] * q[3, 1] - q[3, 0] * q[2, 1])
        + (q[3, 0] * q[0, 1] - q[0, 0] * q[3, 1])
    )


def _clip_donor_by_plic(donor_quad, nx_p, ny_p, C_p, max_v=MAX_V):
    """Clip donor quadrilateral by PLIC half-plane → fluid polygon."""
    poly = jnp.zeros((max_v, 2)).at[:4].set(donor_quad)
    mag = jnp.sqrt(nx_p**2 + ny_p**2 + 1e-30)
    xc = 0.25 * jnp.sum(donor_quad[:, 0])
    yc = 0.25 * jnp.sum(donor_quad[:, 1])
    p0 = jnp.array([xc + C_p * nx_p / mag, yc + C_p * ny_p / mag])
    # Clip normal: toward fluid = opposite of PLIC normal (toward empty)
    cn = jnp.array([-nx_p / mag, -ny_p / mag])
    _, _, nv, _ = _clip_scan(poly, jnp.int32(4), cn[None, :], p0[None, :], max_v)
    return nv


def _full_clip_single(dq, ar):
    """Clip donor quad against acceptor rect (no PLIC)."""
    poly = jnp.zeros((MAX_V, 2)).at[:4].set(dq)
    _, _, _, area = _clip_scan(
        poly, jnp.int32(4), _ACC_NORMS, jnp.stack([ar[0], ar[1], ar[2], ar[0]]), MAX_V
    )
    return area


def _plic_clip_single(dq, ar, nx_p, ny_p, C_p, F_val):
    """Full PLIC-in-overlay: clip donor by PLIC, then by acceptor.

    Returns raw overlap area (NOT multiplied by F). Caller normalizes.
    """
    eps = 1e-6
    is_empty = F_val <= eps

    # Clip donor directly against acceptor
    poly_full = jnp.zeros((MAX_V, 2)).at[:4].set(dq)
    _, _, _, area_direct = _clip_scan(
        poly_full,
        jnp.int32(4),
        _ACC_NORMS,
        jnp.stack([ar[0], ar[1], ar[2], ar[0]]),
        MAX_V,
    )

    # PLIC clipping (for interface cells with nonzero normal and 0<F<1)
    mag = jnp.sqrt(nx_p**2 + ny_p**2 + 1e-30)
    has_normal = mag > 1e-6
    is_partial = (F_val > eps) & (F_val < 1.0 - eps)

    xc = 0.25 * jnp.sum(dq[:, 0])
    yc = 0.25 * jnp.sum(dq[:, 1])
    cn = jnp.array([-nx_p / mag, -ny_p / mag])
    p0 = jnp.array([xc + C_p * nx_p / mag, yc + C_p * ny_p / mag])

    poly_plic = jnp.zeros((MAX_V, 2)).at[:4].set(dq)
    fp, fv, fnv, _ = _clip_scan(
        poly_plic, jnp.int32(4), cn[None, :], p0[None, :], MAX_V
    )
    _, _, _, area_plic = _clip_scan(
        fp, fnv, _ACC_NORMS, jnp.stack([ar[0], ar[1], ar[2], ar[0]]), MAX_V
    )

    # Use PLIC overlap for partial cells with valid normal
    use_plic = is_partial & has_normal
    area = jnp.where(is_empty, 0.0, jnp.where(use_plic, area_plic, area_direct))

    return area


# vmap versions
_clip_full_vmap = jax.vmap(jax.vmap(_full_clip_single))
_plic_clip_vmap = jax.vmap(jax.vmap(_plic_clip_single))
_quad_area_vmap = jax.vmap(jax.vmap(_quad_area))


def overlay_lagrangian(F, x_verts, y_verts, grid, nx_f=None, ny_f=None, C_f=None):
    """Lagrangian overlay with optional PLIC fluid polygon support."""
    nh = grid.nh
    nx, ny = grid.nx, grid.ny

    x_acc = jnp.linspace(grid.x_range[0], grid.x_range[1], nx + 1)
    y_acc = jnp.linspace(grid.y_range[0], grid.y_range[1], ny + 1)

    F_int = F[nh : nh + nx, nh : nh + ny, 0]
    use_plic = nx_f is not None

    if use_plic:
        nx_int = nx_f[nh : nh + nx, nh : nh + ny, 0]
        ny_int = ny_f[nh : nh + nx, nh : nh + ny, 0]
        C_int = C_f[nh : nh + nx, nh : nh + ny, 0]

    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    def _offset_transfer(di, dj):
        vi = jnp.arange(nx)[:, None] + nh - di
        vj = jnp.arange(ny)[None, :] + nh - dj

        dq = jnp.stack(
            [
                jnp.stack([x_verts[vi, vj, 0], y_verts[vi, vj, 0]], -1),
                jnp.stack([x_verts[vi + 1, vj, 0], y_verts[vi + 1, vj, 0]], -1),
                jnp.stack([x_verts[vi + 1, vj + 1, 0], y_verts[vi + 1, vj + 1, 0]], -1),
                jnp.stack([x_verts[vi, vj + 1, 0], y_verts[vi, vj + 1, 0]], -1),
            ],
            axis=2,
        )

        F_pad = jnp.pad(F_int, ((1, 1), (1, 1)))
        dF = F_pad[1 - di : nx + 1 - di, 1 - dj : ny + 1 - dj]

        ai, aj = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
        ar = jnp.stack(
            [
                jnp.stack([x_acc[ai], y_acc[aj]], -1),
                jnp.stack([x_acc[ai + 1], y_acc[aj]], -1),
                jnp.stack([x_acc[ai + 1], y_acc[aj + 1]], -1),
                jnp.stack([x_acc[ai], y_acc[aj + 1]], -1),
            ],
            axis=2,
        )

        if use_plic:
            nx_pad = jnp.pad(nx_int, ((1, 1), (1, 1)))
            ny_pad = jnp.pad(ny_int, ((1, 1), (1, 1)))
            C_pad = jnp.pad(C_int, ((1, 1), (1, 1)))
            dnx = nx_pad[1 - di : nx + 1 - di, 1 - dj : ny + 1 - dj]
            dny = ny_pad[1 - di : nx + 1 - di, 1 - dj : ny + 1 - dj]
            dC = C_pad[1 - di : nx + 1 - di, 1 - dj : ny + 1 - dj]

            overlap = _plic_clip_vmap(dq, ar, dnx, dny, dC, dF)
            darea = jnp.maximum(_quad_area_vmap(dq), 1e-30)

            # Compute fluid polygon area for normalization
            # For partial cells: area ≈ F * donor_area
            # For full cells: area = donor_area (direct clip)
            # transfer = overlap / fluid_polygon_area * F_donor
            # ≈ overlap / (F * darea) * F = overlap / darea  (for PLIC)
            # ≈ overlap / darea * F  (for full cells, direct clip)
            is_full_mask = dF >= 1.0 - 1e-6
            fluid_area = jnp.where(is_full_mask, darea, jnp.maximum(dF * darea, 1e-30))
            return dF * overlap / fluid_area
        else:
            overlap = _clip_full_vmap(dq, ar)
            darea = jnp.maximum(_quad_area_vmap(dq), 1e-30)
            return dF * overlap / darea

    F_new = jnp.zeros((nx, ny))
    for di, dj in offsets:
        F_new = F_new + _offset_transfer(di, dj)

    return F_new
