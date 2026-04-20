"""Step 3: Vectorized Eulerian Overlay — Zero scatter, zero fori_loop.

Replaces Sutherland-Hodgman clipping with "vertex collect + shoelace":
  1. Collect ALL candidate intersection vertices (pure tensor ops)
  2. Sort by polar angle (jnp.argsort on fixed-size array)
  3. Shoelace formula for area

For two convex polygons (deformed quad vs axis-aligned rect):
  - Donor vertices inside acceptor: 4 candidates (bbox test)
  - Acceptor vertices inside donor: 4 candidates (cross-product test)
  - Edge-edge intersections: 4×4 = 16 candidates (2×2 linear solve)
  Total: 24 candidates max. All computed in parallel.

With PLIC: donor quad first clipped by PLIC plane → up to 5 vertices.
  - PLIC vertices inside acceptor: 5 candidates
  - Acceptor vertices inside PLIC polygon: 4 candidates
  - Edge-edge: 5×4 = 20 candidates
  Total: 29 candidates. Use MAX_CAND = 32 for padding.
"""

import jax
import jax.numpy as jnp
from ...data_types import GridInfo, Array

MAX_CAND = 32  # max candidate vertices for intersection


# ─────────────────────────────────────────────────────────
# Primitives: point-in-polygon, edge-edge intersection
# ─────────────────────────────────────────────────────────

def _point_in_rect(pts, x_lo, y_lo, x_hi, y_hi):
    """Test if points are inside axis-aligned rectangle.

    pts: (..., 2), returns: (...) boolean
    """
    return ((pts[..., 0] >= x_lo) & (pts[..., 0] <= x_hi) &
            (pts[..., 1] >= y_lo) & (pts[..., 1] <= y_hi))


def _point_in_convex_quad(pts, quad):
    """Test if points are inside a convex quadrilateral.

    pts: (..., 2), quad: (4, 2) — vertices in order.
    Uses cross-product sign test: point is inside iff all cross products
    have the same sign (assuming CCW or CW ordering).
    Returns: (...) boolean.
    """
    # For each edge (v[i] → v[i+1]), compute cross product with (pt - v[i])
    # cross = (v[i+1] - v[i]) × (pt - v[i])
    signs = []
    for i in range(4):
        j = (i + 1) % 4
        edge = quad[j] - quad[i]  # (2,)
        to_pt = pts - quad[i]     # (..., 2)
        cross = edge[0] * to_pt[..., 1] - edge[1] * to_pt[..., 0]
        signs.append(cross)
    s = jnp.stack(signs, axis=-1)  # (..., 4)
    # All positive or all negative → inside
    all_pos = jnp.all(s >= -1e-12, axis=-1)
    all_neg = jnp.all(s <= 1e-12, axis=-1)
    return all_pos | all_neg


def _segment_intersections(p1, p2, q1, q2):
    """Compute intersection of segments (p1→p2) and (q1→q2).

    All inputs: (2,) — single segments.
    Returns: (2,) intersection point, bool valid.

    Solves: p1 + t*(p2-p1) = q1 + s*(q2-q1), t,s ∈ [0,1]
    """
    d1 = p2 - p1  # (2,)
    d2 = q2 - q1  # (2,)
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    # Parallel check
    safe_denom = jnp.where(jnp.abs(denom) < 1e-30, 1e-30, denom)

    dp = q1 - p1
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / safe_denom
    s = (dp[0] * d1[1] - dp[1] * d1[0]) / safe_denom

    pt = p1 + t * d1
    valid = (t >= -1e-10) & (t <= 1 + 1e-10) & (s >= -1e-10) & (s <= 1 + 1e-10) & (jnp.abs(denom) > 1e-30)
    return pt, valid


def _batch_segment_intersections(poly_verts, poly_n, rect_verts):
    """Compute all edge-edge intersections between polygon and rectangle.

    poly_verts: (M, 2) — polygon vertices (M = max polygon size)
    poly_n: int — actual number of vertices
    rect_verts: (4, 2) — rectangle vertices

    Returns: points (M*4, 2), valid (M*4,)
    """
    M = poly_verts.shape[0]
    poly_idx = jnp.arange(M)
    poly_nxt = jnp.where(poly_idx < poly_n - 1, poly_idx + 1, 0)

    all_pts = []
    all_valid = []

    # For each rect edge (4 edges)
    for ri in range(4):
        rj = (ri + 1) % 4
        q1 = rect_verts[ri]
        q2 = rect_verts[rj]

        # Vectorize over polygon edges
        p1 = poly_verts  # (M, 2)
        p2 = poly_verts[poly_nxt]  # (M, 2)

        d1 = p2 - p1  # (M, 2)
        d2 = q2 - q1  # (2,) broadcast
        denom = d1[:, 0] * d2[1] - d1[:, 1] * d2[0]  # (M,)
        safe_denom = jnp.where(jnp.abs(denom) < 1e-30, 1e-30, denom)

        dp = q1 - p1  # (M, 2)
        t = (dp[:, 0] * d2[1] - dp[:, 1] * d2[0]) / safe_denom
        s = (dp[:, 0] * d1[:, 1] - dp[:, 1] * d1[:, 0]) / safe_denom

        pts = p1 + t[:, None] * d1  # (M, 2)
        valid = ((t >= -1e-10) & (t <= 1 + 1e-10) &
                 (s >= -1e-10) & (s <= 1 + 1e-10) &
                 (jnp.abs(denom) > 1e-30) &
                 (poly_idx < poly_n))

        all_pts.append(pts)
        all_valid.append(valid)

    return jnp.concatenate(all_pts, axis=0), jnp.concatenate(all_valid, axis=0)


# ─────────────────────────────────────────────────────────
# PLIC clipping: quad → polygon (up to 5 verts)
# ─────────────────────────────────────────────────────────

def _clip_quad_by_plane(quad, n_vec, p0):
    """Clip convex quad by half-plane n·(p - p0) ≤ 0.

    Returns: polygon vertices (5, 2) and vertex count.
    Pure tensor ops — no dynamic indexing, no scatter.

    Strategy: pre-compute ALL possible outputs (8 slots: 4 vertices + 4 intersections),
    then compact via prefix sum.
    """
    # Classify 4 vertices
    d = jnp.sum((quad - p0[None, :]) * n_vec[None, :], axis=1)  # (4,)
    inside = d <= 1e-10  # (4,)

    # For each of 4 edges, compute: should_emit_vertex, should_emit_intersection, intersection_point
    emit_v = jnp.zeros(8, dtype=bool)
    emit_pts = jnp.zeros((8, 2))

    for i in range(4):
        j = (i + 1) % 4
        t = jnp.clip(d[i] / (d[i] - d[j] + 1e-30), 0.0, 1.0)
        inter = quad[i] + t * (quad[j] - quad[i])
        crosses = inside[i] != inside[j]

        # Slot 2*i: vertex i (emitted if inside)
        emit_v = emit_v.at[2*i].set(inside[i])
        emit_pts = emit_pts.at[2*i].set(quad[i])

        # Slot 2*i+1: intersection (emitted if edge crosses)
        emit_v = emit_v.at[2*i+1].set(crosses)
        emit_pts = emit_pts.at[2*i+1].set(inter)

    # Compact: gather emitted points into contiguous array
    # prefix sum gives output positions
    positions = jnp.cumsum(emit_v.astype(jnp.int32)) - 1  # 0-indexed
    n_out = jnp.sum(emit_v).astype(jnp.int32)

    # Scatter to output (fixed 8 slots → 5 output, positions are monotonic)
    result = jnp.zeros((5, 2))
    for k in range(8):
        pos = jnp.clip(positions[k], 0, 4)
        result = jnp.where(emit_v[k] & (pos < 5),
                           result.at[pos].set(emit_pts[k]), result)

    return result, jnp.minimum(n_out, 5)


# ─────────────────────────────────────────────────────────
# Core: convex polygon vs rectangle overlap area
# ─────────────────────────────────────────────────────────

def _convex_poly_rect_area(poly, n_poly, rect):
    """Area of intersection of convex polygon with axis-aligned rectangle.

    poly: (M, 2) vertices, n_poly: vertex count
    rect: (4, 2) rectangle vertices (BL, BR, TR, TL order)

    Returns: scalar area. Pure tensor ops.
    """
    M = poly.shape[0]
    x_lo = rect[0, 0]; y_lo = rect[0, 1]
    x_hi = rect[2, 0]; y_hi = rect[2, 1]

    # ── Collect candidate vertices ──

    # Type 1: polygon vertices inside rectangle
    poly_inside = _point_in_rect(poly, x_lo, y_lo, x_hi, y_hi)
    poly_inside = poly_inside & (jnp.arange(M) < n_poly)
    cand_1 = poly         # (M, 2)
    valid_1 = poly_inside  # (M,)

    # Type 2: rectangle vertices inside polygon
    # Must respect actual vertex count n_poly (NOT hardcoded 4)
    # For nv < 4, unused vertices are (0,0) which would create a
    # degenerate quad and produce wrong point-in-polygon results.
    # Use winding number test with n_poly edges.
    def _point_in_convex_npoly(pt, verts, nv):
        """Test if pt is inside convex polygon with nv vertices."""
        signs = jnp.zeros(M)
        for k in range(M):
            kn = jnp.where(k < nv - 1, k + 1, 0)
            edge = verts[kn] - verts[k]
            to_pt = pt - verts[k]
            cross = edge[0] * to_pt[1] - edge[1] * to_pt[0]
            signs = signs.at[k].set(jnp.where(k < nv, cross, 0.0))
        active = jnp.arange(M) < nv
        all_pos = jnp.all(jnp.where(active, signs >= -1e-12, True))
        all_neg = jnp.all(jnp.where(active, signs <= 1e-12, True))
        return (all_pos | all_neg) & (nv >= 3)

    rect_inside = jnp.array([_point_in_convex_npoly(rect[k], poly, n_poly) for k in range(4)])
    cand_2 = rect          # (4, 2)
    valid_2 = rect_inside  # (4,)

    # Type 3: edge-edge intersections
    cand_3, valid_3 = _batch_segment_intersections(poly, n_poly, rect)

    # ── Combine all candidates ──
    all_cands = jnp.concatenate([cand_1, cand_2, cand_3], axis=0)    # (M+4+M*4, 2)
    all_valid = jnp.concatenate([valid_1, valid_2, valid_3], axis=0)  # (M+4+M*4,)

    n_total = all_cands.shape[0]

    # ── Compute centroid of valid points ──
    n_valid = jnp.sum(all_valid) + 1e-30
    cx = jnp.sum(jnp.where(all_valid, all_cands[:, 0], 0.0)) / n_valid
    cy = jnp.sum(jnp.where(all_valid, all_cands[:, 1], 0.0)) / n_valid

    # ── Sort by polar angle ──
    angles = jnp.where(all_valid,
                       jnp.arctan2(all_cands[:, 1] - cy, all_cands[:, 0] - cx),
                       100.0)  # invalid → large angle (sorted to end)
    order = jnp.argsort(angles)

    sorted_x = all_cands[order, 0]
    sorted_y = all_cands[order, 1]
    sorted_valid = all_valid[order]

    # ── Shoelace formula ──
    # Only use first n_valid_int vertices
    n_valid_int = jnp.sum(all_valid).astype(jnp.int32)
    idx = jnp.arange(n_total)
    nxt = jnp.where(idx < n_valid_int - 1, idx + 1, 0)
    nxt = jnp.where(idx >= n_valid_int, 0, nxt)

    cross = sorted_x * sorted_y[nxt] - sorted_x[nxt] * sorted_y
    cross = jnp.where(sorted_valid & (idx < n_valid_int), cross, 0.0)

    return 0.5 * jnp.abs(jnp.sum(cross))


# ─────────────────────────────────────────────────────────
# Single-cell overlap computation
# ─────────────────────────────────────────────────────────

def _overlap_single(dq, ar, nx_p, ny_p, C_p, F_val):
    """Compute overlap area between donor quad and acceptor rect.

    Handles three cases:
      F ≈ 0: return 0 (empty donor)
      F ≈ 1: clip full quad vs rect
      0 < F < 1: clip PLIC polygon vs rect
    """
    eps = 1e-6

    # Acceptor rect vertices: (4, 2) in BL, BR, TR, TL order
    rect = ar

    # ── Full cell: quad vs rect directly ──
    area_full = _convex_poly_rect_area(dq, jnp.int32(4), rect)

    # ── PLIC cell: clip quad by PLIC plane, then vs rect ──
    mag = jnp.sqrt(nx_p**2 + ny_p**2 + 1e-30)
    xc = 0.25 * jnp.sum(dq[:, 0])
    yc = 0.25 * jnp.sum(dq[:, 1])
    cn = jnp.array([-nx_p / mag, -ny_p / mag])
    p0 = jnp.array([xc + C_p * nx_p / mag, yc + C_p * ny_p / mag])

    plic_poly, plic_nv = _clip_quad_by_plane(dq, cn, p0)
    area_plic = _convex_poly_rect_area(plic_poly, plic_nv, rect)

    # ── Select based on F ──
    is_empty = F_val < eps
    is_full = F_val > 1.0 - eps
    has_normal = mag > 1e-6

    area = jnp.where(is_empty, 0.0,
           jnp.where(is_full | ~has_normal, area_full, area_plic))

    return area


# Vectorize over (nx, ny)
_overlap_vmap = jax.vmap(jax.vmap(_overlap_single))


def _quad_area(q):
    """Area of quadrilateral (4, 2)."""
    return 0.5 * jnp.abs(
        (q[0, 0]*q[1, 1] - q[1, 0]*q[0, 1]) +
        (q[1, 0]*q[2, 1] - q[2, 0]*q[1, 1]) +
        (q[2, 0]*q[3, 1] - q[3, 0]*q[2, 1]) +
        (q[3, 0]*q[0, 1] - q[0, 0]*q[3, 1]))

_quad_area_vmap = jax.vmap(jax.vmap(_quad_area))


# ─────────────────────────────────────────────────────────
# Main overlay function
# ─────────────────────────────────────────────────────────

def overlay_lagrangian(F, x_verts, y_verts, grid, nx_f=None, ny_f=None, C_f=None):
    """Vectorized Lagrangian overlay — zero scatter, zero fori_loop.

    Uses vertex-collect + polar-sort + shoelace for polygon intersection.
    """
    nh = grid.nh
    nx, ny = grid.nx, grid.ny

    x_acc = jnp.linspace(grid.x_range[0], grid.x_range[1], nx + 1)
    y_acc = jnp.linspace(grid.y_range[0], grid.y_range[1], ny + 1)

    F_int = F[nh:nh+nx, nh:nh+ny, 0]
    use_plic = nx_f is not None

    if use_plic:
        nx_int = nx_f[nh:nh+nx, nh:nh+ny, 0]
        ny_int = ny_f[nh:nh+nx, nh:nh+ny, 0]
        C_int = C_f[nh:nh+nx, nh:nh+ny, 0]

    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]

    def _offset_transfer(di, dj):
        vi = jnp.arange(nx)[:, None] + nh - di
        vj = jnp.arange(ny)[None, :] + nh - dj

        dq = jnp.stack([
            jnp.stack([x_verts[vi, vj, 0], y_verts[vi, vj, 0]], -1),
            jnp.stack([x_verts[vi+1, vj, 0], y_verts[vi+1, vj, 0]], -1),
            jnp.stack([x_verts[vi+1, vj+1, 0], y_verts[vi+1, vj+1, 0]], -1),
            jnp.stack([x_verts[vi, vj+1, 0], y_verts[vi, vj+1, 0]], -1),
        ], axis=2)

        F_pad = jnp.pad(F_int, ((1, 1), (1, 1)))
        dF = F_pad[1-di:nx+1-di, 1-dj:ny+1-dj]

        ai, aj = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
        ar = jnp.stack([
            jnp.stack([x_acc[ai], y_acc[aj]], -1),
            jnp.stack([x_acc[ai+1], y_acc[aj]], -1),
            jnp.stack([x_acc[ai+1], y_acc[aj+1]], -1),
            jnp.stack([x_acc[ai], y_acc[aj+1]], -1),
        ], axis=2)

        # Acceptor cell area (= dx * dy for uniform grid)
        acc_area = grid.dx * grid.dy

        if use_plic:
            nx_pad = jnp.pad(nx_int, ((1, 1), (1, 1)))
            ny_pad = jnp.pad(ny_int, ((1, 1), (1, 1)))
            C_pad = jnp.pad(C_int, ((1, 1), (1, 1)))
            dnx = nx_pad[1-di:nx+1-di, 1-dj:ny+1-dj]
            dny = ny_pad[1-di:nx+1-di, 1-dj:ny+1-dj]
            dC = C_pad[1-di:nx+1-di, 1-dj:ny+1-dj]

            # overlap = area of (fluid polygon ∩ acceptor rect)
            # For PLIC: this IS the fluid volume entering the acceptor
            # For full cells: overlap = (deformed quad ∩ acceptor rect)
            overlap = _overlap_vmap(dq, ar, dnx, dny, dC, dF)

            # F_new contribution = overlap / acceptor_area
            # (overlap is already the fluid area, no further scaling needed)
            return overlap / acc_area
        else:
            # No PLIC: overlap = (full quad ∩ rect), scale by donor F
            overlap = _overlap_vmap(dq, ar, jnp.zeros_like(dF), jnp.zeros_like(dF),
                                    jnp.zeros_like(dF), jnp.ones_like(dF))
            return dF * overlap / acc_area

    F_new = jnp.zeros((nx, ny))
    for di, dj in offsets:
        F_new = F_new + _offset_transfer(di, dj)

    return F_new
