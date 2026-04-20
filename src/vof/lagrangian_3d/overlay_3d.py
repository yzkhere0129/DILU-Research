"""3D Overlay — Coordinate-face polyhedron clipping + divergence-theorem volume.

Clips deformed hexahedra (optionally PLIC-truncated) against axis-aligned
acceptor cubes using Sutherland-Hodgman.  All ops use jax.lax.scan / vmap
with fixed-size tensors — **zero Python for-loops in traced code**.

Polyhedron representation (coordinate-face):
    faces:   (MAX_F, MAX_FV, 3) float32 — vertex positions per face
    face_nv: (MAX_F,) int32             — vertex count per face
    n_faces: int32                      — number of active faces

This avoids vertex-index bookkeeping entirely: shared vertices are stored
redundantly.  The divergence-theorem volume formula doesn't care.
"""

import jax
import jax.numpy as jnp

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

MAX_F = 14  # hex(6) + PLIC(1) + box(6) + safety(1)
MAX_FV = 14  # quad(4) grows ≤+1 per clip; cap face ≤ ~13 verts
_EPS = 1e-10
_EPS_BOX = jnp.float32(1e-6)  # float32-appropriate tolerance for bounding-box overlap

# Hex face table: CCW from outside → outward normals (verified for unit cube)
#   v0=(0,0,0) v1=(1,0,0) v2=(1,1,0) v3=(0,1,0)
#   v4=(0,0,1) v5=(1,0,1) v6=(1,1,1) v7=(0,1,1)
HEX_FACES = jnp.array(
    [
        [0, 3, 2, 1],  # bottom  z=0  normal -z
        [4, 5, 6, 7],  # top     z=1  normal +z
        [0, 1, 5, 4],  # front   y=0  normal -y
        [2, 3, 7, 6],  # back    y=1  normal +y
        [0, 4, 7, 3],  # left    x=0  normal -x
        [1, 2, 6, 5],  # right   x=1  normal +x
    ],
    dtype=jnp.int32,
)  # (6, 4)


# ═══════════════════════════════════════════════════════════════
# Polyhedron initialisation
# ═══════════════════════════════════════════════════════════════


def _init_hex_poly(hex_verts):
    """Build coordinate-face polyhedron from 8-vertex hex.

    hex_verts: (8, 3) float32
    Returns: (faces, face_nv, n_faces)
    """
    faces = jnp.zeros((MAX_F, MAX_FV, 3), dtype=jnp.float32)
    face_nv = jnp.zeros(MAX_F, dtype=jnp.int32)

    # Gather face vertex coordinates: (6, 4, 3)
    hex_face_coords = hex_verts[HEX_FACES]
    faces = faces.at[:6, :4, :].set(hex_face_coords)
    face_nv = face_nv.at[:6].set(4)
    return faces, face_nv, jnp.int32(6)


def _init_tet_poly(tet4):
    """Build coordinate-face polyhedron from 4-vertex tet.

    tet4: (4, 3) float32
    Returns: (faces, face_nv, n_faces)
    """
    faces = jnp.zeros((MAX_F, MAX_FV, 3), dtype=jnp.float32)
    face_nv = jnp.zeros(MAX_F, dtype=jnp.int32)

    # 4 triangular faces (outward normals for positive-det tet)
    idx = jnp.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=jnp.int32)
    for fi in range(4):  # compile-time constant loop (4 iterations) — OK
        faces = faces.at[fi, :3, :].set(tet4[idx[fi]])
    face_nv = face_nv.at[:4].set(3)
    return faces, face_nv, jnp.int32(4)


# ═══════════════════════════════════════════════════════════════
# Sutherland-Hodgman: clip ONE convex polygon face by a half-space
# ═══════════════════════════════════════════════════════════════


def _sh_clip_face(face_verts, face_nv, plane_n, plane_d):
    """Clip convex polygon by half-space  n · x ≤ d.

    Args:
        face_verts: (MAX_FV, 3)
        face_nv:    int32 (actual vertex count)
        plane_n:    (3,) outward normal of clipping plane
        plane_d:    scalar

    Returns:
        out_verts:  (MAX_FV, 3)
        out_nv:     int32
        inter_pts:  (2, 3)  — the (≤2) edge-plane intersection points
        n_inter:    int32   — 0 or 2
    """
    dist = jnp.sum(face_verts * plane_n[None, :], axis=1) - plane_d  # (MAX_FV,)

    def step(carry, idx):
        out, n_out, inters, n_int = carry
        i = idx
        j = jnp.where(i < face_nv - 1, i + 1, 0)
        active = i < face_nv

        pi, pj = face_verts[i], face_verts[j]
        di, dj = dist[i], dist[j]
        i_in = (di <= _EPS) & active
        j_in = (dj <= _EPS) & active

        # Intersection point on edge i→j
        sd = jnp.where(jnp.abs(di - dj) < 1e-30, 1e-30, di - dj)
        t = jnp.clip(di / sd, 0.0, 1.0)
        inter = pi + t * (pj - pi)

        # Cases (mutually exclusive)
        case_both_in = i_in & j_in
        case_exit = i_in & ~j_in & active
        case_enter = ~i_in & j_in
        emit_any = case_both_in | case_exit | case_enter

        # First emitted vertex
        v_first = jnp.where(case_both_in, pj, inter)
        s1 = jnp.minimum(n_out, MAX_FV - 1)
        out = jnp.where(emit_any, out.at[s1].set(v_first), out)
        n_out = n_out + jnp.where(emit_any, 1, 0)

        # Second emitted vertex (only for enter case)
        s2 = jnp.minimum(n_out, MAX_FV - 1)
        out = jnp.where(case_enter, out.at[s2].set(pj), out)
        n_out = n_out + jnp.where(case_enter, 1, 0)

        # Record intersection (exit or enter)
        has_inter = case_exit | case_enter
        si = jnp.minimum(n_int, 1)
        inters = jnp.where(has_inter, inters.at[si].set(inter), inters)
        n_int = n_int + jnp.where(has_inter, 1, 0)

        return (out, n_out, inters, n_int), None

    init = (
        jnp.zeros((MAX_FV, 3), jnp.float32),
        jnp.int32(0),
        jnp.zeros((2, 3), jnp.float32),
        jnp.int32(0),
    )
    (out_verts, out_nv, inter_pts, n_inter), _ = jax.lax.scan(
        step, init, jnp.arange(MAX_FV)
    )
    return out_verts, out_nv, inter_pts, n_inter


# ═══════════════════════════════════════════════════════════════
# Clip entire polyhedron by one half-space
# ═══════════════════════════════════════════════════════════════


def _clip_poly_by_plane(poly, plane_n, plane_d):
    """Clip convex polyhedron by half-space  n · x ≤ d.

    poly = (faces, face_nv, n_faces)
    Returns updated poly in the same format.
    """
    faces, face_nv, n_faces = poly

    # 1. vmap S-H clip over all faces
    clip_fn = lambda fv, fnv: _sh_clip_face(fv, fnv, plane_n, plane_d)
    clipped_faces, clipped_nv, inter_pts, n_inter = jax.vmap(clip_fn)(faces, face_nv)

    # 2. Collect intersection points for cap face
    face_active = jnp.arange(MAX_F) < n_faces
    has_inter = (n_inter > 0) & face_active

    # Flatten: (MAX_F*2, 3)
    all_inters = inter_pts.reshape(-1, 3)
    k_idx = jnp.tile(jnp.arange(2), MAX_F)
    f_idx = jnp.repeat(jnp.arange(MAX_F), 2)
    all_valid = has_inter[f_idx] & (k_idx < n_inter[f_idx])

    # 3. Deduplicate (vectorised pairwise distance, O(N²) with N=MAX_F*2=28)
    N = MAX_F * 2
    pdist2 = jnp.sum(
        (all_inters[:, None, :] - all_inters[None, :, :]) ** 2, axis=-1
    )  # (N, N)
    lower = jnp.tril(jnp.ones((N, N), dtype=bool), k=-1)
    is_dup = jnp.any(lower & all_valid[None, :] & (pdist2 < 1e-10), axis=1)
    unique_valid = all_valid & ~is_dup
    n_unique = jnp.sum(unique_valid).astype(jnp.int32)

    # 4. Polar sort unique cap vertices on the clipping plane
    cap_sum = jnp.sum(jnp.where(unique_valid[:, None], all_inters, 0.0), axis=0)
    cap_center = cap_sum / (jnp.float32(n_unique) + 1e-30)

    # Build local 2D axes on plane
    e1 = jnp.array([1.0, 0.0, 0.0])
    e1 = jnp.where(jnp.abs(jnp.dot(plane_n, e1)) > 0.9, jnp.array([0.0, 1.0, 0.0]), e1)
    e1 = e1 - jnp.dot(e1, plane_n) * plane_n
    e1 = e1 / (jnp.linalg.norm(e1) + 1e-30)
    e2 = jnp.cross(plane_n, e1)

    rel = all_inters - cap_center[None, :]
    angles = jnp.where(
        unique_valid,
        jnp.arctan2(jnp.sum(rel * e2[None, :], axis=1), jnp.sum(rel * e1[None, :], axis=1)),
        jnp.float32(1e10),
    )
    order = jnp.argsort(angles)

    cap_face = jnp.zeros((MAX_FV, 3), jnp.float32)
    # Gather first MAX_FV sorted unique points
    sorted_pts = all_inters[order]
    cap_face = cap_face.at[: min(N, MAX_FV)].set(sorted_pts[: min(N, MAX_FV)])
    cap_nv = jnp.minimum(n_unique, jnp.int32(MAX_FV))

    # 5. Assemble: clipped faces + cap
    add_cap = (n_unique >= 3).astype(jnp.int32)
    cap_slot = jnp.minimum(n_faces, jnp.int32(MAX_F - 1))
    new_faces = jnp.where(
        add_cap.astype(bool), clipped_faces.at[cap_slot].set(cap_face), clipped_faces
    )
    new_nv = jnp.where(
        add_cap.astype(bool), clipped_nv.at[cap_slot].set(cap_nv), clipped_nv
    )
    new_n_faces = n_faces + add_cap

    return new_faces, new_nv, new_n_faces


# ═══════════════════════════════════════════════════════════════
# Clip polyhedron by axis-aligned box (6 planes via scan)
# ═══════════════════════════════════════════════════════════════


def _clip_poly_by_box(poly, box_min, box_max):
    """Clip convex polyhedron by axis-aligned box [box_min, box_max].

    box_min, box_max: (3,) float32
    """
    normals = jnp.array(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
        dtype=jnp.float32,
    )
    dvals = jnp.array(
        [-box_min[0], box_max[0], -box_min[1], box_max[1], -box_min[2], box_max[2]],
        dtype=jnp.float32,
    )

    def step(poly, inp):
        n, d = inp[0], inp[1]
        return _clip_poly_by_plane(poly, n, d), None

    planes = jnp.stack([normals, jnp.broadcast_to(dvals[:, None], (6, 3))], axis=1)
    # Hmm, normals and dvals have different shapes. Let me use a different approach.
    poly_out, _ = jax.lax.scan(
        lambda p, i: (_clip_poly_by_plane(p, normals[i], dvals[i]), None),
        poly,
        jnp.arange(6),
    )
    return poly_out


# ═══════════════════════════════════════════════════════════════
# Volume via divergence theorem
# ═══════════════════════════════════════════════════════════════


def _poly_volume(poly):
    """Volume of closed convex polyhedron via divergence theorem.

    V = (1/6) |Σ_faces Σ_tris v_a · (v_b × v_c)|

    Fan triangulation: face f → triangles (f[0], f[t+1], f[t+2]) for t=0..nv-3.

    All vertices are shifted to local coordinates (relative to faces[0,0])
    before computing triple products.  The divergence-theorem volume is
    translation-invariant for closed polyhedra, so the result is unchanged.
    Without this shift, float32 cancellation is catastrophic: vertices at
    ~0.5 with cell volume ~1e-6 means 5 orders of magnitude must cancel.
    """
    faces, face_nv, n_faces = poly

    # Shift to local coordinates: keeps values O(dx) instead of O(domain)
    ref = faces[0, 0, :]
    local = faces - ref[None, None, :]

    v0 = local[:, 0, :]  # (MAX_F, 3) — fan center

    vb = local[:, 1 : MAX_FV - 1, :]
    vc = local[:, 2:MAX_FV, :]

    cross = jnp.cross(vb, vc)  # (MAX_F, MAX_FV-2, 3)
    triple = jnp.sum(v0[:, None, :] * cross, axis=-1)  # (MAX_F, MAX_FV-2)

    tri_idx = jnp.arange(MAX_FV - 2)
    face_active = jnp.arange(MAX_F)[:, None] < n_faces
    tri_active = tri_idx[None, :] < (face_nv[:, None] - 2)
    mask = face_active & tri_active

    return jnp.abs(jnp.sum(jnp.where(mask, triple, 0.0))) / 6.0


# ═══════════════════════════════════════════════════════════════
# Analytic tet volume (fast path for enclosed tets)
# ═══════════════════════════════════════════════════════════════


def _analytic_vol4(tet4):
    """Volume of tetrahedron with 4 vertices."""
    a = tet4[1] - tet4[0]
    b = tet4[2] - tet4[0]
    c = tet4[3] - tet4[0]
    det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
    return jnp.abs(det) / 6.0


# ═══════════════════════════════════════════════════════════════
# Hex-box volume (the main per-cell computation)
# ═══════════════════════════════════════════════════════════════


def _hex_box_volume(hex_verts, box_min, box_max):
    """Volume of (deformed hex) ∩ (axis-aligned box).  No PLIC."""
    # Shift to local coordinates (ref = first vertex of hex) so ALL downstream
    # operations (SH clip distances, intersections, volumes) use O(dx) values
    # instead of O(domain), dramatically improving float32 precision.
    ref = hex_verts[0]
    hv = hex_verts - ref[None, :]
    blo = box_min - ref
    bhi = box_max - ref

    inside = (
        (hv[:, 0] >= blo[0] - _EPS_BOX)
        & (hv[:, 0] <= bhi[0] + _EPS_BOX)
        & (hv[:, 1] >= blo[1] - _EPS_BOX)
        & (hv[:, 1] <= bhi[1] + _EPS_BOX)
        & (hv[:, 2] >= blo[2] - _EPS_BOX)
        & (hv[:, 2] <= bhi[2] + _EPS_BOX)
    )
    all_inside = jnp.all(inside)
    all_outside = (
        (jnp.max(hv[:, 0]) <= blo[0] + _EPS_BOX)
        | (jnp.min(hv[:, 0]) >= bhi[0] - _EPS_BOX)
        | (jnp.max(hv[:, 1]) <= blo[1] + _EPS_BOX)
        | (jnp.min(hv[:, 1]) >= bhi[1] - _EPS_BOX)
        | (jnp.max(hv[:, 2]) <= blo[2] + _EPS_BOX)
        | (jnp.min(hv[:, 2]) >= bhi[2] - _EPS_BOX)
    )
    poly = _init_hex_poly(hv)
    hex_vol = _poly_volume(poly)
    clipped = _clip_poly_by_box(poly, blo, bhi)
    clip_vol = _poly_volume(clipped)
    return jnp.where(all_inside, hex_vol, jnp.where(all_outside, 0.0, clip_vol))


def _hex_plic_box_volume(hex_verts, plic_n, plic_d, F_val, box_min, box_max):
    """Volume of (hex ∩ PLIC fluid half-space) ∩ box, AND fluid poly volume.

    Returns: (overlap_vol, fluid_vol) — both scalars.
    """
    # Shift to local coordinates for float32 precision
    ref = hex_verts[0]
    hv = hex_verts - ref[None, :]
    blo = box_min - ref
    bhi = box_max - ref
    pd = plic_d - jnp.dot(plic_n, ref)

    all_outside = (
        (jnp.max(hv[:, 0]) <= blo[0] + _EPS_BOX)
        | (jnp.min(hv[:, 0]) >= bhi[0] - _EPS_BOX)
        | (jnp.max(hv[:, 1]) <= blo[1] + _EPS_BOX)
        | (jnp.min(hv[:, 1]) >= bhi[1] - _EPS_BOX)
        | (jnp.max(hv[:, 2]) <= blo[2] + _EPS_BOX)
        | (jnp.min(hv[:, 2]) >= bhi[2] - _EPS_BOX)
    )
    is_empty = F_val < 1e-8
    skip = all_outside | is_empty

    poly = _init_hex_poly(hv)

    # PLIC clip → fluid polyhedron
    fluid_poly = _clip_poly_by_plane(poly, plic_n, pd)
    fluid_vol = _poly_volume(fluid_poly)

    # Box clip (using local coordinates)
    clipped = _clip_poly_by_box(fluid_poly, blo, bhi)
    overlap_vol = _poly_volume(clipped)

    return (jnp.where(skip, 0.0, overlap_vol),
            jnp.where(skip, 1e-30, fluid_vol))


# ═══════════════════════════════════════════════════════════════
# Tet-box volume (backward compat with existing 6-tet approach)
# ═══════════════════════════════════════════════════════════════


def _clip_tet_by_box(tet4, box):
    """Volume of tet ∩ box.  box = [xlo, xhi, ylo, yhi, zlo, zhi]."""
    box_min = jnp.array([box[0], box[2], box[4]])
    box_max = jnp.array([box[1], box[3], box[5]])

    analytic = _analytic_vol4(tet4)
    inside = (
        (tet4[:, 0] >= box_min[0] - _EPS)
        & (tet4[:, 0] <= box_max[0] + _EPS)
        & (tet4[:, 1] >= box_min[1] - _EPS)
        & (tet4[:, 1] <= box_max[1] + _EPS)
        & (tet4[:, 2] >= box_min[2] - _EPS)
        & (tet4[:, 2] <= box_max[2] + _EPS)
    )
    all_inside = jnp.all(inside)
    all_outside = (
        jnp.all(tet4[:, 0] < box_min[0] - _EPS)
        | jnp.all(tet4[:, 0] > box_max[0] + _EPS)
        | jnp.all(tet4[:, 1] < box_min[1] - _EPS)
        | jnp.all(tet4[:, 1] > box_max[1] + _EPS)
        | jnp.all(tet4[:, 2] < box_min[2] - _EPS)
        | jnp.all(tet4[:, 2] > box_max[2] + _EPS)
    )

    poly = _init_tet_poly(tet4)
    clipped = _clip_poly_by_box(poly, box_min, box_max)
    clip_vol = _poly_volume(clipped)

    return jnp.where(all_inside, analytic, jnp.where(all_outside, 0.0, clip_vol))


_clip_tet_by_box_jit = jax.jit(_clip_tet_by_box)
_clip_6tets = jax.vmap(_clip_tet_by_box, in_axes=(0, None))


def clip_tets_to_box(tets_6x4x3, box_bounds):
    return _clip_6tets(tets_6x4x3, box_bounds)


# ═══════════════════════════════════════════════════════════════
# FAST PATH: Tet-pool analytical splitting (no polyhedron overhead)
# ═══════════════════════════════════════════════════════════════

POOL = 10  # max sub-tets per original tet (tight but sufficient for 6 axis-aligned clips)


def _split_tet_by_plane(tet4, plane_n, plane_d):
    """Split tet by half-space n·x ≤ d → up to 3 sub-tets.

    Returns: (3, 4, 3) sub-tets and (3,) validity flags.
    All cases handled branchlessly via jnp.where.
    """
    dist = jnp.sum(tet4 * plane_n[None, :], axis=1) - plane_d  # (4,)
    inside = dist <= _EPS
    n_in = jnp.sum(inside).astype(jnp.int32)

    # All 6 edge intersection points (always computed, cheap)
    ei = jnp.array([0, 0, 0, 1, 1, 2])
    ej = jnp.array([1, 2, 3, 2, 3, 3])
    di_e, dj_e = dist[ei], dist[ej]
    sd = jnp.where(jnp.abs(di_e - dj_e) < 1e-30, 1e-30, di_e - dj_e)
    t = jnp.clip(di_e / sd, 0.0, 1.0)
    P = tet4[ei] + t[:, None] * (tet4[ej] - tet4[ei])  # (6, 3)
    # Edge map: (0,1)=0  (0,2)=1  (0,3)=2  (1,2)=3  (1,3)=4  (2,3)=5

    v = tet4
    empty = jnp.zeros((4, 3), jnp.float32)

    # ── n_in = 1: one vertex inside → 1 sub-tet ──
    # Inside vertex k, edges from k to the other 3 produce P_k*
    t1_v0 = jnp.stack([v[0], P[0], P[1], P[2]])  # edges 01,02,03
    t1_v1 = jnp.stack([v[1], P[0], P[3], P[4]])  # edges 10,12,13
    t1_v2 = jnp.stack([v[2], P[1], P[3], P[5]])  # edges 20,21,23
    t1_v3 = jnp.stack([v[3], P[2], P[4], P[5]])  # edges 30,31,32
    tet_1 = jnp.where(inside[0], t1_v0,
            jnp.where(inside[1], t1_v1,
            jnp.where(inside[2], t1_v2, t1_v3)))

    # ── n_in = 3: one vertex outside → 3 sub-tets (truncated corner) ──
    # Outside vertex k removed. Inside verts = {a,b,c}. Intersections on edges ka,kb,kc.
    def _tets_for_3in(k, a, b, c, Pka, Pkb, Pkc):
        return jnp.stack([
            jnp.stack([v[a], v[b], v[c], Pka]),
            jnp.stack([v[b], v[c], Pka, Pkb]),
            jnp.stack([v[c], Pka, Pkb, Pkc]),
        ])  # (3, 4, 3)

    tets_3_k0 = _tets_for_3in(0, 1, 2, 3, P[0], P[1], P[2])  # k=0 out
    tets_3_k1 = _tets_for_3in(1, 0, 2, 3, P[0], P[3], P[4])  # k=1 out
    tets_3_k2 = _tets_for_3in(2, 0, 1, 3, P[1], P[3], P[5])  # k=2 out
    tets_3_k3 = _tets_for_3in(3, 0, 1, 2, P[2], P[4], P[5])  # k=3 out

    tets_3 = jnp.where(~inside[0], tets_3_k0,
             jnp.where(~inside[1], tets_3_k1,
             jnp.where(~inside[2], tets_3_k2, tets_3_k3)))

    # ── n_in = 2: two vertices inside → 3 sub-tets (wedge) ──
    # Inside pair (a,b), outside pair (c,d).
    # Intersections: Pac, Pad, Pbc, Pbd.
    def _edge_idx(i, j):
        """Return the index in the 6-edge array for edge (i,j), i<j."""
        return jnp.where(
            (i == 0) & (j == 1), 0,
            jnp.where((i == 0) & (j == 2), 1,
            jnp.where((i == 0) & (j == 3), 2,
            jnp.where((i == 1) & (j == 2), 3,
            jnp.where((i == 1) & (j == 3), 4, 5)))))

    def _tets_for_2in(a, b, c, d):
        i_ac = _edge_idx(jnp.minimum(a, c), jnp.maximum(a, c))
        i_ad = _edge_idx(jnp.minimum(a, d), jnp.maximum(a, d))
        i_bc = _edge_idx(jnp.minimum(b, c), jnp.maximum(b, c))
        i_bd = _edge_idx(jnp.minimum(b, d), jnp.maximum(b, d))
        Pac, Pad, Pbc, Pbd = P[i_ac], P[i_ad], P[i_bc], P[i_bd]
        return jnp.stack([
            jnp.stack([v[a], v[b], Pac, Pad]),
            jnp.stack([v[b], Pac, Pad, Pbc]),
            jnp.stack([Pad, Pac, Pbc, Pbd]),
        ])

    # 6 possible inside-pairs:
    pair_01 = inside[0] & inside[1] & ~inside[2] & ~inside[3]
    pair_02 = inside[0] & inside[2] & ~inside[1] & ~inside[3]
    pair_03 = inside[0] & inside[3] & ~inside[1] & ~inside[2]
    pair_12 = inside[1] & inside[2] & ~inside[0] & ~inside[3]
    pair_13 = inside[1] & inside[3] & ~inside[0] & ~inside[2]
    pair_23 = inside[2] & inside[3] & ~inside[0] & ~inside[1]

    tets_2 = jnp.where(pair_01, _tets_for_2in(0, 1, 2, 3),
             jnp.where(pair_02, _tets_for_2in(0, 2, 1, 3),
             jnp.where(pair_03, _tets_for_2in(0, 3, 1, 2),
             jnp.where(pair_12, _tets_for_2in(1, 2, 0, 3),
             jnp.where(pair_13, _tets_for_2in(1, 3, 0, 2),
                                _tets_for_2in(2, 3, 0, 1))))))

    # ── Assemble output (3, 4, 3) ──
    out = jnp.where(
        n_in == 0, jnp.stack([empty, empty, empty]),
        jnp.where(n_in == 4, jnp.stack([tet4, empty, empty]),
        jnp.where(n_in == 1, jnp.stack([tet_1, empty, empty]),
        jnp.where(n_in == 3, tets_3, tets_2))))

    out_valid = jnp.where(
        n_in == 0, jnp.array([False, False, False]),
        jnp.where(n_in == 4, jnp.array([True, False, False]),
        jnp.where(n_in == 1, jnp.array([True, False, False]),
        jnp.array([True, True, True]))))  # n_in = 2 or 3

    return out, out_valid


def _clip_pool_by_plane(pool, valid, plane_n, plane_d):
    """Clip all tets in pool by one half-space.

    pool:  (POOL, 4, 3)
    valid: (POOL,)
    Returns: updated (pool, valid) with sub-tets compacted via prefix-sum.
    """
    # Split each tet → up to 3 sub-tets
    sub_tets, sub_valid = jax.vmap(
        lambda t, v: _split_tet_by_plane(t, plane_n, plane_d)
    )(pool, valid)
    sub_valid = sub_valid & valid[:, None]

    # Flatten: (POOL*3, 4, 3) and (POOL*3,)
    flat_tets = sub_tets.reshape(-1, 4, 3)
    flat_valid = sub_valid.reshape(-1)

    # Compact via prefix-sum + sequential scatter (cheaper than argsort)
    positions = jnp.cumsum(flat_valid.astype(jnp.int32)) - 1

    new_pool = jnp.zeros((POOL, 4, 3), jnp.float32)
    new_valid = jnp.zeros(POOL, dtype=bool)

    def _scatter(carry, idx):
        np_, nv_ = carry
        pos = jnp.minimum(positions[idx], POOL - 1)
        is_v = flat_valid[idx]
        np_ = jnp.where(is_v, np_.at[pos].set(flat_tets[idx]), np_)
        nv_ = jnp.where(is_v, nv_.at[pos].set(True), nv_)
        return (np_, nv_), None

    (new_pool, new_valid), _ = jax.lax.scan(
        _scatter, (new_pool, new_valid), jnp.arange(POOL * 3)
    )
    return new_pool, new_valid


def _tet_box_vol_fast(tet4, box_min, box_max):
    """Volume of tet ∩ box via tet-pool splitting (fast path)."""
    pool = jnp.zeros((POOL, 4, 3), jnp.float32).at[0].set(tet4)
    valid = jnp.zeros(POOL, dtype=bool).at[0].set(True)

    norms = jnp.array(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
        dtype=jnp.float32,
    )
    dvals = jnp.array(
        [-box_min[0], box_max[0], -box_min[1], box_max[1], -box_min[2], box_max[2]],
        dtype=jnp.float32,
    )

    def step(state, i):
        p, v = state
        p, v = _clip_pool_by_plane(p, v, norms[i], dvals[i])
        return (p, v), None

    (pool, valid), _ = jax.lax.scan(step, (pool, valid), jnp.arange(6))

    vols = jax.vmap(_analytic_vol4)(pool)
    return jnp.sum(jnp.where(valid, vols, 0.0))


def _hex_box_volume_tetpool(hex_verts, box_min, box_max):
    """Volume of hex ∩ box via 6-tet decomposition + tet-pool clipping (experimental)."""
    from .move_3d import TET_INDICES

    # Fast paths
    inside = (
        (hex_verts[:, 0] >= box_min[0] - _EPS)
        & (hex_verts[:, 0] <= box_max[0] + _EPS)
        & (hex_verts[:, 1] >= box_min[1] - _EPS)
        & (hex_verts[:, 1] <= box_max[1] + _EPS)
        & (hex_verts[:, 2] >= box_min[2] - _EPS)
        & (hex_verts[:, 2] <= box_max[2] + _EPS)
    )
    all_inside = jnp.all(inside)
    all_outside = (
        jnp.all(hex_verts[:, 0] < box_min[0] - _EPS)
        | jnp.all(hex_verts[:, 0] > box_max[0] + _EPS)
        | jnp.all(hex_verts[:, 1] < box_min[1] - _EPS)
        | jnp.all(hex_verts[:, 1] > box_max[1] + _EPS)
        | jnp.all(hex_verts[:, 2] < box_min[2] - _EPS)
        | jnp.all(hex_verts[:, 2] > box_max[2] + _EPS)
    )

    # Decompose hex into 6 tets
    tets = hex_verts[TET_INDICES]  # (6, 4, 3)
    hex_vol = jnp.sum(jax.vmap(_analytic_vol4)(tets))

    # Clip each tet independently
    clip_vol = jnp.sum(jax.vmap(
        lambda t: _tet_box_vol_fast(t, box_min, box_max)
    )(tets))

    return jnp.where(all_inside, hex_vol, jnp.where(all_outside, 0.0, clip_vol))


# ═══════════════════════════════════════════════════════════════
# Overlay function (tet-pool fast path)
# ═══════════════════════════════════════════════════════════════


def overlay_lagrangian_3d(
    F, x_verts, y_verts, z_verts, grid, nx_f=None, ny_f=None, nz_f=None, C_f=None
):
    """Lagrangian overlay with PLIC-in-overlay (Barkhudarov proper).

    When PLIC data (nx_f, ny_f, nz_f, C_f) is provided:
      For each donor cell, clip hex by PLIC plane FIRST → fluid polyhedron,
      then clip against acceptor box → overlap IS the fluid volume transferred.

    Without PLIC: falls back to dF × geometric_overlap (diffusive).
    """
    from .move_3d import build_deformed_hexahedra

    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    hex_v = build_deformed_hexahedra(x_verts, y_verts, z_verts, grid)

    x_acc = jnp.linspace(grid.x_range[0], grid.x_range[1], nx + 1)
    y_acc = jnp.linspace(grid.y_range[0], grid.y_range[1], ny + 1)
    z_acc = jnp.linspace(grid.z_range[0], grid.z_range[1], nz + 1)

    F_int = F[nh : nh + nx, nh : nh + ny, nh : nh + nz, 0]
    cell_vol = dx * dy * dz

    use_plic = nx_f is not None

    if use_plic:
        # Build PLIC clip plane parameters for each interior cell:
        #   plane_n = (nx, ny, nz)  (unit normal, fluid→empty)
        #   plane_d = C + n · hex_centroid
        # For full cells (F≈1): set d = +1e6 → PLIC clips nothing
        # For empty cells (F≈0): set d = -1e6 → PLIC clips everything
        nx_int = nx_f[nh : nh + nx, nh : nh + ny, nh : nh + nz, 0]
        ny_int = ny_f[nh : nh + nx, nh : nh + ny, nh : nh + nz, 0]
        nz_int = nz_f[nh : nh + nx, nh : nh + ny, nh : nh + nz, 0]
        C_int = C_f[nh : nh + nx, nh : nh + ny, nh : nh + nz, 0]

        # C from compute_intercept_C_3d is CENTER-RELATIVE (see _volume_below_3d
        # line 150: "d = C_phys + 0.5*(|a|+|b|+|c|)" maps center-relative C
        # to unit-cube coordinate).  So d_global = C + n · centroid.
        hex_cx = jnp.mean(hex_v[..., 0], axis=-1)
        hex_cy = jnp.mean(hex_v[..., 1], axis=-1)
        hex_cz = jnp.mean(hex_v[..., 2], axis=-1)

        plic_d_int = C_int + nx_int * hex_cx + ny_int * hex_cy + nz_int * hex_cz

        # Handle full/empty: override d to make PLIC inactive
        is_full = F_int > 1.0 - 1e-6
        is_empty = F_int < 1e-6
        plic_d_int = jnp.where(is_full, 1e6, jnp.where(is_empty, -1e6, plic_d_int))

        # Stack PLIC normals: (Nx, Ny, Nz, 3)
        plic_n_int = jnp.stack([nx_int, ny_int, nz_int], axis=-1)
        # For full/empty: set normal to (0,0,1) (arbitrary, d handles it)
        has_normal = ~is_full & ~is_empty
        plic_n_int = jnp.where(has_normal[..., None], plic_n_int,
                               jnp.array([0.0, 0.0, 1.0]))

        _overlap_vvv = jax.vmap(jax.vmap(jax.vmap(_hex_plic_box_volume)))
    else:
        _overlap_vvv = jax.vmap(jax.vmap(jax.vmap(_hex_box_volume)))

    # Pre-pad once
    hp = jnp.pad(hex_v, ((1, 1), (1, 1), (1, 1), (0, 0), (0, 0)))
    Fp = jnp.pad(F_int, ((1, 1), (1, 1), (1, 1)))

    ai, aj, ak = jnp.meshgrid(
        jnp.arange(nx), jnp.arange(ny), jnp.arange(nz), indexing="ij"
    )
    bmin = jnp.stack([x_acc[ai], y_acc[aj], z_acc[ak]], axis=-1)
    bmax = jnp.stack([x_acc[ai + 1], y_acc[aj + 1], z_acc[ak + 1]], axis=-1)

    offsets = jnp.array(
        [[di, dj, dk] for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)],
        dtype=jnp.int32,
    )

    if use_plic:
        pn_pad = jnp.pad(plic_n_int, ((1, 1), (1, 1), (1, 1), (0, 0)))
        pd_pad = jnp.pad(plic_d_int, ((1, 1), (1, 1), (1, 1)))

    def _scan_body(F_acc, offset):
        di, dj, dk = offset[0], offset[1], offset[2]
        dh = jax.lax.dynamic_slice(hp, (1 - di, 1 - dj, 1 - dk, 0, 0), (nx, ny, nz, 8, 3))
        dF = jax.lax.dynamic_slice(Fp, (1 - di, 1 - dj, 1 - dk), (nx, ny, nz))

        if use_plic:
            d_pn = jax.lax.dynamic_slice(pn_pad, (1 - di, 1 - dj, 1 - dk, 0), (nx, ny, nz, 3))
            d_pd = jax.lax.dynamic_slice(pd_pad, (1 - di, 1 - dj, 1 - dk), (nx, ny, nz))
            overlap, fluid_vol = _overlap_vvv(dh, d_pn, d_pd, dF, bmin, bmax)
            contrib = dF * overlap / jnp.maximum(fluid_vol, jnp.float32(1e-20))
        else:
            overlap = _overlap_vvv(dh, bmin, bmax)
            contrib = dF * overlap / cell_vol

        return F_acc + contrib, None

    F_new, _ = jax.lax.scan(_scan_body, jnp.zeros((nx, ny, nz)), offsets)

    return jnp.clip(F_new, 0.0, 1.0)
