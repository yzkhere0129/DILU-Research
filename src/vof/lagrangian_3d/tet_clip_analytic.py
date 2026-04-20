"""Analytical Tet-Box volume — FULLY VECTORIZED, no trace-time loops on pool.

V3: all inner loops replaced with jax.vmap + vectorized scatter.
XLA graph: ~50 ops per tet_box_vol (was 2000+ in v2).
"""

import jax
import jax.numpy as jnp

_EPS = jnp.float32(1e-7)


def _tet_vol(v0, v1, v2, v3):
    a = v1-v0; b = v2-v0; c = v3-v0
    det = a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0])
    return jnp.abs(det) / 6.0


def _clip_one_tet(tet, plane_n, plane_d):
    """Clip one tet → up to 3 sub-tets. Pure arithmetic + 4-element argsort."""
    dist = tet @ plane_n - plane_d
    n_in = jnp.sum(dist <= _EPS).astype(jnp.int32)
    order = jnp.argsort(dist)  # 4-element sort: tiny sorting network
    ds = dist[order]; vs = tet[order]
    empty = jnp.zeros((4, 3), jnp.float32)

    def _t(i, j):
        di, dj = ds[i], ds[j]
        denom = di - dj
        safe = jnp.where(jnp.abs(denom) < _EPS, jnp.copysign(_EPS, denom+1e-30), denom)
        return jnp.clip(di / safe, 0.0, 1.0)

    t01=_t(0,1); t02=_t(0,2); t03=_t(0,3)
    t12=_t(1,2); t13=_t(1,3)
    t30=_t(3,0); t31=_t(3,1); t32=_t(3,2)

    P01=vs[0]+t01*(vs[1]-vs[0]); P02=vs[0]+t02*(vs[2]-vs[0]); P03=vs[0]+t03*(vs[3]-vs[0])
    P12=vs[1]+t12*(vs[2]-vs[1]); P13=vs[1]+t13*(vs[3]-vs[1])
    P30=vs[3]+t30*(vs[0]-vs[3]); P31=vs[3]+t31*(vs[1]-vs[3]); P32=vs[3]+t32*(vs[2]-vs[3])

    tet_1 = jnp.stack([vs[0], P01, P02, P03])
    tet_3a = jnp.stack([vs[0],vs[1],vs[2],P30])
    tet_3b = jnp.stack([vs[1],vs[2],P30,P31])
    tet_3c = jnp.stack([vs[2],P30,P31,P32])
    tet_2a = jnp.stack([vs[0],vs[1],P02,P03])
    tet_2b = jnp.stack([vs[1],P02,P03,P12])
    tet_2c = jnp.stack([P03,P12,P13,vs[1]])

    out = jnp.where(n_in==0, jnp.stack([empty,empty,empty]),
          jnp.where(n_in==4, jnp.stack([tet,empty,empty]),
          jnp.where(n_in==1, jnp.stack([tet_1,empty,empty]),
          jnp.where(n_in==3, jnp.stack([tet_3a,tet_3b,tet_3c]),
          jnp.stack([tet_2a,tet_2b,tet_2c])))))

    out_valid = jnp.where(n_in==0, jnp.array([False,False,False]),
                jnp.where(n_in==4, jnp.array([True,False,False]),
                jnp.where(n_in==1, jnp.array([True,False,False]),
                jnp.array([True,True,True]))))
    return out, out_valid  # (3, 4, 3), (3,)


# Vectorized clip over pool: one vmap call processes ALL pool entries
_clip_pool_vmap = jax.vmap(_clip_one_tet, in_axes=(0, None, None))


def _compact_pool(flat_tets, flat_valid, POOL):
    """Vectorized prefix-sum compaction. No Python loops, no argsort.

    Scatters valid entries to their cumsum positions using a junk slot
    to absorb invalid writes without data corruption.
    """
    positions = jnp.cumsum(flat_valid.astype(jnp.int32)) - 1
    N = flat_tets.shape[0]

    # Invalid entries write to slot POOL (junk slot, discarded afterwards)
    safe_pos = jnp.where(flat_valid & (positions < POOL), positions, POOL)

    # Scatter into POOL+1 array (extra junk slot absorbs invalid writes)
    new_pool = jnp.zeros((POOL + 1, 4, 3), jnp.float32).at[safe_pos].set(flat_tets)
    new_valid = jnp.zeros(POOL + 1, dtype=bool).at[safe_pos].set(
        flat_valid & (positions < POOL))

    return new_pool[:POOL], new_valid[:POOL]


POOL = 32  # module-level constant for consistency


def tet_box_vol(tet4, box_min, box_max):
    """Volume of tet ∩ axis-aligned box.

    FULLY VECTORIZED: vmap over pool + vectorized scatter compaction.
    Only 6 trace-time iterations (one per box face), each with ~10 XLA ops.
    """
    inside = (
        (tet4[:,0]>=box_min[0]-_EPS) & (tet4[:,0]<=box_max[0]+_EPS) &
        (tet4[:,1]>=box_min[1]-_EPS) & (tet4[:,1]<=box_max[1]+_EPS) &
        (tet4[:,2]>=box_min[2]-_EPS) & (tet4[:,2]<=box_max[2]+_EPS))
    all_inside = jnp.all(inside)
    all_outside = (
        jnp.all(tet4[:,0]<box_min[0]-_EPS) | jnp.all(tet4[:,0]>box_max[0]+_EPS) |
        jnp.all(tet4[:,1]<box_min[1]-_EPS) | jnp.all(tet4[:,1]>box_max[1]+_EPS) |
        jnp.all(tet4[:,2]<box_min[2]-_EPS) | jnp.all(tet4[:,2]>box_max[2]+_EPS))
    V_full = _tet_vol(tet4[0], tet4[1], tet4[2], tet4[3])

    norms = jnp.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1.]])
    dvals = jnp.array([-box_min[0],box_max[0],-box_min[1],box_max[1],-box_min[2],box_max[2]])

    pool = jnp.zeros((POOL, 4, 3), jnp.float32).at[0].set(tet4)
    valid = jnp.zeros(POOL, dtype=bool).at[0].set(True)

    for pi in range(6):  # 6 iterations only (trace-time unroll, acceptable)
        # ── VECTORIZED clip: one vmap over entire pool ──
        all_sub, all_sub_v = _clip_pool_vmap(pool, norms[pi], dvals[pi])
        # all_sub: (POOL, 3, 4, 3), all_sub_v: (POOL, 3)

        # Mask by parent validity
        all_sub_v = all_sub_v & valid[:, None]

        # ── VECTORIZED compact: flatten + prefix-sum scatter ──
        flat_tets = all_sub.reshape(POOL * 3, 4, 3)
        flat_valid = all_sub_v.reshape(POOL * 3)
        pool, valid = _compact_pool(flat_tets, flat_valid, POOL)

    # Sum volumes of valid sub-tets
    vols = jax.vmap(lambda t: _tet_vol(t[0], t[1], t[2], t[3]))(pool)
    clip_vol = jnp.sum(jnp.where(valid, vols, 0.0))

    return jnp.where(all_inside, V_full, jnp.where(all_outside, 0.0, clip_vol))


def hex_box_vol_analytic(hex_verts, box_min, box_max):
    """Volume of hex ∩ box via 6-tet decomposition + vectorized analytical clip."""
    from .move_3d import TET_INDICES
    inside = (
        (hex_verts[:,0]>=box_min[0]-_EPS) & (hex_verts[:,0]<=box_max[0]+_EPS) &
        (hex_verts[:,1]>=box_min[1]-_EPS) & (hex_verts[:,1]<=box_max[1]+_EPS) &
        (hex_verts[:,2]>=box_min[2]-_EPS) & (hex_verts[:,2]<=box_max[2]+_EPS))
    all_inside = jnp.all(inside)
    all_outside = (
        jnp.all(hex_verts[:,0]<box_min[0]-_EPS) | jnp.all(hex_verts[:,0]>box_max[0]+_EPS) |
        jnp.all(hex_verts[:,1]<box_min[1]-_EPS) | jnp.all(hex_verts[:,1]>box_max[1]+_EPS) |
        jnp.all(hex_verts[:,2]<box_min[2]-_EPS) | jnp.all(hex_verts[:,2]>box_max[2]+_EPS))

    tets = hex_verts[TET_INDICES]
    hex_vol = jnp.sum(jax.vmap(lambda t: _tet_vol(t[0],t[1],t[2],t[3]))(tets))
    clip_vol = jnp.sum(jax.vmap(lambda t: tet_box_vol(t, box_min, box_max))(tets))
    return jnp.where(all_inside, hex_vol, jnp.where(all_outside, 0.0, clip_vol))
