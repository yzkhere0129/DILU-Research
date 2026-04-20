/*
 * hex_box_clip.c — Native C/OpenMP kernel for Lagrangian VOF overlay.
 *
 * Replaces JAX S-H polyhedron clipping with:
 *   - 6-tet decomposition of hex
 *   - Sequential 6-plane analytical tet clipping (stack-allocated pool)
 *   - OpenMP parallel over cells, true conditional skips
 *   - double accumulator for volume conservation at scale
 *
 * Build: gcc -O3 -march=native -fopenmp -fPIC -shared -o libhexboxclip.so hex_box_clip.c -lm
 * NO -ffast-math: breaks IEEE 754 coplanar/near-degenerate geometry judgments.
 */

#include "hex_box_clip.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ══════════════════════════════════════════════════════════════
 * Constants
 * ══════════════════════════════════════════════════════════════ */

#define GEO_EPS   1e-6f   /* geometric inside/outside decisions (float32-safe) */
#define DENOM_EPS 1e-12f  /* denominator guard (div-by-zero prevention only) */
#define MAX_POOL  48      /* max sub-tets per original tet (stack-allocated) */

/*
 * 6-tet decomposition of a hexahedron.
 * Exact copy of move_3d.py TET_INDICES (lines 47-57).
 * Internal diagonals: (0,2), (0,4), (2,5).
 * Verified: sum of 6 tet volumes = hex volume for unit cube.
 *
 * NOTE: first-order geometric approximation — see header for caveat.
 */
static const int TET_INDICES[6][4] = {
    {0, 1, 2, 4},  /* T1 */
    {1, 2, 4, 5},  /* T2 */
    {2, 4, 5, 6},  /* T3 */
    {2, 3, 4, 7},  /* T4 */
    {2, 4, 6, 7},  /* T5 */
    {0, 2, 3, 4},  /* T6 */
};

/* 6 box clipping planes: -x, +x, -y, +y, -z, +z */
static const float BOX_NORMALS[6][3] = {
    {-1, 0, 0}, {1, 0, 0},
    {0, -1, 0}, {0, 1, 0},
    {0, 0, -1}, {0, 0, 1},
};

/* ══════════════════════════════════════════════════════════════
 * Helpers
 * ══════════════════════════════════════════════════════════════ */

static inline float dot3(const float a[3], const float b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline float tet_vol(const float v[4][3]) {
    float a[3], b[3], c[3];
    for (int d = 0; d < 3; d++) {
        a[d] = v[1][d] - v[0][d];
        b[d] = v[2][d] - v[0][d];
        c[d] = v[3][d] - v[0][d];
    }
    float det = a[0]*(b[1]*c[2] - b[2]*c[1])
              - a[1]*(b[0]*c[2] - b[2]*c[0])
              + a[2]*(b[0]*c[1] - b[1]*c[0]);
    return fabsf(det) / 6.0f;
}

static inline void lerp_v(float out[3], const float a[3], const float b[3],
                           float da, float db) {
    float denom = da - db;
    if (fabsf(denom) < DENOM_EPS)
        denom = (denom >= 0.0f) ? DENOM_EPS : -DENOM_EPS;
    float t = da / denom;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    for (int d = 0; d < 3; d++)
        out[d] = a[d] + t * (b[d] - a[d]);
}

/* ══════════════════════════════════════════════════════════════
 * Classify hex vs box
 * ══════════════════════════════════════════════════════════════ */

#define CLS_OUTSIDE 0
#define CLS_PARTIAL 1
#define CLS_INSIDE  2

static int classify_hex_box(const float hex[8][3],
                            const float bmin[3], const float bmax[3]) {
    int all_in = 1;
    for (int v = 0; v < 8; v++) {
        for (int d = 0; d < 3; d++) {
            if (hex[v][d] < bmin[d] - GEO_EPS || hex[v][d] > bmax[d] + GEO_EPS)
                all_in = 0;
        }
    }
    if (all_in) return CLS_INSIDE;

    /* Separating axis test (per-axis) */
    for (int d = 0; d < 3; d++) {
        int all_below = 1, all_above = 1;
        for (int v = 0; v < 8; v++) {
            if (hex[v][d] >= bmin[d] - GEO_EPS) all_below = 0;
            if (hex[v][d] <= bmax[d] + GEO_EPS) all_above = 0;
        }
        if (all_below || all_above) return CLS_OUTSIDE;
    }
    return CLS_PARTIAL;
}

/* ══════════════════════════════════════════════════════════════
 * Clip one tet by one half-space: n·x ≤ d
 *
 * Sorts 4 vertices by signed distance, produces 0-3 sub-tets.
 * Returns number of output tets written to out_pool.
 * ══════════════════════════════════════════════════════════════ */

static int clip_tet_by_plane(const float tet[4][3],
                             const float n[3], float d,
                             float out_pool[][4][3]) {
    /* Signed distances */
    float dist[4];
    for (int v = 0; v < 4; v++)
        dist[v] = dot3(tet[v], n) - d;

    /* Sort indices by distance (ascending: most-inside first) */
    int idx[4] = {0, 1, 2, 3};
    /* Simple insertion sort on 4 elements */
    for (int i = 1; i < 4; i++) {
        int key = idx[i];
        float kd = dist[key];
        int j = i - 1;
        while (j >= 0 && dist[idx[j]] > kd) {
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = key;
    }

    /* Count inside vertices (dist ≤ GEO_EPS) */
    int n_in = 0;
    for (int i = 0; i < 4; i++)
        if (dist[idx[i]] <= GEO_EPS) n_in++;

    if (n_in == 0) return 0;  /* all outside */
    if (n_in == 4) {           /* all inside: keep tet */
        memcpy(out_pool[0], tet, 4 * 3 * sizeof(float));
        return 1;
    }

    /* Sorted vertices and distances */
    const float *vs[4];
    float ds[4];
    for (int i = 0; i < 4; i++) {
        vs[i] = tet[idx[i]];
        ds[i] = dist[idx[i]];
    }

    /* Intersection points */
    float P01[3], P02[3], P03[3], P12[3], P13[3];
    float P30[3], P31[3], P32[3];

    if (n_in == 1) {
        /* Only vs[0] inside → 1 sub-tet */
        lerp_v(P01, vs[0], vs[1], ds[0], ds[1]);
        lerp_v(P02, vs[0], vs[2], ds[0], ds[2]);
        lerp_v(P03, vs[0], vs[3], ds[0], ds[3]);
        memcpy(out_pool[0][0], vs[0], 3 * sizeof(float));
        memcpy(out_pool[0][1], P01,   3 * sizeof(float));
        memcpy(out_pool[0][2], P02,   3 * sizeof(float));
        memcpy(out_pool[0][3], P03,   3 * sizeof(float));
        return 1;
    }

    if (n_in == 3) {
        /* vs[3] outside → 3 sub-tets (truncated corner) */
        lerp_v(P30, vs[3], vs[0], ds[3], ds[0]);
        lerp_v(P31, vs[3], vs[1], ds[3], ds[1]);
        lerp_v(P32, vs[3], vs[2], ds[3], ds[2]);

        /* T0: (vs[0], vs[1], vs[2], P30) */
        memcpy(out_pool[0][0], vs[0], 3 * sizeof(float));
        memcpy(out_pool[0][1], vs[1], 3 * sizeof(float));
        memcpy(out_pool[0][2], vs[2], 3 * sizeof(float));
        memcpy(out_pool[0][3], P30,   3 * sizeof(float));
        /* T1: (vs[1], vs[2], P30, P31) */
        memcpy(out_pool[1][0], vs[1], 3 * sizeof(float));
        memcpy(out_pool[1][1], vs[2], 3 * sizeof(float));
        memcpy(out_pool[1][2], P30,   3 * sizeof(float));
        memcpy(out_pool[1][3], P31,   3 * sizeof(float));
        /* T2: (vs[2], P30, P31, P32) */
        memcpy(out_pool[2][0], vs[2], 3 * sizeof(float));
        memcpy(out_pool[2][1], P30,   3 * sizeof(float));
        memcpy(out_pool[2][2], P31,   3 * sizeof(float));
        memcpy(out_pool[2][3], P32,   3 * sizeof(float));
        return 3;
    }

    /* n_in == 2: vs[0],vs[1] inside; vs[2],vs[3] outside → 3 sub-tets (wedge) */
    lerp_v(P02, vs[0], vs[2], ds[0], ds[2]);
    lerp_v(P03, vs[0], vs[3], ds[0], ds[3]);
    lerp_v(P12, vs[1], vs[2], ds[1], ds[2]);
    lerp_v(P13, vs[1], vs[3], ds[1], ds[3]);

    /* T0: (vs[0], vs[1], P02, P03) */
    memcpy(out_pool[0][0], vs[0], 3 * sizeof(float));
    memcpy(out_pool[0][1], vs[1], 3 * sizeof(float));
    memcpy(out_pool[0][2], P02,   3 * sizeof(float));
    memcpy(out_pool[0][3], P03,   3 * sizeof(float));
    /* T1: (vs[1], P02, P03, P12) */
    memcpy(out_pool[1][0], vs[1], 3 * sizeof(float));
    memcpy(out_pool[1][1], P02,   3 * sizeof(float));
    memcpy(out_pool[1][2], P03,   3 * sizeof(float));
    memcpy(out_pool[1][3], P12,   3 * sizeof(float));
    /* T2: (P03, P12, P13, vs[1]) */
    memcpy(out_pool[2][0], P03,   3 * sizeof(float));
    memcpy(out_pool[2][1], P12,   3 * sizeof(float));
    memcpy(out_pool[2][2], P13,   3 * sizeof(float));
    memcpy(out_pool[2][3], vs[1], 3 * sizeof(float));
    return 3;
}

/* ══════════════════════════════════════════════════════════════
 * Clip one tet by axis-aligned box (6 planes)
 * ══════════════════════════════════════════════════════════════ */

static float tet_box_vol_single(const float tet[4][3],
                                const float bmin[3], const float bmax[3]) {
    /* Quick AABB reject/accept for single tet */
    int all_in = 1;
    for (int v = 0; v < 4; v++)
        for (int d = 0; d < 3; d++)
            if (tet[v][d] < bmin[d] - GEO_EPS || tet[v][d] > bmax[d] + GEO_EPS)
                all_in = 0;
    if (all_in) return tet_vol(tet);

    /* Separating axis test */
    for (int d = 0; d < 3; d++) {
        int below = 1, above = 1;
        for (int v = 0; v < 4; v++) {
            if (tet[v][d] >= bmin[d] - GEO_EPS) below = 0;
            if (tet[v][d] <= bmax[d] + GEO_EPS) above = 0;
        }
        if (below || above) return 0.0f;
    }

    /* Stack-allocated double-buffered tet pool */
    float poolA[MAX_POOL][4][3];
    float poolB[MAX_POOL][4][3];
    int nA = 1;
    memcpy(poolA[0], tet, 4 * 3 * sizeof(float));

    float dvals[6] = {
        -bmin[0], bmax[0], -bmin[1], bmax[1], -bmin[2], bmax[2]
    };

    for (int pi = 0; pi < 6; pi++) {
        float (*src)[4][3] = (pi % 2 == 0) ? poolA : poolB;
        float (*dst)[4][3] = (pi % 2 == 0) ? poolB : poolA;
        int nSrc = nA;
        int nDst = 0;

        for (int t = 0; t < nSrc; t++) {
            int added = clip_tet_by_plane(src[t], BOX_NORMALS[pi], dvals[pi],
                                          &dst[nDst]);
            nDst += added;
            if (nDst >= MAX_POOL - 3) break;  /* safety: prevent overflow */
        }
        nA = nDst;
        if (nA == 0) return 0.0f;  /* all clipped away */
    }

    /* Sum surviving tet volumes */
    float (*final_pool)[4][3] = (5 % 2 == 0) ? poolB : poolA;  /* after 6 clips */
    float total = 0.0f;
    for (int t = 0; t < nA; t++)
        total += tet_vol(final_pool[t]);

    return total;
}

/* ══════════════════════════════════════════════════════════════
 * Hex-box volume via 6-tet decomposition
 * ══════════════════════════════════════════════════════════════ */

static float hex_box_vol_single(const float hex[8][3],
                                const float bmin[3], const float bmax[3]) {
    int cls = classify_hex_box(hex, bmin, bmax);
    if (cls == CLS_OUTSIDE) return 0.0f;

    /* Decompose hex into 6 tets */
    float tets[6][4][3];
    for (int t = 0; t < 6; t++)
        for (int v = 0; v < 4; v++)
            memcpy(tets[t][v], hex[TET_INDICES[t][v]], 3 * sizeof(float));

    float hex_vol = 0.0f;
    for (int t = 0; t < 6; t++)
        hex_vol += tet_vol(tets[t]);

    if (cls == CLS_INSIDE) return hex_vol;

    /* PARTIAL: clip each tet by box */
    float clip_vol = 0.0f;
    for (int t = 0; t < 6; t++)
        clip_vol += tet_box_vol_single(tets[t], bmin, bmax);

    return clip_vol;
}

/* ══════════════════════════════════════════════════════════════
 * Public API: batch hex-box volumes (for unit testing)
 * ══════════════════════════════════════════════════════════════ */

void hex_box_vol_batch(const float* hex_verts, const float* box_min,
                       const float* box_max, float* out_vols, int32_t N) {
    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < N; i++) {
        const float (*hex)[3] = (const float (*)[3])&hex_verts[i * 24];
        const float *blo = &box_min[i * 3];
        const float *bhi = &box_max[i * 3];
        out_vols[i] = hex_box_vol_single(hex, blo, bhi);
    }
}

/* ══════════════════════════════════════════════════════════════
 * Public API: full overlay with 27 offsets
 * ══════════════════════════════════════════════════════════════ */

void overlay_all_offsets(
    const float* donor_hexes,   /* (Px*Py*Pz*8*3) where Px=nx+2 etc. */
    const float* donor_F,       /* (Px*Py*Pz) */
    const float* box_min_grid,  /* (nx*ny*nz*3) */
    const float* box_max_grid,  /* (nx*ny*nz*3) */
    float        cell_vol,
    int32_t      nx,
    int32_t      ny,
    int32_t      nz,
    float*       out_F          /* (nx*ny*nz) */
) {
    int32_t N = nx * ny * nz;
    int32_t Py = ny + 2, Pz = nz + 2;

    /* Double accumulator to prevent big-eats-small at millions of cells */
    double* acc = (double*)calloc(N, sizeof(double));

    for (int di = -1; di <= 1; di++)
    for (int dj = -1; dj <= 1; dj++)
    for (int dk = -1; dk <= 1; dk++) {

        #pragma omp parallel for schedule(dynamic, 256)
        for (int32_t idx = 0; idx < N; idx++) {
            int i = idx / (ny * nz);
            int j = (idx / nz) % ny;
            int k = idx % nz;

            /* Donor cell in padded array */
            int si = i + 1 - di;
            int sj = j + 1 - dj;
            int sk = k + 1 - dk;

            int pad_idx = (si * Py + sj) * Pz + sk;
            float dF = donor_F[pad_idx];

            /* TRUE SKIP: empty donor contributes nothing */
            if (dF < GEO_EPS) continue;

            /* Extract hex vertices from padded array */
            float hex[8][3];
            int hex_base = pad_idx * 24;  /* 8 verts × 3 coords */
            for (int v = 0; v < 8; v++)
                for (int d = 0; d < 3; d++)
                    hex[v][d] = donor_hexes[hex_base + v * 3 + d];

            /* Acceptor box bounds */
            float bmin[3], bmax[3];
            for (int d = 0; d < 3; d++) {
                bmin[d] = box_min_grid[idx * 3 + d];
                bmax[d] = box_max_grid[idx * 3 + d];
            }

            float vol = hex_box_vol_single(hex, bmin, bmax);

            /* Atomic add (safe for future non-uniform grids) */
            #pragma omp atomic
            acc[idx] += (double)(dF * vol / cell_vol);
        }
    }

    /* Cast to float32 + clip to [0, 1] */
    for (int32_t i = 0; i < N; i++) {
        float val = (float)acc[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        out_F[i] = val;
    }

    free(acc);
}


/* ══════════════════════════════════════════════════════════════
 * PLIC Sutherland-Hodgman Overlay
 *
 * Adds PLIC support via coordinate-face polyhedron clipping.
 * See PLIC_SH_SPEC.md for the full specification.
 * ══════════════════════════════════════════════════════════════ */

#define POLY_MAX_F    14
#define POLY_MAX_FV   14
#define SH_EPS        1e-10f   /* match JAX _EPS for SH classification */
#define DEDUP_EPS2    1e-10f   /* match JAX dedup threshold (squared)  */
#define AABB_EPS      1e-6f
#define VOL_GUARD     1e-30f

typedef struct {
    float verts[POLY_MAX_F][POLY_MAX_FV][3];
    int   nv[POLY_MAX_F];
    int   nf;
} Poly;

static const int HEX_FACE_IDX[6][4] = {
    {0,3,2,1}, {4,5,6,7}, {0,1,5,4},
    {2,3,7,6}, {0,4,7,3}, {1,2,6,5}
};

/* ── Helpers ─────────────────────────────────────────────── */

static inline float dot3f(const float *a, const float *b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline void lerp3f(const float *a, const float *b, float t, float *out) {
    out[0] = a[0] + t * (b[0] - a[0]);
    out[1] = a[1] + t * (b[1] - a[1]);
    out[2] = a[2] + t * (b[2] - a[2]);
}

static inline void cross3f(const float *a, const float *b, float *out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static inline void copy3f(float *dst, const float *src) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
}

/* ── Function 1: init_hex_poly ───────────────────────────── */

static void init_hex_poly(Poly *p, const float hex[][3]) {
    p->nf = 6;
    for (int f = 0; f < 6; f++) {
        p->nv[f] = 4;
        for (int v = 0; v < 4; v++)
            copy3f(p->verts[f][v], hex[HEX_FACE_IDX[f][v]]);
    }
    for (int f = 6; f < POLY_MAX_F; f++)
        p->nv[f] = 0;
}

/* ── Function 2: clip_face_by_plane ──────────────────────── */

static int clip_face_by_plane(
    const float vin[][3], int n_in,
    const float n[3],     float d,
    float       vout[][3], int max_out,
    float       cap[][3],  int *n_cap, int max_cap)
{
    if (n_in < 3 || n_in > POLY_MAX_FV) return 0;

    float sd[POLY_MAX_FV];
    for (int i = 0; i < n_in; i++)
        sd[i] = dot3f(n, vin[i]) - d;

    int n_out = 0;
    for (int i = 0; i < n_in; i++) {
        int j = (i + 1) % n_in;
        float si = sd[i], sj = sd[j];

        if (si <= SH_EPS) {
            /* v_i INSIDE → emit it */
            if (n_out < max_out) copy3f(vout[n_out++], vin[i]);
            if (sj > SH_EPS) {
                /* crossing out → emit intersection + cap */
                float t = si / (si - sj);
                float p[3]; lerp3f(vin[i], vin[j], t, p);
                if (n_out < max_out) copy3f(vout[n_out++], p);
                if (*n_cap < max_cap) copy3f(cap[(*n_cap)++], p);
            }
        } else {
            if (sj <= SH_EPS) {
                /* crossing in → emit intersection + cap */
                float t = si / (si - sj);
                float p[3]; lerp3f(vin[i], vin[j], t, p);
                if (n_out < max_out) copy3f(vout[n_out++], p);
                if (*n_cap < max_cap) copy3f(cap[(*n_cap)++], p);
            }
        }
    }
    return n_out;
}

/* ── Function 3: clip_poly_by_plane ──────────────────────── */

static void clip_poly_by_plane(Poly *p, const float n[3], float d) {
    Poly tmp = {0};  /* zero-init prevents UB from uninitialized verts */
    /* Larger cap buffer: up to 2 intersections per face × max faces */
    float cap_buf[POLY_MAX_F * 2][3];
    int n_cap = 0;

    /* Clip each active face */
    int orig_nf = p->nf;
    for (int f = 0; f < orig_nf; f++) {
        if (p->nv[f] >= 3) {
            tmp.nv[f] = clip_face_by_plane(
                (const float (*)[3])p->verts[f], p->nv[f],
                n, d,
                tmp.verts[f], POLY_MAX_FV,
                cap_buf, &n_cap, POLY_MAX_F * 2);
        } else {
            tmp.nv[f] = 0;
        }
    }
    for (int f = orig_nf; f < POLY_MAX_F; f++)
        tmp.nv[f] = 0;
    tmp.nf = orig_nf;

    /* Copy clipped faces back */
    for (int f = 0; f < orig_nf; f++) {
        p->nv[f] = tmp.nv[f];
        if (tmp.nv[f] > 0)
            memcpy(p->verts[f], tmp.verts[f], tmp.nv[f] * 3 * sizeof(float));
    }

    if (n_cap < 3) return;  /* no valid cap face */

    /* Deduplicate cap vertices */
    int keep[POLY_MAX_F * 2];
    memset(keep, 0, sizeof(keep));  /* zero-init prevents UB */
    int n_unique = 0;
    for (int i = 0; i < n_cap; i++) {
        int dup = 0;
        for (int j = 0; j < i; j++) {
            if (!keep[j]) continue;
            float dx = cap_buf[i][0] - cap_buf[j][0];
            float dy = cap_buf[i][1] - cap_buf[j][1];
            float dz = cap_buf[i][2] - cap_buf[j][2];
            if (dx*dx + dy*dy + dz*dz < DEDUP_EPS2) { dup = 1; break; }
        }
        keep[i] = !dup;
        if (!dup) n_unique++;
    }
    if (n_unique < 3) return;

    /* Compact unique points */
    float upts[POLY_MAX_FV][3];
    int nu = 0;
    for (int i = 0; i < n_cap && nu < POLY_MAX_FV; i++)
        if (keep[i]) copy3f(upts[nu++], cap_buf[i]);

    /* Polar sort: build in-plane frame */
    /* Centroid */
    float cx = 0, cy = 0, cz = 0;
    for (int i = 0; i < nu; i++) { cx += upts[i][0]; cy += upts[i][1]; cz += upts[i][2]; }
    cx /= nu; cy /= nu; cz /= nu;

    /* Normalize n (guard against zero-length degenerate normals) */
    float nlen = sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    if (nlen < 1e-12f) nlen = 1.0f;
    float nn[3] = {n[0]/nlen, n[1]/nlen, n[2]/nlen};

    /* Gram-Schmidt for e1, e2 (in-plane orthonormal basis) */
    float e1[3] = {1, 0, 0};
    if (fabsf(dot3f(nn, e1)) > 0.9f) { e1[0] = 0; e1[1] = 1; }
    float proj = dot3f(nn, e1);
    e1[0] -= proj * nn[0]; e1[1] -= proj * nn[1]; e1[2] -= proj * nn[2];
    float e1len = sqrtf(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
    if (e1len < 1e-12f) { e1[0] = 0; e1[1] = 0; e1[2] = 1; e1len = 1.0f; }
    e1[0] /= e1len; e1[1] /= e1len; e1[2] /= e1len;
    float e2[3]; cross3f(nn, e1, e2);

    /* Compute angles and sort (insertion sort) */
    float angles[POLY_MAX_FV];
    for (int i = 0; i < nu; i++) {
        float rel[3] = {upts[i][0]-cx, upts[i][1]-cy, upts[i][2]-cz};
        angles[i] = atan2f(dot3f(rel, e2), dot3f(rel, e1));
    }
    for (int i = 1; i < nu; i++) {
        float ka = angles[i];
        float kv[3]; copy3f(kv, upts[i]);
        int j = i - 1;
        while (j >= 0 && angles[j] > ka) {
            angles[j+1] = angles[j];
            copy3f(upts[j+1], upts[j]);
            j--;
        }
        angles[j+1] = ka;
        copy3f(upts[j+1], kv);
    }

    /* Add cap face */
    int slot = p->nf;
    if (slot >= POLY_MAX_F) return;
    p->nv[slot] = nu;
    for (int i = 0; i < nu; i++)
        copy3f(p->verts[slot][i], upts[i]);
    p->nf++;
}

/* ── Function 4: poly_volume ─────────────────────────────── */

static float poly_volume(const Poly *p) {
    double sum = 0.0;
    for (int f = 0; f < p->nf; f++) {
        if (p->nv[f] < 3) continue;
        const float *v0 = p->verts[f][0];
        for (int j = 1; j <= p->nv[f] - 2; j++) {
            const float *vb = p->verts[f][j];
            const float *vc = p->verts[f][j + 1];
            float cr[3]; cross3f(vb, vc, cr);
            sum += (double)dot3f(v0, cr);
        }
    }
    return fabsf((float)(sum / 6.0));
}

/* ── Function 5: hex_plic_box_volume ─────────────────────── */

static void hex_plic_box_volume(
    const float hex[8][3],
    const float plic_n[3], float plic_d,
    float F_donor,
    const float box_min[3], const float box_max[3],
    float *out_overlap, float *out_fluid)
{
    /* 1. Local coordinate shift */
    float hv[8][3], blo[3], bhi[3];
    for (int v = 0; v < 8; v++)
        for (int d = 0; d < 3; d++)
            hv[v][d] = hex[v][d] - hex[0][d];
    for (int d = 0; d < 3; d++) {
        blo[d] = box_min[d] - hex[0][d];
        bhi[d] = box_max[d] - hex[0][d];
    }
    float pd = plic_d - dot3f(plic_n, hex[0]);

    /* 2. AABB separating-axis test */
    for (int a = 0; a < 3; a++) {
        float mn = hv[0][a], mx = hv[0][a];
        for (int v = 1; v < 8; v++) {
            if (hv[v][a] < mn) mn = hv[v][a];
            if (hv[v][a] > mx) mx = hv[v][a];
        }
        if (mx <= blo[a] + AABB_EPS || mn >= bhi[a] - AABB_EPS) {
            *out_overlap = 0; *out_fluid = VOL_GUARD; return;
        }
    }

    /* 3. Empty donor skip */
    if (F_donor < 1e-8f) {
        *out_overlap = 0; *out_fluid = VOL_GUARD; return;
    }

    /* 4. Init polyhedron */
    Poly poly;
    init_hex_poly(&poly, (const float (*)[3])hv);

    /* 5. PLIC clip → fluid polyhedron */
    clip_poly_by_plane(&poly, plic_n, pd);
    *out_fluid = poly_volume(&poly);
    if (*out_fluid < VOL_GUARD) *out_fluid = VOL_GUARD;

    /* 6. Box clip (6 sequential planes) */
    static const float bn[6][3] = {
        {-1,0,0},{1,0,0}, {0,-1,0},{0,1,0}, {0,0,-1},{0,0,1}
    };
    float bd[6] = {-blo[0], bhi[0], -blo[1], bhi[1], -blo[2], bhi[2]};
    for (int i = 0; i < 6; i++)
        clip_poly_by_plane(&poly, bn[i], bd[i]);

    /* 7. Overlap volume */
    *out_overlap = poly_volume(&poly);
}

/* ── Function 6: overlay_all_offsets_plic ────────────────── */

void overlay_all_offsets_plic(
    const float *donor_hexes,
    const float *donor_F,
    const float *donor_plic_n,
    const float *donor_plic_d,
    const float *box_min_grid,
    const float *box_max_grid,
    int32_t nx, int32_t ny, int32_t nz,
    float *out_F)
{
    const int32_t N  = nx * ny * nz;
    const int32_t Py = ny + 2, Pz = nz + 2;

    double *acc = (double *)calloc(N, sizeof(double));
    if (!acc) return;

    for (int di = -1; di <= 1; di++)
    for (int dj = -1; dj <= 1; dj++)
    for (int dk = -1; dk <= 1; dk++) {

        #pragma omp parallel for schedule(dynamic, 256)
        for (int32_t idx = 0; idx < N; idx++) {
            int i = idx / (ny * nz);
            int j = (idx / nz) % ny;
            int k = idx % nz;

            int si = i + 1 - di;
            int sj = j + 1 - dj;
            int sk = k + 1 - dk;
            int donor_idx = (si * Py + sj) * Pz + sk;

            float dF = donor_F[donor_idx];
            if (dF < 1e-8f) continue;

            /* Extract hex vertices */
            float hex[8][3];
            const float *hp = &donor_hexes[donor_idx * 24];
            for (int v = 0; v < 8; v++)
                for (int d = 0; d < 3; d++)
                    hex[v][d] = hp[v * 3 + d];

            /* Extract PLIC data */
            float pn[3] = {
                donor_plic_n[donor_idx * 3 + 0],
                donor_plic_n[donor_idx * 3 + 1],
                donor_plic_n[donor_idx * 3 + 2]
            };
            float pd = donor_plic_d[donor_idx];

            /* Acceptor box */
            float bmin[3] = {
                box_min_grid[idx * 3 + 0],
                box_min_grid[idx * 3 + 1],
                box_min_grid[idx * 3 + 2]
            };
            float bmax[3] = {
                box_max_grid[idx * 3 + 0],
                box_max_grid[idx * 3 + 1],
                box_max_grid[idx * 3 + 2]
            };

            float overlap_vol, fluid_vol;
            hex_plic_box_volume(hex, pn, pd, dF, bmin, bmax,
                                &overlap_vol, &fluid_vol);

            float transfer = dF * overlap_vol / fmaxf(fluid_vol, 1e-20f);

            #pragma omp atomic
            acc[idx] += (double)transfer;
        }
    }

    /* No clip — handled downstream by Python caller */
    for (int32_t i = 0; i < N; i++)
        out_F[i] = (float)acc[i];

    free(acc);
}
