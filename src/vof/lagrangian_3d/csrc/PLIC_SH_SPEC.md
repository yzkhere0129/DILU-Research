# PLIC Sutherland-Hodgman Overlay — C Implementation Specification

**Version**: 2.0 (complete, self-contained)
**Target**: Another AI can reproduce the exact C code from this document alone.

## 1. Problem Statement

Given a deformed hexahedron (donor cell) and an axis-aligned box (acceptor cell),
compute two volumes:

- `fluid_vol` = volume of (hex ∩ PLIC_halfspace)
- `overlap_vol` = volume of (hex ∩ PLIC_halfspace ∩ box)

The transfer formula is: `transfer = F × overlap_vol / max(fluid_vol, 1e-20f)`

The algorithm: init hex as polyhedron → clip by PLIC plane → measure `fluid_vol`
→ clip by 6 box planes → measure `overlap_vol`.

## 2. Data Structures

### 2.1 Polyhedron (Coordinate-Face Representation)

```c
#define POLY_MAX_F   14   /* 6 hex + 1 PLIC + 6 box + 1 safety */
#define POLY_MAX_FV  14   /* quad(4) + up to 7 clips each adding ≤1 vertex */

typedef struct {
    float verts[POLY_MAX_F][POLY_MAX_FV][3];  /* vertex coords per face  */
    int   nv[POLY_MAX_F];                      /* vertex count per face   */
    int   nf;                                   /* active face count       */
} Poly;
```

Stack size: 14 × 14 × 3 × 4 + 14 × 4 + 4 = **2408 bytes**. No heap allocation.

Vertices are stored **redundantly** per face (shared vertices duplicated).
The divergence-theorem volume formula does not require shared-vertex topology.

### 2.2 Hex Vertex Ordering

```
    v4 -------- v5          v0 = (i,   j,   k  )  bottom-near-left
   /|          /|           v1 = (i+1, j,   k  )  bottom-near-right
  / |         / |           v2 = (i+1, j+1, k  )  bottom-far-right
 v7 -------- v6 |          v3 = (i,   j+1, k  )  bottom-far-left
 |  |        |  |           v4 = (i,   j,   k+1)  top-near-left
 |  v0 ------|-- v1         v5 = (i+1, j,   k+1)  top-near-right
 | /         | /            v6 = (i+1, j+1, k+1)  top-far-right
 |/          |/             v7 = (i,   j+1, k+1)  top-far-left
 v3 -------- v2
```

### 2.3 Hex Face Table (CCW from outside → outward normals)

| Face | Vertices    | Normal direction |
|------|-------------|------------------|
| 0    | 0, 3, 2, 1  | −z (bottom)      |
| 1    | 4, 5, 6, 7  | +z (top)         |
| 2    | 0, 1, 5, 4  | −y (front)       |
| 3    | 2, 3, 7, 6  | +y (back)        |
| 4    | 0, 4, 7, 3  | −x (left)        |
| 5    | 1, 2, 6, 5  | +x (right)       |

```c
static const int HEX_FACE_IDX[6][4] = {
    {0,3,2,1}, {4,5,6,7}, {0,1,5,4},
    {2,3,7,6}, {0,4,7,3}, {1,2,6,5}
};
```

### 2.4 Constants

```c
#define SH_EPS       1e-7f    /* inside/outside classification for SH clip */
#define DEDUP_EPS2   1e-6f    /* squared distance for cap vertex dedup     */
#define AABB_EPS     1e-6f    /* bounding-box separating-axis tolerance    */
#define VOL_GUARD    1e-30f   /* denominator guard for fluid_vol           */
```

## 3. PLIC Plane Convention

The PLIC plane equation:

$$\mathbf{n} \cdot \mathbf{x} = d$$

- $\mathbf{n}$ = outward normal (fluid → gas direction), **not necessarily unit length**
- $d$ = intercept
- **Fluid half-space** (kept side): $\mathbf{n} \cdot \mathbf{x} \leq d$

The caller provides `plic_n[3]` and `plic_d` in **global coordinates**.
The implementation shifts to local coordinates before clipping (§8).

## 4. Function 1: `init_hex_poly`

### 4.1 Purpose
Convert 8 hex vertices into the `Poly` coordinate-face representation.

### 4.2 Signature
```c
void init_hex_poly(Poly *p, const float hex[8][3]);
```

### 4.3 Algorithm
```
1. p->nf = 6
2. For f = 0..5:
     p->nv[f] = 4
     For v = 0..3:
       copy hex[HEX_FACE_IDX[f][v]] → p->verts[f][v]
3. For f = 6..POLY_MAX_F-1:
     p->nv[f] = 0    /* zero unused faces */
```

## 5. Function 2: `clip_face_by_plane`

### 5.1 Purpose
Sutherland-Hodgman clip of a single convex polygon by the half-space $\mathbf{n} \cdot \mathbf{x} \leq d$.

### 5.2 Signature
```c
int clip_face_by_plane(
    const float verts_in[][3], int n_in,        /* input polygon          */
    const float n[3],          float d,          /* clipping plane         */
    float       verts_out[][3], int max_out,     /* output polygon         */
    float       cap_buf[][3],  int *n_cap,       /* cap vertex accumulator */
    int         max_cap                          /* cap buffer capacity    */
);
/* Returns: number of output vertices (0 = entirely outside) */
```

### 5.3 Algorithm (Absolute Steps)

```
INPUT:  polygon P = {v_0, ..., v_{n-1}}, plane (n, d)
OUTPUT: clipped polygon Q in verts_out, cap intersection points in cap_buf

1. Pre-compute signed distances:
     sd[i] = dot(n, v_i) - d     for i = 0..n_in-1

2. n_out = 0

3. For each edge i = 0..n_in-1:
     j = (i + 1) % n_in
     si = sd[i],  sj = sd[j]

     CASE A: si ≤ SH_EPS  (v_i INSIDE)
       3a-1. Emit v_i → verts_out[n_out++]
       3a-2. IF sj > SH_EPS  (v_j OUTSIDE, crossing out):
               t = si / (si - sj)
               p = v_i + t × (v_j − v_i)
               Emit p → verts_out[n_out++]
               Append p → cap_buf[(*n_cap)++]

     CASE B: si > SH_EPS  (v_i OUTSIDE)
       3b-1. IF sj ≤ SH_EPS  (v_j INSIDE, crossing in):
               t = si / (si - sj)
               p = v_i + t × (v_j − v_i)
               Emit p → verts_out[n_out++]
               Append p → cap_buf[(*n_cap)++]
             (ELSE: outside→outside, emit nothing)

4. Return n_out
```

### 5.4 Edge Cases
- All inside: every vertex emitted, no cap points, n_out = n_in
- All outside: no vertices emitted, n_out = 0
- Vertex exactly on plane (sd = 0): treated as inside (≤ SH_EPS)
- Safety: if n_in > POLY_MAX_FV, return 0

### 5.5 Interpolation Precision
$t = s_i / (s_i - s_j)$ where $s_i$ and $s_j$ have opposite signs by case logic,
so $|s_i - s_j| \geq \max(|s_i|, |s_j|)$, ensuring $|t| \leq 1$. No clamping needed.

## 6. Function 3: `clip_poly_by_plane`

### 6.1 Purpose
Clip an entire `Poly` by a single half-space. Updates the `Poly` in-place.

### 6.2 Signature
```c
void clip_poly_by_plane(Poly *p, const float n[3], float d);
```

### 6.3 Algorithm

```
INPUT:  Poly p with nf active faces, plane (n, d)
OUTPUT: p updated (faces clipped, cap face added)

1. Allocate scratch:
     Poly tmp;                         /* clipped faces */
     float cap_buf[POLY_MAX_F * 2][3]; /* max 2 intersections per face */
     int n_cap = 0;

2. For each active face f = 0..p->nf-1:
     tmp.nv[f] = clip_face_by_plane(
         p->verts[f], p->nv[f],
         n, d,
         tmp.verts[f], POLY_MAX_FV,
         cap_buf, &n_cap, POLY_MAX_F * 2
     );

3. Copy clipped faces back: tmp → p (faces 0..nf-1)
   (or just swap: *p = tmp after setting nf)

4. IF n_cap < 3:
     /* No valid cap face (polygon entirely inside or degenerate) */
     p->nf = original nf (no new face added)
     RETURN

5. DEDUPLICATE cap_buf:
     For i = 0..n_cap-1:
       For j = i+1..n_cap-1:
         IF ||cap_buf[i] - cap_buf[j]||² < DEDUP_EPS2:
           mark j as duplicate
     Compact non-duplicates → unique_pts[], n_unique
     IF n_unique < 3: RETURN (degenerate cap)

6. ORDER cap vertices by polar angle (§6.4):
     centroid = mean(unique_pts)
     Build orthonormal frame (e1, e2) on plane (§6.5)
     For each unique point:
       rel = point - centroid
       angle = atan2f(dot(rel, e2), dot(rel, e1))
     Sort unique_pts by angle (ascending = CCW from e1)

7. ADD cap face to p:
     cap_slot = p->nf
     p->nv[cap_slot] = n_unique  (capped at POLY_MAX_FV)
     copy sorted unique_pts → p->verts[cap_slot]
     p->nf++
```

### 6.4 Cap Vertex Ordering

Cap vertices lie on the clipping plane $\mathbf{n} \cdot \mathbf{x} = d$.
They must be ordered **CCW when viewed from the $+\mathbf{n}$ direction** (outward normal
= clipping plane normal, pointing toward the removed half-space).

This ensures the divergence theorem gives the correct sign contribution.

### 6.5 In-Plane Orthonormal Frame Construction

```
INPUT:  plane normal n (any length)
OUTPUT: orthonormal basis (e1, e2) spanning the plane

1. Normalize: n_hat = n / ||n||
2. Choose seed: e1_seed = (1, 0, 0)
   IF |dot(n_hat, e1_seed)| > 0.9:
     e1_seed = (0, 1, 0)
3. Gram-Schmidt:
   e1_raw = e1_seed − dot(n_hat, e1_seed) × n_hat
   e1 = e1_raw / ||e1_raw||
4. e2 = n_hat × e1   (right-hand rule)
```

### 6.6 Sorting

Simple insertion sort on angles (n_unique ≤ 28 = POLY_MAX_F×2, typically ≤ 6).
No need for quicksort.

## 7. Function 4: `poly_volume`

### 7.1 Purpose
Compute volume of a closed polyhedron using the divergence theorem.

### 7.2 Mathematical Formula

For a polyhedron with faces $F_k$, each face a polygon with vertices
$(v_0^k, v_1^k, \ldots, v_{m-1}^k)$ in CCW order (outward normal):

$$V = \frac{1}{6} \left| \sum_{k=0}^{n_f - 1} \sum_{j=1}^{m_k - 2}
\mathbf{v}_0^{(k)} \cdot \left( \mathbf{v}_j^{(k)} \times \mathbf{v}_{j+1}^{(k)} \right) \right|$$

Each face is triangulated as a fan from vertex 0: triangles $(v_0, v_j, v_{j+1})$.

### 7.3 Signature
```c
float poly_volume(const Poly *p);
```

### 7.4 Algorithm

```
1. sum = 0.0  (use double accumulator for precision)
2. For each active face f = 0..p->nf-1:
     IF p->nv[f] < 3: skip
     v0 = p->verts[f][0]
     For j = 1..p->nv[f]-2:
       vb = p->verts[f][j]
       vc = p->verts[f][j+1]
       cross = vb × vc
       sum += dot(v0, cross)
3. Return fabsf((float)(sum / 6.0))
```

### 7.5 Precision Note
The `double` accumulator is critical. The divergence-theorem formula involves
cancellation of positive and negative face contributions. With float32 vertices
at O(dx) scale after local coordinate shift, the terms are O(dx³), and the
final volume is also O(dx³), so cancellation is mild. But `double` sum prevents
accumulated roundoff across many faces.

## 8. Function 5: `hex_plic_box_volume`

### 8.1 Purpose
Full pipeline: compute `(overlap_vol, fluid_vol)` for one donor-acceptor pair.

### 8.2 Signature
```c
void hex_plic_box_volume(
    const float hex[8][3],           /* donor hex vertices (global coords)  */
    const float plic_n[3],           /* PLIC normal (fluid→gas)             */
    float       plic_d,              /* PLIC intercept (global coords)      */
    float       F_donor,             /* donor volume fraction               */
    const float box_min[3],          /* acceptor box lower corner           */
    const float box_max[3],          /* acceptor box upper corner           */
    float      *out_overlap_vol,     /* OUTPUT: overlap volume              */
    float      *out_fluid_vol        /* OUTPUT: fluid volume in hex         */
);
```

### 8.3 Algorithm

```
1. LOCAL COORDINATE SHIFT (mandatory for float32 precision):
     ref = hex[0]
     hv[v] = hex[v] - ref        for v = 0..7
     blo = box_min - ref
     bhi = box_max - ref
     pd = plic_d - dot(plic_n, ref)

2. EARLY EXIT — Bounding-box separating axis test:
     For each axis a ∈ {0, 1, 2}:
       hex_max_a = max(hv[0..7][a])
       hex_min_a = min(hv[0..7][a])
       IF hex_max_a ≤ blo[a] + AABB_EPS  OR  hex_min_a ≥ bhi[a] - AABB_EPS:
         *out_overlap_vol = 0;  *out_fluid_vol = VOL_GUARD;  RETURN

3. EARLY EXIT — Empty donor:
     IF F_donor < 1e-8f:
       *out_overlap_vol = 0;  *out_fluid_vol = VOL_GUARD;  RETURN

4. INIT POLYHEDRON:
     Poly poly;
     init_hex_poly(&poly, hv);

5. PLIC CLIP:
     clip_poly_by_plane(&poly, plic_n, pd);
     *out_fluid_vol = poly_volume(&poly);

6. BOX CLIP (6 sequential planes):
     The box defines 6 half-spaces:
       n = (-1,0,0), d = -blo[0]   (x ≥ blo_x)
       n = (+1,0,0), d = +bhi[0]   (x ≤ bhi_x)
       n = (0,-1,0), d = -blo[1]   (y ≥ blo_y)
       n = (0,+1,0), d = +bhi[1]   (y ≤ bhi_y)
       n = (0,0,-1), d = -blo[2]   (z ≥ blo_z)
       n = (0,0,+1), d = +bhi[2]   (z ≤ bhi_z)

     For each of 6 planes:
       clip_poly_by_plane(&poly, n_i, d_i);

7. OVERLAP VOLUME:
     *out_overlap_vol = poly_volume(&poly);

8. GUARD fluid_vol:
     IF *out_fluid_vol < VOL_GUARD:
       *out_fluid_vol = VOL_GUARD;
```

### 8.4 Box Plane Convention

Each box face defines a half-space that retains the **interior** of the box.
The normals point **inward** (toward the box center):

| Plane | Normal $\mathbf{n}$ | Intercept $d$ | Meaning |
|-------|---------------------|---------------|---------|
| x-lo  | (−1, 0, 0)         | −blo_x        | $-x \leq -blo_x$ → $x \geq blo_x$ |
| x-hi  | (+1, 0, 0)         | +bhi_x        | $+x \leq +bhi_x$ → $x \leq bhi_x$ |
| y-lo  | (0, −1, 0)         | −blo_y        | $y \geq blo_y$ |
| y-hi  | (0, +1, 0)         | +bhi_y        | $y \leq bhi_y$ |
| z-lo  | (0, 0, −1)         | −blo_z        | $z \geq blo_z$ |
| z-hi  | (0, 0, +1)         | +bhi_z        | $z \leq bhi_z$ |

## 9. Function 6: `overlay_all_offsets_plic`

### 9.1 Purpose
OpenMP-parallel overlay: loop over all 27 donor offsets × all acceptor cells.

### 9.2 Signature
```c
void overlay_all_offsets_plic(
    const float *donor_hexes,    /* flat: (Px*Py*Pz) * 8 * 3, padded       */
    const float *donor_F,        /* flat: (Px*Py*Pz), padded               */
    const float *donor_plic_n,   /* flat: (Px*Py*Pz) * 3, padded           */
    const float *donor_plic_d,   /* flat: (Px*Py*Pz), padded               */
    const float *box_min_grid,   /* flat: (nx*ny*nz) * 3                   */
    const float *box_max_grid,   /* flat: (nx*ny*nz) * 3                   */
    int32_t      nx, int32_t ny, int32_t nz,
    float       *out_F           /* flat: (nx*ny*nz), OUTPUT                */
);
```

### 9.3 Padding Convention

Donor arrays are padded with 1 ghost layer: `Px = nx+2, Py = ny+2, Pz = nz+2`.

For acceptor cell `(i, j, k)` with offset `(di, dj, dk)`:
```c
int si = i + 1 - di;   /* +1 for padding, -di for offset */
int sj = j + 1 - dj;
int sk = k + 1 - dk;
int donor_idx = (si * Py + sj) * Pz + sk;
```

Hex vertices: `&donor_hexes[donor_idx * 24]` (24 = 8 verts × 3 coords)
PLIC normal: `&donor_plic_n[donor_idx * 3]`
PLIC d: `donor_plic_d[donor_idx]`
Donor F: `donor_F[donor_idx]`

### 9.4 Algorithm

```
1. N = nx * ny * nz
   Px = nx + 2, Py = ny + 2, Pz = nz + 2

2. Allocate double accumulator:
     double *acc = calloc(N, sizeof(double));

3. For each offset (di, dj, dk) ∈ {-1,0,1}³:

     #pragma omp parallel for schedule(dynamic, 256)
     For idx = 0..N-1:
       i = idx / (ny * nz)
       j = (idx / nz) % ny
       k = idx % nz

       Compute donor index (si, sj, sk) → donor_idx
       dF = donor_F[donor_idx]
       IF dF < 1e-8f: continue  /* skip empty donors */

       Extract hex[8][3] from donor_hexes
       Extract plic_n[3], plic_d from donor arrays
       Extract bmin[3], bmax[3] from acceptor grid

       hex_plic_box_volume(hex, plic_n, plic_d, dF, bmin, bmax,
                           &overlap_vol, &fluid_vol);

       transfer = dF * overlap_vol / fmaxf(fluid_vol, 1e-20f);

       #pragma omp atomic
       acc[idx] += (double)transfer;

4. Copy to output (no clip — handled downstream):
     For i = 0..N-1:
       out_F[i] = (float)acc[i];

5. free(acc);
```

### 9.5 Thread Safety
- `acc` uses `double` for precision across 27 atomic additions per cell.
- `#pragma omp atomic` on the accumulator (not the hex/clip computation).
- All local variables (Poly, cap_buf, etc.) are stack-allocated per-thread.

## 10. Numerical Test Cases

### Test A: Unit cube, no PLIC, self-overlap

```
hex = unit cube [0,1]³
box = [0,1]³ (same cell)
plic_n = (0, 0, 1), plic_d = 1e6  (full cell, no clip)
F = 1.0

Expected: overlap_vol = 1.0, fluid_vol = 1.0
```

### Test B: Unit cube, PLIC at half

```
hex = unit cube [0,1]³
box = [0,1]³
plic_n = (0, 1, 0), plic_d = 0.5  (keep y ≤ 0.5)
F = 0.5

Expected: fluid_vol = 0.5, overlap_vol = 0.5
```

### Test C: Unit cube, PLIC diagonal, box = half

```
hex = unit cube [0,1]³
box = [0, 0.5] × [0, 1] × [0, 1]  (left half)
plic_n = (0, 1, 0), plic_d = 0.5
F = 0.5

Expected: fluid_vol = 0.5, overlap_vol = 0.25
```

### Test D: Displaced hex, per-donor conservation

```
hex shifted by (0.0045, 0.0045, 0) from cell (0.5, 0.5, 0)
dx = dy = dz = 0.01
plic_n = (0, 1, 0), plic_d = C + n·centroid  (for F = 0.5)

Sum overlap_vol over all 27 acceptor boxes should equal fluid_vol.
Tolerance: |Σ overlap / fluid_vol - 1| < 0.001 (0.1%)
```

### Test E: Full cell (F=1), no PLIC effect

```
hex = any deformed hex
plic_d = 1e6  (effectively no clip)
F = 1.0

Expected: fluid_vol = hex volume (from divergence theorem)
Σ overlap over acceptors = fluid_vol
```

### Test F: Empty cell (F=0), skip

```
F = 0.0 (or < 1e-8)

Expected: overlap_vol = 0, fluid_vol = 1e-30 (guard)
Transfer = 0 (skipped)
```

## 11. JAX ↔ C Function Mapping

| JAX function (overlay_3d.py) | C function | Notes |
|------------------------------|------------|-------|
| `_init_hex_poly` | `init_hex_poly` | Identical logic |
| `_clip_poly_by_plane` | `clip_poly_by_plane` | C uses explicit loops vs JAX scan/vmap |
| `_poly_volume` | `poly_volume` | C uses double accumulator |
| `_clip_poly_by_box` | Inlined in `hex_plic_box_volume` | 6 sequential calls to `clip_poly_by_plane` |
| `_hex_plic_box_volume` | `hex_plic_box_volume` | Local coord shift identical |
| `overlay_lagrangian_3d_batched` | `overlay_all_offsets_plic` | C uses OpenMP vs JAX scan+vmap |

### Key Differences from JAX

1. **No vmap/scan overhead**: C uses explicit `for` loops with OpenMP parallelism.
2. **Double accumulator**: C accumulates in float64, casts to float32 at the end.
3. **No final clip**: Output `out_F` is raw (can be > 1). Clipping handled by Python caller.
4. **Cap dedup threshold**: C uses `1e-6f` (squared distance) vs JAX `1e-10` (JAX uses float64 intermediates for dedup).
5. **Empty donor skip**: C skips `F < 1e-8` at the outer loop level (before any geometry). JAX skips inside `_hex_plic_box_volume`.

## 12. Compilation

```bash
gcc -O3 -march=native -fopenmp -shared -fPIC \
    -o libhexboxclip.so hex_box_clip.c
```

**No `-ffast-math`**: breaks IEEE 754, can cause incorrect SH classification.

**Link**: `-lm` for `atan2f`, `fabsf`, `sqrtf`, `fmaxf`.
