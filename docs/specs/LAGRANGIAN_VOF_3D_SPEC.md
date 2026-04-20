# 3D Lagrangian VOF Advection -- Complete Technical Specification (Module A-D)

**Version**: 2.0 | **Date**: 2026-04-16 | **Branch**: `yzk/plic-research`

> **Goal**: Another AI that has never seen the code should be able to perfectly
> reproduce every line of production code solely from this document (plus the
> cross-referenced `EULERIAN_PLIC_SPEC.md` Sections 1-2 for shared volume/normal
> formulas).
>
> **Coverage**: 5 production modules under `src/jax_laseram/vof/lagrangian_3d/`:
> `reconstruction_3d.py`, `move_3d.py`, `overlay_3d.py`, `__init__.py`,
> `overlay_native.py`.
>
> **Excluded** (dead code, not used in production pipeline):
> `overlay_batched.py`, `overlay_fast.py`, `overlay_sparse.py`,
> `tet_clip_analytic.py`.

---

## Table of Contents

- [Module A: Mathematics and Physics](#module-a-mathematics-and-physics)
  - [A.1 Governing Equation](#a1-governing-equation)
  - [A.2 Lagrangian Advection (Discrete)](#a2-lagrangian-advection-discrete)
  - [A.3 Transfer Formula (PLIC path)](#a3-transfer-formula-plic-path)
  - [A.4 Transfer Formula (Non-PLIC path)](#a4-transfer-formula-non-plic-path)
  - [A.5 PLIC Normal (Parker-Youngs / Sobel 3D)](#a5-plic-normal-parker-youngs--sobel-3d)
  - [A.6 Scardovelli-Zaleski Volume Formula](#a6-scardovelli-zaleski-volume-formula)
  - [A.7 PLIC Intercept Bisection](#a7-plic-intercept-bisection)
  - [A.8 Second-Order Face Displacement](#a8-second-order-face-displacement)
  - [A.9 Vertex Displacement from Face Averages](#a9-vertex-displacement-from-face-averages)
  - [A.10 Divergence-Theorem Polyhedron Volume](#a10-divergence-theorem-polyhedron-volume)
  - [A.11 Sutherland-Hodgman Polygon Clipping](#a11-sutherland-hodgman-polygon-clipping)
  - [A.12 Cap Face Construction](#a12-cap-face-construction)
  - [A.13 Hex-PLIC-Box Volume Computation](#a13-hex-plic-box-volume-computation)
  - [A.14 27-Offset Overlay Accumulation](#a14-27-offset-overlay-accumulation)
  - [A.15 Complete Variable-to-Code Mapping](#a15-complete-variable-to-code-mapping)
  - [A.16 Complete Tolerance Table](#a16-complete-tolerance-table)
- [Module B: Data Structures and Memory Layout](#module-b-data-structures-and-memory-layout)
  - [B.1 GridInfo NamedTuple](#b1-gridinfo-namedtuple)
  - [B.2 F Array](#b2-f-array)
  - [B.3 Face Velocities (MAC Staggered Grid)](#b3-face-velocities-mac-staggered-grid)
  - [B.4 Hex Vertex Ordering and Face Table](#b4-hex-vertex-ordering-and-face-table)
  - [B.5 Tetrahedron Decomposition Table](#b5-tetrahedron-decomposition-table)
  - [B.6 Polyhedron Coordinate-Face Representation](#b6-polyhedron-coordinate-face-representation)
  - [B.7 Compile-Time Shape Constants](#b7-compile-time-shape-constants)
  - [B.8 Padding Convention](#b8-padding-convention)
  - [B.9 NCDHW Convolution Layout](#b9-ncdhw-convolution-layout)
- [Module C: Interface and I/O Contract](#module-c-interface-and-io-contract)
  - [C.1 advect_vof_lagrangian_3d](#c1-advect_vof_lagrangian_3d)
  - [C.2 compute_plic_normals_3d](#c2-compute_plic_normals_3d)
  - [C.3 compute_intercept_C_3d](#c3-compute_intercept_c_3d)
  - [C.4 _volume_below_3d and _volume_below_3d_2dmap](#c4-_volume_below_3d-and-_volume_below_3d_2dmap)
  - [C.5 _vertex_displacement_from_faces](#c5-_vertex_displacement_from_faces)
  - [C.6 lagrangian_move_faces_3d](#c6-lagrangian_move_faces_3d)
  - [C.7 build_deformed_hexahedra](#c7-build_deformed_hexahedra)
  - [C.8 _init_hex_poly](#c8-_init_hex_poly)
  - [C.9 _sh_clip_face](#c9-_sh_clip_face)
  - [C.10 _clip_poly_by_plane](#c10-_clip_poly_by_plane)
  - [C.11 _clip_poly_by_box](#c11-_clip_poly_by_box)
  - [C.12 _poly_volume](#c12-_poly_volume)
  - [C.13 _hex_box_volume](#c13-_hex_box_volume)
  - [C.14 _hex_plic_box_volume](#c14-_hex_plic_box_volume)
  - [C.15 overlay_lagrangian_3d](#c15-overlay_lagrangian_3d)
  - [C.16 overlay_lagrangian_3d_native](#c16-overlay_lagrangian_3d_native)
- [Module D: Absolute Execution Constraint Flow](#module-d-absolute-execution-constraint-flow)
  - [D.1 Top-Level Pipeline](#d1-top-level-pipeline)
  - [D.2 PLIC Normal Computation (compute_plic_normals_3d)](#d2-plic-normal-computation)
  - [D.3 Gradient-Based Interface Gating](#d3-gradient-based-interface-gating)
  - [D.4 PLIC Intercept Bisection (compute_intercept_C_3d)](#d4-plic-intercept-bisection)
  - [D.5 Volume Formula Execution (_volume_below_3d)](#d5-volume-formula-execution)
  - [D.6 Lagrangian Face Move](#d6-lagrangian-face-move)
  - [D.7 Vertex Displacement from Face Average](#d7-vertex-displacement-from-face-average)
  - [D.8 Build Deformed Hexahedra](#d8-build-deformed-hexahedra)
  - [D.9 Vertex Clamping](#d9-vertex-clamping)
  - [D.10 Overlay Dispatch](#d10-overlay-dispatch)
  - [D.11 JAX Overlay Path (overlay_lagrangian_3d)](#d11-jax-overlay-path)
  - [D.12 Sutherland-Hodgman Clip Face (_sh_clip_face) Scan Body](#d12-sutherland-hodgman-clip-face-scan-body)
  - [D.13 Clip Polyhedron by Plane (_clip_poly_by_plane)](#d13-clip-polyhedron-by-plane)
  - [D.14 Clip Polyhedron by Box (_clip_poly_by_box)](#d14-clip-polyhedron-by-box)
  - [D.15 Polyhedron Volume (_poly_volume)](#d15-polyhedron-volume)
  - [D.16 Hex-Box Volume (_hex_box_volume)](#d16-hex-box-volume)
  - [D.17 Hex-PLIC-Box Volume (_hex_plic_box_volume)](#d17-hex-plic-box-volume)
  - [D.18 Native C Overlay Path (overlay_native.py)](#d18-native-c-overlay-path)
  - [D.19 Obstacle Redistribution](#d19-obstacle-redistribution)
  - [D.20 Clip and Pack Halos](#d20-clip-and-pack-halos)
- [Appendix E: Tet-Pool Fast Path (Alternative Code, Used by Compat Functions)](#appendix-e-tet-pool-fast-path)
- [Appendix F: Benchmark Results](#appendix-f-benchmark-results)
- [Appendix G: Cross-Reference to EULERIAN_PLIC_SPEC.md](#appendix-g-cross-reference)

---

# Module A: Mathematics and Physics

## A.1 Governing Equation

The VOF kinematic equation (Barkhudarov 2004, Eq. 2):

$$V_f \frac{\partial F}{\partial t} + \nabla \cdot (\mathbf{A} \mathbf{U} F) = 0$$

where $F \in [0,1]$ is the volume fraction, $V_f$ is the cell volume open to
flow, $\mathbf{A}$ is the area fraction vector, and $\mathbf{U}$ is the
velocity field.

## A.2 Lagrangian Advection (Discrete)

Instead of computing fluxes through fixed cell faces, the Lagrangian method:

1. **Reconstructs** the interface in each cell using a PLIC plane:
   $\mathbf{n} \cdot \mathbf{x} = d$, where $\mathbf{n}$ is the
   Parker-Youngs/Sobel gradient of $F$, and $d$ is the intercept found by
   bisection to match the cell's volume fraction $F$.

2. **Moves** each cell face by its velocity with second-order correction
   (Barkhudarov Eq. 6-7), creating a deformed hexahedron.

3. **Overlays** each deformed hexahedron (optionally PLIC-truncated) onto the
   Eulerian grid via Sutherland-Hodgman polyhedron clipping, computing the
   volume fraction transferred to each acceptor cell.

## A.3 Transfer Formula (PLIC path)

For a donor cell with volume fraction $F$ and deformed hexahedron $H$:

$$\text{transfer}_j = F \cdot \frac{\text{overlap}_j}{\text{fluid\_vol}}$$

where:
- $\text{overlap}_j = \mathrm{Vol}((H \cap P^-) \cap B_j)$: volume of
  PLIC-truncated hex intersected with acceptor box $B_j$
- $\text{fluid\_vol} = \mathrm{Vol}(H \cap P^-)$: volume of hex below PLIC
  plane
- $P^- = \{\mathbf{x} : \mathbf{n} \cdot \mathbf{x} \le d\}$ is the PLIC
  fluid half-space

**Conservation**: $\sum_j \text{transfer}_j = F$ (exact per donor, before
clipping).

**Important**: The $dV = V_\text{old} / V_\text{new}$ factor from Barkhudarov
Eq. 8 is NOT used in the PLIC formula. The overlap/fluid_vol normalization
already handles volume correctly; using dV causes over-deposition.

Code reference (`overlay_3d.py` line 746):
```
contrib = dF * overlap / jnp.maximum(fluid_vol, jnp.float32(1e-20))
```

## A.4 Transfer Formula (Non-PLIC path)

$$\text{transfer}_j = F \cdot \frac{\text{vol}_j}{V_\text{cell}}$$

where $\text{vol}_j = \mathrm{Vol}(H \cap B_j)$ (no PLIC clip).

Code reference (`overlay_3d.py` line 749):
```
contrib = dF * overlap / cell_vol
```

## A.5 PLIC Normal (Parker-Youngs / Sobel 3D)

> **Cross-reference**: `EULERIAN_PLIC_SPEC.md` Section 2 documents the
> identical Youngs normal computation used by the Eulerian PLIC pipeline.
> The Lagrangian pipeline uses `compute_plic_normals_3d` from
> `reconstruction_3d.py`, which is the SAME mathematical kernel but with
> Lagrangian-specific NCDHW layout handling documented below.

### A.5.1 3D Sobel (Parker-Youngs) Kernel

The cross-section weight matrix for orthogonal axes:

$$W_{2D} = \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix},
\qquad w_\text{sum} = \sum W_{2D} = 16$$

For the x-gradient (kernel axis 1 = H = x in NCDHW layout):

$$K_x[d, :, w] = \frac{1}{2 \cdot w_\text{sum} \cdot \Delta x}
\begin{cases}
-W_{2D}[d, w] & \text{slot}=0 \\
0              & \text{slot}=1 \\
+W_{2D}[d, w] & \text{slot}=2
\end{cases}$$

Equivalently in code (`reconstruction_3d.py` lines 75-78):
```python
ch0 = zeros(3,3,3)
ch0[:, 0, :] = -w2d   # kernel axis-1 slot 0
ch0[:, 2, :] = +w2d   # kernel axis-1 slot 2
ch0 /= (2.0 * w_sum * dx)
```

The y-gradient varies along kernel axis 2 (W = y):
```python
ch1[:, :, 0] = -w2d
ch1[:, :, 2] = +w2d
ch1 /= (2.0 * w_sum * dy)
```

The z-gradient varies along kernel axis 0 (D = z):
```python
ch2[0, :, :] = -w2d
ch2[2, :, :] = +w2d
ch2 /= (2.0 * w_sum * dz)
```

### A.5.2 Kernel Assembly

```python
kernel = jnp.stack([ch0, ch1, ch2])[:, None, :, :, :]
```

Resulting shape: `(3, 1, 3, 3, 3)` -- **OIDHW format**:
- `O=3`: three output channels ($\partial F/\partial x$, $\partial F/\partial y$, $\partial F/\partial z$)
- `I=1`: one input channel
- `D=3, H=3, W=3`: kernel spatial extent

### A.5.3 NCDHW Transpose (CRITICAL)

The F array has spatial layout `(Nx, Ny, Nz)` corresponding to `(x, y, z)`.
The NCDHW conv format maps `D=z, H=x, W=y`:

**Before conv** (`reconstruction_3d.py` lines 101-103):
```python
F_spatial = F[:, :, :, 0]                       # (Nx, Ny, Nz)
lhs = F_spatial.transpose(2, 0, 1)[None, None]  # (1, 1, Nz, Nx, Ny)
```

The transpose `(2, 0, 1)` maps:
- old axis 0 (x=Nx) to new axis 1 (H)
- old axis 1 (y=Ny) to new axis 2 (W)
- old axis 2 (z=Nz) to new axis 0 (D)

**After conv** (`reconstruction_3d.py` lines 117-128):
```python
dFdx_raw = grad[0, 0]  # (Nz-2, Nx-2, Ny-2) in (D,H,W) order
# transpose (1,2,0) to get (Nx-2, Ny-2, Nz-2) in (x,y,z) order
dFdx = zeros(Nx, Ny, Nz)
dFdx[1:-1, 1:-1, 1:-1] = dFdx_raw.transpose(1, 2, 0)
```

The transpose `(1, 2, 0)` maps:
- old axis 0 (D=Nz-2) to new axis 2
- old axis 1 (H=Nx-2) to new axis 0
- old axis 2 (W=Ny-2) to new axis 1

### A.5.4 Normalization

```python
mag = sqrt(dFdx**2 + dFdy**2 + dFdz**2 + 1e-30)
nx = -dFdx / mag
ny = -dFdy / mag
nz = -dFdz / mag
```

The **negation** ensures the normal points from fluid ($F=1$) toward empty
($F=0$): if $\nabla F$ points from empty to full (increasing F), then
$-\nabla F$ points from full to empty.

Normalization epsilon: **`1e-30`** (prevents division by zero; NOT the
interface classification threshold).

### A.5.5 Interface Mask

```python
is_interface = (F[..., 0] > 1e-6) & (F[..., 0] < 1.0 - 1e-6)
nx = where(is_interface, nx, 0.0)
ny = where(is_interface, ny, 0.0)
nz = where(is_interface, nz, 0.0)
```

Cells with $F \le 10^{-6}$ or $F \ge 1 - 10^{-6}$ are considered pure
(empty/full) and their normals are zeroed.

### A.5.6 Output Shape

Each of `nx, ny, nz` has shape `(Nx, Ny, Nz, 1)` (trailing singleton preserved
via `nx[..., None]`).

The result is zero at boundary cells (1 layer lost to VALID convolution on
each side), and zero in pure cells.

### A.5.7 Dtype Propagation

The weight matrix `w2d` is created with `dtype=F.dtype`, so the entire
computation inherits F's precision (float32 or float64).

## A.6 Scardovelli-Zaleski Volume Formula

> **Cross-reference**: `EULERIAN_PLIC_SPEC.md` Section 1 documents the
> identical `_volume_below_3d` function shared between the Eulerian and
> Lagrangian pipelines. The Lagrangian pipeline imports and uses the SAME
> function from `reconstruction_3d.py`.

### A.6.1 Problem Definition

Compute the exact volume truncated by a plane in a unit cube:

$$V(C; a, b, c) = \mathrm{Vol}\{(x,y,z) \in [0,1]^3 : ax + by + cz \le C\}$$

where $(a, b, c) = (n_x \Delta x,\; n_y \Delta y,\; n_z \Delta z)$, and $C$
is the center-relative intercept.

### A.6.2 Coordinate Transform

Step 1 -- reflect to non-negative octant:
$$a_0 = |a|, \quad b_0 = |b|, \quad c_0 = |c|$$

Step 2 -- map center-relative $C$ to unit-cube parameter $d$:
$$d = C + \tfrac{1}{2}(a_0 + b_0 + c_0)$$

Physical meaning: when $d = 0$ the plane passes through the origin corner and
$V = 0$; when $d = a_0 + b_0 + c_0$ the plane passes through the diagonal
corner and $V = 1$.

### A.6.3 Degeneracy Detection

An absolute floor prevents division by zero:
$$\text{thr} = 10^{-10}$$

A **relative threshold** determines whether a coefficient is effectively zero:
$$\text{max\_abc} = \max(a_0, b_0, c_0) + 10^{-30}$$
$$\text{thr\_rel} = 10^{-4}$$

The boolean flags:
$$\text{nz\_a} = (a_0 > \text{thr\_rel} \cdot \text{max\_abc})$$
$$\text{nz\_b} = (b_0 > \text{thr\_rel} \cdot \text{max\_abc})$$
$$\text{nz\_c} = (c_0 > \text{thr\_rel} \cdot \text{max\_abc})$$

Coefficient count:
$$n_\text{nz} = \text{nz\_a} + \text{nz\_b} + \text{nz\_c} \in \{0, 1, 2, 3\}$$

### A.6.4 Branch 0D ($n_\text{nz} = 0$): Uniform

$$V_{0D} = \begin{cases} 1 & d \ge 0 \\ 0 & d < 0 \end{cases}$$

### A.6.5 Branch 1D ($n_\text{nz} = 1$): Slab

Let $m$ be the single nonzero coefficient (selected by priority: $a_0$ if
nz_a, else $b_0$ if nz_b, else $c_0$), guarded:
$$m_{1D} = m + \text{thr}$$

$$V_{1D} = \mathrm{clip}\!\left(\frac{d}{m_{1D}},\; 0,\; 1\right)$$

### A.6.6 Branch 2D ($n_\text{nz} = 2$): Triangle/Trapezoid

**CRITICAL -- `axis=0` sort**: The nonzero coefficients are packed into a
3-element array and sorted:

```python
vals = jnp.sort(jnp.array([
    where(nz_a, a0, 0.0),
    where(nz_b, b0, 0.0),
    where(nz_c, c0, 0.0),
]), axis=0)
```

**The `axis=0` is essential.** When called via `vmap`, the inputs are `(Nz,)`
arrays, so `jnp.array([...])` produces shape `(3, Nz)`. The default `sort`
would sort along `axis=-1` (the `Nz` axis), which would sort spatially
adjacent cells against each other instead of sorting the three coefficients.
Using `axis=0` correctly sorts the coefficient axis.

After sorting, `vals[0] = 0` (the zero coefficient), `vals[1] = p2`,
`vals[2] = q2` are the two nonzero coefficients in ascending order.

Guarded values:
$$p_2 = \max(\text{vals}[1],\; \text{thr}), \quad q_2 = \max(\text{vals}[2],\; \text{thr})$$
$$S_2 = p_2 + q_2, \quad pq_2 = 2 p_2 q_2$$

Piecewise formula:
$$V_{2D} = \begin{cases}
0 & d \le 0 \\
d^2 / pq_2 & 0 < d \le p_2 \\
(2d - p_2) / (2 q_2) & p_2 < d \le q_2 \\
1 - (S_2 - d)^2 / pq_2 & q_2 < d < S_2 \\
1 & d \ge S_2
\end{cases}$$

### A.6.7 Branch 3D ($n_\text{nz} = 3$): Full Inclusion-Exclusion

Sort the three coefficients:
$$m_1 = \max(\min(a_0, b_0, c_0),\; \text{thr})$$
$$m_3 = \max(\max(a_0, b_0, c_0),\; \text{thr})$$
$$m_2 = \max(a_0 + b_0 + c_0 - m_1 - m_3,\; \text{thr})$$

Derived quantities:
$$S_3 = m_1 + m_2 + m_3, \quad p_6 = 6 \cdot m_1 \cdot m_2 \cdot m_3$$

Clamped differences (using $\max(\cdot, 0)$):
$$d_{m_1} = (d - m_1)^+, \quad d_{m_2} = (d - m_2)^+, \quad d_{m_3} = (d - m_3)^+$$
$$d_{m_{12}} = (d - m_1 - m_2)^+, \quad d_{m_{13}} = (d - m_1 - m_3)^+, \quad d_{m_{23}} = (d - m_2 - m_3)^+$$
$$d_S = (d - S_3)^+$$

Inclusion-exclusion numerator:
$$N = d^3 - d_{m_1}^3 - d_{m_2}^3 - d_{m_3}^3 + d_{m_{12}}^3 + d_{m_{13}}^3 + d_{m_{23}}^3 - d_S^3$$

$$V_{3D} = \begin{cases}
0 & d \le 0 \\
N / p_6 & 0 < d < S_3 \\
1 & d \ge S_3
\end{cases}$$

### A.6.8 Final Selection and Clipping

$$V = \begin{cases}
V_{0D} & n_\text{nz} = 0 \\
V_{1D} & n_\text{nz} = 1 \\
V_{2D} & n_\text{nz} = 2 \\
V_{3D} & n_\text{nz} = 3
\end{cases}$$

$$V = \mathrm{clip}(V, 0, 1)$$

Implemented as a `jnp.where` cascade (no Python branches).

### A.6.9 Double-Vmap Batching

The scalar `_volume_below_3d` is batched over the full grid:

```python
_volume_below_3d_2dmap = jax.vmap(jax.vmap(_volume_below_3d))
```

When called in `compute_intercept_C_3d`, the inputs have shape `(Nx, Ny, Nz)`
after stripping the trailing singleton. The double vmap maps over axes 0 and 1
(Nx, Ny), leaving axis 2 (Nz) as the vectorized axis within each vmap'd call.

**This is why `axis=0` in the sort is critical**: inside the innermost
vmap, the inputs to `_volume_below_3d` are `(Nz,)` arrays. The
`jnp.array([...])` in the 2D branch creates shape `(3, Nz)`, and sorting
along `axis=0` correctly sorts the three-coefficient axis.

## A.7 PLIC Intercept Bisection

### A.7.1 Physical Coordinates

The intercept $C$ is in center-relative physical coordinates:
$$a = n_x \cdot \Delta x, \quad b = n_y \cdot \Delta y, \quad c = n_z \cdot \Delta z$$

The bracket:
$$C_\text{half} = \tfrac{1}{2}(|a| + |b| + |c|)$$
$$C_\text{lo} = -C_\text{half}, \quad C_\text{hi} = +C_\text{half}$$

At $C = -C_\text{half}$: the plane is at the cell corner, $V = 0$.
At $C = +C_\text{half}$: the plane is at the opposite corner, $V = 1$.

### A.7.2 Target Clipping

$$F_\text{target} = \mathrm{clip}(F, 10^{-10}, 1 - 10^{-10})$$

This prevents the bisection from targeting exactly 0 or 1, which would make
convergence degenerate.

### A.7.3 Bisection Loop

20 iterations via `jax.lax.fori_loop(0, 20, body_fn, init)`:

```python
C_init = zeros(Nx, Ny, Nz)
state = (C_init, C_lo_s, C_hi_s)

def body_fn(i, state):
    C_mid, lo, hi = state
    vol = _volume_below_3d_2dmap(C_mid, a_s, b_s, c_s)
    below = (vol < F_s)
    lo_new = where(below, C_mid, lo)
    hi_new = where(below, hi, C_mid)
    mid_new = 0.5 * (lo_new + hi_new)
    return (mid_new, lo_new, hi_new)
```

After 20 iterations: $|C - C_\text{true}| \le C_\text{half} / 2^{20} \approx
10^{-6} \cdot C_\text{half}$.

### A.7.4 Interface Masking

After bisection, non-interface cells are zeroed:

```python
is_interface = (F[..., 0] > 1e-6) & (F[..., 0] < 1.0 - 1e-6)
C_final = where(is_interface, C_final, 0.0)
```

### A.7.5 Trailing Singleton

Input/output shapes strip and re-add the trailing dimension:

```python
a_s = a[..., 0]     # (Nx, Ny, Nz)
# ... bisection on (Nx, Ny, Nz) arrays ...
return C_final[..., None]  # (Nx, Ny, Nz, 1)
```

### A.7.6 Lagrangian-Specific: No Newton Refinement

Unlike the Eulerian pipeline (`analytic_intercept.py`) which uses the
Scardovelli-Zaleski analytic formula + Newton refinement, the Lagrangian
pipeline uses pure bisection. This is simpler but converges more slowly
($O(2^{-N})$ vs $O(\epsilon_\text{machine})$).

[OBSERVATION] 20 bisection steps give ~6 decimal digits, which is near the
float32 limit. Increasing to 24 would add negligible cost but provide margin.

## A.8 Second-Order Face Displacement

Barkhudarov (2004) Eq. 6-7:

$$\Delta x_\text{face}[i] = \frac{u[i] \cdot \Delta t}{1 + 0.5 \cdot \frac{\partial u}{\partial x}[i] \cdot \Delta t}$$

The velocity gradient is computed via central differences:

$$\frac{\partial u}{\partial x}[i] = \frac{u[i+1] - u[i-1]}{2 \Delta x}$$

At boundaries ($i=0$ and $i=N_\text{face}-1$), the gradient is zero (due to
initialization of `dudx` as `jnp.zeros_like`).

The same formula applies for all three axes with $(v, \Delta y)$ and
$(w, \Delta z)$.

## A.9 Vertex Displacement from Face Averages

Each vertex displacement is the average of the two adjacent face displacements
along the primary axis. The algorithm has three stages:

**Stage 1: Boundary padding**

For a given axis (say x, with $N_\text{face} = N_\text{tot} - 1$ faces):
- Create array of size $(N_\text{tot} + 1)$ along the axis
- `disp_all[0] = face_disp[0]` (copy first face)
- `disp_all[1:-1] = face_disp[:]` (all interior faces)
- `disp_all[-1] = face_disp[-1]` (copy last face)

**Stage 2: Average**

$$\text{vert\_raw}[i] = \tfrac{1}{2}(\text{disp\_all}[i] + \text{disp\_all}[i+1])$$

Result shape: $(N_\text{tot,x}, N_\text{tot,y}, N_\text{tot,z})$.

**Stage 3: Pad remaining axes**

For each of the three axes (in order 0, 1, 2):
- `jnp.pad` with `(0, 1)` along that axis (adds one element at the end)
- Copy second-to-last slice to last slice along that axis

Final shape: $(N_\text{tot,x}+1, N_\text{tot,y}+1, N_\text{tot,z}+1)$.

[OBSERVATION] Stage 3 pads ALL three axes, including the primary axis that was
already $(N_\text{tot})$ in Stage 2. This means the primary axis goes from
$N_\text{tot}$ to $N_\text{tot}+1$, with the last vertex being a copy of the
second-to-last. The other two axes also get padded with copies.

## A.10 Divergence-Theorem Polyhedron Volume

For a closed convex polyhedron with faces $\{f_k\}$, each face having vertices
$(v_0^{(k)}, v_1^{(k)}, \ldots, v_{m_k-1}^{(k)})$ in CCW order (outward
normal):

$$V = \frac{1}{6} \left| \sum_{k} \sum_{j=0}^{m_k-3}
\tilde{v}_0^{(k)} \cdot
\left( \tilde{v}_{j+1}^{(k)} \times \tilde{v}_{j+2}^{(k)} \right) \right|$$

where $\tilde{v} = v - \text{ref}$ are coordinates shifted by a reference
point $\text{ref} = \text{faces}[0, 0, :]$ (the first vertex of the first
face).

**The local shift is CRITICAL for float32 precision**: without it, vertices at
positions $\sim 0.5$ with cell volumes $\sim 10^{-6}$ would require 5 orders
of magnitude to cancel in the triple product, exhausting float32's 7-digit
mantissa.

The fan triangulation: face $k$ with vertices $(v_0, v_1, \ldots, v_{m-1})$
is decomposed into $m-2$ triangles:
$(v_0, v_1, v_2), (v_0, v_2, v_3), \ldots, (v_0, v_{m-2}, v_{m-1})$.

In code, the cross product is computed as:
```python
vb = local[:, 1:MAX_FV-1, :]     # (MAX_F, MAX_FV-2, 3)
vc = local[:, 2:MAX_FV, :]       # (MAX_F, MAX_FV-2, 3)
cross_val = cross(vb, vc)        # (MAX_F, MAX_FV-2, 3)
triple = sum(v0[:, None, :] * cross_val, axis=-1)  # (MAX_F, MAX_FV-2)
```

[OBSERVATION] The triple product uses `v0 . (vb x vc)` rather than
`(vb - v0) x (vc - v0) . v0`. The divergence theorem form requires
`v0 . (v1 x v2)` for the shifted vertices, not
`(v1-v0) x (v2-v0) . v0`. Both give the same result for closed polyhedra.
The code uses the simpler form.

## A.11 Sutherland-Hodgman Polygon Clipping

Clips a convex polygon by the half-space $\mathbf{n} \cdot \mathbf{x} \le d$.

For each edge $(v_i, v_j)$ where $j = (i+1) \bmod n_v$:

1. Compute signed distances:
   $$d_i = \mathbf{n} \cdot v_i - d, \quad d_j = \mathbf{n} \cdot v_j - d$$

2. Classification (using tolerance $\epsilon = 10^{-10}$):
   - $d_i \le \epsilon$: vertex $i$ is **inside**
   - $d_i > \epsilon$: vertex $i$ is **outside**

3. Intersection point on edge $i \to j$:
   $$t = \frac{d_i}{d_i - d_j + 10^{-30}}, \quad t = \mathrm{clip}(t, 0, 1)$$
   $$\text{inter} = v_i + t \cdot (v_j - v_i)$$

   The $10^{-30}$ guard prevents division by zero when both vertices are on
   the plane. The clip ensures the intersection is within the edge.

4. Emit rules (for edge $i \to j$):
   - **both_in** ($i$ inside, $j$ inside): emit $v_j$
   - **exit** ($i$ inside, $j$ outside): emit intersection
   - **enter** ($i$ outside, $j$ inside): emit intersection, then $v_j$
   - **both_out**: emit nothing

**CRITICAL CODE DETAIL**: The emit logic uses $v_j$ (the NEXT vertex), not
$v_i$ (the current vertex). This is the standard SH convention where each
edge is responsible for emitting its endpoint. The first vertex is emitted
by the last edge's both_in or enter case.

## A.12 Cap Face Construction

When a clipping plane intersects the polyhedron, it creates a new "cap" face
from the intersection points. The construction:

1. **Collect**: For each face that was clipped, extract the edge-plane
   intersection points (up to 2 per face, stored in `inter_pts`).

2. **Validate**: Only intersections from active faces (`face_idx < n_faces`)
   with actual crossings (`n_inter > 0`) are used.

3. **Deduplicate**: Pairwise distance check with threshold $10^{-10}$
   (squared distance). Uses lower-triangular mask to avoid self-comparison.
   $N = \text{MAX\_F} \times 2 = 28$ candidate points.

4. **Polar sort**: Build a 2D coordinate frame on the clipping plane:
   - Start with `e1 = (1, 0, 0)`.
   - If `|dot(plane_n, e1)| > 0.9`, switch to `e1 = (0, 1, 0)`.
   - Orthogonalize: `e1 = e1 - dot(e1, plane_n) * plane_n`.
   - Normalize: `e1 /= (norm(e1) + 1e-30)`.
   - `e2 = cross(plane_n, e1)`.
   - Sort by `atan2(rel . e2, rel . e1)` where `rel = pt - cap_center`.
   - Non-valid points get angle `1e10` to sort them to the end.

5. **Add cap**: If $n_\text{unique} \ge 3$, add cap face at slot
   `min(n_faces, MAX_F - 1)` with vertex count `min(n_unique, MAX_FV)`.

## A.13 Hex-PLIC-Box Volume Computation

For a single donor cell, the volume computation pipeline:

1. **Local shift** (float32 precision): `ref = hex_verts[0]`, shift all
   coordinates by `-ref`.

2. **AABB separating-axis early exit**: check if hex bounding box is entirely
   outside the acceptor box (using `_EPS_BOX = 1e-6`).

3. **Empty donor early exit**: skip if `F_val < 1e-8`.

4. **Init hex polyhedron**: 6 quad faces from `HEX_FACES` table.

5. **PLIC clip**: `_clip_poly_by_plane(poly, plic_n, pd)` -- clips hex to
   fluid half-space. Compute `fluid_vol = _poly_volume(fluid_poly)`.

6. **Box clip**: `_clip_poly_by_box(fluid_poly, blo, bhi)` -- clips PLIC'd
   poly against 6 box planes.

7. **Overlap volume**: `overlap_vol = _poly_volume(clipped)`.

Returns: `(overlap_vol, fluid_vol)` as pair of scalars.

For skip conditions: `overlap_vol = 0.0`, `fluid_vol = 1e-30` (guard against
division by zero in the transfer formula).

## A.14 27-Offset Overlay Accumulation

The overlay iterates over all 27 neighbor offsets
$(d_i, d_j, d_k) \in \{-1, 0, 1\}^3$. For each offset:

1. `dynamic_slice` the padded donor subgrid with shape `(nx, ny, nz)`.
2. Call the overlap function (vmap'd over all cells).
3. Accumulate: `F_new += contribution`.

The final result is clipped: `F_new = clip(F_new, 0, 1)`.

The accumulation is implemented as `jax.lax.scan` over the 27 offsets (not a
Python loop), which enables XLA compilation.

---

# Module B: Data Structures and Memory Layout

## B.1 GridInfo NamedTuple

```python
class GridInfo(NamedTuple):
    nx: int            # interior cell count, x-axis
    ny: int            # interior cell count, y-axis
    nz: int            # interior cell count, z-axis (1 for 2D)
    nh: int            # halo layers (always 1 for 2nd order)
    dx: float          # cell size, x
    dy: float          # cell size, y
    dz: float          # cell size, z
    x_range: (float, float)  # physical domain extents, x
    y_range: (float, float)  # physical domain extents, y
    z_range: (float, float)  # physical domain extents, z
```

Derived quantities:
- `Ntot_x = nx + 2*nh`, `Ntot_y = ny + 2*nh`, `Ntot_z = nz + 2*nh`
- `cell_vol = dx * dy * dz`

## B.2 F Array

```
Shape: (Ntot_x, Ntot_y, Ntot_z, 1) -- 4D with trailing singleton
Type:  float32 (or float64 if JAX_ENABLE_X64)
Layout: row-major (C order), JAX default
Index: F[i, j, k, 0]

Interior region: F[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]
Halo region:     nearest-neighbor extrapolation from interior boundary
```

## B.3 Face Velocities (MAC Staggered Grid)

```
u_face: float32[Ntot_x-1, Ntot_y, Ntot_z]  -- x-face velocities
v_face: float32[Ntot_x, Ntot_y-1, Ntot_z]  -- y-face velocities
w_face: float32[Ntot_x, Ntot_y, Ntot_z-1]  -- z-face velocities

u_face[i, j, k] = velocity at face between cells (i, j, k) and (i+1, j, k)
```

## B.4 Hex Vertex Ordering and Face Table

### Vertex Numbering

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

Here "bottom" = low-z, "top" = high-z, "near" = low-y, "far" = high-y,
"left" = low-x, "right" = high-x.

### Hex Face Table (CCW from outside, outward normals)

```python
HEX_FACES = jnp.array([
    [0, 3, 2, 1],  # Face 0: bottom (z=0), normal -z
    [4, 5, 6, 7],  # Face 1: top    (z=1), normal +z
    [0, 1, 5, 4],  # Face 2: front  (y=0), normal -y
    [2, 3, 7, 6],  # Face 3: back   (y=1), normal +y
    [0, 4, 7, 3],  # Face 4: left   (x=0), normal -x
    [1, 2, 6, 5],  # Face 5: right  (x=1), normal +x
], dtype=jnp.int32)  # shape: (6, 4)
```

Verified: for a unit cube, each face's vertices are in CCW order when viewed
from outside the cube, producing outward normals consistent with the
divergence-theorem volume formula.

### Face Construction in `_init_hex_poly`

```python
hex_face_coords = hex_verts[HEX_FACES]  # (6, 4, 3)
faces[:6, :4, :] = hex_face_coords
face_nv[:6] = 4
n_faces = 6
```

## B.5 Tetrahedron Decomposition Table

6-tet decomposition of the hexahedron (used by compat/fast-path functions,
NOT by the primary SH clipping path):

```python
TET_INDICES = jnp.array([
    [0, 1, 2, 4],  # T1
    [1, 2, 4, 5],  # T2
    [2, 4, 5, 6],  # T3
    [2, 3, 4, 7],  # T4
    [2, 4, 6, 7],  # T5
    [0, 2, 3, 4],  # T6
], dtype=jnp.int32)  # shape: (6, 4)
```

Three internal face diagonals: (0,2), (0,4), (2,5). Each tet has exactly 2
boundary faces + 2 internal faces. Verified: $\sum_{i=1}^{6} V(T_i) = 1.0$
for unit cube.

## B.6 Polyhedron Coordinate-Face Representation

The polyhedron is stored as a **coordinate-face** representation (no shared
vertex indices):

```
faces:   float32[MAX_F, MAX_FV, 3]  -- vertex positions per face
face_nv: int32[MAX_F]               -- vertex count per face
n_faces: int32                       -- number of active faces
```

Shared vertices are stored redundantly in each face. This simplifies the SH
clipping algorithm (no index bookkeeping) at the cost of memory.

The tuple `poly = (faces, face_nv, n_faces)` is the standard polyhedron
representation throughout the overlay module.

## B.7 Compile-Time Shape Constants

```python
MAX_F  = 14   # maximum number of faces in a polyhedron
MAX_FV = 14   # maximum number of vertices per face
```

### Derivation of MAX_F = 14

| Component | Faces |
|-----------|-------|
| Hex       | 6     |
| PLIC cap  | 1     |
| Box caps  | 6     |
| Safety    | 1     |
| **Total** | **14** |

The hex starts with 6 faces. PLIC clipping adds 1 cap face. Box clipping
(6 planes) can add up to 6 cap faces. Plus 1 safety margin.

### Derivation of MAX_FV = 14

A quad face (4 vertices) can gain at most 1 new vertex per SH clip. With up to
$6 + 1 = 7$ clips (PLIC + box), a face could theoretically grow to
$4 + 7 = 11$ vertices. Cap faces from polar sort can have up to
$\sim 13$ vertices. 14 provides margin.

### Hard Limits

These are **compile-time constants** that define fixed tensor shapes for JAX
tracing. **Exceeding them silently corrupts** results: vertex writes beyond
`MAX_FV` use `jnp.minimum(n_out, MAX_FV - 1)` which clamps the write index
to the last slot, overwriting previous data.

## B.8 Padding Convention

Donor arrays are padded with 1 ghost layer of zeros on each side:

```python
hp = jnp.pad(hex_v, ((1, 1), (1, 1), (1, 1), (0, 0), (0, 0)))
Fp = jnp.pad(F_int, ((1, 1), (1, 1), (1, 1)))
```

Padded shapes:
- `hp`: `(nx+2, ny+2, nz+2, 8, 3)`
- `Fp`: `(nx+2, ny+2, nz+2)`
- `pn_pad`: `(nx+2, ny+2, nz+2, 3)` (PLIC normals)
- `pd_pad`: `(nx+2, ny+2, nz+2)` (PLIC intercepts)

For offset `(di, dj, dk)`, the donor subgrid is extracted via:
```python
dh = dynamic_slice(hp, (1-di, 1-dj, 1-dk, 0, 0), (nx, ny, nz, 8, 3))
```

This convention means:
- Offset `(0, 0, 0)`: self-overlap (donor = acceptor)
- Offset `(-1, 0, 0)`: donor is one cell to the left
- Zero padding = outflow BC (fluid exiting the domain is lost)

## B.9 NCDHW Convolution Layout

The 3D convolution for PLIC normals uses the NCDHW format:

```
N = batch size = 1
C = channels = 1 (input), 3 (output)
D = depth = z-axis (Nz)
H = height = x-axis (Nx)
W = width = y-axis (Ny)
```

**Axis mapping**:

| Physical | Array axis | NCDHW role | Kernel axis |
|----------|-----------|------------|-------------|
| x (Nx)  | 0         | H (axis 3) | axis 1      |
| y (Ny)  | 1         | W (axis 4) | axis 2      |
| z (Nz)  | 2         | D (axis 2) | axis 0      |

Kernel shape: `(3, 1, 3, 3, 3)` = `(O, I, D, H, W)`.

Conv output shape: `(1, 3, Nz-2, Nx-2, Ny-2)` = `(N, O, D', H', W')`.

VALID padding loses 1 on each side per spatial dimension.

---

# Module C: Interface and I/O Contract

## C.1 advect_vof_lagrangian_3d

**File**: `__init__.py`, lines 37-207

```python
def advect_vof_lagrangian_3d(
    F: Array,            # float32[Ntx, Nty, Ntz, 1], read-only input
    u_face: Array,       # float32[Ntx-1, Nty, Ntz], read-only
    v_face: Array,       # float32[Ntx, Nty-1, Ntz], read-only
    w_face: Array,       # float32[Ntx, Nty, Ntz-1], read-only
    grid: GridInfo,      # immutable NamedTuple
    dt: float,           # timestep
    obs_x_max: float = float("-inf"),  # obstacle x-bound, -inf = no obstacle
    obs_y_max: float = float("-inf"),
    obs_z_max: float = float("-inf"),
) -> Array:              # float32[Ntx, Nty, Ntz, 1], new array
```

**Pure function**: No side effects. Allocates new output array.

**Dispatch**: Calls `overlay_lagrangian_3d_native` (C path) if
`is_native_available()` returns True, otherwise falls back to
`overlay_lagrangian_3d` (JAX path).

**Output**: New F array with interior clipped to [0, 1] and halos
extrapolated via nearest-neighbor from interior boundary cells.

## C.2 compute_plic_normals_3d

**File**: `reconstruction_3d.py`, lines 50-142

```python
def compute_plic_normals_3d(
    F: Array,     # float32[Nx, Ny, Nz, 1]
    dx: float,
    dy: float,
    dz: float,
) -> tuple:       # (nx, ny, nz) each float32[Nx, Ny, Nz, 1]
```

**Pure function**. The normals point from fluid ($F=1$) toward empty ($F=0$).

Boundary cells (outermost layer) have zero normals due to VALID convolution.
Pure cells ($F \le 10^{-6}$ or $F \ge 1 - 10^{-6}$) have zero normals.

## C.3 compute_intercept_C_3d

**File**: `reconstruction_3d.py`, lines 235-281

```python
def compute_intercept_C_3d(
    F: Array,      # float32[Nx, Ny, Nz, 1]
    nx: Array,     # float32[Nx, Ny, Nz, 1]
    ny: Array,
    nz: Array,
    dx: float, dy: float, dz: float,
    n_iter: int = 20,
) -> Array:        # float32[Nx, Ny, Nz, 1], center-relative intercept
```

**Pure function**. Uses `jax.lax.fori_loop` internally (no Python loop).

The returned intercept $C$ is in center-relative coordinates:
$C = 0$ means the plane passes through the cell center.

Non-interface cells ($F \le 10^{-6}$ or $F \ge 1 - 10^{-6}$) get $C = 0$.

## C.4 _volume_below_3d and _volume_below_3d_2dmap

**File**: `reconstruction_3d.py`, lines 150-232

```python
def _volume_below_3d(
    C: Array,    # scalar or (Nz,) via vmap
    a: Array,    # scalar or (Nz,) via vmap
    b: Array,
    c: Array,
    **_kw,       # ignored keyword args for compat
) -> Array:      # same shape as inputs
```

**Pure function**. Vectorizable via `jnp.where` cascade (no Python branches).

```python
_volume_below_3d_2dmap = jax.vmap(jax.vmap(_volume_below_3d))
```

When called with `(Nx, Ny, Nz)` arrays, the double vmap maps over axes 0 (Nx)
and 1 (Ny), leaving axis 2 (Nz) as the innermost vectorized dimension.

## C.5 _vertex_displacement_from_faces

**File**: `move_3d.py`, lines 80-159

```python
def _vertex_displacement_from_faces(
    face_disp: Array,     # varies by axis (see below)
    axis: int,            # 0=x, 1=y, 2=z
    Ntot_x: int,
    Ntot_y: int,
    Ntot_z: int,
) -> Array:               # float32[Ntot_x+1, Ntot_y+1, Ntot_z+1]
```

**Input shapes by axis**:
- `axis=0`: `face_disp` has shape `(Ntot_x-1, Ntot_y, Ntot_z)`
- `axis=1`: `face_disp` has shape `(Ntot_x, Ntot_y-1, Ntot_z)`
- `axis=2`: `face_disp` has shape `(Ntot_x, Ntot_y, Ntot_z-1)`

**Pure function**. Contains a Python for-loop over 3 axes (unrolled at trace
time since the loop is over a compile-time constant [0, 1, 2]).

## C.6 lagrangian_move_faces_3d

**File**: `move_3d.py`, lines 162-228

```python
def lagrangian_move_faces_3d(
    u_face: Array,    # float32[Ntot_x-1, Ntot_y, Ntot_z]
    v_face: Array,    # float32[Ntot_x, Ntot_y-1, Ntot_z]
    w_face: Array,    # float32[Ntot_x, Ntot_y, Ntot_z-1]
    grid: GridInfo,
    dt: float,
) -> tuple:           # (x_verts, y_verts, z_verts)
                      # each float32[Ntot_x+1, Ntot_y+1, Ntot_z+1]
```

**Pure function**. Returns deformed vertex coordinates (base grid + displacement).

## C.7 build_deformed_hexahedra

**File**: `move_3d.py`, lines 236-297

```python
def build_deformed_hexahedra(
    x_verts: Array,   # float32[Ntot_x+1, Ntot_y+1, Ntot_z+1]
    y_verts: Array,
    z_verts: Array,
    grid: GridInfo,
) -> Array:           # float32[nx, ny, nz, 8, 3]
```

**Pure function**. Extracts the `(nx, ny, nz)` interior hexahedra from the
vertex grids, with halo offset `nh`.

The 8 vertices are stacked as `[v0, v1, ..., v7]` using the ordering in B.4:
```python
_vert(di, dj, dk) = stack([
    x_verts[ii+di, jj+dj, kk+dk],
    y_verts[ii+di, jj+dj, kk+dk],
    z_verts[ii+di, jj+dj, kk+dk],
], axis=-1)

hex_verts = stack([
    _vert(0,0,0), _vert(1,0,0), _vert(1,1,0), _vert(0,1,0),
    _vert(0,0,1), _vert(1,0,1), _vert(1,1,1), _vert(0,1,1),
], axis=3)
```

Where `ii = arange(nx)[:, None, None] + nh`, etc.

## C.8 _init_hex_poly

**File**: `overlay_3d.py`, lines 49-62

```python
def _init_hex_poly(hex_verts):
    # hex_verts: (8, 3) float32
    # Returns: (faces, face_nv, n_faces) tuple
    #   faces:   float32[MAX_F, MAX_FV, 3]
    #   face_nv: int32[MAX_F]
    #   n_faces: int32
```

Initializes the first 6 faces from `HEX_FACES`, sets vertex count to 4 for
each, and `n_faces = 6`. Remaining face slots are zero-initialized.

## C.9 _sh_clip_face

**File**: `overlay_3d.py`, lines 87-154

```python
def _sh_clip_face(
    face_verts,   # float32[MAX_FV, 3]
    face_nv,      # int32 (actual vertex count)
    plane_n,      # float32[3] (clipping plane normal)
    plane_d,      # float32 (clipping plane offset)
) -> tuple:
    # out_verts:  float32[MAX_FV, 3]
    # out_nv:     int32
    # inter_pts:  float32[2, 3]  -- up to 2 intersection points
    # n_inter:    int32           -- 0, 1, or 2
```

**Implementation**: `jax.lax.scan` over `jnp.arange(MAX_FV)` (14 steps).

## C.10 _clip_poly_by_plane

**File**: `overlay_3d.py`, lines 162-230

```python
def _clip_poly_by_plane(
    poly,       # (faces, face_nv, n_faces) tuple
    plane_n,    # float32[3]
    plane_d,    # float32
) -> tuple:     # (faces, face_nv, n_faces) updated tuple
```

**Implementation**: vmap's `_sh_clip_face` over all `MAX_F` faces, then
constructs cap face from intersection points.

## C.11 _clip_poly_by_box

**File**: `overlay_3d.py`, lines 238-263

```python
def _clip_poly_by_box(
    poly,       # (faces, face_nv, n_faces) tuple
    box_min,    # float32[3]
    box_max,    # float32[3]
) -> tuple:     # (faces, face_nv, n_faces) updated tuple
```

**Implementation**: `jax.lax.scan` over 6 planes (indices 0-5).

## C.12 _poly_volume

**File**: `overlay_3d.py`, lines 271-303

```python
def _poly_volume(
    poly,       # (faces, face_nv, n_faces) tuple
) -> float32:   # scalar volume
```

**Pure function**. Returns non-negative volume via `jnp.abs(...)`.

## C.13 _hex_box_volume

**File**: `overlay_3d.py`, lines 325-356

```python
def _hex_box_volume(
    hex_verts,  # float32[8, 3]
    box_min,    # float32[3]
    box_max,    # float32[3]
) -> float32:   # scalar overlap volume
```

Includes local coordinate shift and all_inside/all_outside fast paths.

## C.14 _hex_plic_box_volume

**File**: `overlay_3d.py`, lines 359-393

```python
def _hex_plic_box_volume(
    hex_verts,  # float32[8, 3]
    plic_n,     # float32[3] (PLIC normal)
    plic_d,     # float32 (global PLIC intercept)
    F_val,      # float32 (donor volume fraction)
    box_min,    # float32[3]
    box_max,    # float32[3]
) -> tuple:     # (overlap_vol: float32, fluid_vol: float32)
```

Includes local coordinate shift, AABB early exit, and empty cell skip.

## C.15 overlay_lagrangian_3d

**File**: `overlay_3d.py`, lines 654-755

```python
def overlay_lagrangian_3d(
    F,                           # float32[Ntx, Nty, Ntz, 1]
    x_verts, y_verts, z_verts,  # each float32[Ntx+1, Nty+1, Ntz+1]
    grid,                        # GridInfo
    nx_f=None, ny_f=None,       # float32[Ntx, Nty, Ntz, 1] or None
    nz_f=None, C_f=None,        # PLIC data, all-or-nothing
) -> Array:                      # float32[nx, ny, nz] (interior only)
```

**Returns interior-only array** (no halos). The caller must pack into the
full array and extrapolate halos.

When PLIC data is provided (`nx_f is not None`):
- `_overlap_vvv = vmap(vmap(vmap(_hex_plic_box_volume)))`: triple-vmap over
  `(nx, ny, nz)` spatial dimensions.
- Returns `(overlap, fluid_vol)` pair per cell.
- Transfer: `dF * overlap / max(fluid_vol, 1e-20)`.

When PLIC data is None:
- `_overlap_vvv = vmap(vmap(vmap(_hex_box_volume)))`: triple-vmap.
- Returns scalar overlap per cell.
- Transfer: `dF * overlap / cell_vol`.

## C.16 overlay_lagrangian_3d_native

**File**: `overlay_native.py`, lines 154-260

```python
def overlay_lagrangian_3d_native(
    F,                           # float32[Ntx, Nty, Ntz, 1]
    x_verts, y_verts, z_verts,  # each float32[Ntx+1, Nty+1, Ntz+1]
    grid,                        # GridInfo
    nx_f=None, ny_f=None,       # PLIC data (optional)
    nz_f=None, C_f=None,
    obs_mask=None,               # unused in current code
) -> Array:                      # float32[nx, ny, nz] (interior only)
```

**Side effects**: Calls C library via `jax.pure_callback`. The C library uses
OpenMP parallel execution and allocates/frees a `double[N]` accumulator.

Falls back to JAX batched overlay if the shared library is not available.

---

# Module D: Absolute Execution Constraint Flow

## D.1 Top-Level Pipeline

`advect_vof_lagrangian_3d` (`__init__.py` lines 37-207) executes these steps
in **strict sequential order**:

```
Step 1: PLIC normals                  (GPU, compute_plic_normals_3d)
Step 2: Gradient-based interface gate (GPU, raw central-difference)
Step 3: PLIC intercept bisection      (GPU, compute_intercept_C_3d)
Step 4: Lagrangian face move          (GPU, lagrangian_move_faces_3d)
Step 5: Vertex clamping               (GPU, jnp.clip + obstacle push)
Step 6: Overlay dispatch              (CPU/C or GPU/JAX)
Step 7: Obstacle redistribution       (GPU, conditional)
Step 8: Clip F to [0, 1]             (GPU)
Step 9: Pack into full array + halos  (GPU)
```

## D.2 PLIC Normal Computation

**Entry**: `__init__.py` line 76:
```python
nx_f, ny_f, nz_f = compute_plic_normals_3d(F, dx, dy, dz)
```

**Detailed execution** (`reconstruction_3d.py` lines 50-142):

D.2.1: Build weight matrix:
```python
dt_ = F.dtype                                         # preserve dtype
w2d = jnp.array([[1,2,1],[2,4,2],[1,2,1]], dtype=dt_)
w_sum = jnp.sum(w2d)                                  # = 16
```

D.2.2: Build three gradient kernels:
```python
# x-gradient: varies along kernel axis 1 (H=x)
ch0 = zeros(3,3,3, dtype=dt_)
ch0[:, 0, :] = -w2d          # negative weight at slot 0
ch0[:, 2, :] = +w2d          # positive weight at slot 2
ch0 /= (2.0 * w_sum * dx)    # = ch0 / (32 * dx)

# y-gradient: varies along kernel axis 2 (W=y)
ch1 = zeros(3,3,3, dtype=dt_)
ch1[:, :, 0] = -w2d
ch1[:, :, 2] = +w2d
ch1 /= (2.0 * w_sum * dy)

# z-gradient: varies along kernel axis 0 (D=z)
ch2 = zeros(3,3,3, dtype=dt_)
ch2[0, :, :] = -w2d
ch2[2, :, :] = +w2d
ch2 /= (2.0 * w_sum * dz)
```

D.2.3: Assemble kernel:
```python
kernel = jnp.stack([ch0, ch1, ch2])[:, None, :, :, :]
# shape: (3, 1, 3, 3, 3) = (O, I, D, H, W)
```

D.2.4: Reshape F to NCDHW:
```python
F_spatial = F[:, :, :, 0]                              # (Nx, Ny, Nz)
lhs = jnp.transpose(F_spatial, (2, 0, 1))[None, None]  # (1,1,Nz,Nx,Ny)
```

D.2.5: 3D convolution:
```python
grad = jax.lax.conv_general_dilated(
    lhs, kernel,
    window_strides=(1, 1, 1),
    padding="VALID",
    dimension_numbers=("NCDHW", "OIDHW", "NCDHW"),
)
# grad shape: (1, 3, Nz-2, Nx-2, Ny-2)
```

D.2.6: Unpack and transpose back:
```python
dFdx_raw = grad[0, 0]   # (Nz-2, Nx-2, Ny-2) = (D', H', W')
dFdy_raw = grad[0, 1]
dFdz_raw = grad[0, 2]

# Initialize full-size arrays with zeros
dFdx = zeros(Nx, Ny, Nz, dtype=dt_)
dFdy = zeros(Nx, Ny, Nz, dtype=dt_)
dFdz = zeros(Nx, Ny, Nz, dtype=dt_)

# Transpose (1,2,0) maps (D',H',W') -> (H',W',D') = (Nx-2,Ny-2,Nz-2)
dFdx[1:-1, 1:-1, 1:-1] = jnp.transpose(dFdx_raw, (1, 2, 0))
dFdy[1:-1, 1:-1, 1:-1] = jnp.transpose(dFdy_raw, (1, 2, 0))
dFdz[1:-1, 1:-1, 1:-1] = jnp.transpose(dFdz_raw, (1, 2, 0))
```

D.2.7: Normalize with negation:
```python
mag = sqrt(dFdx**2 + dFdy**2 + dFdz**2 + 1e-30)
nx = -dFdx / mag
ny = -dFdy / mag
nz = -dFdz / mag
```

D.2.8: Zero normals in pure cells:
```python
is_interface = (F[..., 0] > 1e-6) & (F[..., 0] < 1.0 - 1e-6)
nx = where(is_interface, nx, 0.0)
ny = where(is_interface, ny, 0.0)
nz = where(is_interface, nz, 0.0)
```

D.2.9: Add trailing singleton:
```python
return nx[..., None], ny[..., None], nz[..., None]
# each: (Nx, Ny, Nz, 1)
```

## D.3 Gradient-Based Interface Gating

**Entry**: `__init__.py` lines 79-95

This is a SEPARATE gradient computation from the Sobel normals, using raw
central differences:

D.3.1: Compute raw gradient (central difference, 2-point stencil):
```python
raw_gx = zeros_like(F)    # (Ntx, Nty, Ntz, 1)
raw_gy = zeros_like(F)
raw_gz = zeros_like(F)

raw_gx[1:-1, :, :, :] = (F[2:, :, :, :] - F[:-2, :, :, :]) / (2.0 * dx)
raw_gy[:, 1:-1, :, :] = (F[:, 2:, :, :] - F[:, :-2, :, :]) / (2.0 * dy)
raw_gz[:, :, 1:-1, :] = (F[:, :, 2:, :] - F[:, :, :-2, :]) / (2.0 * dz)
```

D.3.2: Compute gradient magnitude:
```python
raw_grad = sqrt(raw_gx**2 + raw_gy**2 + raw_gz**2)
```

D.3.3: Apply threshold:
```python
interface_mask = (raw_grad > 0.5)
```

[OBSERVATION] The threshold is a **fixed value 0.5**, not relative to
`raw_grad.max()`. This differs from the description in the task prompt
("gate = grad_mag > 0.01 * grad_mag.max()"). The actual code uses 0.5.
For a unit-cell grid ($\Delta x = \Delta y = \Delta z$), this means only cells
with $|\nabla F| > 0.5$ (i.e., roughly a sharp interface step) are flagged.

D.3.4: Gate the normals:
```python
nx_f = where(interface_mask, nx_f, 0.0)
ny_f = where(interface_mask, ny_f, 0.0)
nz_f = where(interface_mask, nz_f, 0.0)
```

Cells that fail the gradient gate get zero normals, which means:
- The intercept bisection will produce $C = 0$ for these cells.
- The PLIC overlay will treat them as full or empty based on their F value
  (via the full/empty override in the overlay).

## D.4 PLIC Intercept Bisection

**Entry**: `__init__.py` line 97:
```python
C_f = compute_intercept_C_3d(F, nx_f, ny_f, nz_f, dx, dy, dz)
```

**Detailed execution** (`reconstruction_3d.py` lines 235-281):

D.4.1: Compute physical-space coefficients:
```python
a = nx * dx    # (Nx, Ny, Nz, 1)
b = ny * dy
c = nz * dz
```

D.4.2: Compute bisection bracket:
```python
C_half = 0.5 * (abs(a) + abs(b) + abs(c))   # (Nx, Ny, Nz, 1)
C_lo = -C_half
C_hi = +C_half
```

D.4.3: Clip target away from exact 0/1:
```python
F_target = clip(F, 1e-10, 1.0 - 1e-10)
```

D.4.4: Strip trailing singleton for bisection:
```python
a_s = a[..., 0]        # (Nx, Ny, Nz)
b_s = b[..., 0]
c_s = c[..., 0]
C_lo_s = C_lo[..., 0]
C_hi_s = C_hi[..., 0]
F_s = F_target[..., 0]
```

D.4.5: Bisection loop (20 iterations):
```python
C_init = zeros_like(a_s)  # (Nx, Ny, Nz) all zeros

def body_fn(i, state):
    C_mid, lo, hi = state
    vol = _volume_below_3d_2dmap(C_mid, a_s, b_s, c_s)
    below = (vol < F_s)
    lo_new = where(below, C_mid, lo)
    hi_new = where(below, hi, C_mid)
    mid_new = 0.5 * (lo_new + hi_new)
    return (mid_new, lo_new, hi_new)

C_final, _, _ = jax.lax.fori_loop(0, 20, body_fn, (C_init, C_lo_s, C_hi_s))
```

**Note**: `C_init = 0` (cell center), not `0.5*(C_lo+C_hi)`. The first
iteration always evaluates at the cell center, then refines.

D.4.6: Zero non-interface cells:
```python
is_interface = (F[..., 0] > 1e-6) & (F[..., 0] < 1.0 - 1e-6)
C_final = where(is_interface, C_final, 0.0)
```

D.4.7: Restore trailing singleton:
```python
return C_final[..., None]  # (Nx, Ny, Nz, 1)
```

## D.5 Volume Formula Execution

**Called from**: `body_fn` in D.4.5 via `_volume_below_3d_2dmap`.

**Detailed execution** (`reconstruction_3d.py` lines 150-228):

D.5.1: Reflect to non-negative octant:
```python
a0 = abs(a)
b0 = abs(b)
c0 = abs(c)
d = C + 0.5 * (a0 + b0 + c0)
```

D.5.2: Compute degeneracy thresholds:
```python
thr = jnp.asarray(1e-10, a0.dtype)           # absolute floor
max_abc = maximum(a0, maximum(b0, c0)) + jnp.asarray(1e-30, a0.dtype)
thr_rel = jnp.asarray(1e-4, a0.dtype)        # relative threshold
```

D.5.3: Count nonzero coefficients:
```python
nz_a = (a0 > thr_rel * max_abc)
nz_b = (b0 > thr_rel * max_abc)
nz_c = (c0 > thr_rel * max_abc)
n_nz = nz_a.astype(int32) + nz_b.astype(int32) + nz_c.astype(int32)
```

D.5.4: Compute all four branches (always, using `jnp.where`):

**0D branch**:
```python
V_0d = where(d >= 0, 1.0, 0.0)
```

**1D branch**:
```python
m_1d = where(nz_a, a0, where(nz_b, b0, c0)) + thr
V_1d = clip(d / m_1d, 0.0, 1.0)
```

**2D branch**:
```python
vals = jnp.sort(jnp.array([
    where(nz_a, a0, 0.0),
    where(nz_b, b0, 0.0),
    where(nz_c, c0, 0.0),
]), axis=0)                     # CRITICAL: axis=0, NOT default

p2 = maximum(vals[1], thr)
q2 = maximum(vals[2], thr)
S2 = p2 + q2
pq2 = 2.0 * p2 * q2

V_2d = where(d <= 0, 0.0,
        where(d >= S2, 1.0,
        where(d <= p2, d**2 / pq2,
        where(d <= q2, (2.0*d - p2) / (2.0*q2),
                       1.0 - (S2 - d)**2 / pq2))))
```

**3D branch**:
```python
lo = minimum(a0, minimum(b0, c0))
hi = maximum(a0, maximum(b0, c0))
mi = a0 + b0 + c0 - lo - hi

m1 = maximum(lo, thr)
m2 = maximum(mi, thr)
m3 = maximum(hi, thr)
S3 = m1 + m2 + m3
p6 = 6.0 * m1 * m2 * m3

dm1  = maximum(d - m1, 0.0)
dm2  = maximum(d - m2, 0.0)
dm3  = maximum(d - m3, 0.0)
dm12 = maximum(d - m1 - m2, 0.0)
dm13 = maximum(d - m1 - m3, 0.0)
dm23 = maximum(d - m2 - m3, 0.0)
dS   = maximum(d - S3, 0.0)

numer = d**3 - dm1**3 - dm2**3 - dm3**3 + dm12**3 + dm13**3 + dm23**3 - dS**3

V_3d = where(d <= 0, 0.0, where(d >= S3, 1.0, numer / p6))
```

D.5.5: Select by dimensionality:
```python
V = where(n_nz == 0, V_0d,
    where(n_nz == 1, V_1d,
    where(n_nz == 2, V_2d, V_3d)))
```

D.5.6: Final clip:
```python
return clip(V, 0.0, 1.0)
```

## D.6 Lagrangian Face Move

**Entry**: `__init__.py` lines 100-102:
```python
x_verts, y_verts, z_verts = lagrangian_move_faces_3d(
    u_face, v_face, w_face, grid, dt)
```

**Detailed execution** (`move_3d.py` lines 162-228):

D.6.1: Extract grid parameters:
```python
nh = grid.nh
dx, dy, dz = grid.dx, grid.dy, grid.dz
Ntot_x = grid.nx + 2*nh
Ntot_y = grid.ny + 2*nh
Ntot_z = grid.nz + 2*nh
```

D.6.2: x-face displacement:
```python
dudx = zeros_like(u_face)                   # (Ntot_x-1, Ntot_y, Ntot_z)
dudx[1:-1, :, :] = (u_face[2:,:,:] - u_face[:-2,:,:]) / (2.0*dx)
dx_face = u_face * dt / (1.0 + 0.5 * dudx * dt)
```

D.6.3: y-face displacement:
```python
dvdy = zeros_like(v_face)                   # (Ntot_x, Ntot_y-1, Ntot_z)
dvdy[:, 1:-1, :] = (v_face[:,2:,:] - v_face[:,:-2,:]) / (2.0*dy)
dy_face = v_face * dt / (1.0 + 0.5 * dvdy * dt)
```

D.6.4: z-face displacement:
```python
dwdz = zeros_like(w_face)                   # (Ntot_x, Ntot_y, Ntot_z-1)
dwdz[:, :, 1:-1] = (w_face[:,:,2:] - w_face[:,:,:-2]) / (2.0*dz)
dz_face = w_face * dt / (1.0 + 0.5 * dwdz * dt)
```

D.6.5: Convert face displacements to vertex displacements:
```python
dx_verts = _vertex_displacement_from_faces(dx_face, axis=0, Ntot_x, Ntot_y, Ntot_z)
dy_verts = _vertex_displacement_from_faces(dy_face, axis=1, Ntot_x, Ntot_y, Ntot_z)
dz_verts = _vertex_displacement_from_faces(dz_face, axis=2, Ntot_x, Ntot_y, Ntot_z)
```

D.6.6: Compute original vertex positions:
```python
x_orig = grid.x_range[0] - nh*dx + arange(Ntot_x + 1) * dx
y_orig = grid.y_range[0] - nh*dy + arange(Ntot_y + 1) * dy
z_orig = grid.z_range[0] - nh*dz + arange(Ntot_z + 1) * dz

x_grid, y_grid, z_grid = meshgrid(x_orig, y_orig, z_orig, indexing="ij")
```

D.6.7: Add displacement to get deformed vertices:
```python
x_verts = x_grid + dx_verts   # (Ntot_x+1, Ntot_y+1, Ntot_z+1)
y_verts = y_grid + dy_verts
z_verts = z_grid + dz_verts
```

## D.7 Vertex Displacement from Face Average

**Called from**: D.6.5

**Detailed execution** (`move_3d.py` lines 80-159):

For `_vertex_displacement_from_faces(face_disp, axis, Ntot_x, Ntot_y, Ntot_z)`:

D.7.1: Create extended array along primary axis:
```python
full = [Ntot_x, Ntot_y, Ntot_z]
n_face = full[axis] - 1   # number of faces along this axis

shape_all = [Ntot_x, Ntot_y, Ntot_z]
shape_all[axis] = full[axis] + 1   # Ntot+1 entries
disp_all = zeros(shape_all)
```

D.7.2: Fill boundary and interior:
```python
# Example for axis=0:
disp_all[0, :, :] = face_disp[0, :, :]        # copy first face
disp_all[1:-1, :, :] = face_disp[:, :, :]      # all N-1 faces
disp_all[-1, :, :] = face_disp[-1, :, :]       # copy last face
```

Note: `disp_all` has $N_\text{tot}+1$ entries along the primary axis. Entry 0
is a copy of the first face, entries $1$ through $N_\text{tot}-1$ are the
$N_\text{tot}-1$ face values, and entry $N_\text{tot}$ is a copy of the last
face.

D.7.3: Average adjacent entries:
```python
vert_raw = 0.5 * (disp_all[:-1, :, :] + disp_all[1:, :, :])
# shape: (Ntot_x, Ntot_y, Ntot_z)
```

D.7.4: Pad all three axes (in order 0, 1, 2):
```python
for ax in [0, 1, 2]:
    # Pad 1 at end along axis ax
    result = jnp.pad(result, [(0,0), (0,0), (0,0)])  # with (0,1) at axis ax
    # Copy second-to-last to last along axis ax
    result[..., -1] = result[..., -2]   # (pseudocode for axis ax)
```

After all three pads: shape = $(N_\text{tot,x}+1, N_\text{tot,y}+1, N_\text{tot,z}+1)$.

The Python `for ax in [0, 1, 2]` loop is unrolled at JAX trace time (3
constant iterations), so it does NOT create a traced loop.

## D.8 Build Deformed Hexahedra

**Called from**: `overlay_lagrangian_3d` line 671 and `overlay_lagrangian_3d_native`
line 179.

**Detailed execution** (`move_3d.py` lines 236-297):

D.8.1: Compute index arrays with halo offset:
```python
ii = arange(nx)[:, None, None] + nh   # (nx, 1, 1)
jj = arange(ny)[None, :, None] + nh   # (1, ny, 1)
kk = arange(nz)[None, None, :] + nh   # (1, 1, nz)
```

D.8.2: Define vertex extraction helper:
```python
def _vert(di, dj, dk):
    return stack([
        x_verts[ii+di, jj+dj, kk+dk],
        y_verts[ii+di, jj+dj, kk+dk],
        z_verts[ii+di, jj+dj, kk+dk],
    ], axis=-1)  # (nx, ny, nz, 3)
```

D.8.3: Stack 8 vertices in standard ordering:
```python
hex_verts = stack([
    _vert(0,0,0),  # v0 = corner (i,   j,   k  )
    _vert(1,0,0),  # v1 = corner (i+1, j,   k  )
    _vert(1,1,0),  # v2 = corner (i+1, j+1, k  )
    _vert(0,1,0),  # v3 = corner (i,   j+1, k  )
    _vert(0,0,1),  # v4 = corner (i,   j,   k+1)
    _vert(1,0,1),  # v5 = corner (i+1, j,   k+1)
    _vert(1,1,1),  # v6 = corner (i+1, j+1, k+1)
    _vert(0,1,1),  # v7 = corner (i,   j+1, k+1)
], axis=3)  # (nx, ny, nz, 8, 3)
```

The `axis=3` stacks along the vertex dimension.

## D.9 Vertex Clamping

**Entry**: `__init__.py` lines 108-143

D.9.1: Domain boundary clamp:
```python
xv = clip(x_verts, grid.x_range[0], grid.x_range[1])
yv = clip(y_verts, grid.y_range[0], grid.y_range[1])
zv = clip(z_verts, grid.z_range[0], grid.z_range[1])
```

D.9.2: Obstacle detection (Python-level, trace-time):
```python
has_obstacle = (obs_x_max > float("-inf") or
                obs_y_max > float("-inf") or
                obs_z_max > float("-inf"))
```

D.9.3: If has_obstacle, compute original vertex grid:
```python
x0_grid = grid.x_range[0] - nh*dx + arange(Ntot_x+1)*dx
y0_grid = grid.y_range[0] - nh*dy + arange(Ntot_y+1)*dy
z0_grid = grid.z_range[0] - nh*dz + arange(Ntot_z+1)*dz
xv0, yv0, zv0 = meshgrid(x0_grid, y0_grid, z0_grid, indexing="ij")
eps = 1e-6   # float32-appropriate tolerance
```

D.9.4: **Sequential** obstacle face pushback (ORDER-DEPENDENT):

**x-face push** (executed first):
```python
push_x = ((xv0 >= obs_x_max - eps) &   # originally at/past obstacle x-face
           (xv < obs_x_max) &            # deformed INTO obstacle
           (yv < obs_y_max) &            # within obstacle y-extent
           (zv < obs_z_max))             # within obstacle z-extent
xv = where(push_x, obs_x_max, xv)
```

**y-face push** (uses updated `xv` from x-push):
```python
push_y = ((yv0 >= obs_y_max - eps) &
           (yv < obs_y_max) &
           (xv <= obs_x_max + eps) &     # NOTE: <= not < (catches x-pushed vertices)
           (zv < obs_z_max))
yv = where(push_y, obs_y_max, yv)
```

**z-face push** (uses updated `xv` and `yv`):
```python
push_z = ((zv0 >= obs_z_max - eps) &
           (zv < obs_z_max) &
           (xv <= obs_x_max + eps) &     # NOTE: <= (catches x-pushed)
           (yv <= obs_y_max + eps))      # NOTE: <= (catches y-pushed)
zv = where(push_z, obs_z_max, zv)
```

The `<=` with `+ eps` in the dependent axes ensures corner vertices that were
already pushed by a prior axis are still caught by subsequent pushes.

## D.10 Overlay Dispatch

**Entry**: `__init__.py` lines 146-156

```python
if is_native_available():
    F_new = overlay_lagrangian_3d_native(
        F, x_verts, y_verts, z_verts, grid,
        nx_f=nx_f, ny_f=ny_f, nz_f=nz_f, C_f=C_f,
        obs_mask=None,
    )
else:
    F_new = overlay_lagrangian_3d(
        F, x_verts, y_verts, z_verts, grid,
        nx_f=nx_f, ny_f=ny_f, nz_f=nz_f, C_f=C_f,
    )
```

`is_native_available()` checks if `libhexboxclip.so` exists at
`<module_dir>/csrc/libhexboxclip.so`. This is a trace-time Python check, not
a runtime branch.

## D.11 JAX Overlay Path (overlay_lagrangian_3d)

**Entry**: `overlay_3d.py` lines 654-755

D.11.1: Build deformed hexahedra:
```python
hex_v = build_deformed_hexahedra(x_verts, y_verts, z_verts, grid)
# shape: (nx, ny, nz, 8, 3)
```

D.11.2: Build acceptor box grid:
```python
x_acc = linspace(grid.x_range[0], grid.x_range[1], nx+1)
y_acc = linspace(grid.y_range[0], grid.y_range[1], ny+1)
z_acc = linspace(grid.z_range[0], grid.z_range[1], nz+1)
```

D.11.3: Extract interior F:
```python
F_int = F[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]   # (nx, ny, nz)
cell_vol = dx * dy * dz
```

D.11.4: Prepare PLIC data (if `nx_f is not None`):
```python
nx_int = nx_f[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]
ny_int = ny_f[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]
nz_int = nz_f[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]
C_int  = C_f [nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]
```

D.11.5: Compute global PLIC intercept:
```python
# Hex centroid = mean of 8 vertices
hex_cx = mean(hex_v[..., 0], axis=-1)   # (nx, ny, nz)
hex_cy = mean(hex_v[..., 1], axis=-1)
hex_cz = mean(hex_v[..., 2], axis=-1)

# C (center-relative) + n . centroid = global d
plic_d_int = C_int + nx_int*hex_cx + ny_int*hex_cy + nz_int*hex_cz
```

D.11.6: Override full/empty cells:
```python
is_full  = (F_int > 1.0 - 1e-6)
is_empty = (F_int < 1e-6)

plic_d_int = where(is_full, 1e6, where(is_empty, -1e6, plic_d_int))

plic_n_int = stack([nx_int, ny_int, nz_int], axis=-1)   # (nx,ny,nz,3)
has_normal = ~is_full & ~is_empty
plic_n_int = where(has_normal[..., None], plic_n_int, array([0.0, 0.0, 1.0]))
```

For full cells: `d = 1e6` means the PLIC plane is far above the hex, so the
entire hex is "below" the plane (all fluid). No clipping occurs.

For empty cells: `d = -1e6` means the PLIC plane is far below the hex, so
nothing is "below" the plane (no fluid). Everything is clipped away.

D.11.7: Construct triple-vmap overlap function:
```python
_overlap_vvv = vmap(vmap(vmap(_hex_plic_box_volume)))
```

This triple-vmap maps over the `(nx, ny, nz)` spatial dimensions. Each
innermost call processes a single cell.

D.11.8: Pad donor arrays:
```python
hp = pad(hex_v, ((1,1), (1,1), (1,1), (0,0), (0,0)))    # (nx+2, ny+2, nz+2, 8, 3)
Fp = pad(F_int, ((1,1), (1,1), (1,1)))                    # (nx+2, ny+2, nz+2)
pn_pad = pad(plic_n_int, ((1,1), (1,1), (1,1), (0,0)))   # (nx+2, ny+2, nz+2, 3)
pd_pad = pad(plic_d_int, ((1,1), (1,1), (1,1)))           # (nx+2, ny+2, nz+2)
```

D.11.9: Build acceptor box arrays:
```python
ai, aj, ak = meshgrid(arange(nx), arange(ny), arange(nz), indexing="ij")
bmin = stack([x_acc[ai], y_acc[aj], z_acc[ak]], axis=-1)       # (nx,ny,nz,3)
bmax = stack([x_acc[ai+1], y_acc[aj+1], z_acc[ak+1]], axis=-1) # (nx,ny,nz,3)
```

D.11.10: Construct 27 offsets:
```python
offsets = array([
    [di, dj, dk]
    for di in (-1, 0, 1)
    for dj in (-1, 0, 1)
    for dk in (-1, 0, 1)
], dtype=int32)   # (27, 3)
```

D.11.11: Scan over 27 offsets:
```python
def _scan_body(F_acc, offset):
    di, dj, dk = offset[0], offset[1], offset[2]

    # Extract donor subgrid for this offset
    dh = dynamic_slice(hp, (1-di, 1-dj, 1-dk, 0, 0), (nx, ny, nz, 8, 3))
    dF = dynamic_slice(Fp, (1-di, 1-dj, 1-dk), (nx, ny, nz))

    # Extract PLIC data for this offset
    d_pn = dynamic_slice(pn_pad, (1-di, 1-dj, 1-dk, 0), (nx, ny, nz, 3))
    d_pd = dynamic_slice(pd_pad, (1-di, 1-dj, 1-dk), (nx, ny, nz))

    # Compute overlap for all cells simultaneously
    overlap, fluid_vol = _overlap_vvv(dh, d_pn, d_pd, dF, bmin, bmax)

    # Transfer formula
    contrib = dF * overlap / maximum(fluid_vol, float32(1e-20))

    return F_acc + contrib, None

F_new, _ = jax.lax.scan(_scan_body, zeros((nx, ny, nz)), offsets)
```

D.11.12: Final clip:
```python
return clip(F_new, 0.0, 1.0)
```

## D.12 Sutherland-Hodgman Clip Face Scan Body

**Entry**: `overlay_3d.py` lines 87-154, called from `_clip_poly_by_plane`

The scan iterates over `MAX_FV = 14` vertex indices. For each index `i`:

D.12.1: Compute next vertex index (wrapping):
```python
j = where(i < face_nv - 1, i + 1, 0)
active = (i < face_nv)
```

D.12.2: Get vertex pair and distances:
```python
pi, pj = face_verts[i], face_verts[j]
di, dj = dist[i], dist[j]      # precomputed: dist = sum(face_verts * plane_n, axis=1) - plane_d
```

D.12.3: Classify vertices:
```python
i_in = (di <= _EPS) & active     # _EPS = 1e-10
j_in = (dj <= _EPS) & active
```

D.12.4: Compute intersection point on edge i->j:
```python
sd = where(abs(di - dj) < 1e-30, 1e-30, di - dj)
t = clip(di / sd, 0.0, 1.0)
inter = pi + t * (pj - pi)
```

[OBSERVATION] The formula `t = di / (di - dj)` gives the parameter where
the edge crosses the plane. When `di = dj` (edge parallel to plane), the
guard `1e-30` prevents division by zero, and `t` is clamped to [0,1].

D.12.5: Determine cases (mutually exclusive):
```python
case_both_in = i_in & j_in
case_exit    = i_in & ~j_in & active
case_enter   = ~i_in & j_in
emit_any     = case_both_in | case_exit | case_enter
```

D.12.6: Emit first vertex:
```python
v_first = where(case_both_in, pj, inter)
# For both_in: emit pj (the next vertex)
# For exit:    emit inter (intersection, leaving)
# For enter:   emit inter (intersection, entering)

s1 = minimum(n_out, MAX_FV - 1)
out = where(emit_any, out.at[s1].set(v_first), out)
n_out = n_out + where(emit_any, 1, 0)
```

D.12.7: Emit second vertex (only for enter case):
```python
s2 = minimum(n_out, MAX_FV - 1)
out = where(case_enter, out.at[s2].set(pj), out)
n_out = n_out + where(case_enter, 1, 0)
```

For the **enter** case, TWO vertices are emitted: first the intersection
point, then `pj` (the next vertex which is inside). This is the standard
SH convention.

D.12.8: Record intersection for cap face:
```python
has_inter = case_exit | case_enter
si = minimum(n_int, 1)          # clamp to index 0 or 1
inters = where(has_inter, inters.at[si].set(inter), inters)
n_int = n_int + where(has_inter, 1, 0)
```

**Scan carry**: `(out_verts: float32[MAX_FV, 3], n_out: int32,
inters: float32[2, 3], n_int: int32)`

**Scan init**:
```python
init = (zeros((MAX_FV, 3)), int32(0), zeros((2, 3)), int32(0))
```

**Scan indices**: `arange(MAX_FV)` = [0, 1, ..., 13]

## D.13 Clip Polyhedron by Plane

**Entry**: `overlay_3d.py` lines 162-230

D.13.1: vmap SH clip over all faces:
```python
clip_fn = lambda fv, fnv: _sh_clip_face(fv, fnv, plane_n, plane_d)
clipped_faces, clipped_nv, inter_pts, n_inter = vmap(clip_fn)(faces, face_nv)
# clipped_faces: (MAX_F, MAX_FV, 3)
# clipped_nv:    (MAX_F,)
# inter_pts:     (MAX_F, 2, 3)
# n_inter:       (MAX_F,)
```

D.13.2: Identify faces with intersections:
```python
face_active = (arange(MAX_F) < n_faces)
has_inter = (n_inter > 0) & face_active
```

D.13.3: Flatten intersection points:
```python
all_inters = inter_pts.reshape(-1, 3)    # (MAX_F*2, 3) = (28, 3)
k_idx = tile(arange(2), MAX_F)           # [0,1,0,1,...] length 28
f_idx = repeat(arange(MAX_F), 2)         # [0,0,1,1,...] length 28
all_valid = has_inter[f_idx] & (k_idx < n_inter[f_idx])
```

D.13.4: Deduplicate via pairwise distance:
```python
N = MAX_F * 2   # = 28
pdist2 = sum((all_inters[:, None, :] - all_inters[None, :, :])**2, axis=-1)  # (28, 28)
lower = tril(ones((N, N), dtype=bool), k=-1)
is_dup = any(lower & all_valid[None, :] & (pdist2 < 1e-10), axis=1)
unique_valid = all_valid & ~is_dup
n_unique = sum(unique_valid).astype(int32)
```

D.13.5: Polar sort:
```python
# Cap center
cap_sum = sum(where(unique_valid[:, None], all_inters, 0.0), axis=0)
cap_center = cap_sum / (float32(n_unique) + 1e-30)

# Build local 2D axes on clipping plane
e1 = array([1.0, 0.0, 0.0])
e1 = where(abs(dot(plane_n, e1)) > 0.9, array([0.0, 1.0, 0.0]), e1)
e1 = e1 - dot(e1, plane_n) * plane_n
e1 = e1 / (norm(e1) + 1e-30)
e2 = cross(plane_n, e1)

# Compute angles
rel = all_inters - cap_center[None, :]
angles = where(unique_valid,
    arctan2(sum(rel * e2[None, :], axis=1), sum(rel * e1[None, :], axis=1)),
    float32(1e10))     # non-valid get large angle -> sorted to end

order = argsort(angles)
```

D.13.6: Assemble cap face:
```python
cap_face = zeros((MAX_FV, 3), float32)
sorted_pts = all_inters[order]
cap_face[:min(N, MAX_FV)] = sorted_pts[:min(N, MAX_FV)]
cap_nv = minimum(n_unique, int32(MAX_FV))
```

Note: `min(N, MAX_FV)` where $N = 28$ and `MAX_FV = 14`, so this takes the
first 14 sorted points. This is a potential overflow if more than 14 unique
cap vertices exist.

[OBSERVATION] With `MAX_FV = 14`, cap faces with more than 14 vertices would
be truncated. In practice, hex-box clipping produces at most ~6-8 cap vertices.

D.13.7: Add cap to polyhedron:
```python
add_cap = (n_unique >= 3).astype(int32)
cap_slot = minimum(n_faces, int32(MAX_F - 1))

new_faces = where(add_cap, clipped_faces.at[cap_slot].set(cap_face), clipped_faces)
new_nv    = where(add_cap, clipped_nv.at[cap_slot].set(cap_nv), clipped_nv)
new_n_faces = n_faces + add_cap
```

If fewer than 3 unique cap vertices exist, no cap is added (degenerate clip).

## D.14 Clip Polyhedron by Box

**Entry**: `overlay_3d.py` lines 238-263

D.14.1: Define 6 clipping planes (inward-pointing normals for a box):
```python
normals = array([
    [-1,0,0], [+1,0,0],   # x-min, x-max
    [0,-1,0], [0,+1,0],   # y-min, y-max
    [0,0,-1], [0,0,+1],   # z-min, z-max
], dtype=float32)

dvals = array([
    -box_min[0], +box_max[0],
    -box_min[1], +box_max[1],
    -box_min[2], +box_max[2],
], dtype=float32)
```

The half-space for the x-min plane is: $-x \le -x_\text{min}$, i.e.,
$x \ge x_\text{min}$ (inside the box).

D.14.2: Scan over 6 planes:
```python
poly_out, _ = jax.lax.scan(
    lambda p, i: (_clip_poly_by_plane(p, normals[i], dvals[i]), None),
    poly,
    arange(6),
)
```

Each iteration clips by one plane. The scan ensures sequential execution
(each clip depends on the previous result).

## D.15 Polyhedron Volume

**Entry**: `overlay_3d.py` lines 271-303

D.15.1: Unpack polyhedron:
```python
faces, face_nv, n_faces = poly
```

D.15.2: Local coordinate shift:
```python
ref = faces[0, 0, :]                    # (3,) -- first vertex of first face
local = faces - ref[None, None, :]       # (MAX_F, MAX_FV, 3)
```

D.15.3: Fan center for each face:
```python
v0 = local[:, 0, :]                     # (MAX_F, 3)
```

D.15.4: Fan triangulation vertices:
```python
vb = local[:, 1:MAX_FV-1, :]            # (MAX_F, MAX_FV-2, 3) = (14, 12, 3)
vc = local[:, 2:MAX_FV, :]              # (MAX_F, MAX_FV-2, 3) = (14, 12, 3)
```

D.15.5: Triple product:
```python
cross_val = cross(vb, vc)               # (MAX_F, MAX_FV-2, 3)
triple = sum(v0[:, None, :] * cross_val, axis=-1)   # (MAX_F, MAX_FV-2)
```

D.15.6: Masking:
```python
tri_idx = arange(MAX_FV - 2)            # [0, 1, ..., 11]
face_active = arange(MAX_F)[:, None] < n_faces     # (MAX_F, 1) bool
tri_active = tri_idx[None, :] < (face_nv[:, None] - 2)   # (MAX_F, MAX_FV-2) bool
mask = face_active & tri_active
```

D.15.7: Volume:
```python
return abs(sum(where(mask, triple, 0.0))) / 6.0
```

The absolute value handles both CW and CCW face orderings.

## D.16 Hex-Box Volume

**Entry**: `overlay_3d.py` lines 325-356

D.16.1: Local coordinate shift:
```python
ref = hex_verts[0]                 # (3,)
hv = hex_verts - ref[None, :]      # (8, 3)
blo = box_min - ref
bhi = box_max - ref
```

D.16.2: All-inside check (with tolerance `_EPS_BOX = 1e-6`):
```python
inside = (hv[:, 0] >= blo[0] - _EPS_BOX) & (hv[:, 0] <= bhi[0] + _EPS_BOX) &
         (hv[:, 1] >= blo[1] - _EPS_BOX) & (hv[:, 1] <= bhi[1] + _EPS_BOX) &
         (hv[:, 2] >= blo[2] - _EPS_BOX) & (hv[:, 2] <= bhi[2] + _EPS_BOX)
all_inside = all(inside)
```

D.16.3: All-outside check (separating axis, using `_EPS_BOX = 1e-6`):
```python
all_outside = (max(hv[:, 0]) <= blo[0] + _EPS_BOX) |
              (min(hv[:, 0]) >= bhi[0] - _EPS_BOX) |
              (max(hv[:, 1]) <= blo[1] + _EPS_BOX) |
              (min(hv[:, 1]) >= bhi[1] - _EPS_BOX) |
              (max(hv[:, 2]) <= blo[2] + _EPS_BOX) |
              (min(hv[:, 2]) >= bhi[2] - _EPS_BOX)
```

D.16.4: Compute volumes:
```python
poly = _init_hex_poly(hv)
hex_vol = _poly_volume(poly)                         # full hex volume
clipped = _clip_poly_by_box(poly, blo, bhi)
clip_vol = _poly_volume(clipped)
```

D.16.5: Select result:
```python
return where(all_inside, hex_vol, where(all_outside, 0.0, clip_vol))
```

[OBSERVATION] When `all_inside`, the function returns `hex_vol` (computed via
the divergence theorem on the FULL hex), not `clip_vol`. This avoids the
overhead of 6 SH clips for cells entirely within their acceptor box.

## D.17 Hex-PLIC-Box Volume

**Entry**: `overlay_3d.py` lines 359-393

D.17.1: Local coordinate shift:
```python
ref = hex_verts[0]
hv = hex_verts - ref[None, :]
blo = box_min - ref
bhi = box_max - ref
pd = plic_d - dot(plic_n, ref)    # shift PLIC intercept to local coords
```

D.17.2: All-outside check (same as D.16.3):
```python
all_outside = (max(hv[:,0]) <= blo[0]+_EPS_BOX) | ... (6 conditions)
```

D.17.3: Empty cell check:
```python
is_empty = (F_val < 1e-8)
skip = all_outside | is_empty
```

D.17.4: PLIC clip:
```python
poly = _init_hex_poly(hv)
fluid_poly = _clip_poly_by_plane(poly, plic_n, pd)
fluid_vol = _poly_volume(fluid_poly)
```

D.17.5: Box clip:
```python
clipped = _clip_poly_by_box(fluid_poly, blo, bhi)
overlap_vol = _poly_volume(clipped)
```

D.17.6: Return with skip guard:
```python
return (where(skip, 0.0, overlap_vol),
        where(skip, 1e-30, fluid_vol))
```

Note: `fluid_vol` returns `1e-30` (not `0.0`) for skipped cells to prevent
division by zero in the transfer formula.

[OBSERVATION] There is no `all_inside` fast path for the PLIC case. Even when
all 8 hex vertices are inside the box, the PLIC clip still needs to run to
compute `fluid_vol`. This is correct behavior.

## D.18 Native C Overlay Path

**Entry**: `overlay_native.py` lines 154-260

D.18.1: Library loading (lazy, once):
```python
lib_path = os.path.join(os.path.dirname(__file__), "csrc", "libhexboxclip.so")
lib = ctypes.CDLL(lib_path)
```

D.18.2: Build arrays (same as JAX path, D.11.1-D.11.9).

D.18.3: Flatten to contiguous 1D:
```python
hp_flat = array(hp).ravel()       # float32, C-contiguous
Fp_flat = array(Fp).ravel()
bmin_flat = array(bmin_grid).ravel()
bmax_flat = array(bmax_grid).ravel()
```

The `np.ascontiguousarray` call inside the callback ensures proper memory
layout for C.

D.18.4: Call via `jax.pure_callback`:
```python
result_shape = jax.ShapeDtypeStruct((nx, ny, nz), jnp.float32)

# For PLIC path:
F_new = jax.pure_callback(
    _plic_cb, result_shape,
    hp_flat, Fp_flat, pn_flat, pd_flat, bmin_flat, bmax_flat,
)
```

D.18.5: C function signature:
```c
void overlay_all_offsets_plic(
    const float *donor_hexes,     // (Px*Py*Pz*8*3)
    const float *donor_F,         // (Px*Py*Pz)
    const float *donor_plic_n,    // (Px*Py*Pz*3)
    const float *donor_plic_d,    // (Px*Py*Pz)
    const float *box_min_grid,    // (nx*ny*nz*3)
    const float *box_max_grid,    // (nx*ny*nz*3)
    int32_t nx, int32_t ny, int32_t nz,
    float *out_F                  // (nx*ny*nz) OUTPUT
);
```

Where `Px = nx+2`, `Py = ny+2`, `Pz = nz+2` (padded dimensions).

D.18.6: Non-PLIC C function:
```c
void overlay_all_offsets(
    const float *donor_hexes,     // (Px*Py*Pz*8*3)
    const float *donor_F,         // (Px*Py*Pz)
    const float *box_min_grid,    // (nx*ny*nz*3)
    const float *box_max_grid,    // (nx*ny*nz*3)
    float cell_vol,
    int32_t nx, int32_t ny, int32_t nz,
    float *out_F                  // (nx*ny*nz) OUTPUT
);
```

D.18.7: Batch volume computation:
```c
void hex_box_vol_batch(
    const float *hex_verts,   // (N*8*3)
    const float *box_min,     // (N*3)
    const float *box_max,     // (N*3)
    float *out_vols,          // (N,)
    int32_t N
);
```

## D.19 Obstacle Redistribution

**Entry**: `__init__.py` lines 161-186 (only if `has_obstacle`).

D.19.1: Compute obstacle cell indices:
```python
obs_ix = round((obs_x_max - grid.x_range[0]) / dx)
obs_iy = round((obs_y_max - grid.y_range[0]) / dy)
obs_iz = round((obs_z_max - grid.z_range[0]) / dz)
obs_ix = min(obs_ix, nx)
obs_iy = min(obs_iy, ny)
obs_iz = min(obs_iz, nz)
```

These are Python `int` (trace-time constants), not JAX arrays.

D.19.2: Push x-wall boundary fluid:
```python
if 0 < obs_ix < nx:
    F_new[obs_ix, :obs_iy, :obs_iz] += F_new[obs_ix-1, :obs_iy, :obs_iz]
```

The last obstacle column's fluid is redistributed to the first fluid column.

D.19.3: Push y-wall boundary fluid:
```python
if 0 < obs_iy < ny:
    F_new[:obs_ix, obs_iy, :obs_iz] += F_new[:obs_ix, obs_iy-1, :obs_iz]
```

[OBSERVATION] There is no z-wall push. Only x-wall and y-wall boundary fluid
is redistributed. This may cause volume loss at the z-wall of the obstacle.

D.19.4: Build obstacle mask and zero:
```python
x_cc = grid.x_range[0] + (arange(nx) + 0.5) * dx
y_cc = grid.y_range[0] + (arange(ny) + 0.5) * dy
z_cc = grid.z_range[0] + (arange(nz) + 0.5) * dz

obs_mask_3d = ((x_cc[:, None, None] < obs_x_max) &
               (y_cc[None, :, None] < obs_y_max) &
               (z_cc[None, None, :] < obs_z_max))

F_new = where(obs_mask_3d, 0.0, F_new)
```

The mask uses strict `<` (not `<=`), so cells exactly at the obstacle boundary
are NOT zeroed.

## D.20 Clip and Pack Halos

**Entry**: `__init__.py` lines 188-207

D.20.1: Clip to [0, 1]:
```python
F_new = clip(F_new, 0.0, 1.0)
```

D.20.2: Pack into full array:
```python
F_out = zeros_like(F)       # (Ntx, Nty, Ntz, 1)
F_out[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0] = F_new
```

D.20.3: Extrapolate halos (nearest-neighbor):
```python
# x halos
F_out[:nh, :, :, :]   = F_out[nh:nh+1, :, :, :]
F_out[-nh:, :, :, :]  = F_out[-nh-1:-nh, :, :, :]

# y halos
F_out[:, :nh, :, :]   = F_out[:, nh:nh+1, :, :]
F_out[:, -nh:, :, :]  = F_out[:, -nh-1:-nh, :, :]

# z halos
F_out[:, :, :nh, :]   = F_out[:, :, nh:nh+1, :]
F_out[:, :, -nh:, :]  = F_out[:, :, -nh-1:-nh, :]
```

Each halo region is filled by copying the outermost interior cell's value.
With `nh = 1`, this means `F_out[0] = F_out[1]` and `F_out[-1] = F_out[-2]`
(and similarly for y and z).

---

# Appendix E: Tet-Pool Fast Path (Alternative Code)

The following functions exist in `overlay_3d.py` but are NOT called by the
production pipeline (`advect_vof_lagrangian_3d`). They are used by backward-
compatible tet-based overlay functions.

### E.1 _init_tet_poly (lines 65-79)

Builds a 4-face triangular polyhedron from 4 vertices. Face indices:
```python
[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
```

### E.2 _analytic_vol4 (lines 311-317)

Tetrahedron volume via scalar triple product:
$$V = \frac{1}{6} |(\mathbf{v}_1 - \mathbf{v}_0) \cdot ((\mathbf{v}_2 - \mathbf{v}_0) \times (\mathbf{v}_3 - \mathbf{v}_0))|$$

Expanded as explicit determinant (no cross product call).

### E.3 _split_tet_by_plane (lines 447-549)

Branchless tet splitting by a half-space. Returns up to 3 sub-tetrahedra
depending on how many vertices are inside (0, 1, 2, 3, or 4).

### E.4 _clip_pool_by_plane (lines 552-586)

Pool-based tet clipping with `POOL = 10` max sub-tets. Uses prefix-sum
compaction.

### E.5 _tet_box_vol_fast (lines 589-611)

Clips a single tet against a box via 6-plane scan through the pool.

### E.6 _hex_box_volume_tetpool (lines 614-646)

Experimental: hex-box volume via 6-tet decomposition + tet-pool clipping.

### E.7 _clip_tet_by_box (lines 401-429)

SH-based tet-box clipping via polyhedron representation.

### E.8 clip_tets_to_box (lines 436-437)

vmap wrapper for `_clip_tet_by_box` over 6 tets.

---

# Appendix F: Benchmark Results

### F.1 Test 1 -- 45-degree Droplet Advection (100x100x5, CFL=0.45)

| Metric | Value |
|--------|-------|
| abs(dV/V0) after 20 steps | 0.00025% |
| Per-step time (C PLIC) | 1,164 ms |
| JIT compile | 17.9 s |
| Shape | Circular |

### F.2 Test 2 -- Droplet Impacting Obstacle (50x50x5, CFL=0.45)

| Metric | Value |
|--------|-------|
| dV_total after 39 steps | +0.0000% (exact) |
| dV_fluid (after clip) | -2.422% |
| Cumulative overfill | +2.422% |
| Per-step time (C PLIC) | ~900 ms |
| Shape | Symmetric crescent wrapping |
| Volume error sign | Positive (matches paper) |

### F.3 Performance Comparison (1M cells, 250x250x16, local RTX 3050 + Ryzen 5600H)

| Path | Per-step | JIT | Speedup |
|------|----------|-----|---------|
| JAX batched PLIC (old) | 541,000 ms | 2,656 s | baseline |
| C/OpenMP PLIC (current) | **1,621 ms** | 15.5 s | **~334x** |
| C/OpenMP no-PLIC | ~870 ms | 1 s | ~620x |

### F.4 Per-Stage Breakdown (current, 1M cells)

| Stage | Time | % |
|-------|------|---|
| PLIC normals (Sobel, GPU) | 56.6 ms | 3.5% |
| **PLIC intercept (bisection, GPU)** | **602.7 ms** | **37.2%** |
| Face move (GPU) | 89.8 ms | 5.5% |
| Vertex clamp (GPU) | 2.5 ms | 0.2% |
| **Overlay (C/OpenMP SH PLIC)** | **642.6 ms** | **39.6%** |
| Clip + halo (GPU) | 227.3 ms | 14.0% |
| **TOTAL** | **1621.5 ms** | 100% |

---

# Appendix G: Cross-Reference to EULERIAN_PLIC_SPEC.md

The following components are shared between the Lagrangian and Eulerian PLIC
pipelines and are documented in full in `EULERIAN_PLIC_SPEC.md`:

| This spec section | EULERIAN_PLIC_SPEC section | Component |
|-------------------|---------------------------|-----------|
| A.6 | Section 1 | `_volume_below_3d` (Scardovelli-Zaleski formula) |
| A.5 | Section 2 | Youngs/Sobel normal computation |

### Lagrangian-Specific Differences

1. **Double-vmap pattern**: The Lagrangian pipeline uses
   `_volume_below_3d_2dmap = vmap(vmap(_volume_below_3d))` which maps over
   `(Nx, Ny)` and vectorizes over `Nz`. The Eulerian pipeline uses a different
   vmap structure (1D along each sweep axis).

2. **Bisection vs Analytic+Newton**: The Lagrangian pipeline uses 20-step
   bisection for the intercept. The Eulerian pipeline uses the analytic
   Scardovelli-Zaleski inverse formula with optional Newton refinement.

3. **Gradient gating**: The Lagrangian pipeline uses a fixed threshold
   `|grad_F| > 0.5`. The Eulerian pipeline uses the interface mask from the
   volume formula (`F > 1e-6 & F < 1-1e-6`).

4. **Dtype**: The Lagrangian `compute_plic_normals_3d` creates `w2d` with
   `dtype=F.dtype`, matching the Eulerian convention. However, the Lagrangian
   `_prepare_gl_quadrature` casts to `jnp.float32` explicitly (dead code,
   not used in production).

5. **NCDHW layout**: The Lagrangian normal computation uses
   `conv_general_dilated` with explicit NCDHW layout and `(2,0,1)` /
   `(1,2,0)` transposes. The Eulerian pipeline may use a different layout
   convention.

---

## Complete Tolerance Reference (Quick Lookup)

| Constant | Value | File | Line | Purpose |
|----------|-------|------|------|---------|
| `_EPS` | `1e-10` | `overlay_3d.py` | 25 | SH inside/outside classification |
| `_EPS_BOX` | `1e-6` (float32) | `overlay_3d.py` | 26 | AABB overlap tolerance |
| `MAX_F` | `14` | `overlay_3d.py` | 23 | Max polyhedron faces |
| `MAX_FV` | `14` | `overlay_3d.py` | 24 | Max vertices per face |
| `POOL` | `10` | `overlay_3d.py` | 444 | Max sub-tets in pool (compat) |
| Norm eps | `1e-30` | `reconstruction_3d.py` | 131 | Normal magnitude guard |
| Interface lo | `1e-6` | `reconstruction_3d.py` | 137 | Pure empty threshold |
| Interface hi | `1-1e-6` | `reconstruction_3d.py` | 137 | Pure full threshold |
| Vol floor `thr` | `1e-10` | `reconstruction_3d.py` | 168 | Absolute denominator guard |
| Rel threshold | `1e-4` | `reconstruction_3d.py` | 170 | Coefficient zero detection |
| Max_abc guard | `1e-30` | `reconstruction_3d.py` | 169 | Prevent max_abc = 0 |
| F_target clip lo | `1e-10` | `reconstruction_3d.py` | 256 | Bisection target floor |
| F_target clip hi | `1-1e-10` | `reconstruction_3d.py` | 256 | Bisection target ceiling |
| Bisection iters | `20` | `reconstruction_3d.py` | 275 | Default n_iter |
| SH division guard | `1e-30` | `overlay_3d.py` | 116 | Edge-plane intersection |
| SH t clip | `[0, 1]` | `overlay_3d.py` | 117 | Intersection parameter |
| Dedup eps2 | `1e-10` | `overlay_3d.py` | 190 | Cap vertex squared distance |
| Polar norm eps | `1e-30` | `overlay_3d.py` | 202 | Gram-Schmidt normalization |
| Polar axis flip | `0.9` | `overlay_3d.py` | 200 | Threshold for switching e1 |
| Non-valid angle | `1e10` | `overlay_3d.py` | 209 | Sort sentinel |
| Cap center eps | `1e-30` | `overlay_3d.py` | 196 | Cap centroid denominator |
| Empty donor | `1e-8` | `overlay_3d.py` | 379 | Skip threshold in PLIC |
| Fluid vol guard | `1e-30` | `overlay_3d.py` | 393 | fluid_vol skip return |
| Transfer denom | `1e-20` | `overlay_3d.py` | 746 | PLIC transfer formula |
| Full cell | `1-1e-6` | `overlay_3d.py` | 703 | PLIC d override |
| Empty cell | `1e-6` | `overlay_3d.py` | 704 | PLIC d override |
| PLIC full d | `1e6` | `overlay_3d.py` | 705 | Large positive d |
| PLIC empty d | `-1e6` | `overlay_3d.py` | 705 | Large negative d |
| Default normal | `(0,0,1)` | `overlay_3d.py` | 712 | Full/empty n override |
| Gradient gate | `0.5` | `__init__.py` | 92 | Interface mask threshold |
| Obstacle eps | `1e-6` | `__init__.py` | 127 | Vertex push tolerance |
