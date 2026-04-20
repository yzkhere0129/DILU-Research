# Eulerian PLIC Technical Specification -- Complete Module A-D

> **Goal**: Another AI that has never seen the code should be able to perfectly reproduce every line of code solely from this document.
>
> **Coverage**: 8 modules under `src/jax_laseram/vof/plic/`, plus two core functions called from the shared implementation
> `lagrangian_3d/reconstruction_3d.py`.
>
> **Dependency order**:
> `volume_formula` -> `normal_youngs` -> `intercept_solver` -> `analytic_intercept` -> `geometric_flux` -> `conservative_bounds` -> `strang_sweep` -> `diagnostics`

---

## 0. Precision Mode — Dtype Parametric Interface

All pipeline functions are **dtype-parametric**: the dtype of the output is
determined by the dtype of the input volume-fraction array `F` (and, for
velocity-aware operators, the dtype of the face-velocity array `u_face` /
`v_face` / `w_face`). The module constants (Prewitt kernel weights, degeneracy
thresholds, clip bounds) are cast via `jnp.asarray(value, F.dtype)` or
`jnp.array(..., dtype=F.dtype)` so that no hidden dtype promotion occurs during
convolution, analytic intercept, or Newton refinement.

### 0.1 Supported dtypes

| `F.dtype` | JAX activation | Typical use |
|-----------|---------------|-------------|
| `float32` (default) | always active | Performance-critical production |
| `float64`           | requires `jax.config.update("jax_enable_x64", True)` or env `JAX_ENABLE_X64=1` set **before** any `jnp.array` call | Verification / conservation validation |

### 0.2 Numerical behavior per dtype (Zalesak 3D 128³, 90 steps)

| Config                     | V drift          | L1 shape | ms/step |
|----------------------------|-----------------:|---------:|--------:|
| f32 + clip                 | $-7.2 \times 10^{-3}\%$ | 142.16% | 194     |
| f32 + redistribute (K=3)   | $-6.9 \times 10^{-6}\%$ | 142.16% | 120     |
| f64 + clip                 | $+3.7 \times 10^{-6}\%$ | 142.16% | 734     |
| **f64 + redistribute (K=3)** | **$\pm 1.3 \times 10^{-14}\%$ (machine zero)** | 142.16% | 689     |

L1 shape error is identical across dtypes because it is dominated by the
$O(\Delta x)$ truncation error of the Youngs normal reconstruction, not by
floating-point rounding. Every numerical claim in this spec is dtype-parametric
unless an explicit `[OBSERVATION-f32]` or `[OBSERVATION-f64]` tag says otherwise.

### 0.3 Hardcoded dtype sources (MUST match F.dtype at runtime)

| File | Line | Value | How to keep parametric |
|------|------|-------|-----------------------|
| `analytic_intercept.py` | 59 | `thr = jnp.asarray(1e-10, F.dtype)` | automatic |
| `conservative_bounds.py` | 112, 127, 138 | `dtype = u_face.dtype if hasattr(u_face,'dtype') else jnp.float32` | automatic |
| `conservative_bounds.py` | 121, 132, 143 | `.astype(u_face.dtype)` (fallback to `F.dtype`) | automatic |
| `lagrangian_3d/reconstruction_3d.py` | 69 | `w2d = jnp.array([[1,2,1],[2,4,2],[1,2,1]], dtype=F.dtype)` | automatic |
| `lagrangian_3d/reconstruction_3d.py` | 167, 169 | `thr = jnp.asarray(1e-10, a0.dtype)`, `thr_rel = jnp.asarray(1e-4, a0.dtype)` | automatic |

---

## Table of Contents

- [1. volume\_formula.py](#1-volume_formulapy)
- [2. normal\_youngs.py (+ compute\_plic\_normals\_3d)](#2-normal_youngspy)
- [3. intercept\_solver.py](#3-intercept_solverpy)
- [4. geometric\_flux.py](#4-geometric_fluxpy)
- [5. strang\_sweep.py](#5-strang_sweeppy) **(now supports dual bounds modes: clip OR conservative redistribute)**
- [6. conservative\_bounds.py](#6-conservative_boundspy) **(Weymouth-Zaleski 2010 redistribute — see `specs/CONSERVATIVE_REDISTRIBUTE_SPEC.md` for full Module A-D)**
- [7. diagnostics.py](#7-diagnosticspy)
- [8. analytic\_intercept.py](#8-analytic_interceptpy)

---

# 1. volume\_formula.py

## 1.1 Module A: Mathematics

### 1.1.1 Problem Definition

Given a unit cube $[0,1]^3$ and a plane $ax + by + cz = C$, compute the volume truncated by the plane:

$$V(C; a, b, c) = \mathrm{Vol}\{(x,y,z) \in [0,1]^3 : ax + by + cz \le C\}$$

where $(a, b, c) = (n_x \Delta x,\; n_y \Delta y,\; n_z \Delta z)$, and $C$ is the intercept in the physical cell-center coordinate system.

### 1.1.2 Coordinate Transform: Physical Intercept -> Unit Cube Parameter d

The code first reflects the coefficients into the non-negative octant, then applies a center offset:

$$a_0 = |a|, \quad b_0 = |b|, \quad c_0 = |c|$$

$$d = C + \frac{1}{2}(a_0 + b_0 + c_0)$$

**Physical meaning**: $d$ is the signed distance from the plane to the unit cube "origin corner" $(0,0,0)$ (scaled by the normal vector).
- When $d = 0$, the plane passes through the origin corner, $V = 0$
- When $d = a_0 + b_0 + c_0$, the plane passes through the diagonal corner, $V = 1$
- $d = \frac{1}{2}(a_0 + b_0 + c_0)$ corresponds to $C = 0$, i.e., the plane passes through the cell center, $V = 0.5$

### 1.1.3 Degeneracy Detection

Whether a coefficient is effectively zero is determined via a **relative threshold**:

$$\mathrm{max\_abc} = \max(a_0, b_0, c_0) + 10^{-30}$$

$$\mathrm{nz\_a} = (a_0 > 10^{-4} \cdot \mathrm{max\_abc})$$

$$n_{\mathrm{nz}} = \mathrm{nz\_a} + \mathrm{nz\_b} + \mathrm{nz\_c} \in \{0, 1, 2, 3\}$$

### 1.1.4 Branch 0D ($n_{\mathrm{nz}} = 0$): Uniform Field

$$V_{0D} = \begin{cases} 1 & d \ge 0 \\ 0 & d < 0 \end{cases}$$

### 1.1.5 Branch 1D ($n_{\mathrm{nz}} = 1$): Slab

Only one direction has a nonzero coefficient. Take that nonzero coefficient $m_{1D}$:

$$m_{1D} = \begin{cases} a_0 & \mathrm{nz\_a} \\ b_0 & \mathrm{nz\_b} \wedge \neg\mathrm{nz\_a} \\ c_0 & \text{otherwise} \end{cases} + \epsilon_{\mathrm{abs}}$$

$$V_{1D} = \mathrm{clip}\!\left(\frac{d}{m_{1D}},\; 0,\; 1\right)$$

where $\epsilon_{\mathrm{abs}} = 10^{-10}$ (absolute lower bound to prevent division by zero).

### 1.1.6 Branch 2D ($n_{\mathrm{nz}} = 2$): Triangle/Trapezoid

Sort the three coefficients (with zero coefficients set to 0) and take the two larger ones:

$$\text{vals} = \mathrm{sort}\!\left(\begin{bmatrix} a_0 \cdot \mathrm{nz\_a} \\ b_0 \cdot \mathrm{nz\_b} \\ c_0 \cdot \mathrm{nz\_c} \end{bmatrix}, \; \text{axis}=0\right)$$

$$p = \max(\text{vals}[1],\; \epsilon_{\mathrm{abs}}), \quad q = \max(\text{vals}[2],\; \epsilon_{\mathrm{abs}})$$

$$S_2 = p + q, \quad P_2 = 2pq$$

$$V_{2D} = \begin{cases}
0 & d \le 0 \\
\dfrac{d^2}{P_2} & 0 < d \le p \\
\dfrac{2d - p}{2q} & p < d \le q \\
1 - \dfrac{(S_2 - d)^2}{P_2} & q < d < S_2 \\
1 & d \ge S_2
\end{cases}$$

[OBSERVATION] The 2D branch assumes $p \le q$ (guaranteed by sorting); when $p > q$, the third segment condition $d \le q$ is never satisfied, and the fourth segment takes over directly. The formula is still correct due to the symmetry of inclusion-exclusion.

### 1.1.7 Branch 3D ($n_{\mathrm{nz}} = 3$): Full Scardovelli-Zaleski Inclusion-Exclusion

Sort the three coefficients:

$$m_1 = \min(a_0, b_0, c_0), \quad m_3 = \max(a_0, b_0, c_0), \quad m_2 = a_0 + b_0 + c_0 - m_1 - m_3$$

Safe lower bounds:

$$m_1' = \max(m_1, \epsilon_{\mathrm{abs}}), \quad m_2' = \max(m_2, \epsilon_{\mathrm{abs}}), \quad m_3' = \max(m_3, \epsilon_{\mathrm{abs}})$$

$$S_3 = m_1' + m_2' + m_3', \quad P_6 = 6 m_1' m_2' m_3'$$

Define the truncated positive part function $[\cdot]^+ = \max(\cdot, 0)$:

$$d_{m_1} = [d - m_1']^+, \quad d_{m_2} = [d - m_2']^+, \quad d_{m_3} = [d - m_3']^+$$

$$d_{m_{12}} = [d - m_1' - m_2']^+, \quad d_{m_{13}} = [d - m_1' - m_3']^+, \quad d_{m_{23}} = [d - m_2' - m_3']^+$$

$$d_S = [d - S_3]^+$$

**Inclusion-exclusion formula**:

$$\mathrm{numer} = d^3 - d_{m_1}^3 - d_{m_2}^3 - d_{m_3}^3 + d_{m_{12}}^3 + d_{m_{13}}^3 + d_{m_{23}}^3 - d_S^3$$

$$V_{3D} = \begin{cases}
0 & d \le 0 \\
1 & d \ge S_3 \\
\mathrm{numer} / P_6 & \text{otherwise}
\end{cases}$$

### 1.1.8 Final Selection and Clamping

$$V = \begin{cases}
V_{0D} & n_{\mathrm{nz}} = 0 \\
V_{1D} & n_{\mathrm{nz}} = 1 \\
V_{2D} & n_{\mathrm{nz}} = 2 \\
V_{3D} & n_{\mathrm{nz}} = 3
\end{cases}$$

$$V_{\mathrm{out}} = \mathrm{clip}(V, 0, 1)$$

### 1.1.9 Variable-to-Code Mapping Table

| Math Symbol | Code Variable | Description |
|---------|---------|------|
| $a_0, b_0, c_0$ | `a0, b0, c0` | Reflected coefficient absolute values |
| $d$ | `d` | Intercept in unit cube coordinate system |
| $\epsilon_{\mathrm{abs}}$ | `thr` | `1e-10`, absolute denominator lower bound |
| $\epsilon_{\mathrm{rel}}$ | `thr_rel` | `1e-4`, relative zero-detection threshold |
| $10^{-30}$ | (inline) | Addend to prevent `max_abc` from being zero |
| $n_{\mathrm{nz}}$ | `n_nz` | Count of nonzero coefficients |
| $p, q$ | `p2, q2` | Two nonzero coefficients after sorting in the 2D branch |
| $m_1, m_2, m_3$ | `lo, mi, hi` (before sorting) / `m1, m2, m3` (after safe lower bound) | Sorted coefficients in the 3D branch |
| $S_3, P_6$ | `S3, p6` | Coefficient sum and six-fold product |
| $d_{m_1}$ etc. | `dm1, dm2, dm3, dm12, dm13, dm23, dS` | Truncated positive part terms |

### 1.1.10 Summary of All Epsilon/Tolerance Values

| Value | Variable | Purpose |
|---|------|------|
| `1e-30` | (inline in `max_abc`) | Prevents `max_abc` from being exactly zero, which would cause `thr_rel * max_abc = 0` |
| `1e-10` | `thr` | Absolute denominator lower bound, prevents $m_{1D}$, $p$, $q$, $m_1'$, $m_2'$, $m_3'$ from being exactly zero |
| `1e-4` | `thr_rel` | Relative zero detection for coefficients: values less than $0.01\%$ of $\max(|a|,|b|,|c|)$ are treated as zero |

## 1.2 Module B: Data Structures

### 1.2.1 Shape Polymorphism of `volume_below_plane_3d(C, a, b, c)`

All inputs `C, a, b, c` are broadcast-compatible JAX arrays, which can be:
- Scalar (0-d)
- 1-D `(N,)`
- 2-D `(M, N)`
- 3-D `(Nx, Ny, Nz)`

Output has the same shape as the broadcast result of the inputs; dtype = `C.dtype` (which in turn inherits from `F.dtype`: float32 by default, float64 when x64 mode is active).

### 1.2.2 Implementation Location

`volume_formula.py::volume_below_plane_3d(C, a, b, c)` is a **thin wrapper** that directly calls:

```
lagrangian_3d/reconstruction_3d.py::_volume_below_3d(C, a, b, c)
```

Zero logic, only does a re-export.

## 1.3 Module C: Interface

```python
def volume_below_plane_3d(C, a, b, c) -> jnp.ndarray
```

- **Input range**: $C \in \mathbb{R}$, $a,b,c \in \mathbb{R}$
- **Output range**: $[0, 1]$ (final `clip`)
- **Side effects**: None
- **Purity**: Pure function, `jit`/`vmap`/`grad` safe
- **JAX semantics**: All 4 branches are implemented via nested `jnp.where`, both sides are always evaluated (no short-circuiting), as required by the JAX tracer

## 1.4 Module D: Execution Flow

**D.1** Take absolute values: `a0 = |a|`, `b0 = |b|`, `c0 = |c|`. ORDER-INDEPENDENT.

**D.2** Compute `d = C + 0.5*(a0+b0+c0)`. Depends on D.1.

**D.3** Compute `max_abc = max(a0, max(b0, c0)) + 1e-30`. Depends on D.1.

**D.4** Compute `nz_a = a0 > 1e-4 * max_abc` and `nz_b`, `nz_c`, `n_nz`. Depends on D.3.

**D.5** Compute all four branches $V_{0D}$, $V_{1D}$, $V_{2D}$, $V_{3D}$ in parallel. JAX has no short-circuiting; all branches are evaluated. ORDER-INDEPENDENT (among each other).

**D.5.1 (Critical constraint on `jnp.sort(axis=0)` in the 2D branch)**: `jnp.array([v_a, v_b, v_c])` creates an array of shape `(3,) + input_shape`, and `axis=0` sorts along the coefficient axis. When inputs are `(Nz,)` vectors (produced by `vmap`), `jnp.array([...])` has shape `(3, Nz)`. If the default `axis=-1` were used, it would sort along the `Nz` dimension (wrong!) instead of along the 3-coefficient dimension. **This `axis=0` is load-bearing and must not be omitted or changed.**

**D.6** Select the final value based on `n_nz`, using nested `jnp.where(n_nz == 0, ..., jnp.where(n_nz == 1, ..., jnp.where(n_nz == 2, V_2d, V_3d)))`.

**D.7** Final `jnp.clip(V, 0, 1)`.

---

# 2. normal\_youngs.py (+ compute\_plic\_normals\_3d)

## 2.1 Module A: Mathematics

### 2.1.1 Youngs Normal Vector

The PLIC interface normal vector is approximated via the gradient of the volume fraction field:

$$\mathbf{n} = -\frac{\nabla F}{\|\nabla F\|}$$

The negative sign makes the normal point from fluid ($F=1$) toward empty ($F=0$).

### 2.1.2 Parker-Youngs 27-Point Weighted Stencil

The gradient components are computed via 3D convolution. The cross-sectional weight matrix:

$$W_{2D} = \begin{pmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{pmatrix}, \quad W_{\mathrm{sum}} = 16$$

This is the outer product of two one-dimensional $(1, 2, 1)$ kernels, providing better isotropy than uniform Prewitt weights $(1,1,1)$.

### 2.1.3 $\partial F / \partial x$ Convolution Kernel Construction

The 3D kernel for the $x$-direction gradient has shape $(3, 3, 3)$ (kernel axes: $D=z$, $H=x$, $W=y$):

$$K_x[\cdot, 0, \cdot] = -W_{2D}, \quad K_x[\cdot, 1, \cdot] = 0, \quad K_x[\cdot, 2, \cdot] = +W_{2D}$$

$$K_x \leftarrow \frac{K_x}{2 \cdot W_{\mathrm{sum}} \cdot \Delta x}$$

Similarly:
- $K_y$: antisymmetric along the $W=y$ axis, $K_y[\cdot, \cdot, 0] = -W_{2D}$, $K_y[\cdot, \cdot, 2] = +W_{2D}$
- $K_z$: antisymmetric along the $D=z$ axis, $K_z[0, \cdot, \cdot] = -W_{2D}$, $K_z[2, \cdot, \cdot] = +W_{2D}$

### 2.1.4 Normalization and Interface Mask

$$\mathrm{mag} = \sqrt{(\nabla_x F)^2 + (\nabla_y F)^2 + (\nabla_z F)^2 + 10^{-30}}$$

$$n_x = -\nabla_x F / \mathrm{mag}, \quad n_y = -\nabla_y F / \mathrm{mag}, \quad n_z = -\nabla_z F / \mathrm{mag}$$

Interface detection:

$$\mathrm{is\_interface} = (F > 10^{-6}) \wedge (F < 1 - 10^{-6})$$

Normal vectors of pure cells ($F \le 10^{-6}$ or $F \ge 1 - 10^{-6}$) are forced to zero.

### 2.1.5 Variable-to-Code Mapping Table

| Math Symbol | Code Variable | Description |
|---------|---------|------|
| $W_{2D}$ | `w2d` | `[[1,2,1],[2,4,2],[1,2,1]]`, dtype = `F.dtype` |
| $W_{\mathrm{sum}}$ | `w_sum` | $= 16$ |
| $K_x, K_y, K_z$ | `ch0, ch1, ch2` | 3D convolution kernels for each component |
| $10^{-30}$ | (inline in `mag`) | Prevents gradient explosion from `sqrt(0)` |
| $10^{-6}$ | (inline in `is_interface`) | Interface/pure cell threshold |

### 2.1.6 All Epsilon/Tolerance Values

| Value | Purpose |
|---|------|
| `1e-30` | Prevents `sqrt(0)` in `mag` computation |
| `1e-6` | Interface detection lower bound |
| `1-1e-6` | Interface detection upper bound |

## 2.2 Module B: Data Structures

### 2.2.1 NCDHW Layout Conversion (Critical, Most Error-Prone)

```
Input F: shape (Nx, Ny, Nz)
  | add channel dimension
F_ch: shape (Nx, Ny, Nz, 1)
  | remove channel: F_spatial = F_ch[:,:,:,0]
F_spatial: shape (Nx, Ny, Nz)  <- F[:,:,:,0] in code

  Meaning: F_spatial[i, j, k] = F at cell (x_i, y_j, z_k)
           axis 0 = x, axis 1 = y, axis 2 = z
```

Convert to NCDHW format:

```
lhs = transpose(F_spatial, (2, 0, 1))[None, None, :, :, :]

  F_spatial axes:  0=x, 1=y, 2=z
  after transpose: 0=z, 1=x, 2=y
  after [None,None]: shape = (1, 1, Nz, Nx, Ny)

  NCDHW correspondence:
    N=1 (batch)
    C=1 (channel)
    D=Nz (z axis)
    H=Nx (x axis)
    W=Ny (y axis)
```

ASCII diagram:

```
Physical space       NCDHW space
=========           ============
x (axis 0)  -->  H (axis 3)
y (axis 1)  -->  W (axis 4)
z (axis 2)  -->  D (axis 2)

F_spatial[x, y, z] = lhs[0, 0, z, x, y]
```

Convolution kernel shape: `(3, 1, 3, 3, 3)` = `(O=3_outputs, I=1_input, D=3, H=3, W=3)`

```
kernel[0] = ch0: dF/dx kernel, antisymmetric along H=x axis
kernel[1] = ch1: dF/dy kernel, antisymmetric along W=y axis
kernel[2] = ch2: dF/dz kernel, antisymmetric along D=z axis
```

Convolution output: `grad` shape = `(1, 3, Nz-2, Nx-2, Ny-2)` (VALID padding, 1 less on each side)

```
grad[0, 0] = dF/dx, in (D,H,W)=(z,x,y) layout
grad[0, 1] = dF/dy, same layout
grad[0, 2] = dF/dz, same layout
```

Inverse transpose back to physical space:

```
dFdx_raw: shape (Nz-2, Nx-2, Ny-2)  <- axes: z, x, y
  | transpose(1, 2, 0)
shape (Nx-2, Ny-2, Nz-2)  <- axes: x, y, z

Embed into full-size zero array:
  dFdx = zeros(Nx, Ny, Nz)
  dFdx[1:-1, 1:-1, 1:-1] = transposed_raw
```

### 2.2.2 Shape Adaptation of the Wrapper `compute_youngs_normal_3d`

```
Input: F shape (Nx, Ny, Nz)
  | F_ch = F[..., None]
F_ch shape (Nx, Ny, Nz, 1)
  | compute_plic_normals_3d(F_ch, dx, dy, dz)
Returns: nx_ch, ny_ch, nz_ch, each shape (Nx, Ny, Nz, 1)
  | [..., 0] remove channel dimension
Output: nx, ny, nz, each shape (Nx, Ny, Nz)
```

### 2.2.3 Halo Convention

- Input `F` must have at least 1 layer of halo cells on each side ($3 \times 3 \times 3$ stencil requirement)
- VALID padding means the output is 1 smaller than the input on each side: `(Nz-2, Nx-2, Ny-2)`
- Normal vectors on the outermost ring are always zero (guaranteed by zero-initialized array)

## 2.3 Module C: Interface

### 2.3.1 Public API

```python
def compute_youngs_normal_3d(
    F: jnp.ndarray,       # (Nx, Ny, Nz), dtype = F.dtype (float32 default; float64 if x64 active)
    dx: float,
    dy: float,
    dz: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    # Returns (nx, ny, nz), each (Nx, Ny, Nz)
```

- **Input range**: `F` $\in [0, 1]$ (no clipping done; caller is responsible)
- **Output range**: $\|\mathbf{n}\| = 1$ on interface cells, $\mathbf{n} = \mathbf{0}$ on pure cells
- **Side effects**: None
- **Purity**: Pure function

### 2.3.2 Internal API

```python
def compute_plic_normals_3d(
    F: Array,       # (Nx, Ny, Nz, 1), dtype = F.dtype (float32 default; float64 if x64 active), with channel dimension
    dx: float,
    dy: float,
    dz: float,
) -> tuple[Array, Array, Array]
    # Returns (nx, ny, nz), each (Nx, Ny, Nz, 1)
```

**JAX-specific semantics**:
- `jax.lax.conv_general_dilated` uses `padding="VALID"`, no zero padding
- `dimension_numbers=("NCDHW", "OIDHW", "NCDHW")` specifies the axis semantics for input/kernel/output
- `window_strides=(1,1,1)` stride of 1

## 2.4 Module D: Execution Flow

**D.1** Construct the `w2d` weight matrix: `[[1,2,1],[2,4,2],[1,2,1]]`, dtype = `F.dtype`. All kernel channel arrays (`ch0`, `ch1`, `ch2`) are allocated with `jnp.zeros((3,3,3), dtype=F.dtype)` so convolution inputs and weights share dtype (otherwise `lax.conv_general_dilated` raises `TypeError: requires arguments to have the same dtypes`).

**D.2** Construct three convolution kernels `ch0`, `ch1`, `ch2`, each of shape `(3,3,3)`. Each kernel sets the $-W_{2D}$ and $+W_{2D}$ faces along its corresponding axis, then divides by $2 W_{\mathrm{sum}} \Delta x$ (or $\Delta y$, $\Delta z$). ORDER-INDEPENDENT among each other.

**D.3** Stack and reshape: `kernel = stack([ch0, ch1, ch2])[:, None, :, :, :]`, shape `(3, 1, 3, 3, 3)`.

**D.4** Transpose `F` to NCDHW: `F_spatial[i,j,k] = F[i,j,k,0]`; `lhs = transpose(F_spatial, (2,0,1))[None, None]`, shape `(1, 1, Nz, Nx, Ny)`. **ORDER-DEPENDENT on D.3.**

**D.5** Execute `conv_general_dilated(lhs, kernel, strides=(1,1,1), padding=VALID)`, output shape `(1, 3, Nz-2, Nx-2, Ny-2)`.

**D.6** Extract three components from the convolution output: `dFdx_raw = grad[0,0]` etc., each shape `(Nz-2, Nx-2, Ny-2)`.

**D.7** Initialize three full-zero arrays of shape `(Nx, Ny, Nz)`, write `transpose(raw, (1,2,0))` into `[1:-1, 1:-1, 1:-1]`. ORDER-DEPENDENT: must transpose before set.

**D.8** Normalize: `mag = sqrt(dFdx^2 + dFdy^2 + dFdz^2 + 1e-30)`, `nx = -dFdx/mag` etc.

**D.9** Interface mask: `is_interface = (F[...,0] > 1e-6) & (F[...,0] < 1-1e-6)`; use `jnp.where` to set normal vectors of non-interface cells to zero.

**D.10** Add/remove channel dimension (wrapper): input `F[..., None]`, output `nx[..., 0]`.

---

# 3. intercept\_solver.py

## 3.1 Module A: Mathematics

### 3.1.1 Problem Definition

Given a normal vector $(n_x, n_y, n_z)$ and a target volume fraction $F$, find the intercept $C$ such that:

$$V(C;\; n_x \Delta x,\; n_y \Delta y,\; n_z \Delta z) = F$$

where $V$ is `volume_below_plane_3d` from Section 1.

### 3.1.2 Regula Falsi + Bisection Fallback

**Initial bracket**:

$$C_{\mathrm{half}} = \frac{1}{2}(|a| + |b| + |c|), \quad \text{where} \; a = n_x \Delta x, \; b = n_y \Delta y, \; c = n_z \Delta z$$

$$\mathrm{lo}_0 = -C_{\mathrm{half}}, \quad \mathrm{hi}_0 = +C_{\mathrm{half}}$$

$$V(\mathrm{lo}_0) = 0, \quad V(\mathrm{hi}_0) = 1$$

[OBSERVATION] $V(\mathrm{lo}_0) = 0$ and $V(\mathrm{hi}_0) = 1$ are **assumptions**, not computed values. The code hard-codes `v_lo0 = zeros_like(F)`, `v_hi0 = ones_like(F)` instead of calling `volume_below_plane_3d` to verify. This assumption is mathematically rigorous under the Scardovelli-Zaleski formula: when $C = -C_{\mathrm{half}}$, $d = 0$ so $V = 0$; when $C = +C_{\mathrm{half}}$, $d = a_0+b_0+c_0$ so $V = 1$. Omitting these two `volume_below_plane_3d` calls saves 2 forward model evaluations.

**Target clipping**:

$$F_{\mathrm{safe}} = \mathrm{clip}(F,\; 10^{-6},\; 1 - 10^{-6})$$

**Iteration body** (fixed loop of $N_{\mathrm{iter}} = 15$ steps):

Given bracket $(\mathrm{lo}, \mathrm{hi}, V_{\mathrm{lo}}, V_{\mathrm{hi}})$:

1. **Regula Falsi interpolation**:

$$\mathrm{denom} = V_{\mathrm{hi}} - V_{\mathrm{lo}}$$

$$\mathrm{slope\_ok} = (|\mathrm{denom}| > 10^{-6})$$

$$t_{\mathrm{RF}} = \frac{F_{\mathrm{safe}} - V_{\mathrm{lo}}}{\mathrm{safe\_denom}}$$

where $\mathrm{safe\_denom} = \begin{cases} \mathrm{denom} & \mathrm{slope\_ok} \\ 1 & \text{otherwise} \end{cases}$

2. **Clip to prevent sticking at endpoints**:

$$t_{\mathrm{RF}}^{\mathrm{clip}} = \mathrm{clip}(t_{\mathrm{RF}},\; 0.01,\; 0.99)$$

3. **Select t**:

$$t = \begin{cases} t_{\mathrm{RF}}^{\mathrm{clip}} & \mathrm{slope\_ok} \\ 0.5 & \text{otherwise} \end{cases}$$

4. **New probe point**:

$$C_{\mathrm{new}} = \mathrm{lo} + t \cdot (\mathrm{hi} - \mathrm{lo})$$

5. **Forward model evaluation**:

$$V_{\mathrm{new}} = V(C_{\mathrm{new}};\; a, b, c)$$

6. **Bracket update**: If $V_{\mathrm{new}} < F_{\mathrm{safe}}$, replace the lower bound; otherwise replace the upper bound.

**Final interpolation** (after loop):

On the final bracket $(\mathrm{lo}_f, \mathrm{hi}_f, V_{\mathrm{lo},f}, V_{\mathrm{hi},f})$, perform one more Regula Falsi interpolation (without updating state), but this time the clip range for $t$ is $[0, 1]$ instead of $[0.01, 0.99]$:

$$C_{\mathrm{final}} = \mathrm{lo}_f + t_{\mathrm{final}} \cdot (\mathrm{hi}_f - \mathrm{lo}_f)$$

**Pure cell zeroing**:

$$||\mathbf{n}||^2 = n_x^2 + n_y^2 + n_z^2$$

$$C_{\mathrm{out}} = \begin{cases} C_{\mathrm{final}} & ||\mathbf{n}||^2 > 10^{-12} \\ 0 & \text{otherwise} \end{cases}$$

### 3.1.3 Variable-to-Code Mapping Table

| Math Symbol | Code Variable | Description |
|---------|---------|------|
| $a, b, c$ | `a, b, c` | $n_x \Delta x$, $n_y \Delta y$, $n_z \Delta z$ |
| $C_{\mathrm{half}}$ | `C_half` | Bracket half-width |
| $\mathrm{lo}_0, \mathrm{hi}_0$ | `lo0, hi0` | Initial bracket |
| $V_{\mathrm{lo}}$ etc. | `v_lo, v_hi` | Volume fraction values at bracket endpoints |
| $F_{\mathrm{safe}}$ | `F_safe` | Clipped target value |
| $t$ | `t` | Interpolation parameter |
| $N_{\mathrm{iter}}$ | `_DEFAULT_N_ITER` | 15 |

### 3.1.4 All Epsilon/Tolerance Values

| Value | Code Constant | Purpose |
|---|---------|------|
| `1e-6` | `_SLOPE_EPS` | Regula Falsi slope threshold; below this, falls back to bisection |
| `0.01` | `_T_CLIP_LO` | Interpolation parameter lower bound, prevents Regula Falsi from sticking at one end |
| `0.99` | `_T_CLIP_HI` | Interpolation parameter upper bound |
| `1e-6` | `_F_CLIP_LO` | Target volume fraction lower bound |
| `1-1e-6` | `_F_CLIP_HI` | Target volume fraction upper bound |
| `1e-12` | `_N_EPS2` | Normal vector magnitude-squared threshold; below this, treated as pure cell |
| `15` | `_DEFAULT_N_ITER` | Fixed iteration count |

## 3.2 Module B: Data Structures

### 3.2.1 Input/Output Shape

All inputs `nx, ny, nz, F` have the same shape, typically `(Nx, Ny, Nz)`.
Output `C` has the same shape.

### 3.2.2 `fori_loop` State Tuple

```
state = (lo, hi, v_lo, v_hi)
```

Each element has the same shape as the input, dtype = `F.dtype`.

## 3.3 Module C: Interface

```python
def solve_intercept(
    nx: jnp.ndarray,    # Normal vector x component
    ny: jnp.ndarray,    # Normal vector y component
    nz: jnp.ndarray,    # Normal vector z component
    F: jnp.ndarray,     # Target volume fraction
    dx: float,          # Grid spacing in x direction
    dy: float,          # Grid spacing in y direction
    dz: float,          # Grid spacing in z direction
    n_iter: int = 15,   # Fixed iteration count (compile-time constant)
) -> jnp.ndarray        # Intercept C, same shape
```

- **Input range**: `F` $\in [0, 1]$, normal vector is unit vector or zero vector
- **Output range**: $C \in [-C_{\mathrm{half}}, +C_{\mathrm{half}}]$, 0 at pure cells
- **Side effects**: None
- **Purity**: Pure function, `jit` safe
- **JAX-specific semantics**:
  - `jax.lax.fori_loop(0, n_iter, body, init)` unrolls into a fixed 15 steps; each step's body is fully executed (no early exit)
  - `jnp.where` evaluates both sides -- the `below` branch and `!below` branch of bracket update are both computed
  - `n_iter` must be a Python int (compile-time constant), not a JAX tracer

## 3.4 Module D: Execution Flow

**D.1** Compute plane coefficients: $a = n_x \Delta x$, $b = n_y \Delta y$, $c = n_z \Delta z$.

**D.2** Compute bracket half-width: $C_{\mathrm{half}} = 0.5 \cdot (|a| + |b| + |c|)$.

**D.3** Initialize: `lo0 = -C_half`, `hi0 = +C_half`, `v_lo0 = zeros`, `v_hi0 = ones`.

[OBSERVATION] In D.3, `v_lo0` and `v_hi0` are **directly assigned** to 0 and 1, rather than computed by calling `volume_below_plane_3d(lo0, a, b, c)` and `volume_below_plane_3d(hi0, a, b, c)`. This is a correct mathematical simplification (see explanation in 3.1.2), but if someone modifies the bracket initialization in the future without also updating `v_lo0`/`v_hi0`, it would introduce a subtle bug.

**D.4** Clip target: $F_{\mathrm{safe}} = \mathrm{clip}(F, 10^{-6}, 1-10^{-6})$.

**D.5** `fori_loop(0, 15, body, (lo0, hi0, v_lo0, v_hi0))`: exactly 15 iterations.

**D.5.1** (Inside loop body) Compute `denom = v_hi - v_lo`.

**D.5.2** Determine `slope_ok = |denom| > 1e-6`.

**D.5.3** Compute `t_rf = (F_safe - v_lo) / safe_denom`, where `safe_denom = where(slope_ok, denom, 1.0)`.

**D.5.4** Clip: `t_rf_clipped = clip(t_rf, 0.01, 0.99)`.

**D.5.5** Select: `t = where(slope_ok, t_rf_clipped, 0.5)`.

**D.5.6** Probe: `C_new = lo + t * (hi - lo)`.

**D.5.7** Forward model: `v_new = volume_below_plane_3d(C_new, a, b, c)`. ORDER-DEPENDENT on D.5.6.

**D.5.8** Update: `below = v_new < F_safe`; conditionally replace the corresponding end of the bracket.

**D.6** (After loop) Final Regula Falsi interpolation, clip range $[0, 1]$.

**D.7** Pure cell zeroing: determined by `n_sq > 1e-12`.

---

# 4. geometric\_flux.py

## 4.1 Module A: Mathematics

### 4.1.1 Sub-box Concept

In Eulerian PLIC advection, the flux through each face at each time step is computed via the plane-cube intersection on a **sub-box** (donor region). The sub-box is the rectangular sub-region of the donor cell swept by the velocity field across the face.

For x-direction velocity $u$ (scalar):

- Swept width: $|u| \Delta t$ (along the x axis)
- Cross-section: full cell width $\Delta y \times \Delta z$

### 4.1.2 Sub-box Center Coordinate Derivation

**Key formula**: `center_x = sign(u) * dx/2 - u * dt/2`

**Derivation**:

With the donor cell center as origin, the x axis range is $[-\Delta x/2, +\Delta x/2]$.

**Case 1: $u \ge 0$ (rightward flow)**, donor is the left cell (index $i$):
- Fluid flows out from the donor's right edge; the sub-box occupies $[+\Delta x/2 - |u|\Delta t, +\Delta x/2]$
- Sub-box center:

$$x_c = \frac{(+\Delta x/2 - |u|\Delta t) + (+\Delta x/2)}{2} = +\frac{\Delta x}{2} - \frac{|u|\Delta t}{2} = +\frac{\Delta x}{2} - \frac{u\Delta t}{2}$$

$$= \mathrm{sign}(u) \cdot \frac{\Delta x}{2} - \frac{u\Delta t}{2}$$

**Case 2: $u < 0$ (leftward flow)**, donor is the right cell (index $i+1$):
- Fluid flows out from the donor's left edge; the sub-box occupies $[-\Delta x/2, -\Delta x/2 + |u|\Delta t]$
- Sub-box center:

$$x_c = \frac{(-\Delta x/2) + (-\Delta x/2 + |u|\Delta t)}{2} = -\frac{\Delta x}{2} + \frac{|u|\Delta t}{2}$$

Note that when $u < 0$, $|u| = -u$, so $|u|\Delta t/2 = -u\Delta t/2$:

$$x_c = -\frac{\Delta x}{2} - \frac{u\Delta t}{2} = \mathrm{sign}(u) \cdot \frac{\Delta x}{2} - \frac{u\Delta t}{2}$$

Both cases unify to:

$$\boxed{x_c = \mathrm{sign}(u) \cdot \frac{\Delta x}{2} - \frac{u\Delta t}{2}}$$

### 4.1.3 Plane Truncation on the Sub-box

In the sub-box's local coordinate system, the PLIC plane coefficients need to be rescaled:

$$a_{\mathrm{sub}} = n_{\mathrm{axis}} \cdot |u|\Delta t$$

$$b_{\mathrm{sub}} = n_{\mathrm{other1}} \cdot \Delta x_{\mathrm{other1}}$$

$$c_{\mathrm{sub}} = n_{\mathrm{other2}} \cdot \Delta x_{\mathrm{other2}}$$

Intercept relative to the sub-box center:

$$C_{\mathrm{sub}} = C_{\mathrm{donor}} - n_{\mathrm{axis}} \cdot x_c$$

Volume fraction:

$$V_{\mathrm{frac}} = \begin{cases}
V(C_{\mathrm{sub}};\; a_{\mathrm{sub}}, b_{\mathrm{sub}}, c_{\mathrm{sub}}) & \text{interface cell} \\
F_{\mathrm{donor}} & \text{pure cell}
\end{cases}$$

### 4.1.4 Flux Computation

Taking the x-direction as an example:

$$\mathrm{flux\_mag} = V_{\mathrm{frac}} \cdot |u|\Delta t \cdot \Delta y \cdot \Delta z$$

$$\mathrm{flux} = \mathrm{sign}(u) \cdot \mathrm{flux\_mag}$$

**Meaning**: `flux[i,j,k]` is the signed volume flowing rightward through face $(i+1/2, j, k)$ during time $\Delta t$.

### 4.1.5 Conservative Update

$$F_i^{n+1} = F_i^n + \frac{\mathrm{flux}_{i-1/2} - \mathrm{flux}_{i+1/2}}{\Delta x \Delta y \Delta z}$$

Code implementation (`apply_flux_x`):

$$\Delta F_{\mathrm{interior}} = \frac{\mathrm{flux}[:-1] - \mathrm{flux}[1:]}{\mathrm{cell\_vol}}$$

$$F[1:-1] \mathrel{+}= \Delta F_{\mathrm{interior}}$$

Boundary cells (index 0 and -1) are not modified; they are handled by subsequent halo updates.

### 4.1.6 Variable-to-Code Mapping Table

| Math Symbol | Code Variable | Description |
|---------|---------|------|
| $|u|\Delta t$ | `abs_udt` | Swept distance |
| $\mathrm{sign}(u)$ | `sign_u` | Velocity sign |
| $x_c$ | `center_x` | Sub-box center coordinate along the sweep axis |
| $a_{\mathrm{sub}}$ | `a_sub` | Plane coefficient along the sweep axis on the sub-box |
| $C_{\mathrm{sub}}$ | `C_sub` | Intercept on the sub-box |
| $V_{\mathrm{frac}}$ | `V_frac` | Volume fraction in the sub-box |
| $\mathrm{flux\_mag}$ | `flux_mag` | Unsigned flux volume |

### 4.1.7 All Epsilon/Tolerance Values

| Value | Code Constant | Purpose |
|---|---------|------|
| `1e-6` | `_INTERFACE_EPS` | Interface/pure cell detection threshold, consistent with Stage 1 |

## 4.2 Module B: Data Structures

### 4.2.1 Input Arrays

| Array | Shape | Description |
|------|-------|------|
| `F` | `(Nx, Ny, Nz)` | Volume fraction, halo-padded |
| `nx, ny, nz` | `(Nx, Ny, Nz)` | Normal vector components |
| `C` | `(Nx, Ny, Nz)` | Intercept |

### 4.2.2 Flux Arrays

| Function | Output Shape | Description |
|------|-----------|------|
| `sweep_flux_x` | `(Nx-1, Ny, Nz)` | x-face flux |
| `sweep_flux_y` | `(Nx, Ny-1, Nz)` | y-face flux |
| `sweep_flux_z` | `(Nx, Ny, Nz-1)` | z-face flux |

### 4.2.3 Donor Selection Slicing Pattern

Taking the x-direction as an example:

```
F_left  = F[:-1]   # shape (Nx-1, Ny, Nz) -- cell to the left of the face
F_right = F[1:]    # shape (Nx-1, Ny, Nz) -- cell to the right of the face

F_donor = where(u >= 0, F_left, F_right)  # upwind cell
```

### 4.2.4 Update Pattern of `apply_flux_x`

```
dF_interior = (flux_x[:-1] - flux_x[1:]) / cell_vol
  shape: (Nx-2, Ny, Nz)

F = F.at[1:-1].add(dF_interior)  # only update interior, boundaries untouched
```

[OBSERVATION] `F.at[1:-1].add(...)` uses JAX's in-place update semantics, which actually returns a new array. Boundary cells (index 0 and -1) remain unchanged and must be updated by `halo_fn`.

## 4.3 Module C: Interface

### 4.3.1 Internal Core

```python
def _sub_box_flux_kernel(
    F_donor: jnp.ndarray,          # Donor cell's F
    n_axis: jnp.ndarray,           # Normal component along the sweep axis
    n_other1: jnp.ndarray,         # Normal component along the first transverse axis
    n_other2: jnp.ndarray,         # Normal component along the second transverse axis
    C_donor: jnp.ndarray,          # Donor's intercept
    sub_box_center_axis: jnp.ndarray,  # Sub-box center along the sweep axis
    abs_sub_box_axis: jnp.ndarray, # Sub-box width = |u|*dt
    dx_other1: float,              # Grid spacing of transverse axis 1
    dx_other2: float,              # Grid spacing of transverse axis 2
) -> jnp.ndarray                   # V_frac, same shape
```

### 4.3.2 Public Functions (x as example; y/z are mirrors)

```python
def sweep_flux_x(
    F, nx, ny, nz, C,   # each (Nx, Ny, Nz)
    u: float,           # Scalar x velocity
    dt: float,
    dx: float, dy: float, dz: float,
) -> jnp.ndarray       # (Nx-1, Ny, Nz) signed volume flux

def apply_flux_x(
    F: jnp.ndarray,      # (Nx, Ny, Nz)
    flux_x: jnp.ndarray, # (Nx-1, Ny, Nz)
    dx: float, dy: float, dz: float,
) -> jnp.ndarray         # (Nx, Ny, Nz), interior updated, boundaries untouched
```

### 4.3.3 Re-exports from `conservative_bounds.py`

`geometric_flux.py` lines 54-61 re-export all six public symbols from `conservative_bounds.py`:

```python
from .conservative_bounds import (  # noqa: F401  (re-export)
    redistribute_bounds_x,
    redistribute_bounds_y,
    redistribute_bounds_z,
    apply_flux_x_conservative,
    apply_flux_y_conservative,
    apply_flux_z_conservative,
)
```

These re-exports are included in `geometric_flux.__all__`:

```python
__all__ = [
    "sweep_flux_x",
    "sweep_flux_y",
    "sweep_flux_z",
    "apply_flux_x",
    "apply_flux_y",
    "apply_flux_z",
    "apply_flux_x_conservative",
    "apply_flux_y_conservative",
    "apply_flux_z_conservative",
]
```

`strang_sweep.py` imports the conservative functions via `geometric_flux`, NOT directly from `conservative_bounds`:

```python
from .geometric_flux import (
    apply_flux_x,
    apply_flux_y,
    apply_flux_x_conservative,
    apply_flux_y_conservative,
    sweep_flux_x,
    sweep_flux_y,
)
```

A reimplementer MUST ensure that `geometric_flux.py` re-exports these names so that `strang_sweep.py`'s import chain resolves correctly.

### 4.3.4 Normal Vector Component Passing Convention for Three Directions

| Sweep Direction | `n_axis` | `n_other1` | `n_other2` | `dx_other1` | `dx_other2` |
|-----------|----------|-----------|-----------|------------|------------|
| x | `nx` | `ny` | `nz` | `dy` | `dz` |
| y | `ny` | `nx` | `nz` | `dx` | `dz` |
| z | `nz` | `nx` | `ny` | `dx` | `dy` |

## 4.4 Module D: Execution Flow

### 4.4.1 `sweep_flux_x` Execution Sequence

**D.1** `abs_udt = |u| * dt`, `sign_u = sign(u)`.

**D.2** Slice: `F_left = F[:-1]`, `F_right = F[1:]`, similarly slice `nx, ny, nz, C`.

**D.3** Donor selection: `F_donor = where(u >= 0, F_left, F_right)`, similarly select other quantities.

**D.4** Sub-box center: `center_x = sign_u * (dx/2) - (u*dt)/2`.

**D.5** Call `_sub_box_flux_kernel(...)` to get `V_frac`. ORDER-DEPENDENT on D.3 and D.4.

**D.6** `flux_mag = V_frac * abs_udt * dy * dz`.

**D.7** `return sign_u * flux_mag`.

### 4.4.2 `_sub_box_flux_kernel` Execution Sequence

**D.1** Rescale: `a_sub = n_axis * abs_sub_box_axis`.

**D.2** `b_sub = n_other1 * dx_other1`, `c_sub = n_other2 * dx_other2`.

**D.3** Intercept offset: `C_sub = C_donor - n_axis * sub_box_center_axis`. ORDER-DEPENDENT on D.1 (conceptually).

**D.4** `V_frac_plic = volume_below_plane_3d(C_sub, a_sub, b_sub, c_sub)`.

**D.5** Interface detection: `is_interface = (F_donor > 1e-6) & (F_donor < 1-1e-6)`.

**D.6** Selection: `V_frac = where(is_interface, V_frac_plic, F_donor)`. Pure cells directly use $F_{\mathrm{donor}}$ as the volume fraction.

### 4.4.3 `apply_flux_x` Execution Sequence

**D.1** `cell_vol = dx * dy * dz`.

**D.2** `dF_interior = (flux_x[:-1] - flux_x[1:]) / cell_vol`.

- `flux_x[:-1]` shape `(Nx-2, Ny, Nz)`: flux entering from the left face
- `flux_x[1:]` shape `(Nx-2, Ny, Nz)`: flux leaving from the right face

**D.3** `F.at[1:-1].add(dF_interior)`. Only updates $i \in [1, N_x-2]$.

---

# 5. strang\_sweep.py

## 5.1 Module A: Mathematics

### 5.1.1 Strang Splitting

2D Strang splitting decomposes one time step $\Delta t$ of 2D advection into three sub-steps:

$$\mathcal{L}(\Delta t) = \mathcal{L}_x(\Delta t/2) \circ \mathcal{L}_y(\Delta t) \circ \mathcal{L}_x(\Delta t/2) + O(\Delta t^3)$$

This is second-order accurate provided: the velocity field is exactly divergence-free. For the droplet\_45deg benchmark ($u = v = \mathrm{const}$), this condition is exactly satisfied.

### 5.1.2 Sub-step Sequence

The internal flow of each sub-step $\mathcal{L}_d(dt_{\mathrm{sub}})$:

1. **Halo update**: `F = halo_fn(F)` -- fill ghost cells
2. **PLIC reconstruction** (sparse): Youngs normal (full field) + padded-gather interface cells + analytic intercept (compact) + scatter back
3. **Flux computation**: `sweep_flux_d(...)` -- sub-box geometric flux
4. **Conservative update**: `apply_flux_d(...)` -- divergence-form update
5. **Boundedness enforcement** — one of two modes (runtime-selectable):
   - **`bounds='clip'`**: `F = clip(F, 0, 1)` -- simple truncation. NOT volume-conservative (discards overshoot/undershoot mass).
   - **`bounds='redistribute'`** (default in production): `F = apply_flux_d_conservative(F, flux, dt, dd, u_face, n_iter=3)` -- Weymouth-Zaleski 2010 conservative redistribute: `K=3` fixed `fori_loop` iterations that push overshoot to the sweep-downstream neighbor and pull undershoot from the upstream neighbor, followed by a final safety `clip`. See `specs/CONSERVATIVE_REDISTRIBUTE_SPEC.md` for the full Module A-D specification.
6. **Halo update**: `F = halo_fn(F)` -- prepare for next sub-step

[OBSERVATION] Each sub-step calls `halo_fn` at both the beginning and end, meaning that between two consecutive sub-steps `halo_fn` is called twice (end of previous step + beginning of next step). This is redundant but guarantees correctness -- if someone optimizes away one of them in the future, they must ensure the intermediate state is not corrupted.

[OBSERVATION] Mode comparison on Zalesak 3D 128³ (90 steps, 1/10 revolution):
- `clip`: V drift −7.2e−3% (f32) / +3.7e−6% (f64). Fast (~65 ms/step @ f32) but leaks volume at float32 ε-accumulation level.
- `redistribute` (K=3): V drift −6.9e−6% (f32) / ±1.3e−14% (f64, machine zero). ~+17 ms/step @ f32 overhead. Shape L1 error (142.16%) identical between modes — L1 is dominated by Youngs-normal $O(\Delta x)$ truncation, not by rounding.

The `clip(F, 0, 1)` in `redistribute` mode is still applied at the end as a
safety net: the K=3 iteration cascades overshoot up to 3 cells downstream; any
residual `|F-1| < 10^{-5}` (f32) / `10^{-14}` (f64) is absorbed by the final
clip and costs proportionally negligible volume.

### 5.1.3 Full Time Step Orchestration

```
Input:  F^n, u, v, dt, dx, dy, dz, halo_fn

Step 1: F^* = plic_subsweep_x(F^n, u, dt/2, dx, dy, dz, halo_fn)
Step 2: F^** = plic_subsweep_y(F^*, v, dt,   dx, dy, dz, halo_fn)
Step 3: F^{n+1} = plic_subsweep_x(F^**, u, dt/2, dx, dy, dz, halo_fn)

Output: F^{n+1}
```

### 5.1.4 CFL Condition

The documentation states: when $|u|\Delta t/\Delta x + |v|\Delta t/\Delta y \le 0.5$, the local CFL for each sub-step is $\le 0.5$. The formula allows CFL $\le 1$ but recommends $\le 0.5$.

## 5.2 Module B: Data Structures

### 5.2.1 Module-Level Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `_MAX_INTERFACE_FRAC` | `0.15` | Maximum fraction of total cells allocated for the padded-gather compact array. At 15% headroom vs a typical 5-8% interface fraction, overflow is unlikely. |
| `_INTERFACE_EPS` | `1e-6` | Interface detection threshold. Must match `intercept_solver._F_CLIP_LO`. |

### 5.2.2 `halo_fn` Type

```python
HaloFn = Callable[[jnp.ndarray], jnp.ndarray]
```

Accepts `F` of shape `(Nx, Ny, Nz)`, returns the same shape with ghost cells filled.

### 5.2.3 Output of Internal Function `_reconstruct`

```python
def _reconstruct(F, dx, dy, dz) -> (nx, ny, nz, C)
```

All outputs have shape `(Nx, Ny, Nz)`, dtype = `F.dtype`. Internally, only the interface cells (typically 5-8% of total) are gathered into a compact 1D array, solved via `analytic_intercept`, and scattered back. The full-field `C` is zero-initialized with `dtype=F.dtype`, so pure cells get `C = 0`.

### 5.2.4 Padded-Gather Compact Array

The compact array has a **fixed** size `max_n = int(total * _MAX_INTERFACE_FRAC)`, where `total` is the product of all spatial dimensions. This fixed size is a JAX requirement: `jnp.where(..., size=...)` needs a compile-time constant output size for XLA shape inference. If fewer than `max_n` cells are interface cells, the remaining slots are filled with index 0 (a halo cell). If more cells are interface cells than `max_n`, the excess cells silently receive `C = 0` -- a graceful degradation, not a crash.

### 5.2.5 Legacy Dense Path `_reconstruct_dense`

A fallback function `_reconstruct_dense` is retained that runs `solve_intercept` (Regula Falsi, 15 iterations) on ALL cells. This is not called in the current pipeline but remains for debugging and A/B comparison. It has the same signature and output as `_reconstruct`.

## 5.3 Module C: Interface

```python
def plic_subsweep_x(
    F: jnp.ndarray,       # (Nx, Ny, Nz)
    u: float,             # Scalar x velocity
    dt_sub: float,        # Sub-step time step (dt/2 or dt)
    dx: float,
    dy: float,
    dz: float,
    halo_fn: HaloFn,
) -> jnp.ndarray          # (Nx, Ny, Nz)

def plic_subsweep_y(
    F: jnp.ndarray,
    v: float,
    dt_sub: float,
    dx: float,
    dy: float,
    dz: float,
    halo_fn: HaloFn,
) -> jnp.ndarray

def plic_time_step_2d_strang(
    F: jnp.ndarray,       # (Nx, Ny, Nz)
    u: float,             # Scalar x velocity
    v: float,             # Scalar y velocity
    dt: float,            # Full time step
    dx: float,
    dy: float,
    dz: float,
    halo_fn: HaloFn,
) -> jnp.ndarray          # (Nx, Ny, Nz)
```

- **Side effects**: None (halo_fn should also be a pure function)
- **Purity**: Chain of pure functions

## 5.4 Module D: Execution Flow

### 5.4.1 `plic_subsweep_x` Detailed Sequence (ORDER-DEPENDENT, strictly serial)

**D.1** `F = halo_fn(F)` -- fill ghost cells.

**D.2** `nx, ny, nz, C = _reconstruct(F, dx, dy, dz)` -- sparse PLIC reconstruction (see 5.4.5). ORDER-DEPENDENT on D.1.

**D.3** `flux = sweep_flux_x(F, nx, ny, nz, C, u, dt_sub, dx, dy, dz)` -- geometric flux. ORDER-DEPENDENT on D.2.

**D.4** `F = apply_flux_x(F, flux, dx, dy, dz)` -- conservative update. ORDER-DEPENDENT on D.3.

**D.5** `F = apply_flux_x_conservative(F, flux, dt_sub, dx, u, n_iter=3)` -- conservative bound enforcement (Weymouth-Zaleski redistribute). ORDER-DEPENDENT on D.4. This calls `redistribute_bounds_x(F, u, n_iter=3)` internally; the `flux`, `dt_sub`, and `dx` arguments are accepted for API compatibility but are unused. See Section 6 and `specs/CONSERVATIVE_REDISTRIBUTE_SPEC.md` for the full algorithm.

[OBSERVATION] The test scripts in `examples/plic_eulerian_tests/` (e.g., `run_zalesak_3d.py`, `run_comparison.py`) use inline Strang steps with `jnp.clip(F, 0.0, 1.0)` instead of calling `strang_sweep.py` or `conservative_bounds.py`. They import only `compute_youngs_normal_3d`, `analytic_intercept`, `sweep_flux_x/y/z`, `apply_flux_x/y/z`, `compute_stats`, and `format_stats`. A reimplementer reproducing only the test scripts can use `jnp.clip`; a reimplementer reproducing `strang_sweep.py` MUST use `apply_flux_x_conservative`.

**D.6** `return halo_fn(F)` -- prepare for next sub-step. ORDER-DEPENDENT on D.5.

### 5.4.2 `plic_subsweep_y` -- Exact Mirror of x

Same 6 steps, replacing `sweep_flux_x` with `sweep_flux_y`, `apply_flux_x` with `apply_flux_y`, `apply_flux_x_conservative` with `apply_flux_y_conservative`, and the velocity parameter is `v`. In D.5, the call becomes `apply_flux_y_conservative(F, flux, dt_sub, dy, v, n_iter=3)`.

### 5.4.3 `plic_time_step_2d_strang` Full Orchestration

**D.1** `F = plic_subsweep_x(F, u, dt/2, dx, dy, dz, halo_fn)` -- half-step x sweep.

**D.2** `F = plic_subsweep_y(F, v, dt, dx, dy, dz, halo_fn)` -- full-step y sweep. ORDER-DEPENDENT on D.1.

**D.3** `F = plic_subsweep_x(F, u, dt/2, dx, dy, dz, halo_fn)` -- half-step x sweep. ORDER-DEPENDENT on D.2.

**D.4** `return F`.

All three steps are **strictly serial** and cannot be parallelized or reordered.

### 5.4.4 Sub-step Time Step Sizes

| Sub-step | Function | `dt_sub` |
|------|------|----------|
| 1 | `plic_subsweep_x` | `dt / 2.0` |
| 2 | `plic_subsweep_y` | `dt` |
| 3 | `plic_subsweep_x` | `dt / 2.0` |

### 5.4.5 `_reconstruct` (Sparse Padded-Gather Path) Internal Execution Flow

**D.R.1** Compute `shape = F.shape` and `total = product of all dimensions in shape` (computed via a Python for-loop over `shape`). Then `max_n = int(total * 0.15)`.

**D.R.2** Compute Youngs normals on the **full** field: `nx, ny, nz = compute_youngs_normal_3d(F, dx, dy, dz)`. This is cheap (~2.6 ms at 1M cells via cuDNN).

**D.R.3** Identify interface cells: `is_interface = (F > 1e-6) & (F < 1.0 - 1e-6)`.

**D.R.4** Padded gather: `flat_idx = jnp.where(is_interface.ravel(), size=max_n, fill_value=0)[0]`.
- `is_interface.ravel()` flattens the 3D boolean mask to 1D.
- `jnp.where(..., size=max_n, fill_value=0)` returns a tuple of 1D index arrays; `[0]` extracts the first (and only, since input is 1D).
- `size=max_n` fixes the output length at compile time (JAX requirement). If fewer than `max_n` cells are interface cells, remaining slots are filled with index `0` (a halo cell).

**D.R.5** Extract compact arrays via flat indexing:
- `F_c = F.ravel()[flat_idx]`
- `nx_c = nx.ravel()[flat_idx]`
- `ny_c = ny.ravel()[flat_idx]`
- `nz_c = nz.ravel()[flat_idx]`

Each has shape `(max_n,)`.

**D.R.6** Solve intercept on compact arrays: `C_c = analytic_intercept(nx_c, ny_c, nz_c, F_c, dx, dy, dz)`. This calls the Phase B analytic solver (Section 7) on the compact 1D arrays only. ORDER-DEPENDENT on D.R.5.

**D.R.7** Scatter back to full field:
- `C_flat = jnp.zeros(total, dtype=F.dtype)`
- `C_flat = C_flat.at[flat_idx].set(C_c)`
- `C = C_flat.reshape(shape)`

Pure cells (index not in `flat_idx`) retain `C = 0`. Dummy padding slots (with `flat_idx = 0`) write to index 0, which is a halo cell and will be overwritten by `halo_fn`.

**D.R.8** Return `(nx, ny, nz, C)`.

### 5.4.6 `_reconstruct_dense` (Legacy Dense Path) Internal Execution Flow

```python
def _reconstruct_dense(F, dx, dy, dz):
    nx, ny, nz = compute_youngs_normal_3d(F, dx, dy, dz)
    C = solve_intercept(nx, ny, nz, F, dx, dy, dz)
    return nx, ny, nz, C
```

This is the original Phase A path (Regula Falsi on all cells). Retained as a fallback but **not called** by the current pipeline.

---

# 6. diagnostics.py

## 6.1 Module A: Mathematics

### 6.1.1 Statistics Definitions

Given a halo-padded $F$ and halo width $n_h$, the interior region is:

$$F_{\mathrm{int}} = F[n_h:-n_h,\; n_h:-n_h,\; n_h:-n_h]$$

$$\mathrm{volume} = \left(\sum_{i,j,k} F_{\mathrm{int}}[i,j,k]\right) \cdot V_{\mathrm{cell}}$$

$$f_{\min} = \min_{i,j,k} F_{\mathrm{int}}[i,j,k]$$

$$f_{\max} = \max_{i,j,k} F_{\mathrm{int}}[i,j,k]$$

$$n_{\mathrm{below}} = \#\{(i,j,k) : F_{\mathrm{int}}[i,j,k] < -\epsilon\}$$

$$n_{\mathrm{above}} = \#\{(i,j,k) : F_{\mathrm{int}}[i,j,k] > 1 + \epsilon\}$$

$$n_{\mathrm{interface}} = \#\{(i,j,k) : \epsilon < F_{\mathrm{int}}[i,j,k] < 1 - \epsilon\}$$

where $\epsilon = 10^{-6}$.

### 6.1.2 Formatted Output

```
V=<volume:.6e> dV/V=<drift:+.2e> F=[<f_min:+.3e>, <f_max:+.3e>] n_if=<n_interface> n_below=<n_below> n_above=<n_above>
```

`drift` is only computed when `v_ref` is not None and not 0: `drift = (volume - v_ref) / v_ref`.

### 6.1.3 All Epsilon/Tolerance Values

| Value | Code Constant | Purpose |
|---|---------|------|
| `1e-6` | `_EPS` | Boundedness detection threshold |

## 6.2 Module B: Data Structures

### 6.2.1 `PLICStats` NamedTuple

```python
class PLICStats(NamedTuple):
    volume: float       # Integral volume
    f_min: float        # Interior minimum F
    f_max: float        # Interior maximum F
    n_below: int        # Number of cells violating the lower bound
    n_above: int        # Number of cells violating the upper bound
    n_interface: int    # Number of interface cells
```

**Note**: All fields are Python native types (`float`, `int`), not JAX arrays. The code explicitly converts via `float(...)` and `int(...)`, which triggers JAX's device-to-host transfer and prevents tracing/jit.

### 6.2.2 Input Shape

| Parameter | Type | Description |
|------|------|------|
| `F` | `(Nx, Ny, Nz)` | Halo-padded volume fraction |
| `nh` | `int` | Halo width per side, typical value 1 |
| `cell_vol` | `float` | Single grid cell volume $\Delta x \cdot \Delta y \cdot \Delta z$ |

## 6.3 Module C: Interface

```python
def interior_slice(F: jnp.ndarray, nh: int) -> jnp.ndarray
    # Returns F[nh:-nh, nh:-nh, nh:-nh]; when nh=0, returns F itself

def compute_stats(F: jnp.ndarray, nh: int, cell_vol: float) -> PLICStats
    # Returns Python scalars (not JAX arrays)

def format_stats(stats: PLICStats, v_ref: float | None = None) -> str
    # Returns formatted string
```

- **Side effects**: `compute_stats` triggers device-to-host transfer (`float()`, `int()`)
- **JIT compatibility**: `compute_stats` cannot be called inside `jit` (because `float()` materializes values)
- `interior_slice`: when `nh > 0`, returns `F[nh:-nh, nh:-nh, nh:-nh]`; when `nh == 0`, returns `F` directly (no slicing)

## 6.4 Module D: Execution Flow

### 6.4.1 `compute_stats` Sequence

**D.1** `interior = interior_slice(F, nh)` -- extract interior region.

**D.2** Compute 6 scalar statistics in parallel (ORDER-INDEPENDENT among each other):
- `volume = float(jnp.sum(interior) * cell_vol)`
- `f_min = float(jnp.min(interior))`
- `f_max = float(jnp.max(interior))`
- `n_below = int(jnp.sum(interior < -1e-6))`
- `n_above = int(jnp.sum(interior > 1 + 1e-6))`
- `n_interface = int(jnp.sum((interior > 1e-6) & (interior < 1 - 1e-6)))`

**D.3** Pack into `PLICStats` NamedTuple and return.

### 6.4.2 `format_stats` Sequence

**D.1** If `v_ref is not None and v_ref != 0.0`:
- `drift = (stats.volume - v_ref) / v_ref`
- `vol_str = f"V={stats.volume:.6e} dV/V={drift:+.2e}"`

Otherwise: `vol_str = f"V={stats.volume:.6e}"`

**D.2** Concatenate and return the complete string.

---

# 7. analytic\_intercept.py

## 7.1 Module A: Mathematics

### 7.1.1 Problem Definition

This module solves the **inverse** of the forward volume model (Section 1). Given a unit cell, an interface normal $(n_x, n_y, n_z)$, and a target volume fraction $F$, find the intercept $C$ such that:

$$V(C;\; n_x \Delta x,\; n_y \Delta y,\; n_z \Delta z) = F$$

The forward model $V(d)$ is a piecewise polynomial in the unit-cube parameter $d$ (see Section 1.1.7). Because $V(d)$ is piecewise polynomial, its inverse $d(V)$ is piecewise algebraic -- the inverse can be computed in closed form for each region via square roots, cube roots, or Cardano's formula. This yields an $O(1)$ per-cell solver with no iteration, replacing the $O(N_{\mathrm{iter}})$ Regula Falsi solver of Section 3.

**Reference**: Scardovelli R, Zaleski S. "Analytical Relations Connecting Linear Interfaces and Volume Fractions in Rectangular Grids." J. Comput. Phys. 164 (2000), pp 228-237, Section 3.

### 7.1.2 Coordinate Transform and Complement Symmetry

The same coordinate transform as the forward model is used:

$$a_0 = |n_x \Delta x|, \quad b_0 = |n_y \Delta y|, \quad c_0 = |n_z \Delta z|$$

$$S = a_0 + b_0 + c_0$$

The relationship between the physical cell-centered intercept $C$ and the unit-cube parameter $d$:

$$d = C + \frac{S}{2}$$

**Complement symmetry**: Because $V(S - d) = 1 - V(d)$, if $F > 0.5$ we can solve for $F_w = 1 - F$ and then take $d = S - d_w$. This restricts the working volume fraction to $F_w \le 0.5$, which halves the number of regions that need inverse formulas.

$$F_w = \begin{cases} 1 - F & F > 0.5 \\ F & F \le 0.5 \end{cases}$$

### 7.1.3 Degeneracy Detection (4-Case Dispatch by $n_{\mathrm{nz}}$)

Identical to the forward model (Section 1.1.3):

$$\mathrm{max\_abc} = \max(a_0, b_0, c_0) + 10^{-30}$$

$$\mathrm{nz\_a} = (a_0 > 10^{-4} \cdot \mathrm{max\_abc})$$

$$n_{\mathrm{nz}} = \mathrm{nz\_a} + \mathrm{nz\_b} + \mathrm{nz\_c} \in \{0, 1, 2, 3\}$$

The threshold constants $10^{-30}$ and $10^{-4}$ match the forward model's `_EPS` and `_THR_REL` respectively, ensuring forward-then-inverse and inverse-then-forward consistency.

### 7.1.4 Case 0D ($n_{\mathrm{nz}} = 0$): All Coefficients Near Zero

$$d_{0D} = \frac{S}{2}$$

This is the midpoint; $V(d) = \mathbf{1}\{d \ge 0\}$, so any $d$ is "correct" for $F$ near 0 or 1.

### 7.1.5 Case 1D ($n_{\mathrm{nz}} = 1$): Slab

Only one coefficient is nonzero:

$$m_{1D} = \begin{cases} a_0 & \mathrm{nz\_a} \\ b_0 & \mathrm{nz\_b} \wedge \neg\mathrm{nz\_a} \\ c_0 & \text{otherwise} \end{cases} + \epsilon_{\mathrm{abs}}$$

where $\epsilon_{\mathrm{abs}} = 10^{-10}$. The forward model is $V = d / m_{1D}$, so:

$$d_{1D} = F_{\mathrm{safe}} \cdot m_{1D}$$

No complement symmetry is needed for the 1D case because the formula is already linear.

### 7.1.6 Case 2D ($n_{\mathrm{nz}} = 2$): Triangle/Trapezoid Inverse

Sort the three coefficients (zero for inactive axes) and take the two nonzero ones:

$$\text{vals} = \mathrm{sort}\!\left(\begin{bmatrix} a_0 \cdot \mathrm{nz\_a} \\ b_0 \cdot \mathrm{nz\_b} \\ c_0 \cdot \mathrm{nz\_c} \end{bmatrix}, \; \text{axis}=0\right)$$

$$p = \max(\text{vals}[1],\; \epsilon_{\mathrm{abs}}), \quad q = \max(\text{vals}[2],\; \epsilon_{\mathrm{abs}}), \quad S_2 = p + q$$

Apply complement symmetry:

$$\text{use\_comp}_{2D} = (F_{\mathrm{safe}} > 0.5), \quad F_{w,2D} = \begin{cases} 1 - F_{\mathrm{safe}} & \text{use\_comp}_{2D} \\ F_{\mathrm{safe}} & \text{otherwise} \end{cases}$$

**Transition volume**:

$$V_1^{(2D)} = \frac{p}{2q}$$

**Region 1** ($F_{w,2D} < V_1^{(2D)}$): Corner triangle. Forward: $V = d^2 / (2pq)$. Inverse:

$$d_{\mathrm{r1}} = \sqrt{\max(2pq \cdot F_{w,2D},\; 0)}$$

**Region 2** ($F_{w,2D} \ge V_1^{(2D)}$): Trapezoid. Forward: $V = (2d - p) / (2q)$. Inverse:

$$d_{\mathrm{r2}} = q \cdot F_{w,2D} + \frac{p}{2}$$

**Piecewise selection**:

$$d_{w,2D} = \begin{cases} d_{\mathrm{r1}} & F_{w,2D} < V_1^{(2D)} \\ d_{\mathrm{r2}} & \text{otherwise} \end{cases}$$

**Undo complement**:

$$d_{2D} = \begin{cases} S_2 - d_{w,2D} & \text{use\_comp}_{2D} \\ d_{w,2D} & \text{otherwise} \end{cases}$$

### 7.1.7 Case 3D ($n_{\mathrm{nz}} = 3$): Full Scardovelli-Zaleski Analytic Inverse

#### 7.1.7.1 Branchless 3-Element Sorting Network

Sort $(a_0, b_0, c_0)$ into $m_1 \le m_2 \le m_3$ using a branchless sorting network:

$$\mathrm{lo}_1 = \min(a_0, b_0), \quad \mathrm{hi}_1 = \max(a_0, b_0)$$

$$\mathrm{lo}_2 = \min(\mathrm{hi}_1, c_0), \quad m_3 = \max(\mathrm{hi}_1, c_0)$$

$$m_1 = \min(\mathrm{lo}_1, \mathrm{lo}_2), \quad m_2 = \max(\mathrm{lo}_1, \mathrm{lo}_2)$$

Safe lower bounds: $m_1' = \max(m_1, \epsilon_{\mathrm{abs}})$, $m_2' = \max(m_2, \epsilon_{\mathrm{abs}})$, $m_3' = \max(m_3, \epsilon_{\mathrm{abs}})$.

Derived quantities:

$$m_{12} = m_1 + m_2, \quad S_3 = m_1 + m_2 + m_3, \quad P_6 = 6 m_1' m_2' m_3'$$

#### 7.1.7.2 Complement Symmetry for 3D

$$\text{use\_comp}_{3D} = (F_{\mathrm{safe}} > 0.5), \quad F_{w,3D} = \begin{cases} 1 - F_{\mathrm{safe}} & \text{use\_comp}_{3D} \\ F_{\mathrm{safe}} & \text{otherwise} \end{cases}$$

#### 7.1.7.3 Transition Volumes

$$V_1^{(3D)} = \frac{m_1'^2}{6 m_2' m_3'}$$

$$V_2^{(3D)} = V_1^{(3D)} + \frac{m_2 - m_1}{2 m_3' + 10^{-30}}$$

$$V_3^{(3D)} = \frac{m_3^3 - (m_3 - m_1)^3 - (m_3 - m_2)^3}{P_6 + 10^{-30}}$$

$V_3$ is the volume at the breakpoint $d = m_3$, where the $(d - m_3)^3$ term in the forward inclusion-exclusion formula first becomes nonzero.

#### 7.1.7.4 Region 1 ($F_w < V_1$): Corner Tetrahedron

Forward: $V = d^3 / P_6$. Inverse:

$$d_{\mathrm{r1}} = \sqrt[3]{P_6 \cdot F_w}$$

computed via `_safe_cbrt` (see 7.1.10).

Range clamp: $d_{\mathrm{r1}} = \mathrm{clip}(d_{\mathrm{r1}}, 0, m_1)$.

#### 7.1.7.5 Region 2 ($V_1 \le F_w < V_2$): One Edge Clipped

Forward: $V = V_1 + (d - m_1)^2 / (2 m_2' m_3') + \ldots$ simplifies to a quadratic. Inverse:

$$d_{\mathrm{r2}} = \frac{1}{2}\left(m_1 + \sqrt{\max(m_1^2 + 8 m_2' m_3' (F_w - V_1),\; 0)}\right)$$

Range clamp: $d_{\mathrm{r2}} = \mathrm{clip}(d_{\mathrm{r2}}, m_1, m_2)$.

#### 7.1.7.6 Region 3 ($V_2 \le F_w \le 0.5$): Cubic Inverse via Cardano

This is the most complex region. The forward model is a cubic polynomial in $d$. Two sub-cases arise depending on whether $F_w < V_3$ (the plane has not yet reached the $d = m_3$ breakpoint).

**Sub-case A** ($F_w < V_3$, i.e. $d < m_3$):

The forward cubic (from the inclusion-exclusion formula without the $(d-m_3)^3$ term):

$$d^3 - 3 m_{12} d^2 + 3(m_1^2 + m_2^2) d = m_1^3 + m_2^3 - 6 m_1 m_2 m_3 F_w$$

Apply the depressed cubic substitution $d = t + m_{12}$:

$$t^3 + p_A t + q_A = 0$$

where:

$$p_A = -6 m_1' m_2'$$

$$q_A = 3 m_1' m_2' (2 m_3' F_w - m_{12})$$

**[SIGN CRITICAL]**: The sign of $q_A$ is derived from the depressed cubic shift $d = t + m_{12}$ applied to the Region 3A equation. Getting this sign wrong selects the wrong Cardano root and produces catastrophic errors. The expression $q_A = 3 m_1 m_2 (2 m_3 F_w - m_{12})$ has the $(2 m_3 F_w - m_{12})$ factor, which is negative when $F_w$ is small and positive when $F_w$ is large.

Since $p_A < 0$ (always, because $m_1' m_2' > 0$), the discriminant $\Delta = -4p_A^3 - 27q_A^2 > 0$ for physical volume fractions, giving three real roots via the trigonometric Cardano formula:

$$D_A = \sqrt{2 m_1' m_2'}, \quad D_A^3 = D_A \cdot 2 m_1' m_2'$$

$$\cos \theta_A = \mathrm{clip}\!\left(\frac{-q_A}{2 D_A^3 + 10^{-30}},\; -1,\; 1\right)$$

$$\theta_A = \arccos(\cos \theta_A)$$

Three roots:

$$t_{A,0} = 2 D_A \cos\!\left(\frac{\theta_A}{3}\right), \quad t_{A,1} = 2 D_A \cos\!\left(\frac{\theta_A + 2\pi}{3}\right), \quad t_{A,2} = 2 D_A \cos\!\left(\frac{\theta_A + 4\pi}{3}\right)$$

$$d_{A,k} = t_{A,k} + m_{12} \quad (k = 0, 1, 2)$$

**Root selection**: Rather than analytically determining which root lies in the valid range $[m_2, \min(m_{12}, m_3)]$, the code evaluates the forward model $V(d_{A,k})$ for all three roots and picks the one closest to $F_w$:

$$C_{A,k} = d_{A,k} - S_3/2 \quad \text{(convert to physical intercept)}$$

$$V_{A,k} = V(C_{A,k};\; a_0, b_0, c_0) \quad \text{(forward model call)}$$

$$e_k = |V_{A,k} - F_w|$$

$$d_{\mathrm{r3,A}} = d_{A,\mathrm{argmin}(e_0, e_1, e_2)}$$

The argmin is implemented as nested `jnp.where`:
```
where(e0 < e1, where(e0 < e2, d_A0, d_A2), where(e1 < e2, d_A1, d_A2))
```

Range clamp: $d_{\mathrm{r3,A}} = \mathrm{clip}(d_{\mathrm{r3,A}}, m_2, \min(m_{12}, m_3))$.

**Sub-case B** ($F_w \ge V_3$, i.e. $d \ge m_3$):

The full cubic including the $(d - m_3)^3$ term from the forward formula:

$$-2d^3 + 3 S_3 d^2 - 3(m_1^2 + m_2^2 + m_3^2) d + (m_1^3 + m_2^3 + m_3^3) = 6 m_1 m_2 m_3 F_w$$

Apply the depressed cubic substitution $d = t + S_3/2$:

$$t^3 + p_B t + q_B = 0$$

where:

$$S_{\mathrm{sq}} = m_1^2 + m_2^2 + m_3^2$$

$$P_2 = m_1' m_2' + m_1' m_3' + m_2' m_3'$$

$$m_{123} = m_1' m_2' m_3'$$

$$p_B = \frac{3}{2} S_{\mathrm{sq}} - \frac{3}{4} S_3^2$$

The sum of cubes is computed as:

$$\Sigma_3 = S_3^3 - 3 S_3 P_2 + 3 m_{123}$$

$$q_B = -\frac{\Sigma_3 - 6 m_{123} F_w}{2} + \frac{3 S_3 S_{\mathrm{sq}} - S_3^3}{4}$$

**$p_B > 0$ guard**: The trigonometric Cardano formula requires $p_B < 0$ (three real roots). When $p_B \ge 0$ (which occurs for near-axis normals where $m_1 \approx 0$), there is only one real root and the trigonometric formula breaks. In this case, a direct cube-root fallback is used:

$$\text{p\_ok} = (p_B < -10^{-10})$$

When `p_ok` is true, the trigonometric Cardano proceeds:

$$D_B^2 = \max(-p_B/3,\; 10^{-30}), \quad D_B = \sqrt{D_B^2}, \quad D_B^3 = D_B \cdot D_B^2$$

$$\cos \theta_B = \mathrm{clip}\!\left(\frac{-q_B}{2 D_B^3 + 10^{-30}},\; -1,\; 1\right)$$

$$\theta_B = \arccos(\cos \theta_B)$$

Three roots:

$$t_{B,k} = 2 D_B \cos\!\left(\frac{\theta_B + 2k\pi}{3}\right), \quad k = 0, 1, 2$$

When `p_ok` is false, all three roots collapse to:

$$t_{\mathrm{dir}} = \mathrm{cbrt\_safe}(-q_B)$$

$$t_{B,k} = \begin{cases} t_{B,k}^{\mathrm{trig}} & \text{p\_ok} \\ t_{\mathrm{dir}} & \text{otherwise} \end{cases}$$

$$d_{B,k} = t_{B,k} + S_3/2 \quad (k = 0, 1, 2)$$

**Root selection**: Same V(d)-based strategy as Sub-case A:

$$C_{B,k} = d_{B,k} - S_3/2, \quad V_{B,k} = V(C_{B,k};\; a_0, b_0, c_0)$$

$$e_k = |V_{B,k} - F_w|, \quad d_{\mathrm{r3,B}} = d_{B,\mathrm{argmin}(e_0, e_1, e_2)}$$

Range clamp: $d_{\mathrm{r3,B}} = \mathrm{clip}(d_{\mathrm{r3,B}}, m_3, S_3/2)$.

[OBSERVATION] The range clamping of $d_{\mathrm{r3,A}}$ and $d_{\mathrm{r3,B}}$ occurs **after** the sub-case selection `d_3d_r3 = where(F_w < V3, d_3d_r3_A, d_3d_r3_B)` on line 235 of the source, and then the clamped versions are re-selected on line 241. The first selection on line 235 uses unclamped values. The second selection on line 241 uses clamped values and overwrites the first. This means the first `jnp.where` on line 235 is dead code -- its result is never used. Both sides of both `jnp.where` are evaluated (JAX semantics), so this is harmless but wasteful.

#### 7.1.7.7 Piecewise Selection for 3D

$$d_{w,3D} = \begin{cases} d_{\mathrm{r1}} & F_w < V_1 \\ d_{\mathrm{r2}} & V_1 \le F_w < V_2 \\ d_{\mathrm{r3}} & F_w \ge V_2 \end{cases}$$

where $d_{\mathrm{r3}} = \begin{cases} d_{\mathrm{r3,A}} & F_w < V_3 \\ d_{\mathrm{r3,B}} & F_w \ge V_3 \end{cases}$ (with range clamping applied).

**Undo complement**:

$$d_{3D} = \begin{cases} S_3 - d_{w,3D} & \text{use\_comp}_{3D} \\ d_{w,3D} & \text{otherwise} \end{cases}$$

### 7.1.8 Near-2D Fallback for Degenerate 3D Normals

When $n_{\mathrm{nz}} = 3$ but $m_1$ is tiny relative to $m_3$, the 3D cubic becomes ill-conditioned ($p_B$ goes positive, Cardano selects the wrong root). The code detects this and falls back to the 2D formula:

$$\mathrm{near\_2d} = (n_{\mathrm{nz}} = 3) \;\wedge\; (m_1 < 0.05 \cdot m_3)$$

**Threshold justification**: The 5% threshold was empirically chosen. Higher thresholds (e.g. 15%) catch more degenerate cubics but also misclassify cells where $m_1$ is physically significant (observed: $m_1/m_3 = 0.13$ with threshold $0.15$ gave 2% error because the 2D approximation was poor).

### 7.1.9 Final Dispatch

$$d_{\mathrm{unit}} = \begin{cases} d_{0D} & n_{\mathrm{nz}} = 0 \\ d_{1D} & n_{\mathrm{nz}} = 1 \\ d_{2D} & n_{\mathrm{nz}} = 2 \;\text{OR}\; \mathrm{near\_2d} \\ d_{3D} & n_{\mathrm{nz}} = 3 \;\text{AND}\; \neg\mathrm{near\_2d} \end{cases}$$

### 7.1.10 Helper: `_safe_cbrt(x)`

Branchless cube root safe for $x \le 0$:

$$\mathrm{cbrt\_safe}(x) = \mathrm{sign}(x) \cdot (|x| + 10^{-30})^{1/3}$$

The $10^{-30}$ addend prevents $0^{1/3}$ from producing NaN gradients under JAX autodiff.

### 7.1.11 Newton Refinement (5 Steps)

After the analytic solve, the intercept is refined via Newton's method to correct floating-point precision gaps (Cardano cubic-root residual ~$5 \times 10^{-3}$ in float32; ~$5 \times 10^{-15}$ in float64) and any wrong-root Cardano selections. The Newton step uses **exact** derivatives computed via JAX forward-mode automatic differentiation (`jax.jvp`). Because Newton is quadratically convergent, **5 steps in float32** (current fixed count) and **3 steps in float64** both drive the residual to the machine-epsilon floor.

**Initial guess**: Convert from unit-cube parameter to physical intercept:

$$C^{(0)} = d_{\mathrm{unit}} - \frac{S}{2}$$

**Newton iteration** (5 steps, unrolled Python for-loop):

For $k = 0, 1, 2, 3, 4$:

$$V^{(k)} = V(C^{(k)};\; a_0, b_0, c_0)$$

$$r^{(k)} = V^{(k)} - F_{\mathrm{safe}}$$

$$\frac{dV}{dC}\bigg|_{C^{(k)}} = \mathrm{jax.jvp}\!\left(\lambda c : V(c;\; a_0, b_0, c_0),\; (C^{(k)},),\; (\mathbf{1},)\right)_1$$

The derivative is computed via `jax.jvp` with a tangent vector of ones. This is exact (not finite-difference) and equivalent to symbolic differentiation of the piecewise polynomial forward model.

**Safe denominator**:

$$\frac{dV}{dC}_{\mathrm{safe}} = \begin{cases} \frac{dV}{dC} & \frac{dV}{dC} > 10^{-10} \\ 1 & \text{otherwise} \end{cases}$$

**Update**:

$$C^{(k+1)} = \mathrm{clip}\!\left(C^{(k)} - \frac{r^{(k)}}{(dV/dC)_{\mathrm{safe}}},\; -\frac{S}{2},\; +\frac{S}{2}\right)$$

The domain clamp $[-S/2, +S/2]$ ensures $C$ stays in the valid range where $V \in [0, 1]$.

**Cost**: 5 Newton steps = 10 forward model evaluations (5 for $V$, 5 for $dV/dC$ via JVP). Combined with the 6 forward evaluations for root selection (3 for sub-case A, 3 for sub-case B), the total is 16 forward model evaluations per cell. This is comparable to the 15-iteration Regula Falsi solver but produces a significantly more accurate result.

### 7.1.12 Pure Cell Zeroing

$$||\mathbf{n}||^2 = n_x^2 + n_y^2 + n_z^2$$

$$C_{\mathrm{out}} = \begin{cases} C^{(5)} & ||\mathbf{n}||^2 > 10^{-12} \\ 0 & \text{otherwise} \end{cases}$$

### 7.1.13 Variable-to-Code Mapping Table

| Math Symbol | Code Variable | Description |
|---------|---------|------|
| $a_0, b_0, c_0$ | `a0, b0, c0` | $\|n_x \Delta x\|$, $\|n_y \Delta y\|$, $\|n_z \Delta z\|$ |
| $S$ | `S` | $a_0 + b_0 + c_0$ |
| $F_{\mathrm{safe}}$ | `F_safe` | $\mathrm{clip}(F, 10^{-6}, 1 - 10^{-6})$ |
| $n_{\mathrm{nz}}$ | `n_nz` | Count of nonzero coefficients |
| $\epsilon_{\mathrm{abs}}$ | `thr` | `jnp.asarray(1e-10, F.dtype)` — dtype-parametric |
| $m_1, m_2, m_3$ | `m1, m2, m3` | Sorted coefficients (unsafened) |
| $m_1', m_2', m_3'$ | `m1s, m2s, m3s` | $\max(m_k, \epsilon_{\mathrm{abs}})$ |
| $m_{12}$ | `m12` | $m_1 + m_2$ |
| $S_3$ | `S3` | $m_1 + m_2 + m_3$ |
| $P_6$ | `p6` | $6 m_1' m_2' m_3'$ |
| $F_w$ | `F_w3` (3D), `F_w2` (2D) | Working volume fraction $\le 0.5$ |
| $V_1, V_2, V_3$ | `V1_3d, V2_3d, V3_3d` | Transition volumes |
| $p_A$ | `p_A` | $-6 m_1' m_2'$ |
| $q_A$ | `q_A` | $3 m_1' m_2' (2 m_3' F_w - m_{12})$ |
| $D_A$ | `D_A` | $\sqrt{2 m_1' m_2'}$ |
| $\theta_A$ | `theta_A` | $\arccos(\cos \theta_A)$ |
| $S_{\mathrm{sq}}$ | `S3_sq` | $m_1^2 + m_2^2 + m_3^2$ |
| $P_2$ | `P2` | $m_1' m_2' + m_1' m_3' + m_2' m_3'$ |
| $m_{123}$ | `m123` | $m_1' m_2' m_3'$ |
| $p_B$ | `p_B` | $\frac{3}{2} S_{\mathrm{sq}} - \frac{3}{4} S_3^2$ |
| $q_B$ | `q_B` | See formula in 7.1.7.6 |
| $\text{p\_ok}$ | `p_ok` | $(p_B < -10^{-10})$ |

### 7.1.14 All Epsilon/Tolerance Values

| Value | Code Constant | Purpose |
|---|---------|------|
| `1e-30` | `_EPS` | Addend in `_safe_cbrt`, `max_abc`, `V3_3d` denominator, `D_A_cubed`/`D_B_cubed` denominator, `V2_3d` denominator; prevents division by zero or NaN from `0^(1/3)` |
| `1e-12` | `_N_EPS2` | Normal vector magnitude-squared threshold for pure cell zeroing |
| `1e-10` | `thr` = `jnp.asarray(1e-10, F.dtype)` | Absolute lower bound on sorted coefficients $m_1', m_2', m_3'$, $p$, $q$, and $m_{1D}$. Dtype-parametric: promotes to `F.dtype` to avoid dtype-mismatch errors in mixed-precision broadcasts |
| `1e-10` | (inline in `p_ok`) | Threshold for testing whether $p_B$ is sufficiently negative for trigonometric Cardano |
| `1e-10` | (inline in `safe_dV`) | Threshold for Newton derivative denominator guard |
| `1e-6` | (inline in `F_safe` clip) | Lower bound for target volume fraction clipping |
| `1-1e-6` | (inline in `F_safe` clip) | Upper bound for target volume fraction clipping |
| `1e-4` | `_THR_REL` | Relative zero detection threshold for coefficient degeneracy (matches forward model) |
| `0.05` | (inline in `near_2d`) | Threshold for near-degenerate 3D-to-2D fallback: $m_1 < 0.05 m_3$ |
| `5` | (inline in `for _ in range(5)`) | Number of Newton refinement steps |

### 7.1.15 Module-Level Constant Dtype Warning

[OBSERVATION] `_EPS = 1e-30`, `_N_EPS2 = 1e-12`, and `_THR_REL = 1e-4` are **plain Python float literals** assigned at module scope (lines 26-28 of `analytic_intercept.py`). They are NOT wrapped in `jnp.asarray`. When they participate in JAX operations (e.g., `jnp.abs(x) + _EPS`, `n_sq > _N_EPS2`, `a0 > _THR_REL * max_abc`), they undergo JAX's implicit Python-float-to-scalar promotion, which always promotes to the dtype of the other operand. This happens to produce correct results in both float32 and float64 modes, but the mechanism is implicit promotion, not explicit casting.

Only `thr` (line 59: `thr = jnp.asarray(1e-10, F.dtype)`) is explicitly cast to `F.dtype` via `jnp.asarray`.

A reimplementer MUST NOT wrap `_EPS`, `_N_EPS2`, or `_THR_REL` in `jnp.asarray` -- doing so would create traced JAX arrays at module load time (before any concrete dtype is known), which would either default to float32 (breaking float64 mode) or cause XLA trace-shape mismatches. The correct pattern is bare Python float literals at module scope, exactly as in the source code.

Section 0.3 of this spec states "cast via `jnp.asarray(value, F.dtype)`" as a general rule; this is an exception. The three module-level constants bypass that rule by design.

## 7.2 Module B: Data Structures

### 7.2.1 Input/Output Shape

All inputs `nx, ny, nz, F` have the same shape (broadcast-compatible). Typical shapes:
- 3D field: `(Nx, Ny, Nz)` when called on full field
- 1D compact: `(max_n,)` when called via padded-gather from `strang_sweep._reconstruct`

Scalar parameters `dx, dy, dz` are Python floats.

Output `C` has the same shape as the broadcast of `nx, ny, nz, F`, dtype = `F.dtype`.

### 7.2.2 F Clipping

$$F_{\mathrm{safe}} = \mathrm{clip}(F, 10^{-6}, 1 - 10^{-6})$$

This matches the `_F_CLIP_LO` / `_F_CLIP_HI` of `intercept_solver.py` (Section 3.1.4).

### 7.2.3 Internal Dependency on Forward Model

The function imports `volume_below_plane_3d` from `.volume_formula` (Section 1) via a **local import** inside the function body (not at module top level). This is used in two places:

1. **Root selection**: 6 calls (3 for sub-case A, 3 for sub-case B) to evaluate $V(d)$ for each Cardano root
2. **Newton refinement**: 5 calls for the residual $V(C) - F$, plus 5 calls implicitly inside `jax.jvp` for the derivative

The local import of `jax` is also done inside the function body.

## 7.3 Module C: Interface

```python
def analytic_intercept(
    nx: jnp.ndarray,    # Normal vector x component
    ny: jnp.ndarray,    # Normal vector y component
    nz: jnp.ndarray,    # Normal vector z component
    F: jnp.ndarray,     # Target volume fraction
    dx: float,          # Grid spacing in x direction
    dy: float,          # Grid spacing in y direction
    dz: float,          # Grid spacing in z direction
) -> jnp.ndarray        # Intercept C in physical cell-centered coordinates
```

- **Input range**: `F` $\in [0, 1]$; normal vector is unit vector or zero vector
- **Output range**: $C \in [-S/2, +S/2]$, where $S = |n_x \Delta x| + |n_y \Delta y| + |n_z \Delta z|$; 0 at pure cells
- **Output convention**: Same as `solve_intercept` -- physical cell-centered intercept
- **Side effects**: None
- **Purity**: Pure function, `jit`/`vmap`/`grad` safe
- **JAX semantics**: All branches computed via `jnp.where` (both sides always evaluated). `jax.jvp` is called INSIDE the function body (not by the caller). The Python for-loop `for _ in range(5)` is unrolled at trace time.

### 7.3.1 Internal Helper

```python
def _safe_cbrt(x) -> jnp.ndarray
```

Not exported (no `__all__` entry). Used internally for cube roots in Region 1 and the $p_B \ge 0$ fallback.

## 7.4 Module D: Execution Flow

### 7.4.1 `analytic_intercept` Detailed Sequence

**D.1** Compute absolute plane coefficients: `a0 = |nx * dx|`, `b0 = |ny * dy|`, `c0 = |nz * dz|`. ORDER-INDEPENDENT.

**D.2** Compute `S = a0 + b0 + c0`.

**D.3** Clip target: `F_safe = clip(F, 1e-6, 1.0 - 1e-6)`.

**D.4** Degeneracy detection: `thr = jnp.asarray(1e-10, F.dtype)` (the ONLY dtype-parametric constant; see Section 7.1.15). `max_abc = max(a0, max(b0, c0)) + _EPS` where `_EPS = 1e-30` is a bare Python float (implicit promotion). Compute `nz_a = a0 > _THR_REL * max_abc` where `_THR_REL = 1e-4` is a bare Python float (implicit promotion), similarly `nz_b`, `nz_c`. `n_nz = nz_a.astype(jnp.int32) + nz_b.astype(jnp.int32) + nz_c.astype(jnp.int32)`. Note: `_EPS` and `_THR_REL` are NOT wrapped in `jnp.asarray` -- they are plain Python float literals at module scope that participate via JAX's implicit promotion (see Section 7.1.15 for the rationale).

**D.5** Compute Case 0D: `d_0d = 0.5 * S`.

**D.6** Compute Case 1D: `m_1d = where(nz_a, a0, where(nz_b, b0, c0)) + thr`, `d_1d = F_safe * m_1d`.

**D.7** Compute Case 2D:
- D.7.1: Sort: `vals = sort(array([where(nz_a, a0, 0), where(nz_b, b0, 0), where(nz_c, c0, 0)]), axis=0)`.
- D.7.2: `p2 = max(vals[1], thr)`, `q2 = max(vals[2], thr)`, `S2 = p2 + q2`.
- D.7.3: Complement: `use_comp_2d = F_safe > 0.5`, `F_w2 = where(use_comp_2d, 1 - F_safe, F_safe)`.
- D.7.4: `V1_2d = p2 / (2 * q2)`.
- D.7.5: Region 1: `d_2d_r1 = sqrt(max(2 * p2 * q2 * F_w2, 0))`.
- D.7.6: Region 2: `d_2d_r2 = q2 * F_w2 + p2/2`.
- D.7.7: Select: `d_2d_work = where(F_w2 < V1_2d, d_2d_r1, d_2d_r2)`.
- D.7.8: Undo complement: `d_2d = where(use_comp_2d, S2 - d_2d_work, d_2d_work)`.

**D.8** Compute Case 3D:
- D.8.1: Branchless sort: `lo1 = min(a0, b0)`, `hi1 = max(a0, b0)`, `lo2 = min(hi1, c0)`, `m3 = max(hi1, c0)`, `m1 = min(lo1, lo2)`, `m2 = max(lo1, lo2)`.
- D.8.2: Safe bounds: `m1s = max(m1, thr)`, `m2s = max(m2, thr)`, `m3s = max(m3, thr)`.
- D.8.3: Derived quantities: `m12 = m1 + m2`, `S3 = m1 + m2 + m3`, `p6 = 6 * m1s * m2s * m3s`.
- D.8.4: Complement: `use_comp_3d = F_safe > 0.5`, `F_w3 = where(use_comp_3d, 1 - F_safe, F_safe)`.
- D.8.5: Transition volumes:
  - `V1_3d = m1s^2 / (6 * m2s * m3s)`
  - `V2_3d = V1_3d + (m2 - m1) / (2 * m3s + 1e-30)`
  - `dm1_at_m3 = m3 - m1`, `dm2_at_m3 = m3 - m2`
  - `V3_3d = (m3^3 - dm1_at_m3^3 - dm2_at_m3^3) / (p6 + 1e-30)`
- D.8.6: Region 1: `d_3d_r1 = _safe_cbrt(p6 * F_w3)`.
- D.8.7: Region 2: `inner_r2 = m1^2 + 8 * m2s * m3s * (F_w3 - V1_3d)`, `d_3d_r2 = 0.5 * (m1 + sqrt(max(inner_r2, 0)))`.
- D.8.8: Region 3, Sub-case A:
  - `m1m2 = m1s * m2s`
  - `p_A = -6 * m1m2`
  - `q_A = 3 * m1m2 * (2 * m3s * F_w3 - m12)` **[SIGN CRITICAL]**
  - `D_A = sqrt(max(2 * m1m2, 1e-30))`
  - `D_A_cubed = D_A * 2 * m1m2`
  - `cos_A = clip(-q_A / (2 * D_A_cubed + 1e-30), -1, 1)`
  - `theta_A = arccos(cos_A)`
  - `t_A0 = 2 * D_A * cos(theta_A / 3)`
  - `t_A1 = 2 * D_A * cos((theta_A + 2*pi) / 3)`
  - `t_A2 = 2 * D_A * cos((theta_A + 4*pi) / 3)`
  - `d_A0 = t_A0 + m12`, `d_A1 = t_A1 + m12`, `d_A2 = t_A2 + m12`
  - Local import: `from .volume_formula import volume_below_plane_3d as _vbp`
  - `C_Ak = d_Ak - 0.5 * S3` for $k = 0, 1, 2$
  - `V_Ak = _vbp(C_Ak, a0, b0, c0)` for $k = 0, 1, 2$
  - `ek = |V_Ak - F_w3|` for $k = 0, 1, 2$
  - `d_3d_r3_A = where(e0 < e1, where(e0 < e2, d_A0, d_A2), where(e1 < e2, d_A1, d_A2))`
- D.8.9: Region 3, Sub-case B:
  - `S3_sq = m1^2 + m2^2 + m3^2`
  - `P2 = m1s * m2s + m1s * m3s + m2s * m3s`
  - `m123 = m1s * m2s * m3s`
  - `p_B = 1.5 * S3_sq - 0.75 * S3 * S3`
  - `sum_cubes = S3^3 - 3 * S3 * P2 + 3 * m123`
  - `q_B = -(sum_cubes - 6 * m123 * F_w3) / 2 + (3 * S3 * S3_sq - S3^3) / 4`
  - `D_sq_B = max(-p_B / 3, 1e-30)`, `D_B = sqrt(D_sq_B)`, `D_B_cubed = D_B * D_sq_B`
  - `p_ok = (p_B < -1e-10)`
  - `cos_B = clip(-q_B / (2 * D_B_cubed + 1e-30), -1, 1)`
  - `theta_B = arccos(cos_B)`
  - `t_B0 = 2 * D_B * cos(theta_B / 3)`
  - `t_B1 = 2 * D_B * cos((theta_B + 2*pi) / 3)`
  - `t_B2 = 2 * D_B * cos((theta_B + 4*pi) / 3)`
  - `t_dir = _safe_cbrt(-q_B)` (degenerate $p \ge 0$ fallback)
  - `t_Bk = where(p_ok, t_Bk, t_dir)` for $k = 0, 1, 2$
  - `d_Bk = t_Bk + S3/2` for $k = 0, 1, 2$
  - `C_Bk = d_Bk - 0.5 * S3` for $k = 0, 1, 2$
  - `V_Bk = _vbp(C_Bk, a0, b0, c0)` for $k = 0, 1, 2$
  - `ebk = |V_Bk - F_w3|` for $k = 0, 1, 2$
  - `d_3d_r3_B = where(eb0 < eb1, where(eb0 < eb2, d_B0, d_B2), where(eb1 < eb2, d_B1, d_B2))`
- D.8.10: Sub-case dispatch (first, dead): `d_3d_r3 = where(F_w3 < V3_3d, d_3d_r3_A, d_3d_r3_B)`.
  [OBSERVATION] This assignment on source line 235 is dead code; its result is overwritten by D.8.11.
- D.8.11: Range clamping and re-dispatch:
  - `d_3d_r3_A = clip(d_3d_r3_A, m2, min(m12, m3))`
  - `d_3d_r3_B = clip(d_3d_r3_B, m3, S3/2)`
  - `d_3d_r3 = where(F_w3 < V3_3d, d_3d_r3_A, d_3d_r3_B)` (overwrites D.8.10)
- D.8.12: Range clamping for regions 1 and 2:
  - `d_3d_r1 = clip(d_3d_r1, 0, m1)`
  - `d_3d_r2 = clip(d_3d_r2, m1, m2)`
- D.8.13: Piecewise selection: `d_3d_work = where(F_w3 < V1_3d, d_3d_r1, where(F_w3 < V2_3d, d_3d_r2, d_3d_r3))`.
- D.8.14: Undo complement: `d_3d = where(use_comp_3d, S3 - d_3d_work, d_3d_work)`.

**D.9** Near-2D fallback: `near_2d = (n_nz == 3) & (m1 < 0.05 * m3)`.

**D.10** Final dispatch: `d_unit = where(n_nz == 0, d_0d, where(n_nz == 1, d_1d, where((n_nz == 2) | near_2d, d_2d, d_3d)))`.

**D.11** Convert to physical intercept: `C_cur = d_unit - 0.5 * S`.

**D.12** Local import: `import jax`.

**D.13** Define `_newton_step(C_in)`:
- D.13.1: `V_at = volume_below_plane_3d(C_in, a0, b0, c0)` (captures `a0, b0, c0` from outer scope via closure).
- D.13.2: `residual = V_at - F_safe`.
- D.13.3: `_, dVdC = jax.jvp(lambda c: volume_below_plane_3d(c, a0, b0, c0), (C_in,), (ones_like(C_in),))`.
- D.13.4: `safe_dV = where(dVdC > 1e-10, dVdC, ones_like(dVdC))`.
- D.13.5: `C_new = C_in - residual / safe_dV`.
- D.13.6: `return clip(C_new, -0.5 * S, 0.5 * S)`.

**D.14** Newton refinement loop: `for _ in range(5): C_cur = _newton_step(C_cur)`. This is an unrolled Python for-loop (5 iterations at trace time), not `jax.lax.fori_loop`.

**D.15** Assign: `C = C_cur`.

**D.16** Pure cell zeroing: `n_sq = nx^2 + ny^2 + nz^2`, `is_interface = n_sq > 1e-12`, `return where(is_interface, C, zeros_like(C))`.

---

# Appendix: Inter-Module Dependency Graph

```
lagrangian_3d/reconstruction_3d.py
  |-- _volume_below_3d()        <- volume_formula.py (re-export)
  +-- compute_plic_normals_3d() <- normal_youngs.py (re-export)

volume_formula.py
  +-- volume_below_plane_3d()   -> intercept_solver.py, geometric_flux.py,
                                   analytic_intercept.py

normal_youngs.py
  +-- compute_youngs_normal_3d() -> strang_sweep.py (via _reconstruct)

intercept_solver.py
  +-- solve_intercept()          -> strang_sweep.py (via _reconstruct_dense, UNUSED)

analytic_intercept.py
  |-- _safe_cbrt()               (internal)
  +-- analytic_intercept()       -> strang_sweep.py (via _reconstruct)
      |-- imports volume_formula.volume_below_plane_3d (local, inside function)
      +-- imports jax (local, inside function)

conservative_bounds.py
  |-- _shift_plus1_{x,y,z}()   (internal, zero-padded neighbor shifts)
  |-- _shift_minus1_{x,y,z}()  (internal, zero-padded neighbor shifts)
  |-- _cell_sign_{x,y,z}()     (internal, dominant-face velocity sign)
  |-- _redistribute_body()      (internal, fori_loop core)
  |-- redistribute_bounds_{x,y,z}()  -> geometric_flux.py (re-export)
  +-- apply_flux_{x,y,z}_conservative()  -> geometric_flux.py (re-export)

geometric_flux.py
  |-- _sub_box_flux_kernel()    (internal)
  |-- sweep_flux_x/y/z()       -> strang_sweep.py
  |-- apply_flux_x/y/z()       -> strang_sweep.py
  +-- RE-EXPORTS from conservative_bounds:
      redistribute_bounds_{x,y,z}, apply_flux_{x,y,z}_conservative
      -> strang_sweep.py (imports conservative functions via geometric_flux)

strang_sweep.py
  |-- _reconstruct()            (internal, calls normal + analytic_intercept)
  |-- _reconstruct_dense()      (internal, calls normal + solve_intercept, UNUSED)
  |-- plic_subsweep_x/y()      (calls apply_flux_{x,y}_conservative after apply_flux_{x,y})
  +-- plic_time_step_2d_strang()

diagnostics.py
  |-- PLICStats
  |-- interior_slice()
  |-- compute_stats()
  +-- format_stats()
```

---

# Appendix: Global Epsilon/Tolerance Overview Table

| Value | Location | Purpose |
|---|------|------|
| `1e-30` | `_volume_below_3d` (`max_abc`), `compute_plic_normals_3d` (`mag`), `analytic_intercept` (`_EPS`: `_safe_cbrt`, `D_A_cubed` denom, `D_B_cubed` denom, `V2_3d` denom, `V3_3d` denom) | Prevents division by zero or NaN from `0^(1/3)` (extremely small addend, does not affect precision) |
| `1e-12` | `intercept_solver` (`_N_EPS2`), `analytic_intercept` (`_N_EPS2`) | Normal vector magnitude-squared threshold for pure cell zeroing |
| `1e-10` | `_volume_below_3d` (`thr`), `analytic_intercept` (`thr`: coefficient lower bound, `p_ok` threshold, `safe_dV` denominator guard) | Absolute denominator lower bound / degeneracy threshold |
| `1e-6` | Multiple locations (`_INTERFACE_EPS`, `_F_CLIP_LO/HI`, `_SLOPE_EPS`, `_EPS`, interface mask, `F_safe` in `analytic_intercept`) | Unified interface/pure cell detection threshold and volume fraction clipping bound |
| `1e-4` | `_volume_below_3d` (`thr_rel`), `analytic_intercept` (`_THR_REL`) | Coefficient relative zero detection |
| `0.05` | `analytic_intercept` (inline in `near_2d`) | Near-degenerate 3D normal: fall back to 2D formula when $m_1 < 0.05 m_3$ |
| `0.15` | `strang_sweep` (`_MAX_INTERFACE_FRAC`) | Maximum fraction of total cells for padded-gather compact array |
| `0.01 / 0.99` | `intercept_solver` (`_T_CLIP_LO/HI`) | Regula Falsi interpolation parameter clip |
| `5` | `analytic_intercept` (inline Newton loop bound) | Number of Newton refinement steps |
| `15` | `intercept_solver` (`_DEFAULT_N_ITER`) | Fixed Regula Falsi iteration count |

---

# Appendix: Phase A vs Phase B Intercept Solver Comparison

| Property | Phase A (`solve_intercept`) | Phase B (`analytic_intercept`) |
|----------|---------------------------|-------------------------------|
| Algorithm | Regula Falsi + bisection fallback | Scardovelli-Zaleski 2000 analytic inverse + Newton polish |
| Iteration count | 15 fixed | 0 (analytic) + 5 Newton |
| Forward model calls | 15 (one per iteration) | 16 (6 root selection + 10 Newton) |
| Accuracy (float32) | ~$10^{-5}$ relative | ~$10^{-7}$ relative (Newton converges quadratically) |
| Accuracy (float64) | ~$10^{-7}$ relative | ~$10^{-14}$ relative (saturates at machine ε after 3 Newton steps) |
| Loop construct | `jax.lax.fori_loop` (XLA while loop) | Python for-loop (unrolled at trace time) |
| Current usage | `_reconstruct_dense` (unused fallback) | `_reconstruct` (active path via padded-gather) |
| Complexity per cell | $O(N_{\mathrm{iter}})$ | $O(1)$ analytic + $O(N_{\mathrm{Newton}})$ polish |

---

# Appendix: `__init__.py` (Package Initializer)

The file `src/jax_laseram/vof/plic/__init__.py` contains ONLY a module-level docstring describing the package's purpose and design goals. It has:

- **No imports** (no `from .xxx import ...`, no `import xxx`)
- **No `__all__`** list
- **No executable code** beyond the docstring

The docstring (37 lines) describes three design goals (research knowledge artefact, reusable code artefact, mathematical rigour) and lists the planned module layout. It does NOT affect runtime behavior in any way.

A reimplementer MUST create this file (Python requires `__init__.py` for a package directory) but its contents can be an empty file or a docstring-only file. The critical constraint is that it MUST NOT contain any imports, as none of the other modules in the package rely on package-level re-exports from `__init__.py`. Each module imports its dependencies directly (e.g., `from .volume_formula import volume_below_plane_3d`).
