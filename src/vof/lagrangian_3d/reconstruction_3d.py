"""Step 1: 3D PLIC Interface Reconstruction — JAX-Native.

Computes piecewise linear interface normals (nx, ny, nz) and intercepts C
for all cells using pure tensor operations (no Python loops, no branches).

Key design:
  - Normal: 3×3×3 Least-Squares via conv_general_dilated (Prewitt 3D)
  - Intercept: bisection with Gauss-Legendre quadrature (zero branches)

Reference: Barkhudarov (2004) §4: "Least Squares method using a 27-point stencil."
"""

import jax
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss

from ...data_types import Array


# ═══════════════════════════════════════════════════════════════
# Gauss-Legendre quadrature on [0,1]³ (compile-time constants)
# ═══════════════════════════════════════════════════════════════


def _prepare_gl_quadrature(n: int = 7):
    """Precompute Gauss-Legendre nodes and weights on [0,1]³."""
    nodes_1d, weights_1d = leggauss(n)
    nodes_1d = 0.5 * (nodes_1d + 1.0)
    weights_1d = 0.5 * weights_1d

    xi, eta, zeta = jnp.meshgrid(nodes_1d, nodes_1d, nodes_1d, indexing="ij")
    nodes_3d = jnp.stack([xi.ravel(), eta.ravel(), zeta.ravel()], axis=-1).astype(
        jnp.float32
    )

    wi, wj, wk = jnp.meshgrid(weights_1d, weights_1d, weights_1d, indexing="ij")
    weights_3d = (wi * wj * wk).ravel().astype(jnp.float32)

    return nodes_3d, weights_3d


gl_nodes_5, gl_weights_5 = _prepare_gl_quadrature(7)  # 343 pts, exact to deg 13


# ═══════════════════════════════════════════════════════════════
# 3D Prewitt convolution for interface normals
# ═══════════════════════════════════════════════════════════════


def compute_plic_normals_3d(F: Array, dx: float, dy: float, dz: float) -> tuple:
    """Compute 3D PLIC interface normal via 27-point Least-Squares gradient.

    NCDHW layout: D=z-axis(0), H=x-axis(1), W=y-axis(2).
    Kernel axis 0=D=z, axis 1=H=x, axis 2=W=y.

    Args:
        F: volume fraction, shape (Nx, Ny, Nz, 1).
        dx, dy, dz: cell sizes.

    Returns:
        nx, ny, nz: normal components, same shape as F.
            Normal points from fluid (F=1) toward empty (F=0).
    """
    # ── Build kernels ──
    # 3D Sobel (Parker-Youngs) weighting for better isotropy:
    #   face=2, edge=1, corner not used for the cross-section weights.
    #   Cross-section weight matrix: [[1,2,1],[2,4,2],[1,2,1]] / sum
    # This gives less grid-aligned bias than uniform Prewitt weights.
    dt_ = F.dtype
    w2d = jnp.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=dt_)
    w_sum = jnp.sum(w2d)  # = 16

    # Output channel 0 → ∂F/∂x (varies along kernel axis 1 = H = x)
    # Cross-section weight is w2d applied in (D=z, W=y) plane
    ch0 = jnp.zeros((3, 3, 3), dtype=dt_)
    ch0 = ch0.at[:, 0, :].set(-w2d)
    ch0 = ch0.at[:, 2, :].set(w2d)
    ch0 = ch0 / (2.0 * w_sum * dx)

    # Output channel 1 → ∂F/∂y (varies along kernel axis 2 = W = y)
    # Cross-section weight is w2d applied in (D=z, H=x) plane
    ch1 = jnp.zeros((3, 3, 3), dtype=dt_)
    ch1 = ch1.at[:, :, 0].set(-w2d)
    ch1 = ch1.at[:, :, 2].set(w2d)
    ch1 = ch1 / (2.0 * w_sum * dy)

    # Output channel 2 → ∂F/∂z (varies along kernel axis 0 = D = z)
    # Cross-section weight is w2d applied in (H=x, W=y) plane
    ch2 = jnp.zeros((3, 3, 3), dtype=dt_)
    ch2 = ch2.at[0, :, :].set(-w2d)
    ch2 = ch2.at[2, :, :].set(w2d)
    ch2 = ch2 / (2.0 * w_sum * dz)

    kernel = jnp.stack([ch0, ch1, ch2])[:, None, :, :, :]
    # kernel shape: (3, 1, 3, 3, 3) [OIDHW]

    # ── Reshape F to NCDHW ──
    # F shape: (Nx, Ny, Nz, 1)
    # F_spatial[i,j,k] = F at cell (x_i, y_j, z_k)
    # NCDHW: F_arr[0,0,d,h,w] = F_spatial[h,w,d]
    F_spatial = F[:, :, :, 0]  # (Nx, Ny, Nz)
    lhs = jnp.transpose(F_spatial, (2, 0, 1))[None, None, :, :, :]
    # lhs: (1, 1, Nz, Nx, Ny) = (N, C, D, H, W)

    # ── 3D convolution ──
    grad = jax.lax.conv_general_dilated(
        lhs,
        kernel,
        window_strides=(1, 1, 1),
        padding="VALID",
        dimension_numbers=("NCDHW", "OIDHW", "NCDHW"),
    )
    # grad: (1, 3, Nz-2, Nx-2, Ny-2)
    # grad[0,0] = ch0 result = ∂F/∂x, in (D,H,W)=(z,x,y) order

    # ── Unpack and transpose back to (Nx, Ny, Nz) ──
    dFdx_raw = grad[0, 0]  # (Nz-2, Nx-2, Ny-2)
    dFdy_raw = grad[0, 1]
    dFdz_raw = grad[0, 2]

    dFdx = jnp.zeros(F.shape[:3], dtype=dt_)
    dFdy = jnp.zeros(F.shape[:3], dtype=dt_)
    dFdz = jnp.zeros(F.shape[:3], dtype=dt_)

    # raw (z,x,y) → transpose (1,2,0) → (x,y,z)
    dFdx = dFdx.at[1:-1, 1:-1, 1:-1].set(jnp.transpose(dFdx_raw, (1, 2, 0)))
    dFdy = dFdy.at[1:-1, 1:-1, 1:-1].set(jnp.transpose(dFdy_raw, (1, 2, 0)))
    dFdz = dFdz.at[1:-1, 1:-1, 1:-1].set(jnp.transpose(dFdz_raw, (1, 2, 0)))

    # ── Normalize ──
    mag = jnp.sqrt(dFdx**2 + dFdy**2 + dFdz**2 + 1e-30)
    nx = -dFdx / mag
    ny = -dFdy / mag
    nz = -dFdz / mag

    # Zero normals in pure cells
    is_interface = (F[..., 0] > 1e-6) & (F[..., 0] < 1.0 - 1e-6)
    nx = jnp.where(is_interface, nx, 0.0)
    ny = jnp.where(is_interface, ny, 0.0)
    nz = jnp.where(is_interface, nz, 0.0)

    return nx[..., None], ny[..., None], nz[..., None]


# ═══════════════════════════════════════════════════════════════
# Truncated cube volume — Scardovelli & Zaleski (2000) exact formula
# ═══════════════════════════════════════════════════════════════


def _volume_below_3d(C: Array, a: Array, b: Array, c: Array, **_kw) -> Array:
    """EXACT Vol{(x,y,z)∈[0,1]³ : ax+by+cz ≤ C}.

    Scardovelli & Zaleski (2000) inclusion-exclusion formula with explicit
    handling of degenerate cases (1 or 2 zero coefficients).
    Pure jnp.where cascade — no branches, fully vectorisable.
    """
    # Step 1: reflect to non-negative coefficients
    a0 = jnp.abs(a)
    b0 = jnp.abs(b)
    c0 = jnp.abs(c)
    # Map from physical C (relative to cell center) to unit-cube coordinate d:
    # d = C_phys + 0.5*(|a|+|b|+|c|)
    # This combines the center-to-corner offset AND the reflection corrections.
    d = C + 0.5 * (a0 + b0 + c0)

    # Count nonzero coefficients — use RELATIVE threshold to avoid
    # catastrophic cancellation in the 3D inclusion-exclusion formula
    thr = jnp.asarray(1e-10, a0.dtype)  # absolute floor for denominators
    max_abc = jnp.maximum(a0, jnp.maximum(b0, c0)) + jnp.asarray(1e-30, a0.dtype)
    thr_rel = jnp.asarray(1e-4, a0.dtype)  # coefficient < 0.01% of max → treat as zero
    nz_a, nz_b, nz_c = a0 > thr_rel * max_abc, b0 > thr_rel * max_abc, c0 > thr_rel * max_abc
    n_nz = nz_a.astype(jnp.int32) + nz_b.astype(jnp.int32) + nz_c.astype(jnp.int32)

    # ── n_nz = 0: uniform → V = (d >= 0) ──
    V_0d = jnp.where(d >= 0, 1.0, 0.0)

    # ── n_nz = 1: 1D slab → V = clamp(d/m, 0, 1) ──
    m_1d = jnp.where(nz_a, a0, jnp.where(nz_b, b0, c0)) + thr
    V_1d = jnp.clip(d / m_1d, 0.0, 1.0)

    # ── n_nz = 2: 2D triangle/trapezoid ──
    # axis=0 is critical: when called via vmap, inputs are (Nz,) arrays,
    # so jnp.array([...]) is (3, Nz). Default sort (axis=-1) would sort
    # along Nz instead of the 3-coefficient axis, completely breaking the
    # 2D formula and causing the PLIC bisection to converge to wrong C.
    vals = jnp.sort(jnp.array([
        jnp.where(nz_a, a0, 0.0),
        jnp.where(nz_b, b0, 0.0),
        jnp.where(nz_c, c0, 0.0),
    ]), axis=0)
    p2 = jnp.maximum(vals[1], thr)
    q2 = jnp.maximum(vals[2], thr)
    S2 = p2 + q2
    pq2 = 2.0 * p2 * q2
    V_2d = jnp.where(d <= 0, 0.0,
            jnp.where(d >= S2, 1.0,
            jnp.where(d <= p2, d ** 2 / pq2,
            jnp.where(d <= q2, (2.0 * d - p2) / (2.0 * q2),
                       1.0 - (S2 - d) ** 2 / pq2))))

    # ── n_nz = 3: full 3D inclusion-exclusion ──
    lo = jnp.minimum(a0, jnp.minimum(b0, c0))
    hi = jnp.maximum(a0, jnp.maximum(b0, c0))
    mi = a0 + b0 + c0 - lo - hi
    m1 = jnp.maximum(lo, thr)
    m2 = jnp.maximum(mi, thr)
    m3 = jnp.maximum(hi, thr)
    S3 = m1 + m2 + m3
    p6 = 6.0 * m1 * m2 * m3

    dm1 = jnp.maximum(d - m1, 0.0)
    dm2 = jnp.maximum(d - m2, 0.0)
    dm3 = jnp.maximum(d - m3, 0.0)
    dm12 = jnp.maximum(d - m1 - m2, 0.0)
    dm13 = jnp.maximum(d - m1 - m3, 0.0)
    dm23 = jnp.maximum(d - m2 - m3, 0.0)
    dS = jnp.maximum(d - S3, 0.0)

    numer = (d ** 3 - dm1 ** 3 - dm2 ** 3 - dm3 ** 3
             + dm12 ** 3 + dm13 ** 3 + dm23 ** 3 - dS ** 3)
    V_3d = jnp.where(d <= 0, 0.0, jnp.where(d >= S3, 1.0, numer / p6))

    # ── Select by dimensionality ──
    V = jnp.where(n_nz == 0, V_0d,
        jnp.where(n_nz == 1, V_1d,
        jnp.where(n_nz == 2, V_2d, V_3d)))

    return jnp.clip(V, 0.0, 1.0)


# Batch version: vmap over (Ny, Nz) for a (Nx, Ny, Nz) field
_volume_below_3d_2dmap = jax.vmap(jax.vmap(_volume_below_3d))


def compute_intercept_C_3d(
    F: Array,
    nx: Array,
    ny: Array,
    nz: Array,
    dx: float,
    dy: float,
    dz: float,
    n_iter: int = 20,
) -> Array:
    """Compute PLIC intercept C via bisection in physical C space."""
    a = nx * dx
    b = ny * dy
    c = nz * dz

    # C_phys ranges from -C_half to +C_half where C_half = 0.5*(|a|+|b|+|c|)
    # At C=-C_half: plane at cell corner → V=0. At C=+C_half: V=1.
    C_half = 0.5 * (jnp.abs(a) + jnp.abs(b) + jnp.abs(c))
    C_lo = -C_half
    C_hi = C_half

    F_target = jnp.clip(F, 1e-10, 1.0 - 1e-10)

    a_s = a[..., 0]
    b_s = b[..., 0]
    c_s = c[..., 0]
    C_lo_s = C_lo[..., 0]
    C_hi_s = C_hi[..., 0]
    F_s = F_target[..., 0]

    def body_fn(i, state):
        C_mid, lo, hi = state
        vol = _volume_below_3d_2dmap(C_mid, a_s, b_s, c_s)
        below = vol < F_s
        lo_new = jnp.where(below, C_mid, lo)
        hi_new = jnp.where(below, hi, C_mid)
        mid_new = 0.5 * (lo_new + hi_new)
        return mid_new, lo_new, hi_new

    C_init = jnp.zeros_like(a_s)
    C_final, _, _ = jax.lax.fori_loop(0, n_iter, body_fn, (C_init, C_lo_s, C_hi_s))

    is_interface = (F[..., 0] > 1e-6) & (F[..., 0] < 1.0 - 1e-6)
    C_final = jnp.where(is_interface, C_final, 0.0)

    return C_final[..., None]
