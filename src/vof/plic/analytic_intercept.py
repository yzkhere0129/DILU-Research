"""Scardovelli-Zaleski (2000) analytic intercept — Phase B.

Given the interface normal and a target volume fraction F, compute the
PLIC plane intercept d analytically (O(1) per cell, no iteration).

This is the **inverse** of ``volume_below_plane_3d``.  The forward
model V(d) is a piecewise polynomial; the inverse d(V) is therefore
piecewise-algebraic.  The implementation mirrors the forward model's
4-case structure (0D / 1D / 2D / 3D) with an ``n_nz`` dispatch and
uses the same degeneracy thresholds so that forward-then-inverse and
inverse-then-forward are consistent to float32 round-off.

Reference
---------
Scardovelli R, Zaleski S. "Analytical Relations Connecting Linear
Interfaces and Volume Fractions in Rectangular Grids."
J. Comput. Phys. 164 (2000), pp 228-237, Section 3.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["analytic_intercept"]

_EPS = 1.0e-30
_N_EPS2 = 1.0e-12
_THR_REL = 1.0e-4   # matches volume_formula's thr_rel for n_nz counting


def _safe_cbrt(x):
    """Branchless cube root safe for x <= 0."""
    return jnp.sign(x) * (jnp.abs(x) + _EPS) ** (1.0 / 3.0)


def analytic_intercept(
    nx: jnp.ndarray,
    ny: jnp.ndarray,
    nz: jnp.ndarray,
    F: jnp.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> jnp.ndarray:
    """Scardovelli-Zaleski 2000 analytic PLIC intercept (O(1), no iteration).

    Returns C in *physical cell-centered* coordinates (same convention
    as ``solve_intercept``).
    """
    # --- Absolute plane coefficients ---
    a0 = jnp.abs(nx * dx)
    b0 = jnp.abs(ny * dy)
    c0 = jnp.abs(nz * dz)
    S = a0 + b0 + c0  # used for coordinate mapping: d = C + S/2

    F_safe = jnp.clip(F, 1e-6, 1.0 - 1e-6)

    # --- Degeneracy detection (mirror of volume_below_plane_3d) ---
    thr = jnp.asarray(1e-10, F.dtype)
    max_abc = jnp.maximum(a0, jnp.maximum(b0, c0)) + _EPS
    nz_a = a0 > _THR_REL * max_abc
    nz_b = b0 > _THR_REL * max_abc
    nz_c = c0 > _THR_REL * max_abc
    n_nz = nz_a.astype(jnp.int32) + nz_b.astype(jnp.int32) + nz_c.astype(jnp.int32)

    # ================================================================
    # Case 0D (n_nz = 0): all coefficients ~0 → d meaningless
    # ================================================================
    d_0d = 0.5 * S  # V(d) = 1{d>=0} → any d is "correct" for F near 0 or 1

    # ================================================================
    # Case 1D (n_nz = 1): V = d / m_active → d = F * m_active
    # ================================================================
    m_1d = jnp.where(nz_a, a0, jnp.where(nz_b, b0, c0)) + thr
    d_1d = F_safe * m_1d

    # ================================================================
    # Case 2D (n_nz = 2): piecewise sqrt / linear inverse
    # ================================================================
    # Sort the non-zero coefficients into (p, q) with p <= q.
    # Same sorting as forward model: build [maybe_a, maybe_b, maybe_c]
    # with zeros for inactive, sort along axis=0, take vals[1]=p, vals[2]=q.
    vals = jnp.sort(jnp.array([
        jnp.where(nz_a, a0, jnp.zeros_like(a0)),
        jnp.where(nz_b, b0, jnp.zeros_like(b0)),
        jnp.where(nz_c, c0, jnp.zeros_like(c0)),
    ]), axis=0)
    p2 = jnp.maximum(vals[1], thr)   # smaller non-zero coeff
    q2 = jnp.maximum(vals[2], thr)   # larger non-zero coeff
    S2 = p2 + q2

    # Symmetry: work with F_w <= 0.5
    use_comp_2d = F_safe > 0.5
    F_w2 = jnp.where(use_comp_2d, 1.0 - F_safe, F_safe)

    # Transition volume: V1_2d = p2 / (2*q2)
    V1_2d = p2 / (2.0 * q2)

    # Region 1: F_w2 < V1_2d → d = sqrt(2*p2*q2*F_w2)
    d_2d_r1 = jnp.sqrt(jnp.maximum(2.0 * p2 * q2 * F_w2, 0.0))

    # Region 2: F_w2 >= V1_2d → d = q2*F_w2 + p2/2
    d_2d_r2 = q2 * F_w2 + p2 / 2.0

    d_2d_work = jnp.where(F_w2 < V1_2d, d_2d_r1, d_2d_r2)
    # Undo complement
    d_2d = jnp.where(use_comp_2d, S2 - d_2d_work, d_2d_work)

    # ================================================================
    # Case 3D (n_nz = 3): piecewise cbrt / sqrt / Cardano inverse
    # ================================================================
    # Branchless sorting network: m1 <= m2 <= m3
    lo1 = jnp.minimum(a0, b0)
    hi1 = jnp.maximum(a0, b0)
    lo2 = jnp.minimum(hi1, c0)
    m3  = jnp.maximum(hi1, c0)
    m1  = jnp.minimum(lo1, lo2)
    m2  = jnp.maximum(lo1, lo2)
    m1s = jnp.maximum(m1, thr)
    m2s = jnp.maximum(m2, thr)
    m3s = jnp.maximum(m3, thr)

    m12 = m1 + m2
    S3  = m1 + m2 + m3
    p6  = 6.0 * m1s * m2s * m3s

    # Symmetry: work with F_w <= 0.5
    use_comp_3d = F_safe > 0.5
    F_w3 = jnp.where(use_comp_3d, 1.0 - F_safe, F_safe)

    # Transition volumes (for F_w <= 0.5):
    V1_3d = m1s * m1s / (6.0 * m2s * m3s)
    V2_3d = V1_3d + (m2 - m1) / (2.0 * m3s + _EPS)
    # V3 = V(d=m3) = exact volume at the d=m3 breakpoint, where the
    # (d-m3)^3 term kicks in. Computed from the forward formula:
    #   V(m3) = [m3^3 - (m3-m1)^3 - (m3-m2)^3] / (6*m1*m2*m3)
    dm1_at_m3 = m3 - m1   # always >= 0
    dm2_at_m3 = m3 - m2   # always >= 0
    V3_3d = (m3 ** 3 - dm1_at_m3 ** 3 - dm2_at_m3 ** 3) / (p6 + _EPS)

    # Region 1: F_w3 < V1_3d → corner tetrahedron
    d_3d_r1 = _safe_cbrt(p6 * F_w3)

    # Region 2: V1 <= F_w3 < V2 → one edge clipped
    inner_r2 = m1 * m1 + 8.0 * m2s * m3s * (F_w3 - V1_3d)
    d_3d_r2 = 0.5 * (m1 + jnp.sqrt(jnp.maximum(inner_r2, 0.0)))

    # Region 3: V2 <= F_w3 <= V3 — depressed cubic via Cardano
    # Two sub-cases: m12 <= m3 and m12 > m3.

    # --- Sub-case A: m12 <= m3 ---
    # Cubic: d^3 - 3*(m1+m2)*d^2 + 3*(m1^2+m2^2)*d = m1^3+m2^3 - 6*m1*m2*m3*F_w
    # Depressed form (d = t + m12):  t^3 - 6*m1*m2*t + q = 0
    #   q = 3*m1*m2*(m12 - 2*m3*F_w)
    m1m2 = m1s * m2s
    p_A = -6.0 * m1m2
    # q = 3*m1*m2*(2*m3*F_w - m12)  [sign-critical: derived from
    #   depressed cubic shift d = t + m12 applied to the region-3a
    #   equation d³ - 3*m12*d² + 3*(m1²+m2²)*d - (C3-6m123F) = 0]
    q_A = 3.0 * m1m2 * (2.0 * m3s * F_w3 - m12)
    D_A = jnp.sqrt(jnp.maximum(2.0 * m1m2, _EPS))
    D_A_cubed = D_A * 2.0 * m1m2
    cos_A = jnp.clip(-q_A / (2.0 * D_A_cubed + _EPS), -1.0, 1.0)
    theta_A = jnp.arccos(cos_A)
    # All 3 Cardano roots — pick the one in valid range [m2, m3]
    t_A0 = 2.0 * D_A * jnp.cos(theta_A / 3.0)
    t_A1 = 2.0 * D_A * jnp.cos((theta_A + 2.0 * jnp.pi) / 3.0)
    t_A2 = 2.0 * D_A * jnp.cos((theta_A + 4.0 * jnp.pi) / 3.0)
    d_A0, d_A1, d_A2 = t_A0 + m12, t_A1 + m12, t_A2 + m12
    # Select root by evaluating V(d) and picking closest to F_w3.
    # This is foolproof: no wrong-root Cardano selection possible.
    from .volume_formula import volume_below_plane_3d as _vbp
    C_A0 = d_A0 - 0.5 * S3
    C_A1 = d_A1 - 0.5 * S3
    C_A2 = d_A2 - 0.5 * S3
    V_A0 = _vbp(C_A0, a0, b0, c0)
    V_A1 = _vbp(C_A1, a0, b0, c0)
    V_A2 = _vbp(C_A2, a0, b0, c0)
    e0 = jnp.abs(V_A0 - F_w3)
    e1 = jnp.abs(V_A1 - F_w3)
    e2 = jnp.abs(V_A2 - F_w3)
    d_3d_r3_A = jnp.where(e0 < e1, jnp.where(e0 < e2, d_A0, d_A2),
                                     jnp.where(e1 < e2, d_A1, d_A2))

    # --- Sub-case B: m12 > m3 ---
    # Full cubic including (d-m3)^3 term:
    #   -2d^3 + 3*S3*d^2 - 3*(m1^2+m2^2+m3^2)*d + (m1^3+m2^3+m3^3) = 6*m1*m2*m3*F_w
    # Depressed (d = t + S3/2):  t^3 + p'*t + q' = 0
    S3_sq = m1 * m1 + m2 * m2 + m3 * m3
    P2 = m1s * m2s + m1s * m3s + m2s * m3s
    m123 = m1s * m2s * m3s

    # Depressed cubic: d³ - (3S/2)d² + (3S2/2)d - (C3-6m123F)/2 = 0
    # Substitution d = t + S/2 → t³ + p·t + q = 0
    # p = 3·S2/2 - 3·S²/4          (must be negative for 3 real roots)
    # q = -(C3-6m123F)/2 + (3S·S2 - S³)/4
    p_B = 1.5 * S3_sq - 0.75 * S3 * S3
    sum_cubes = S3 ** 3 - 3.0 * S3 * P2 + 3.0 * m123  # m1³+m2³+m3³
    q_B = -(sum_cubes - 6.0 * m123 * F_w3) / 2.0 + (3.0 * S3 * S3_sq - S3 ** 3) / 4.0

    D_sq_B = jnp.maximum(-p_B / 3.0, _EPS)
    D_B = jnp.sqrt(D_sq_B)
    D_B_cubed = D_B * D_sq_B
    # Trigonometric Cardano requires p < 0 (three real roots).
    # When p >= 0 (near-axis normals: m1 ≈ 0), use direct formula.
    p_ok = (p_B < -1.0e-10)
    cos_B = jnp.clip(-q_B / (2.0 * D_B_cubed + _EPS), -1.0, 1.0)
    theta_B = jnp.arccos(cos_B)
    # All 3 Cardano roots — pick the one in valid range [m3, S3/2]
    t_B0 = 2.0 * D_B * jnp.cos(theta_B / 3.0)
    t_B1 = 2.0 * D_B * jnp.cos((theta_B + 2.0 * jnp.pi) / 3.0)
    t_B2 = 2.0 * D_B * jnp.cos((theta_B + 4.0 * jnp.pi) / 3.0)
    # Degenerate p≈0 fallback
    t_dir = _safe_cbrt(-q_B)
    t_B0 = jnp.where(p_ok, t_B0, t_dir)
    t_B1 = jnp.where(p_ok, t_B1, t_dir)
    t_B2 = jnp.where(p_ok, t_B2, t_dir)
    d_B0, d_B1, d_B2 = t_B0 + S3/2, t_B1 + S3/2, t_B2 + S3/2
    C_B0 = d_B0 - 0.5 * S3
    C_B1 = d_B1 - 0.5 * S3
    C_B2 = d_B2 - 0.5 * S3
    V_B0 = _vbp(C_B0, a0, b0, c0)
    V_B1 = _vbp(C_B1, a0, b0, c0)
    V_B2 = _vbp(C_B2, a0, b0, c0)
    eb0 = jnp.abs(V_B0 - F_w3)
    eb1 = jnp.abs(V_B1 - F_w3)
    eb2 = jnp.abs(V_B2 - F_w3)
    d_3d_r3_B = jnp.where(eb0 < eb1, jnp.where(eb0 < eb2, d_B0, d_B2),
                                       jnp.where(eb1 < eb2, d_B1, d_B2))

    # Sub-case A is valid whenever d < m3, which corresponds to
    # F_w < V3 (the volume at d = m3). This holds regardless of
    # whether m12 <= m3. For F_w >= V3, the (d-m3)^3 term is
    # non-zero and sub-case B (full 3-term equation) is required.
    d_3d_r3 = jnp.where(F_w3 < V3_3d, d_3d_r3_A, d_3d_r3_B)

    # Clamp cubic roots to their valid physical ranges before Newton.
    # Sub-case A: d ∈ [m2, min(m12, m3)];  Sub-case B: d ∈ [m3, S3/2]
    d_3d_r3_A = jnp.clip(d_3d_r3_A, m2, jnp.minimum(m12, m3))
    d_3d_r3_B = jnp.clip(d_3d_r3_B, m3, S3 / 2.0)
    d_3d_r3 = jnp.where(F_w3 < V3_3d, d_3d_r3_A, d_3d_r3_B)

    # Also clamp region 1 and 2 results to their ranges
    d_3d_r1 = jnp.clip(d_3d_r1, 0.0, m1)
    d_3d_r2 = jnp.clip(d_3d_r2, m1, m2)

    # Piecewise selection for 3D
    d_3d_work = jnp.where(F_w3 < V1_3d, d_3d_r1,
                jnp.where(F_w3 < V2_3d, d_3d_r2,
                          d_3d_r3))

    # Undo complement
    d_3d = jnp.where(use_comp_3d, S3 - d_3d_work, d_3d_work)

    # ================================================================
    # Dispatch by n_nz — with near-degenerate 3D → 2D fallback
    # ================================================================
    # When n_nz=3 but m1 is tiny relative to m2 (near-axis normal),
    # the 3D cubic becomes ill-conditioned (p_B > 0, wrong-root).
    # Fall back to the 2D formula which treats m1 as zero.
    # Degenerate 3D: when m1 is small relative to m3, the 3D cubic's
    # p coefficient goes positive (only 1 real root) and Cardano's
    # trigonometric formula breaks.  Fall back to the robust 2D formula.
    # Threshold: m1 < 5% of m3 → treat as 2D.  Higher thresholds
    # (like 0.15) catch more degenerate cubics but also break cells
    # where m1 is physically significant (observed: m1/m3=0.13 with
    # threshold 0.15 gave 2% error because 2D approximation was bad).
    near_2d = (n_nz == 3) & (m1 < 0.05 * m3)
    d_unit = jnp.where(n_nz == 0, d_0d,
             jnp.where(n_nz == 1, d_1d,
             jnp.where((n_nz == 2) | near_2d, d_2d, d_3d)))

    # ================================================================
    # Newton refinement (2 steps) — fixes float32 precision gaps and
    # wrong-root Cardano selections.  Cost: 4 extra calls to
    # volume_below_plane_3d.  Still O(1) per cell.
    # ================================================================
    from .volume_formula import volume_below_plane_3d

    C_cur = d_unit - 0.5 * S

    import jax

    def _newton_step(C_in):
        V_at = volume_below_plane_3d(C_in, a0, b0, c0)
        residual = V_at - F_safe
        # Exact derivative via JAX forward-mode AD (jvp).
        _, dVdC = jax.jvp(
            lambda c: volume_below_plane_3d(c, a0, b0, c0),
            (C_in,),
            (jnp.ones_like(C_in),),
        )
        safe_dV = jnp.where(dVdC > 1e-10, dVdC, jnp.ones_like(dVdC))
        C_new = C_in - residual / safe_dV
        # Clamp to valid domain: V is only defined for C ∈ [-S/2, S/2]
        return jnp.clip(C_new, -0.5 * S, 0.5 * S)

    for _ in range(5):
        C_cur = _newton_step(C_cur)
    C = C_cur

    # Zero out pure cells
    n_sq = nx * nx + ny * ny + nz * nz
    is_interface = n_sq > _N_EPS2
    return jnp.where(is_interface, C, jnp.zeros_like(C))
