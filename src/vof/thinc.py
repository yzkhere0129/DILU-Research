"""THINC/SW — Slope-Weighted THINC with multi-dimensional normal.

Key improvements over 1D THINC:
  1. Analytical x_tilde (no Newton, no fallback)
  2. 2D interface normal from Youngs gradient (3×3 stencil)
  3. Slope-weighted β: β_eff = β × |n_sweep| / |n|
     → horizontal interface + x-sweep: β_eff ≈ 0 (no reconstruction)
     → interface ⊥ sweep: β_eff = β (full reconstruction)
     This eliminates the directional bias of 1D operator splitting.

Reference: Xiao et al. (2011), "Revisit to the THINC scheme"
           Ii et al. (2012), "THINC method with multi-dimensional reconstruction"
"""

import jax.numpy as jnp
from ..data_types import Array


def _logcosh(x: Array) -> Array:
    """Numerically stable log(cosh(x))."""
    return jnp.where(jnp.abs(x) > 15.0,
                     jnp.abs(x) - jnp.log(2.0),
                     jnp.log(jnp.cosh(x)))


def _solve_xtilde(F_eff: Array, beta_eff: Array) -> Array:
    """Analytical x_tilde for given F_eff and (variable) beta.

    x̃ = atanh([cosh(β) - exp(2β(F-0.5))] / sinh(β)) / β

    When β → 0 (interface parallel to sweep): x̃ → 0.5 - F + 0.5 = 1-F
    which gives H(ξ) ≈ F everywhere (constant, = donor cell).
    """
    # Avoid β = 0 (would give division by zero)
    beta_safe = jnp.maximum(beta_eff, 0.01)

    cosh_b = jnp.cosh(beta_safe)
    sinh_b = jnp.sinh(beta_safe)
    P = jnp.exp(2.0 * beta_safe * (F_eff - 0.5))
    arg = (cosh_b - P) / sinh_b
    arg = jnp.clip(arg, -1.0 + 1e-7, 1.0 - 1e-7)
    return jnp.arctanh(arg) / beta_safe


def compute_youngs_gradient(F: Array, dx: float, dy: float) -> tuple:
    """Compute 2D interface normal using Youngs gradient (3×3 stencil).

    Uses centered differences on the full F array (including ghosts).

    Returns:
        dFdx, dFdy: gradient components at cell centers (same shape as F,
                     but border values are unreliable)
    """
    # Centered differences (2nd order)
    dFdx = jnp.zeros_like(F)
    dFdy = jnp.zeros_like(F)

    dFdx = dFdx.at[1:-1, :, :].set(
        (F[2:, :, :] - F[:-2, :, :]) / (2.0 * dx))
    dFdy = dFdy.at[:, 1:-1, :].set(
        (F[:, 2:, :] - F[:, :-2, :]) / (2.0 * dy))

    return dFdx, dFdy


def compute_flux_volume(F_L: Array, F_R: Array,
                        vel_face: Array, dt: float, dx: float,
                        beta: float, axis: int,
                        dFdx_full: Array, dFdy_full: Array,
                        dy: float = 1.0) -> Array:
    """Compute THINC/SW flux with multi-dimensional normal.

    Args:
        F_L, F_R: volume fractions at left/right cells
        vel_face: face velocity (signed)
        dt, dx: timestep and cell size in sweep direction
        beta: base sharpness parameter
        axis: 0 for x-sweep, 1 for y-sweep
        dFdx_full, dFdy_full: Youngs gradient at ALL cells
        dy: cell size in cross direction (for gradient magnitude)

    Returns:
        flux_vol: signed volume fraction transported
    """
    eps = 1e-10
    C_abs = jnp.abs(vel_face) * dt / dx
    going_right = vel_face >= 0

    # Donor cell
    F_donor = jnp.where(going_right, F_L, F_R)
    is_interface = (F_donor > eps) & (F_donor < 1.0 - eps)

    # ── Multi-dimensional interface normal at donor cell ──
    if axis == 0:
        # x-sweep: donor gradient from the left or right cell
        grad_sweep_L = dFdx_full[:-1, :, :]   # dF/dx at left cell
        grad_sweep_R = dFdx_full[1:, :, :]    # dF/dx at right cell
        grad_cross_L = dFdy_full[:-1, :, :]   # dF/dy at left cell
        grad_cross_R = dFdy_full[1:, :, :]    # dF/dy at right cell
    else:
        grad_sweep_L = dFdy_full[:, :-1, :]
        grad_sweep_R = dFdy_full[:, 1:, :]
        grad_cross_L = dFdx_full[:, :-1, :]
        grad_cross_R = dFdx_full[:, 1:, :]

    # Select donor's gradient
    grad_sweep = jnp.where(going_right, grad_sweep_L, grad_sweep_R)
    grad_cross = jnp.where(going_right, grad_cross_L, grad_cross_R)

    # Interface normal magnitude
    grad_mag = jnp.sqrt(grad_sweep**2 + grad_cross**2 + eps**2)

    # Slope-weighted beta: β_eff = β × |n_sweep| / |n|
    # When interface is parallel to sweep direction: |n_sweep| ≈ 0 → β_eff ≈ 0
    # When interface is perpendicular to sweep: |n_sweep| ≈ |n| → β_eff ≈ β
    beta_eff = beta * jnp.abs(grad_sweep) / grad_mag

    # Interface orientation (sigma) from the sweep-direction gradient
    sigma = jnp.sign(grad_sweep)
    sigma = jnp.where(sigma == 0, 1.0, sigma)

    # ── THINC reconstruction with variable beta ──
    F_eff = jnp.where(sigma >= 0, F_donor, 1.0 - F_donor)
    F_eff = jnp.clip(F_eff, eps, 1.0 - eps)

    x_t = _solve_xtilde(F_eff, beta_eff)

    # Integrated flux for vel >= 0: ∫_{1-C}^{1} H dξ
    a_pos = beta_eff * (1.0 - C_abs - x_t)
    b_pos = beta_eff * (1.0 - x_t)
    int_pos = 0.5 * C_abs + (_logcosh(b_pos) - _logcosh(a_pos)) / (2.0 * beta_eff + eps)

    # Integrated flux for vel < 0: ∫_{0}^{C} H dξ
    a_neg = beta_eff * (0.0 - x_t)
    b_neg = beta_eff * (C_abs - x_t)
    int_neg = 0.5 * C_abs + (_logcosh(b_neg) - _logcosh(a_neg)) / (2.0 * beta_eff + eps)

    # Select direction
    int_raw = jnp.where(going_right, int_pos, int_neg)

    # Undo sigma flip
    flux_thinc = jnp.where(sigma >= 0, int_raw, C_abs - int_raw)

    # Safety clamp
    flux_thinc = jnp.clip(flux_thinc, 0.0, jnp.maximum(F_donor, C_abs))

    # Pure cells: donor flux
    flux_pure = F_donor * C_abs

    # Use THINC only where β_eff is significant (interface crosses this direction).
    # When β_eff < threshold, interface is parallel to sweep → donor-cell is exact.
    # This is a GEOMETRIC criterion (interface orientation), not a volume criterion.
    use_thinc = is_interface & (beta_eff > 0.5)
    flux_unsigned = jnp.where(use_thinc, flux_thinc, flux_pure)

    return jnp.where(going_right, flux_unsigned, -flux_unsigned)
