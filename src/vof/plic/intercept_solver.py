"""PLIC plane-intercept solver — Phase A (Regula Falsi + bisection fallback).

Given the interface normal :math:`\\mathbf{n} = (n_x, n_y, n_z)` and a
target volume fraction :math:`F`, find the plane intercept :math:`C`
(in physical, cell-centered coordinates) such that

.. math::

    V(C; \\mathbf{n}) \\equiv \\mathrm{Vol}\\{(x,y,z)\\in[0,1]^3 :
        a x + b y + c z \\le C\\} = F,

where :math:`(a,b,c) = (n_x\\Delta x, n_y\\Delta y, n_z\\Delta z)`. The
forward model ``V(C)`` is the Scardovelli-Zaleski analytic formula in
:mod:`jax_laseram.vof.plic.volume_formula`.

Strategy (Phase A)
------------------
Fixed-step Regula Falsi with bisection fallback, 15 iterations under
:func:`jax.lax.fori_loop`. Rationale from the plan:

1. **Bracketed root**: :math:`V(C)` is monotone non-decreasing in
   :math:`C`, so the bracket invariant :math:`V(\\text{lo}) \\le F \\le
   V(\\text{hi})` is preserved exactly. Bisection is the degenerate case
   where we always split the bracket in half; Regula Falsi uses linear
   interpolation on :math:`(C_{\\text{lo}}, V_{\\text{lo}})` and
   :math:`(C_{\\text{hi}}, V_{\\text{hi}})` which converges super-linearly
   on smooth-ish :math:`V(C)`.
2. **Bisection fallback**: when the bracket-endpoint slope
   :math:`V_{\\text{hi}} - V_{\\text{lo}}` is smaller than ``1e-6``
   (near-converged), or the linear-interpolation fraction falls outside
   ``[0.01, 0.99]`` (stuck-endpoint pathology), we take a half-step
   instead. This guarantees worst-case bisection behaviour without
   relying on the Illinois state variable.
3. **Fixed step count**: GPU SIMT wants batch-uniform control flow,
   so ``while_loop`` (which waits on the slowest cell) is out.
   ``fori_loop(0, 15, body)`` runs exactly 15 body applications for
   every cell. In float32 the bracket shrinks by at most a factor of
   :math:`2^{-15} \\approx 3 \\times 10^{-5}` per bisection step; with
   Regula Falsi acceleration, ``|V(C) - F|`` typically drops below
   :math:`10^{-6}` (float32 machine epsilon) within 15 steps for
   all non-pathological inputs.

Phase B (Stage 4 of research plan) will add an analytic Scardovelli-
Zaleski inverse via Cardano's formula, replacing the iterative solver
entirely. Phase A then becomes the AD-safe fallback and oracle.

Shape contract
--------------
All inputs are shape-polymorphic; the typical use case is 3-D arrays
``(Nx, Ny, Nz)`` (the PLIC-per-cell pattern), but scalars and 1-D arrays
are also supported for unit testing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .volume_formula import volume_below_plane_3d

__all__ = ["solve_intercept"]

# Number of Regula Falsi iterations. Tuned for float32 convergence with
# the worst-case cube diagonal bracket; see the module docstring. This
# is a module-level constant so ``fori_loop`` can static-trace it.
_DEFAULT_N_ITER = 15

# Safeguard thresholds. ``_SLOPE_EPS`` decides when to fall back from
# Regula Falsi to bisection; ``_T_CLIP`` bounds the linear-interp step
# so a pathological Regula Falsi pattern still shrinks the bracket.
_SLOPE_EPS = 1.0e-6
_T_CLIP_LO = 0.01
_T_CLIP_HI = 0.99

# F clipping — avoids exact 0/1 targets which would make the bracket
# degenerate (lo or hi "already" matches). Caller should also mask out
# pure cells upstream, but this is defensive.
_F_CLIP_LO = 1.0e-6
_F_CLIP_HI = 1.0 - 1.0e-6

# Pure-cell detection threshold on the normal magnitude. Returned
# intercept is zero when ``|n|^2 < _N_EPS2``.
_N_EPS2 = 1.0e-12


def solve_intercept(
    nx: jnp.ndarray,
    ny: jnp.ndarray,
    nz: jnp.ndarray,
    F: jnp.ndarray,
    dx: float,
    dy: float,
    dz: float,
    n_iter: int = _DEFAULT_N_ITER,
) -> jnp.ndarray:
    """Solve :math:`V(C; \\mathbf{n}) = F` for :math:`C` (Phase A).

    Parameters
    ----------
    nx, ny, nz : jnp.ndarray
        Components of the PLIC interface normal, same shape, unit
        vectors on interface cells (zero on pure cells).
    F : jnp.ndarray
        Target volume fraction, same shape as the normals. Values are
        clipped into ``[1e-6, 1 - 1e-6]`` internally to keep the
        Regula-Falsi bracket non-degenerate.
    dx, dy, dz : float
        Cell sizes along x, y, z.
    n_iter : int, optional
        Number of fixed Regula Falsi / bisection iterations (default
        ``15``). Must be a Python int (compile-time constant) since it
        drives a ``fori_loop``.

    Returns
    -------
    C : jnp.ndarray
        Plane intercept in physical cell-centered coordinates, same
        shape as the inputs. Zero in pure cells where
        :math:`|\\mathbf{n}|^2 < 10^{-12}`.

    Notes
    -----
    This routine is pure ``jax.jit``-compatible and does not depend on
    any data-dependent shapes, so it is safe to wrap in higher-level
    ``jax.vmap`` / ``jax.pmap``.
    """
    # Plane coefficients in unit-cube space
    a = nx * dx
    b = ny * dy
    c = nz * dz

    # Symmetric bracket about cell center: V(-C_half) = 0, V(+C_half) = 1
    # (by construction of the Scardovelli-Zaleski forward model).
    C_half = 0.5 * (jnp.abs(a) + jnp.abs(b) + jnp.abs(c))
    lo0 = -C_half
    hi0 = +C_half
    v_lo0 = jnp.zeros_like(F)
    v_hi0 = jnp.ones_like(F)

    F_safe = jnp.clip(F, _F_CLIP_LO, _F_CLIP_HI)

    def body(_: int, state):
        lo, hi, v_lo, v_hi = state
        # --- Regula Falsi linear interpolation ---
        denom = v_hi - v_lo
        slope_ok = jnp.abs(denom) > _SLOPE_EPS
        safe_denom = jnp.where(slope_ok, denom, 1.0)
        t_rf = (F_safe - v_lo) / safe_denom
        # Clip to avoid stuck-endpoint pathologies (Illinois-lite guard)
        t_rf_clipped = jnp.clip(t_rf, _T_CLIP_LO, _T_CLIP_HI)
        # Fall back to bisection when the slope is near-zero
        t = jnp.where(slope_ok, t_rf_clipped, 0.5)
        C_new = lo + t * (hi - lo)

        v_new = volume_below_plane_3d(C_new, a, b, c)

        # Update bracket: below target -> replace low side, else replace high side
        below = v_new < F_safe
        lo_out = jnp.where(below, C_new, lo)
        hi_out = jnp.where(below, hi, C_new)
        v_lo_out = jnp.where(below, v_new, v_lo)
        v_hi_out = jnp.where(below, v_hi, v_new)
        return lo_out, hi_out, v_lo_out, v_hi_out

    lo_f, hi_f, v_lo_f, v_hi_f = jax.lax.fori_loop(
        0, n_iter, body, (lo0, hi0, v_lo0, v_hi0)
    )

    # Final estimate: linear-interpolate once more on the final bracket
    # (this is "free" — one more Regula Falsi step without updating state).
    denom = v_hi_f - v_lo_f
    slope_ok = jnp.abs(denom) > _SLOPE_EPS
    safe_denom = jnp.where(slope_ok, denom, 1.0)
    t_final = jnp.where(
        slope_ok,
        jnp.clip((F_safe - v_lo_f) / safe_denom, 0.0, 1.0),
        0.5,
    )
    C_final = lo_f + t_final * (hi_f - lo_f)

    # Zero out pure cells (degenerate normals)
    n_sq = nx * nx + ny * ny + nz * nz
    is_interface = n_sq > _N_EPS2
    return jnp.where(is_interface, C_final, jnp.zeros_like(C_final))
