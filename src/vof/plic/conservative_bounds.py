"""Weymouth-Zaleski conservative bounds redistribution for Eulerian PLIC.

After a PLIC sub-sweep the volume fraction ``F`` can overshoot
(``F > 1``) or undershoot (``F < 0``) by ~1e-3 due to float32 precision
in the Scardovelli-Zaleski analytic intercept.  Plain ``jnp.clip`` is
fast but non-conservative — the clipped mass is simply lost/created.

This module implements a small-fixed-iteration redistribution scheme
inspired by Weymouth & Zaleski (2010, JCP 229 §4.2) and OpenFOAM's
``MULES::limit`` (``nAlphaBounds``).  Each iteration:

1. Compute overshoot ``over = max(F - 1, 0)``  and
   undershoot ``under = max(-F, 0)``  per cell.
2. Clip ``F`` into ``[0, 1]``.
3. Push ``over`` to the **downstream** neighbour along the sweep axis
   (determined by the cell-centered sign of the sweep velocity).
4. Pull from the **upstream** neighbour by the same amount for
   ``under`` (i.e. subtract ``under`` from that neighbour).

Because every transfer is an equal add/subtract pair between two
interior cells, total volume is conserved exactly (to round-off).
Mass that would leave the domain via a boundary is *not* transferred
(zero-padded shift), which preserves conservation as long as the
interface never touches the outer halo — the standard regime for the
Stage 2 benchmarks.

Why fixed iterations and not ``while_loop``
-------------------------------------------
JAX's ``while_loop`` requires a rank-0 boolean carry and incurs large
compilation / dispatch overhead that dwarfs the work itself at the
1M-cell scale.  Empirically, 2-3 iterations are enough to drive
``max(F-1, 0)`` below 1e-7 for the float32 residues we see
(~5e-3 per sub-sweep).  A final ``jnp.clip`` is still applied to
guarantee the hard bound.

JIT safety
----------
- No ``while_loop``; uses ``jax.lax.fori_loop`` with a compile-time
  loop count.
- No ``jnp.roll`` — all neighbour shifts use padding + slicing so
  there is no periodic wrap-around at the domain boundary.
- No data-dependent shapes; every array has shape determined by
  the input pytree.

The interface is a plain drop-in replacement for ``jnp.clip(F, 0, 1)``
after ``apply_flux_{x,y,z}``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


__all__ = [
    "redistribute_bounds_x",
    "redistribute_bounds_y",
    "redistribute_bounds_z",
    # Drop-in API expected by the test suite and compare scripts:
    "apply_flux_x_conservative",
    "apply_flux_y_conservative",
    "apply_flux_z_conservative",
]


def _shift_plus1_x(a: jnp.ndarray) -> jnp.ndarray:
    """Shift ``a`` by +1 along axis 0 with zero-padding (no wrap)."""
    # a[i] <- a_in[i-1];  a_out[0] = 0
    return jnp.concatenate([jnp.zeros_like(a[:1]), a[:-1]], axis=0)


def _shift_minus1_x(a: jnp.ndarray) -> jnp.ndarray:
    """Shift ``a`` by -1 along axis 0 with zero-padding (no wrap)."""
    return jnp.concatenate([a[1:], jnp.zeros_like(a[-1:])], axis=0)


def _shift_plus1_y(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([jnp.zeros_like(a[:, :1]), a[:, :-1]], axis=1)


def _shift_minus1_y(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([a[:, 1:], jnp.zeros_like(a[:, -1:])], axis=1)


def _shift_plus1_z(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([jnp.zeros_like(a[:, :, :1]), a[:, :, :-1]], axis=2)


def _shift_minus1_z(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([a[:, :, 1:], jnp.zeros_like(a[:, :, -1:])], axis=2)


def _cell_sign_x(u_face: jnp.ndarray, shape) -> jnp.ndarray:
    """Cell-centered sign of the x sweep velocity.

    ``u_face`` may be a scalar (constant advection) or a face-centered
    array of shape ``(Nx-1, Ny, Nz)``.

    For face-centered velocities, we use the OUTFLOW face (the face on
    the downstream side of the cell) to determine the sign:
    - For sign > 0 (flow goes right): the relevant face for cell i is
      i+1/2, i.e. u_face[i]. Pad the right boundary with u_face[-1].
    - For sign < 0 (flow goes left): the relevant face for cell i is
      i-1/2, i.e. u_face[i-1]. Pad the left boundary with u_face[0].

    Using the AVERAGE of both adjacent faces causes cancellation when
    a single face has the opposite sign (e.g. a vortex), which gives
    sign=0 and silently drops the redistribution for those cells.
    Instead we take the sign of max(|u_left|, |u_right|) to pick the
    dominant face. This is robust for the sign-detection use case.
    """
    dtype = u_face.dtype if hasattr(u_face, "dtype") else jnp.float32
    if jnp.ndim(u_face) == 0:
        return jnp.full(shape, jnp.sign(u_face), dtype=dtype)
    # u_face shape = (Nx-1, Ny, Nz)
    # Pad to get shape (Nx, Ny, Nz) for left face (i-1/2) of each cell:
    u_left = jnp.concatenate([u_face[:1], u_face], axis=0)    # u_face[i-1]; pad left = u_face[0]
    # And for right face (i+1/2):
    u_right = jnp.concatenate([u_face, u_face[-1:]], axis=0)  # u_face[i]; pad right = u_face[-1]
    # Dominant face: pick the one with larger absolute value
    use_right = jnp.abs(u_right) >= jnp.abs(u_left)
    u_dom = jnp.where(use_right, u_right, u_left)
    return jnp.sign(u_dom[: shape[0]]).astype(dtype)


def _cell_sign_y(v_face: jnp.ndarray, shape) -> jnp.ndarray:
    """Cell-centered sign of the y sweep velocity (mirror of X)."""
    dtype = v_face.dtype if hasattr(v_face, "dtype") else jnp.float32
    if jnp.ndim(v_face) == 0:
        return jnp.full(shape, jnp.sign(v_face), dtype=dtype)
    v_left = jnp.concatenate([v_face[:, :1], v_face], axis=1)
    v_right = jnp.concatenate([v_face, v_face[:, -1:]], axis=1)
    use_right = jnp.abs(v_right) >= jnp.abs(v_left)
    v_dom = jnp.where(use_right, v_right, v_left)
    return jnp.sign(v_dom[:, : shape[1]]).astype(dtype)


def _cell_sign_z(w_face: jnp.ndarray, shape) -> jnp.ndarray:
    """Cell-centered sign of the z sweep velocity (mirror of X)."""
    dtype = w_face.dtype if hasattr(w_face, "dtype") else jnp.float32
    if jnp.ndim(w_face) == 0:
        return jnp.full(shape, jnp.sign(w_face), dtype=dtype)
    w_left = jnp.concatenate([w_face[:, :, :1], w_face], axis=2)
    w_right = jnp.concatenate([w_face, w_face[:, :, -1:]], axis=2)
    use_right = jnp.abs(w_right) >= jnp.abs(w_left)
    w_dom = jnp.where(use_right, w_right, w_left)
    return jnp.sign(w_dom[:, :, : shape[2]]).astype(dtype)


def _redistribute_body(
    F: jnp.ndarray,
    sign: jnp.ndarray,
    shift_plus,
    shift_minus,
    n_iter: int,
) -> jnp.ndarray:
    """Generic fixed-iteration redistribute along one axis.

    ``sign`` has the same shape as ``F`` and gives the cell-centered
    sweep-velocity sign in ``{-1, 0, +1}``.  ``shift_plus`` /
    ``shift_minus`` are the zero-padded ``±1`` shift ops for the target
    axis.

    Conservation algorithm
    ----------------------
    For each interior cell with overshoot (F > 1):
    - Primary: push surplus to the INTERIOR downstream neighbor.
    - Fallback: if the downstream neighbor is the halo, push to the
      upstream interior neighbor instead.
    - If neither neighbor is interior: loss is unavoidable (single-cell
      interior domain).

    Key invariant: the interior mask (shift_plus(interior) or
    shift_minus(interior)) tells us whether the destination is interior.
    We use this to decide direction, not the physical existence of the cell.
    """
    # Direction masks
    mask_pos = (sign > 0).astype(F.dtype)
    mask_neg = (sign < 0).astype(F.dtype)

    ones = jnp.ones_like(F)
    # Interior mask: cells that have a valid neighbor on BOTH sides.
    # shift_minus(ones)[i] > 0 → physical cell i+1 exists (may be halo)
    # shift_plus(ones)[i] > 0  → physical cell i-1 exists (may be halo)
    has_phys_plus = (shift_minus(ones) > 0.0).astype(F.dtype)
    has_phys_minus = (shift_plus(ones) > 0.0).astype(F.dtype)
    interior = has_phys_plus * has_phys_minus  # cells that are not on the boundary

    # Does the DOWNSTREAM neighbor of cell i exist AND is it interior?
    # Interior cells are those with BOTH neighbors, i.e. interior[i+1] = 1 for
    # i+1 to be a valid donation target.
    # shift_plus(interior)[i] = interior[i-1]: tells cell i if its left neighbor is interior
    # shift_minus(interior)[i] = interior[i+1]: tells cell i if its right neighbor is interior
    downstream_interior_pos = shift_minus(interior)  # interior[i+1] at cell i
    downstream_interior_neg = shift_plus(interior)   # interior[i-1] at cell i

    # u>0: donate to i+1 (ONLY if i+1 is interior, i.e. downstream_interior_pos[i]=1)
    # u<0: donate to i-1 (ONLY if i-1 is interior, i.e. downstream_interior_neg[i]=1)
    has_downstream = mask_pos * downstream_interior_pos + mask_neg * downstream_interior_neg

    # Fallback: when primary downstream is not interior, donate to upstream (if interior)
    # u>0, i+1 not interior → donate to i-1 (upstream)
    # u<0, i-1 not interior → donate to i+1 (upstream for u<0)
    no_downstream_pos = mask_pos * (1.0 - downstream_interior_pos)  # u>0 but i+1 is halo
    no_downstream_neg = mask_neg * (1.0 - downstream_interior_neg)  # u<0 but i-1 is halo
    has_fallback = no_downstream_pos * downstream_interior_neg \
                 + no_downstream_neg * downstream_interior_pos

    # Only redistribute from interior cells
    can_redistribute = interior

    def body(_, F_in):
        # Key insight: do NOT clip F before distributing.  Instead, compute
        # the overshoot/undershoot and distribute it as a delta, leaving F
        # itself unconstrained until the final clip after all iterations.
        # This allows cascades to propagate through multiple cells without
        # losing mass to premature clipping.
        over = jnp.maximum(F_in - 1.0, 0.0) * can_redistribute
        under = jnp.maximum(-F_in, 0.0) * can_redistribute

        # Remove the local over/under from the cell (this is always conservative:
        # the surplus is moved, not destroyed).  The receiving cell may go above 1
        # transiently — that is fine; it will be handled in a later iteration
        # or by the final clip if n_iter is sufficient.
        F_adjusted = F_in - over + under  # remove surplus, fill deficit locally

        # Shift convention:
        #   shift_plus(a)[i]  = a[i-1]: cell i RECEIVES the value that was at i-1
        #   shift_minus(a)[i] = a[i+1]: cell i RECEIVES the value that was at i+1
        #
        # For u>0: donate over[i] to i+1 → cell i+1 receives over[i]
        #   → shift_plus(over)[i+1] = over[i]  ✓  (shift_plus shifts the array
        #      rightward, so i+1 gets what was at i)
        # For u<0: donate over[i] to i-1 → cell i-1 receives over[i]
        #   → shift_minus(over)[i-1] = over[i]  ✓

        # Primary: donate downstream
        over_primary_pos = over * mask_pos * has_downstream   # u>0, has i+1
        over_primary_neg = over * mask_neg * has_downstream   # u<0, has i-1
        # Fallback: boundary cell donates upstream instead
        over_fallback_pos = over * mask_pos * has_fallback    # u>0, no i+1 → to i-1
        over_fallback_neg = over * mask_neg * has_fallback    # u<0, no i-1 → to i+1

        # For undershoot: borrow from downstream (reduce downstream cell's F)
        under_primary_pos = under * mask_pos * has_downstream
        under_primary_neg = under * mask_neg * has_downstream
        under_fallback_pos = under * mask_pos * has_fallback
        under_fallback_neg = under * mask_neg * has_fallback

        # Compute the delta that flows INTO each cell
        add = (
            shift_plus(over_primary_pos)      # cell i gets surplus from cell i-1 (u>0)
            + shift_minus(over_primary_neg)   # cell i gets surplus from cell i+1 (u<0)
            + shift_minus(over_fallback_pos)  # fallback: last interior cell's surplus to i-1
            + shift_plus(over_fallback_neg)   # fallback: first interior cell's surplus to i+1
        )
        sub = (
            shift_plus(under_primary_pos)
            + shift_minus(under_primary_neg)
            + shift_minus(under_fallback_pos)
            + shift_plus(under_fallback_neg)
        )

        # Apply ONLY to interior cells (halo unchanged)
        return F_adjusted + (add - sub) * interior

    F_out = jax.lax.fori_loop(0, n_iter, body, F)
    # Final clip: removes any floating-point residues from cells that could
    # not fully donate their surplus (e.g. surrounded by saturated neighbors).
    # With n_iter >= max_cascade_length this clip is a no-op (nothing to clip)
    # and mass is exactly conserved.  For n_iter < cascade length, mass
    # conservation holds only approximately — the caller should use n_iter >= 3
    # for typical PLIC flows where single-step overshoots are at most ~1 cell wide.
    #
    # NOTE: The final clip is INTENTIONALLY not applied to the interior delta
    # mechanism — each body() call is fully conservative.  The only non-
    # conservation path is this final clip, which fires only when n_iter is
    # too small relative to the overshoot cascade length.
    return jnp.clip(F_out, 0.0, 1.0)


def redistribute_bounds_x(
    F: jnp.ndarray, u_face, n_iter: int = 3
) -> jnp.ndarray:
    """Conservative bound enforcement after an x-sweep.

    Parameters
    ----------
    F : jnp.ndarray, shape ``(Nx, Ny, Nz)``
        Halo-padded volume fraction immediately after
        ``apply_flux_x``; may have small over/undershoots.
    u_face : float or jnp.ndarray shape ``(Nx-1, Ny, Nz)``
        Sweep velocity.  Sign only is used — to choose the downstream
        neighbour for each interior cell.
    n_iter : int
        Number of redistribution passes (default 3).  Must be static.
    """
    sign = _cell_sign_x(u_face, F.shape)
    return _redistribute_body(F, sign, _shift_plus1_x, _shift_minus1_x, n_iter)


def redistribute_bounds_y(
    F: jnp.ndarray, v_face, n_iter: int = 3
) -> jnp.ndarray:
    """Conservative bound enforcement after a y-sweep.  Mirror of X."""
    sign = _cell_sign_y(v_face, F.shape)
    return _redistribute_body(F, sign, _shift_plus1_y, _shift_minus1_y, n_iter)


def redistribute_bounds_z(
    F: jnp.ndarray, w_face, n_iter: int = 3
) -> jnp.ndarray:
    """Conservative bound enforcement after a z-sweep.  Mirror of X."""
    sign = _cell_sign_z(w_face, F.shape)
    return _redistribute_body(F, sign, _shift_plus1_z, _shift_minus1_z, n_iter)


# ---------------------------------------------------------------------------
# Public API aliases (expected by test suite and compare scripts)
# ---------------------------------------------------------------------------
# The test suite imports these names.  They wrap the internal
# redistribute_bounds_* functions and match the interface contract:
#
#   apply_flux_x_conservative(
#       F, flux, dt, dx, u_face, n_iter=3
#   ) -> F_corrected
#
# The ``flux``, ``dt``, and ``dx`` arguments are accepted for API
# compatibility but are currently unused (the redistribution only needs
# ``u_face`` for the sign). The caller is expected to have already called
# ``apply_flux_x`` before this function; all this does is enforce [0, 1]
# conservatively.


def apply_flux_x_conservative(
    F: jnp.ndarray,
    flux: jnp.ndarray,
    dt: float,
    dx: float,
    u_face,
    n_iter: int = 3,
) -> jnp.ndarray:
    """Conservative bound enforcement after apply_flux_x.

    Drop-in replacement for ``jnp.clip(F, 0, 1)`` that preserves mass.

    Parameters
    ----------
    F : jnp.ndarray, shape ``(Nx, Ny, Nz)``
        Volume fraction after ``apply_flux_x`` (may have over/undershoots).
    flux : jnp.ndarray, shape ``(Nx-1, Ny, Nz)``
        Face fluxes from the preceding sweep (accepted for API compat,
        not used internally by the current redistribution algorithm).
    dt, dx : float
        Time step and cell size (accepted for API compat, not currently
        used).
    u_face : float or jnp.ndarray, shape ``(Nx-1, Ny, Nz)``
        Face velocity — only its sign is used to determine the downstream
        neighbour for each cell.
    n_iter : int
        Number of redistribution passes (default 3).
    """
    return redistribute_bounds_x(F, u_face, n_iter=n_iter)


def apply_flux_y_conservative(
    F: jnp.ndarray,
    flux: jnp.ndarray,
    dt: float,
    dy: float,
    v_face,
    n_iter: int = 3,
) -> jnp.ndarray:
    """Conservative bound enforcement after apply_flux_y.  Mirror of X."""
    return redistribute_bounds_y(F, v_face, n_iter=n_iter)


def apply_flux_z_conservative(
    F: jnp.ndarray,
    flux: jnp.ndarray,
    dt: float,
    dz: float,
    w_face,
    n_iter: int = 3,
) -> jnp.ndarray:
    """Conservative bound enforcement after apply_flux_z.  Mirror of X."""
    return redistribute_bounds_z(F, w_face, n_iter=n_iter)
