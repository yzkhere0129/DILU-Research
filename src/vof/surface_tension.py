"""CSF surface tension with Smoothed Color Function (SCF).

Three critical fixes for spurious current elimination:
1. Gaussian kernel smoothing (2 passes) before ALL gradient computations
2. Strict gradient-magnitude mask: |∇F̃| > ε/Δx (ε=0.01)
3. Use smoothed ∇F̃ for BOTH curvature AND force distribution
   (NOT raw ∇F — that creates singular delta-function jets)

Sign convention (F=1 liquid, F=0 gas):
  n = ∇F̃/|∇F̃|  → points INTO liquid (outward from bubble)
  κ = -∇·n      → negative for convex bubble (correct: inward pressure)
  f = σ·κ·∇F̃   → inward force compressing bubble ✓

Reference: Brackbill et al. (1992), Lafaurie et al. (1994).
"""

import jax.numpy as jnp
from ..data_types import GridInfo, Array


def _gaussian_smooth_2d(F: Array, nh: int, n_passes: int = 2) -> Array:
    """2D Gaussian smoothing: [1,2,1; 2,4,2; 1,2,1]/16 kernel."""
    for _ in range(n_passes):
        F = F.at[:nh,:,:].set(F[nh:nh+1,:,:])
        F = F.at[-nh:,:,:].set(F[-nh-1:-nh,:,:])
        F = F.at[:,:nh,:].set(F[:,nh:nh+1,:])
        F = F.at[:,-nh:,:].set(F[:,-nh-1:-nh,:])

        ic = slice(1, -1)
        C  = F[ic, ic, :]
        E  = F[2:, ic, :]; W  = F[:-2, ic, :]
        N  = F[ic, 2:, :]; S  = F[ic, :-2, :]
        NE = F[2:, 2:, :]; NW = F[:-2, 2:, :]
        SE = F[2:, :-2,:]; SW = F[:-2, :-2, :]
        F = F.at[ic, ic, :].set((4*C + 2*(E+W+N+S) + (NE+NW+SE+SW)) / 16.0)
    return F


def compute_surface_tension_force(F: Array, grid: GridInfo,
                                   sigma: float,
                                   smooth_passes: int = 2) -> tuple:
    """CSF force: f = σ·κ·∇F̃ with strict interface masking.

    ALL gradients computed from smoothed F̃ — no raw F derivatives anywhere.
    """
    nh = grid.nh
    dx, dy = grid.dx, grid.dy
    nx, ny = grid.nx, grid.ny
    ic = slice(nh, -nh)

    # ── Step 1: Smooth F ──
    F_s = _gaussian_smooth_2d(F.copy(), nh, n_passes=smooth_passes)

    # BCs on smoothed field
    F_s = F_s.at[:nh,:,:].set(F_s[nh:nh+1,:,:])
    F_s = F_s.at[-nh:,:,:].set(F_s[-nh-1:-nh,:,:])
    F_s = F_s.at[:,:nh,:].set(F_s[:,nh:nh+1,:])
    F_s = F_s.at[:,-nh:,:].set(F_s[:,-nh-1:-nh,:])

    # ── Step 2: ∇F̃ at cell centers (central differences on smoothed field) ──
    dFdx = (F_s[nh+1:nh+nx+1, ic, :] - F_s[nh-1:nh+nx-1, ic, :]) / (2*dx)
    dFdy = (F_s[ic, nh+1:nh+ny+1, :] - F_s[ic, nh-1:nh+ny-1, :]) / (2*dy)

    # |∇F̃|
    eps_reg = 1e-30
    grad_mag = jnp.sqrt(dFdx**2 + dFdy**2 + eps_reg)

    # ── Step 3: Unit normal n = ∇F̃/|∇F̃| ──
    nx_c = dFdx / grad_mag
    ny_c = dFdy / grad_mag

    # ── Step 4: Curvature κ = -∇·n ──
    nx_pad = jnp.pad(nx_c, ((1,1),(1,1),(0,0)), mode='edge')
    ny_pad = jnp.pad(ny_c, ((1,1),(1,1),(0,0)), mode='edge')

    kappa = -((nx_pad[2:,1:-1,:] - nx_pad[:-2,1:-1,:]) / (2*dx) +
              (ny_pad[1:-1,2:,:] - ny_pad[1:-1,:-2,:]) / (2*dy))

    # ── Step 5: HARD BOOLEAN MASK (F-band + gradient threshold) ──
    # F is now STRICTLY in [0,1]. Use both conditions:
    F_int = jnp.clip(F[ic, ic, :], 0.0, 1.0)

    # Condition 1: F must be in interface band (kills bulk phase noise)
    in_band = (F_int > 0.01) & (F_int < 0.99)

    # Condition 2: gradient must be significant (kills numerical dust)
    grad_threshold = 0.1 / max(dx, dy)
    has_grad = grad_mag > grad_threshold

    # Both conditions must be true
    mask = (in_band & has_grad).astype(F.dtype)

    # ── Step 6: CSF force = σ · κ · ∇F̃ · mask ──
    fx = sigma * kappa * dFdx * mask
    fy = sigma * kappa * dFdy * mask

    return fx, fy
