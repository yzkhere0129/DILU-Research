"""Two-phase material property mixing from VOF field.

Density: arithmetic mean (consistent with momentum conservation)
    ρ(F) = F·ρ_L + (1-F)·ρ_G

Viscosity: arithmetic mean (simplest, works for moderate ratios)
    μ(F) = F·μ_L + (1-F)·μ_G

For extreme viscosity ratios (>100:1), harmonic mean is better,
but arithmetic is standard for CSF and simpler to implement.
"""

import jax.numpy as jnp
from ..data_types import Array


def compute_density(F: Array, rho_L: float, rho_G: float) -> Array:
    """Arithmetic mean density from VOF field.

    Args:
        F: volume fraction (0=gas, 1=liquid)
        rho_L: liquid density
        rho_G: gas density

    Returns:
        rho: density field (same shape as F)
    """
    return F * rho_L + (1.0 - F) * rho_G


def compute_viscosity(F: Array, mu_L: float, mu_G: float) -> Array:
    """Arithmetic mean dynamic viscosity from VOF field.

    Args:
        F: volume fraction
        mu_L: liquid viscosity
        mu_G: gas viscosity

    Returns:
        mu: viscosity field (same shape as F)
    """
    return F * mu_L + (1.0 - F) * mu_G


def compute_kinematic_viscosity(F: Array, rho: Array,
                                 mu_L: float, mu_G: float) -> Array:
    """Kinematic viscosity ν = μ/ρ for the diffusion operator."""
    mu = compute_viscosity(F, mu_L, mu_G)
    return mu / rho
