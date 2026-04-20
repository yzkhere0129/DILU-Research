"""Analytic plane-cube intersection volume (Scardovelli-Zaleski 2000).

Forward model :math:`V(d; \\mathbf{n}) = \\mathrm{Vol}\\{(x,y,z) \\in [0,1]^3 :
a x + b y + c z \\le C\\}` where :math:`(a,b,c) = (n_x \\Delta x, n_y \\Delta y,
n_z \\Delta z)` and ``C`` is the plane intercept in cell-centered physical
coordinates. This is the branchless 6+-branch inclusion-exclusion formula
from:

    Scardovelli & Zaleski, "Analytical relations connecting linear
    interfaces and volume fractions in rectangular grids",
    J. Comput. Phys. 164 (2000).

The implementation is shared with the Lagrangian VOF module
(``jax_laseram.vof.lagrangian_3d.reconstruction_3d._volume_below_3d``);
we re-export it here under a public-API name so that the Eulerian PLIC
research module has a single, documented entry point. Pinning tests live
in ``tests/plic_eulerian/test_primitives.py``.

Convention
----------
- ``C`` is a *physical* intercept relative to the cell center; it maps to
  the unit-cube parameter ``d = C + 0.5 * (|a| + |b| + |c|)`` internally.
- ``(a, b, c) = (n_x dx, n_y dy, n_z dz)``; signs may be arbitrary (the
  routine takes absolute values and uses the cell-center reflection
  symmetry of the unit cube).
- At ``C = -0.5 * (|a|+|b|+|c|)`` → ``V = 0`` (plane at cube corner);
  at ``C = +0.5 * (|a|+|b|+|c|)`` → ``V = 1`` (plane at opposite corner);
  at ``C = 0`` → ``V = 0.5`` (plane through cell center).

Shape polymorphism
------------------
All inputs (``C``, ``a``, ``b``, ``c``) are *elementwise*; they may be
Python scalars, 0-d / 1-d / 2-d / 3-d JAX arrays, as long as they
broadcast together. The function is fully ``jax.jit`` / ``jax.vmap`` /
``jax.grad``-safe: no Python branching on tracers, no data-dependent
shapes, no external calls.

Critical subtlety (do not change without running the pinning tests)
-------------------------------------------------------------------
Inside the 2D degenerate case (``n_nz == 2``), the implementation builds
a length-3 array ``jnp.array([a, b, c])`` and sorts along ``axis=0``.
That ``axis=0`` is load-bearing: it refers to the *coefficient axis*,
not the spatial axes. Under any number of ``vmap`` wraps or direct 3D
array calls the coefficient axis is always the leading one because
``jnp.array([...])`` stacks along axis 0. The Lagrangian reference file
carries a lengthy comment explaining this; see
``lagrangian_3d/reconstruction_3d.py`` lines ~181-189.
"""

from ..lagrangian_3d.reconstruction_3d import _volume_below_3d

__all__ = ["volume_below_plane_3d"]


def volume_below_plane_3d(C, a, b, c):
    """Volume of a unit cube below the plane :math:`ax + by + cz = C`.

    Thin public-API wrapper around the Lagrangian module's private
    implementation so the Eulerian PLIC code has a stable import point.
    If the Lagrangian implementation ever drifts, the pinning tests in
    ``tests/plic_eulerian/test_primitives.py`` will catch it.

    Parameters
    ----------
    C : scalar or array
        Plane intercept in *physical, cell-centered* coordinates. Valid
        range is :math:`[-0.5(|a|+|b|+|c|), +0.5(|a|+|b|+|c|)]`; values
        outside are clipped to ``0`` / ``1`` respectively.
    a, b, c : scalar or array
        Plane coefficients, typically ``(n_x dx, n_y dy, n_z dz)`` where
        ``n`` is a unit normal and ``d{x,y,z}`` are cell sizes. Signs
        arbitrary.

    Returns
    -------
    V : same shape as broadcast of inputs
        Fraction of the unit cube below the plane, clipped to [0, 1].

    Notes
    -----
    This is the *forward* model. Inverting it to find ``C`` given a
    target ``V = F`` (the VOF volume fraction) is the job of
    :mod:`jax_laseram.vof.plic.intercept_solver`.
    """
    return _volume_below_3d(C, a, b, c)
