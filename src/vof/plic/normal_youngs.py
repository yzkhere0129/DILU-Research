"""Youngs / Parker-Young interface normal reconstruction (3D).

Computes the PLIC interface normal :math:`\\mathbf{n}` as the normalized
gradient of the volume-fraction field :math:`F`, using a 3x3x3
isotropic-friendly stencil. The normal convention points *from fluid
into empty*, i.e. :math:`\\mathbf{n} = -\\nabla F / \\|\\nabla F\\|`.

Stencil
-------
The gradient components use Parker-Young weighting on the cross-section:

.. code::

    w2d = [[1, 2, 1],
           [2, 4, 2],
           [1, 2, 1]] / 16

which is the discrete convolution of two 1-2-1 one-dimensional kernels
and is considerably more isotropic than plain Prewitt weights
``[1,1,1]``. The 3D kernel for :math:`\\partial F/\\partial x` is the
antisymmetric :math:`(-w2d, 0, +w2d)` stack along the x axis (and
similarly for y, z). Reference: Parker & Youngs (1992) and Barkhudarov
(2004) §4 "Least Squares method using a 27-point stencil".

Implementation note
-------------------
The shared Lagrangian VOF module already implements this kernel in
``jax_laseram.vof.lagrangian_3d.reconstruction_3d.compute_plic_normals_3d``
using ``jax.lax.conv_general_dilated`` with NCDHW layout and a
zero-border output. We re-export it here as a thin wrapper with a
cleaner shape contract for the Eulerian PLIC module: the input ``F``
and the output ``(nx, ny, nz)`` are all 3-D arrays ``(Nx, Ny, Nz)`` with
no channel dimension.

If Stage 4 of the PLIC research plan needs to upgrade to ELVIRA (6
candidate normals + argmin :math:`L^2` fit), we can replace this module
without touching the Lagrangian code.

Shape / halo contract
---------------------
- Input ``F`` must already have *at least one* halo cell on each side
  (the 3x3x3 stencil writes zero at the outermost layer).
- Output normals are zero in pure cells where
  :math:`F \\le 10^{-6}` or :math:`F \\ge 1-10^{-6}`, so the caller can
  freely multiply by an interface mask without double-zeroing.
- Normals are unit vectors where the interface mask is active,
  :math:`\\|\\mathbf{n}\\| = 1` to within float32 round-off.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..lagrangian_3d.reconstruction_3d import compute_plic_normals_3d

__all__ = ["compute_youngs_normal_3d"]


def compute_youngs_normal_3d(
    F: jnp.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the Youngs/Parker-Young PLIC normal for a 3-D VOF field.

    Parameters
    ----------
    F : jnp.ndarray, shape ``(Nx, Ny, Nz)``
        Volume-fraction field, values in ``[0, 1]`` (clipping is *not*
        enforced here — caller's responsibility). Must include at least
        one halo layer on every side so the 3x3x3 stencil fits.
    dx, dy, dz : float
        Cell sizes along the three coordinate axes.

    Returns
    -------
    nx, ny, nz : jnp.ndarray, each shape ``(Nx, Ny, Nz)``
        Components of the interface normal, pointing from fluid
        (``F=1``) toward empty (``F=0``). Zero in pure cells and at the
        outermost 1-cell ring (stencil validity boundary).

    Notes
    -----
    This is a thin wrapper around
    :func:`jax_laseram.vof.lagrangian_3d.reconstruction_3d.compute_plic_normals_3d`
    which uses the NCDHW convolution layout internally. The channel
    dimension is managed here so that Eulerian PLIC callers see only
    3-D arrays.
    """
    F_ch = F[..., None]  # (Nx, Ny, Nz, 1) — Lagrangian API expects a channel
    nx_ch, ny_ch, nz_ch = compute_plic_normals_3d(F_ch, dx, dy, dz)
    return nx_ch[..., 0], ny_ch[..., 0], nz_ch[..., 0]
