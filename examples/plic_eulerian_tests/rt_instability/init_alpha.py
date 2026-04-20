"""Initial alpha.water (VOF) field for the RT air/helium benchmark.

Translates the reference OpenFOAM init script
``examples/RT_air_helium/initAlpha.py`` into a JAX / numpy function
suitable for the PLIC research module. The interface definition

    y_interface(x) = 2.0 + 0.05 * cos(2 * pi * x)

and the sub-grid 8x8 sampling scheme are preserved exactly; the only
substantive change is that the original's Python double ``for``-loop
(``initAlpha.py`` lines 28-40) is replaced by a vectorised JAX
computation to stay fast at 128 x 512 x 64 sub-samples (~4 M points).

Above the interface (``y >= y_interface``): ``alpha = 1`` (heavy / air)
Below the interface (``y <  y_interface``): ``alpha = 0`` (light / helium)
"""

from __future__ import annotations

import jax.numpy as jnp

from jax_laseram.grid import create_grid

from case_config import CASE


def build_grid():
    """Build the GridInfo for this case from CASE dict."""
    g = CASE["grid"]
    d = CASE["domain"]
    return create_grid(
        nx=g["nx"],
        ny=g["ny"],
        nz=g["nz"],
        x_range=d["x"],
        y_range=d["y"],
        z_range=d["z"],
        nh=g["nh"],
    )


def _apply_rt_bcs(F: jnp.ndarray, nh: int, has_z_halo: bool) -> jnp.ndarray:
    """Fill halo cells according to CASE['bcs'].

    - x_min / x_max : symmetry (copy from nearest interior cell)
    - y_min         : fixed_value = 0.0 (light)
    - y_max         : fixed_value = 1.0 (heavy)
    - z_min / z_max : symmetry (quasi-2D: z-homogeneous field)
    """
    # x symmetry
    F = F.at[:nh, :, :].set(F[nh : nh + 1, :, :])
    F = F.at[-nh:, :, :].set(F[-nh - 1 : -nh, :, :])
    # y fixed values
    F = F.at[:, :nh, :].set(0.0)     # bottom = light
    F = F.at[:, -nh:, :].set(1.0)    # top = heavy
    # z symmetry for quasi-2D layouts
    if has_z_halo:
        F = F.at[:, :, :nh].set(F[:, :, nh : nh + 1])
        F = F.at[:, :, -nh:].set(F[:, :, -nh - 1 : -nh])
    return F


def create_rt_vof(grid) -> jnp.ndarray:
    """Create the RT initial VOF field with 8x8 per-cell sub-sampling.

    Parameters
    ----------
    grid : jax_laseram.data_types.GridInfo

    Returns
    -------
    F : jnp.ndarray, shape ``(nx+2*nh, ny+2*nh, max(nz, 1))``
        Volume fraction (dtype follows JAX x64 setting).
    """
    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy

    n_sub = int(CASE["subsample"])                      # 8
    y0 = float(CASE["rt"]["interface_y0"])              # 2.0
    amp = float(CASE["rt"]["amplitude"])                # 0.05
    k = float(CASE["rt"]["wavenumber"])                 # 2 * pi

    # Interior cell centers — matches initAlpha.py:18-19.
    xc = jnp.linspace(dx / 2.0, 1.0 - dx / 2.0, nx)
    yc = jnp.linspace(dy / 2.0, 4.0 - dy / 2.0, ny)

    # Sub-sample offsets (si + 0.5) / n_sub relative to cell lower edge —
    # matches initAlpha.py:34-35 exactly.
    sub_offsets_x = (jnp.arange(n_sub) + 0.5) / n_sub  # shape (n_sub,)
    sub_offsets_y = (jnp.arange(n_sub) + 0.5) / n_sub

    # Absolute sub-sample coordinates: xs[i, si] = xc[i] - dx/2 + dx * off
    xs = xc[:, None] - dx / 2.0 + dx * sub_offsets_x[None, :]   # (nx, n_sub)
    ys = yc[:, None] - dy / 2.0 + dy * sub_offsets_y[None, :]   # (ny, n_sub)

    # Interface y-coordinate at every x sub-sample.
    y_interface_sub = y0 + amp * jnp.cos(k * xs)               # (nx, n_sub)

    # Compare each (i, si) column of interface against every (j, sj) sample.
    # Build a (nx, ny, n_sub, n_sub) boolean array, then average over the
    # sub-sample axes to recover the cell-averaged alpha.
    #
    # is_heavy[i, j, si, sj] = 1 if ys[j, sj] >= y_interface_sub[i, si]
    is_heavy = (
        ys[None, :, None, :]                                 # (1,  ny, 1, n_sub)
        >= y_interface_sub[:, None, :, None]                 # (nx, 1,  n_sub, 1)
    ).astype(jnp.float64)
    # Mean over the two sub-sample axes -> (nx, ny)
    alpha_interior = is_heavy.mean(axis=(2, 3))

    # Embed into the halo-padded array. Quasi-2D: replicate alpha over
    # all nz interior layers (plus z halos via _apply_rt_bcs).
    if nz > 1:
        nz_total = nz + 2 * nh
    else:
        nz_total = 1
    has_z_halo = nz > 1
    shape = (nx + 2 * nh, ny + 2 * nh, nz_total)
    F = jnp.zeros(shape)
    alpha_slab = jnp.broadcast_to(alpha_interior[:, :, None], (nx, ny, nz))
    z_slice = slice(nh, nh + nz) if has_z_halo else slice(0, 1)
    F = F.at[nh : nh + nx, nh : nh + ny, z_slice].set(alpha_slab)
    F = _apply_rt_bcs(F, nh, has_z_halo)
    return F
