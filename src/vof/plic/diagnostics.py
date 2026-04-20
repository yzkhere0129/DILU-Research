"""Per-step conservation and boundedness diagnostics for Eulerian PLIC.

Lightweight helpers that operate on the halo-padded VOF field ``F``
and return scalar metrics for logging / gate-checking. All functions
are pure-functional and cheap enough to call every time step.

The ``PLICStats`` NamedTuple bundles the diagnostics the research plan
calls out in Stage 2/3 gates:

* ``volume``        integrated :math:`\\int F\\,dV` over the interior
* ``f_min``         minimum of ``F`` on the interior (should stay >= 0)
* ``f_max``         maximum of ``F`` on the interior (should stay <= 1)
* ``n_below``       count of interior cells with ``F < -EPS``
* ``n_above``       count of interior cells with ``F > 1 + EPS``
* ``n_interface``   number of cells with ``EPS < F < 1 - EPS``
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


_EPS = 1.0e-6


class PLICStats(NamedTuple):
    volume: float
    f_min: float
    f_max: float
    n_below: int
    n_above: int
    n_interface: int


def interior_slice(F: jnp.ndarray, nh: int) -> jnp.ndarray:
    """Return the interior (halo-stripped) view of ``F``."""
    return F[nh:-nh, nh:-nh, nh:-nh] if nh > 0 else F


def compute_stats(F: jnp.ndarray, nh: int, cell_vol: float) -> PLICStats:
    """Evaluate :class:`PLICStats` on the interior of ``F``.

    Returns plain Python scalars rather than JAX arrays so the caller
    can feed the values into ``print`` / ``str.format`` without
    triggering extra traces.
    """
    interior = interior_slice(F, nh)
    volume = float(jnp.sum(interior) * cell_vol)
    f_min = float(jnp.min(interior))
    f_max = float(jnp.max(interior))
    n_below = int(jnp.sum(interior < -_EPS))
    n_above = int(jnp.sum(interior > 1.0 + _EPS))
    n_interface = int(jnp.sum((interior > _EPS) & (interior < 1.0 - _EPS)))
    return PLICStats(
        volume=volume,
        f_min=f_min,
        f_max=f_max,
        n_below=n_below,
        n_above=n_above,
        n_interface=n_interface,
    )


def format_stats(stats: PLICStats, v_ref: float | None = None) -> str:
    """Return a compact one-line string representation of ``stats``."""
    if v_ref is not None and v_ref != 0.0:
        drift = (stats.volume - v_ref) / v_ref
        vol_str = f"V={stats.volume:.6e} dV/V={drift:+.2e}"
    else:
        vol_str = f"V={stats.volume:.6e}"
    return (
        f"{vol_str} F=[{stats.f_min:+.3e}, {stats.f_max:+.3e}] "
        f"n_if={stats.n_interface} n_below={stats.n_below} "
        f"n_above={stats.n_above}"
    )
