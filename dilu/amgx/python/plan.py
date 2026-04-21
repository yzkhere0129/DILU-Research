"""Plan context manager: pairs amgx_setup + amgx_release around a PCG loop.

Usage:
    from dilu.amgx.python import Plan, CLASSICAL_V_CYCLE

    with Plan(row_ptr, col_idx, values, config_json=CLASSICAL_V_CYCLE) as plan:
        # Solve once:
        x, iters, status = plan.solve(b, x0)
        # Or refresh values next timestep:
        plan.update_coefficients(new_values)
        x, iters, status = plan.solve(b_new, jnp.zeros_like(b_new))

The pattern (row_ptr, col_idx) must not change across calls on the same plan;
`values` may change via `update_coefficients`. The (N, nnz) fingerprint on the
C++ side guards against accidental shape drift.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .wrapper import (
    amgx_setup,
    amgx_update_coefficients,
    amgx_solve,
    amgx_release,
)


class Plan:
    """AMGx plan owning an opaque setup-time token.

    Strong-refs row_ptr / col_idx for the plan's lifetime so JAX's allocator
    does not recycle their device buffers while AMGx (potentially) holds
    pointers into them. AMGx actually copies these into its own device
    buffers at setup, so this guard is defensive — it matches the pattern
    used by Phase 2's cuSPARSE plan.
    """

    def __init__(self, row_ptr, col_idx, values, config_json: str):
        row_ptr = jnp.asarray(row_ptr, dtype=jnp.int32)
        col_idx = jnp.asarray(col_idx, dtype=jnp.int32)
        values = jnp.asarray(values, dtype=jnp.float64)

        self._row_ptr = row_ptr
        self._col_idx = col_idx
        self._values = values  # seed; may be superseded by update_coefficients
        self._config_json = str(config_json)

        self._token = amgx_setup(row_ptr, col_idx, values, self._config_json)
        # Force token materialization so any AMGx setup error surfaces here,
        # not in the first solve call.
        self._token.block_until_ready()
        self._released = False

    @property
    def token(self):
        return self._token

    @property
    def row_ptr(self):
        return self._row_ptr

    @property
    def col_idx(self):
        return self._col_idx

    def update_coefficients(self, new_values):
        """Replace A's values (pattern unchanged) and refresh smoother state."""
        new_values = jnp.asarray(new_values, dtype=jnp.float64)
        status = amgx_update_coefficients(self._token, new_values)
        status.block_until_ready()
        self._values = new_values
        return status

    def solve(self, b, x0=None):
        """Solve A x = b via AMGx's configured PCG. Returns (x, iters, status).

        `iters` is int32[1]: the number of outer PCG iterations. `status`
        is an int32[1] AMGX_SOLVE_STATUS (0=SUCCESS, 1=FAILED, 2=DIVERGED,
        3=NOT_CONVERGED).
        """
        b = jnp.asarray(b, dtype=jnp.float64)
        if x0 is None:
            x0 = jnp.zeros_like(b)
        else:
            x0 = jnp.asarray(x0, dtype=jnp.float64)
        return amgx_solve(self._token, b, x0)

    def release(self):
        if not self._released and self._token is not None:
            status = amgx_release(self._token)
            status.block_until_ready()
            self._released = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
