"""Plan context manager: pairs analyze + release around a user's PCG loop.

Usage:
    with Plan(row_ptr, col_idx, values) as plan:
        d_star = plan.factor(values, diag_offset)
        for k in range(max_iter):
            z = plan.apply(d_star, r)
            ...
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .wrapper import (
    dilu_factor,
    cusparse_dilu_analyze,
    cusparse_dilu_apply,
    cusparse_dilu_release,
    build_diag_offset,
)


class Plan:
    """cuSPARSE DILU plan owning an opaque analyze-descriptor token.

    The pattern buffers `row_ptr` and `col_idx` MUST NOT be mutated or
    re-allocated for the plan's lifetime — the fingerprint the C++ side stores
    at analyze time is `(row_ptr_device_ptr, col_idx_device_ptr, n, nnz)`.
    Feeding `apply` different pattern buffers raises InvalidArgument.

    `values` is allowed to change (that's the whole point of `dilu_factor`
    per-step recomputation). If the pattern changes, create a new `Plan`.
    """

    def __init__(self, row_ptr, col_idx, values, diag_offset=None):
        row_ptr = jnp.asarray(row_ptr, dtype=jnp.int32)
        col_idx = jnp.asarray(col_idx, dtype=jnp.int32)
        values = jnp.asarray(values, dtype=jnp.float64)
        if diag_offset is None:
            diag_offset = build_diag_offset(np.asarray(row_ptr), np.asarray(col_idx))
        diag_offset = jnp.asarray(diag_offset, dtype=jnp.int32)

        # Keep strong refs so XLA never recycles the buffers while the plan is alive.
        self._row_ptr = row_ptr
        self._col_idx = col_idx
        self._diag_offset = diag_offset
        self._values = values  # only used as analyze seed; user updates per step

        self._token = cusparse_dilu_analyze(row_ptr, col_idx, values,
                                            diag_offset)
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

    @property
    def diag_offset(self):
        return self._diag_offset

    def factor(self, values):
        """Recompute D_* on the current `values` (pattern unchanged)."""
        return dilu_factor(self._row_ptr, self._col_idx, values, self._diag_offset)

    def apply(self, values, d_star, r):
        """Run the two-solve DILU preconditioner: z = M^{-1} r.

        Caller must ensure `values` matches the CSR pattern that was analyzed
        at __init__ — the plan owns that pattern and uses it regardless of
        what is passed here, so mismatch produces silently-wrong results.
        """
        return cusparse_dilu_apply(self._token, values, d_star, r)

    def release(self):
        if not self._released and self._token is not None:
            status = cusparse_dilu_release(self._token)
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
