"""High-level Plan context manager for the multicolor DILU preconditioner.

Orchestrates:
  - host-side coloring (`red_black_color` fast path or `greedy_color_csr`),
  - host-side CSR permutation,
  - FFI `multicolor_analyze` token creation,
  - `apply` / `refactor` / `release` lifecycle,
all behind a tiny surface matching Phase 2's `Plan` class.

Usage:
    with MulticolorPlan(row_ptr, col_idx, values,
                        grid_shape=(nx, ny, nz)) as plan:
        d_star = plan.factor(values)              # via refactor
        for k in range(max_iter):
            z = plan.apply(values, d_star, r)     # original-ordering I/O

`grid_shape` triggers the red-black fast path; omit it to fall back to greedy.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from .coloring import color_csr_auto
from .permute import permute_csr, build_diag_offset_permuted
from .wrapper import (
    multicolor_analyze,
    multicolor_apply,
    multicolor_release,
    _multicolor_refactor_with_n,
)


class MulticolorPlan:
    """Multi-color DILU plan.

    Lifecycle (arch §5.3):
        __init__   ->  host coloring + permute + FFI analyze + factor
        apply      ->  hot-path preconditioner
        refactor   ->  values-only update (returns new d_star in original order)
        release    ->  FFI release; idempotent
    """

    def __init__(
        self,
        row_ptr,
        col_idx,
        values,
        grid_shape: Optional[Tuple[int, int, int]] = None,
    ):
        row_ptr_h = np.ascontiguousarray(np.asarray(row_ptr), dtype=np.int32)
        col_idx_h = np.ascontiguousarray(np.asarray(col_idx), dtype=np.int32)
        values_h  = np.ascontiguousarray(np.asarray(values),  dtype=np.float64)

        self._N = int(row_ptr_h.shape[0]) - 1
        self._nnz = int(col_idx_h.shape[0])

        # 1. Host coloring (red-black or greedy) + validation.
        perm, iperm, colors, color_offsets = color_csr_auto(
            row_ptr_h, col_idx_h, grid_shape=grid_shape)
        self._n_colors = int(color_offsets.shape[0]) - 1
        self._perm_host = perm
        self._iperm_host = iperm
        self._color_offsets_host = color_offsets

        # 2. Host-side permutation.
        row_ptr_new, col_idx_new, values_new, nnz_map = permute_csr(
            row_ptr_h, col_idx_h, values_h, perm, iperm)
        diag_offset_new = build_diag_offset_permuted(row_ptr_new, col_idx_new)

        # 3. Stash strong refs to all host arrays (helps debugging + keeps GPU
        # strong-ref'd via the device_put below).
        self._row_ptr_h = row_ptr_h
        self._col_idx_h = col_idx_h
        self._nnz_map_host = nnz_map

        # 4. Push to device.
        def _dp(x, dt):
            return jax.device_put(jnp.asarray(x, dtype=dt))
        rp_tilde_d = _dp(row_ptr_new, jnp.int32)
        ci_tilde_d = _dp(col_idx_new, jnp.int32)
        vv_tilde_d = _dp(values_new, jnp.float64)
        do_tilde_d = _dp(diag_offset_new, jnp.int32)
        perm_d = _dp(perm, jnp.int32)
        iperm_d = _dp(iperm, jnp.int32)
        color_offsets_d = _dp(color_offsets, jnp.int32)
        nnz_map_d = _dp(nnz_map, jnp.int32)

        # Strong refs so the device buffers survive the plan's lifetime.
        self._rp_tilde_d = rp_tilde_d
        self._ci_tilde_d = ci_tilde_d
        self._vv_tilde_d = vv_tilde_d
        self._do_tilde_d = do_tilde_d
        self._perm_d = perm_d
        self._iperm_d = iperm_d
        self._color_offsets_d = color_offsets_d
        self._nnz_map_d = nnz_map_d

        # 5. FFI analyze.
        self._token = multicolor_analyze(
            rp_tilde_d, ci_tilde_d, vv_tilde_d, do_tilde_d,
            perm_d, iperm_d, color_offsets_d, nnz_map_d)
        self._token.block_until_ready()
        self._released = False

    # ---- Public accessors (introspection) ---------------------------------
    @property
    def token(self):
        return self._token

    @property
    def N(self) -> int:
        return self._N

    @property
    def nnz(self) -> int:
        return self._nnz

    @property
    def n_colors(self) -> int:
        return self._n_colors

    @property
    def color_offsets(self) -> np.ndarray:
        return self._color_offsets_host

    @property
    def perm(self) -> np.ndarray:
        return self._perm_host

    @property
    def iperm(self) -> np.ndarray:
        return self._iperm_host

    # ---- Hot path ---------------------------------------------------------
    def factor(self, values):
        """Alias for `refactor` — mirrors Phase 2's `Plan.factor` entry name."""
        return self.refactor(values)

    def refactor(self, values):
        """Recompute d_star from new values.

        Returns d_star in ORIGINAL ordering.
        """
        return _multicolor_refactor_with_n(self._token, values, self._N)

    def apply(self, values, d_star, r):
        """Apply M_mcDILU^{-1} to r. All arrays in ORIGINAL ordering.

        `values` must come from the same CSR pattern the plan was built on
        (the plan's `nnz_map` is used to gather into the permuted layout).
        """
        return multicolor_apply(self._token, values, d_star, r)

    # ---- Lifecycle --------------------------------------------------------
    def release(self):
        if not self._released and self._token is not None:
            status = multicolor_release(self._token)
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
