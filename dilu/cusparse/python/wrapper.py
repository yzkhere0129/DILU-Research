"""JAX-side wrappers for the four Phase 2 FFI primitives.

Design-doc §4 contracts:
    dilu_factor:              (row_ptr, col_idx, values, diag_offset) -> d_star
    cusparse_dilu_analyze:    (row_ptr, col_idx, values) -> token[1] (uint64)
    cusparse_dilu_apply:      (token, row_ptr, col_idx, values, d_star,
                               diag_offset, r) -> z
    cusparse_dilu_release:    (token) -> status[1] (int32)

All inputs/outputs are device-resident JAX arrays. `diag_offset[i]` is the
index into `col_idx` / `values` of A's (i, i) entry; the caller precomputes it
on host via `build_diag_offset` below.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.ffi
import jax.numpy as jnp

from .registration import (
    register_once,
    FACTOR_TARGET, ANALYZE_TARGET, APPLY_TARGET, RELEASE_TARGET,
)


def build_diag_offset(row_ptr, col_idx) -> np.ndarray:
    """Compute diag_offset[i] = index in col_idx of the (i,i) entry.

    Host-side; expected to be called once when the CSR is assembled.
    Raises ValueError if any row lacks a diagonal entry (DILU undefined there).
    """
    row_ptr = np.ascontiguousarray(np.asarray(row_ptr, dtype=np.int32))
    col_idx = np.ascontiguousarray(np.asarray(col_idx, dtype=np.int32))
    n = int(row_ptr.shape[0]) - 1
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        rs = int(row_ptr[i])
        re = int(row_ptr[i + 1])
        # col_idx within a row is ascending (cuSPARSE requirement); use searchsorted.
        k = rs + int(np.searchsorted(col_idx[rs:re], i))
        if k >= re or int(col_idx[k]) != i:
            raise ValueError(
                f"build_diag_offset: row {i} has no diagonal entry"
            )
        out[i] = k
    return np.ascontiguousarray(out)


def dilu_factor(row_ptr, col_idx, values, diag_offset):
    """Compute D_* for DILU factorization. Serial-on-device recurrence."""
    register_once()
    row_ptr = jnp.asarray(row_ptr, dtype=jnp.int32)
    col_idx = jnp.asarray(col_idx, dtype=jnp.int32)
    values = jnp.asarray(values, dtype=jnp.float64)
    diag_offset = jnp.asarray(diag_offset, dtype=jnp.int32)

    n = int(row_ptr.shape[0]) - 1
    out_type = jax.ShapeDtypeStruct((n,), jnp.float64)
    call = jax.ffi.ffi_call(FACTOR_TARGET, out_type, vmap_method="sequential")
    return call(row_ptr, col_idx, values, diag_offset)


def cusparse_dilu_analyze(row_ptr, col_idx, values, diag_offset):
    """Run cuSPARSE analysis; return an opaque uint64[1] token.

    The analyze handler copies row_ptr/col_idx/diag_offset into plan-owned
    buffers so the cuSPARSE descriptors survive XLA buffer shuffles under jit.
    """
    register_once()
    row_ptr = jnp.asarray(row_ptr, dtype=jnp.int32)
    col_idx = jnp.asarray(col_idx, dtype=jnp.int32)
    values = jnp.asarray(values, dtype=jnp.float64)
    diag_offset = jnp.asarray(diag_offset, dtype=jnp.int32)

    out_type = jax.ShapeDtypeStruct((1,), jnp.uint64)
    call = jax.ffi.ffi_call(ANALYZE_TARGET, out_type, vmap_method="sequential")
    return call(row_ptr, col_idx, values, diag_offset)


def cusparse_dilu_apply(token, values, d_star, r):
    """Hot-path apply: z = M_DILU^{-1} r, reusing the plan behind `token`.

    Pattern (row_ptr/col_idx/diag_offset) is NOT re-supplied — the plan owns
    those copies and uses them consistently.  `values` is expected to match
    the CSR pattern that was analyzed; passing mismatched-pattern values
    produces silently-wrong results (no runtime check because a content-hash
    every call would cost a GPU reduction — user-facing risk documented in
    Plan.apply).
    """
    register_once()
    token = jnp.asarray(token, dtype=jnp.uint64)
    values = jnp.asarray(values, dtype=jnp.float64)
    d_star = jnp.asarray(d_star, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)

    n = int(r.shape[0])
    out_type = jax.ShapeDtypeStruct((n,), jnp.float64)
    call = jax.ffi.ffi_call(APPLY_TARGET, out_type, vmap_method="sequential")
    return call(token, values, d_star, r)


def cusparse_dilu_release(token):
    """Free the plan associated with `token`. Returns int32[1] status."""
    register_once()
    token = jnp.asarray(token, dtype=jnp.uint64)
    out_type = jax.ShapeDtypeStruct((1,), jnp.int32)
    call = jax.ffi.ffi_call(RELEASE_TARGET, out_type, vmap_method="sequential")
    return call(token)
