"""JAX-side wrappers for the four Phase 3 multicolor FFI primitives.

Contracts mirror arch doc §3:
    multicolor_analyze(row_ptr_tilde, col_idx_tilde, values_tilde,
                        diag_offset_tilde, perm, iperm, color_offsets,
                        nnz_map) -> token[1] (uint64)
    multicolor_apply(token, values, d_star, r) -> z
    multicolor_refactor(token, values) -> d_star
    multicolor_release(token) -> status[1] (int32)

Caller-side semantics:
  - `analyze` consumes a pre-permuted CSR. The plan-owned buffers are a copy.
    The high-level `Plan` class in `plan.py` drives the permutation from the
    ORIGINAL (row_ptr, col_idx, values) + a coloring.
  - `apply` and `refactor` accept vectors/values in ORIGINAL ordering. The
    handlers permute internally via `perm`/`nnz_map` and inverse-permute
    outputs via `iperm`.
"""
from __future__ import annotations

import jax
import jax.ffi
import jax.numpy as jnp

from .registration import (
    register_once,
    ANALYZE_TARGET, APPLY_TARGET, REFACTOR_TARGET, RELEASE_TARGET,
)


def multicolor_analyze(
    row_ptr_tilde, col_idx_tilde, values_tilde, diag_offset_tilde,
    perm, iperm, color_offsets, nnz_map,
):
    """Register a plan for the pre-permuted CSR Ã and return the token.

    All inputs are expected to be device-resident JAX arrays.
    """
    register_once()
    row_ptr_tilde = jnp.asarray(row_ptr_tilde, dtype=jnp.int32)
    col_idx_tilde = jnp.asarray(col_idx_tilde, dtype=jnp.int32)
    values_tilde  = jnp.asarray(values_tilde,  dtype=jnp.float64)
    diag_offset_tilde = jnp.asarray(diag_offset_tilde, dtype=jnp.int32)
    perm          = jnp.asarray(perm,          dtype=jnp.int32)
    iperm         = jnp.asarray(iperm,         dtype=jnp.int32)
    color_offsets = jnp.asarray(color_offsets, dtype=jnp.int32)
    nnz_map       = jnp.asarray(nnz_map,       dtype=jnp.int32)

    out_type = jax.ShapeDtypeStruct((1,), jnp.uint64)
    call = jax.ffi.ffi_call(ANALYZE_TARGET, out_type, vmap_method="sequential")
    return call(row_ptr_tilde, col_idx_tilde, values_tilde, diag_offset_tilde,
                perm, iperm, color_offsets, nnz_map)


def multicolor_apply(token, values, d_star, r):
    """Hot-path apply: z = M_mcDILU^{-1} r.  I/O in ORIGINAL ordering."""
    register_once()
    token  = jnp.asarray(token,  dtype=jnp.uint64)
    values = jnp.asarray(values, dtype=jnp.float64)
    d_star = jnp.asarray(d_star, dtype=jnp.float64)
    r      = jnp.asarray(r,      dtype=jnp.float64)

    n = int(r.shape[0])
    out_type = jax.ShapeDtypeStruct((n,), jnp.float64)
    call = jax.ffi.ffi_call(APPLY_TARGET, out_type, vmap_method="sequential")
    return call(token, values, d_star, r)


def multicolor_refactor(token, values):
    """Values-only update: compute new d_star (original ordering) from new values."""
    register_once()
    token  = jnp.asarray(token,  dtype=jnp.uint64)
    values = jnp.asarray(values, dtype=jnp.float64)

    # We need N; infer from the token's partner? No — the C++ side has it.
    # But JAX ffi_call needs a static output shape. Solution: keep N in the
    # Plan class and pass it through. For the pure wrapper, accept N as kwarg.
    raise NotImplementedError(
        "multicolor_refactor needs N; use Plan.refactor() instead of this "
        "raw wrapper. See plan.py."
    )


def _multicolor_refactor_with_n(token, values, n: int):
    """Internal variant that knows `n` and can declare the output shape."""
    register_once()
    token  = jnp.asarray(token,  dtype=jnp.uint64)
    values = jnp.asarray(values, dtype=jnp.float64)

    out_type = jax.ShapeDtypeStruct((n,), jnp.float64)
    call = jax.ffi.ffi_call(REFACTOR_TARGET, out_type, vmap_method="sequential")
    return call(token, values)


def multicolor_release(token):
    """Free the plan behind `token`. Idempotent (status == 0 if already gone)."""
    register_once()
    token = jnp.asarray(token, dtype=jnp.uint64)
    out_type = jax.ShapeDtypeStruct((1,), jnp.int32)
    call = jax.ffi.ffi_call(RELEASE_TARGET, out_type, vmap_method="sequential")
    return call(token)
