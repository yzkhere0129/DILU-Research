"""JAX-side wrappers for the four Phase 4 AMGx FFI primitives.

Design-doc §3 contracts:
    amgx_setup:          (row_ptr, col_idx, values; attr config_json) -> token[1]
    amgx_update_coefficients: (token, values) -> status[1]
    amgx_solve:          (token, b, x0) -> (x, iters[1], status[1])
    amgx_release:        (token) -> status[1]

All device-resident inputs/outputs. The config JSON is an FFI *attribute*
(host-side, read once during ffi_call lowering), NOT a traced tensor.
"""
from __future__ import annotations

import jax
import jax.ffi
import jax.numpy as jnp

from .registration import (
    register_once,
    SETUP_TARGET, UPDATE_TARGET, SOLVE_TARGET, RELEASE_TARGET,
)


def amgx_setup(row_ptr, col_idx, values, config_json: str):
    """Build an AMGx plan for the given CSR and config. Returns uint64[1] token.

    `config_json` is a valid AMGx config JSON (see dilu.amgx.python.config).
    The token outlives the Python call; use Plan or call amgx_release to
    free the underlying AMGx resources.
    """
    register_once()
    row_ptr = jnp.asarray(row_ptr, dtype=jnp.int32)
    col_idx = jnp.asarray(col_idx, dtype=jnp.int32)
    values = jnp.asarray(values, dtype=jnp.float64)

    out_type = jax.ShapeDtypeStruct((1,), jnp.uint64)
    call = jax.ffi.ffi_call(
        SETUP_TARGET, out_type, vmap_method="sequential")
    return call(row_ptr, col_idx, values, config_json=str(config_json))


def amgx_update_coefficients(token, values):
    """Refresh AMGx's matrix values + smoother state for an existing plan.

    Pattern (row_ptr, col_idx) is assumed unchanged. Returns int32[1] status
    (0 = OK, non-zero = AMGx return code).
    """
    register_once()
    token = jnp.asarray(token, dtype=jnp.uint64)
    values = jnp.asarray(values, dtype=jnp.float64)

    out_type = jax.ShapeDtypeStruct((1,), jnp.int32)
    call = jax.ffi.ffi_call(
        UPDATE_TARGET, out_type, vmap_method="sequential")
    return call(token, values)


def amgx_solve(token, b, x0):
    """Solve A x = b with AMGx's configured PCG-AMG. Returns (x, iters, status).

    `x0` is the initial guess (use jnp.zeros(n) for cold start).
    `iters` is int32[1]: the iteration count AMGx took.
    `status` is int32[1]: AMGX_SOLVE_STATUS (0=SUCCESS, 2=DIVERGED,
    3=NOT_CONVERGED, 1=FAILED).
    """
    register_once()
    token = jnp.asarray(token, dtype=jnp.uint64)
    b = jnp.asarray(b, dtype=jnp.float64)
    x0 = jnp.asarray(x0, dtype=jnp.float64)

    n = int(b.shape[0])
    out_types = (
        jax.ShapeDtypeStruct((n,), jnp.float64),  # x
        jax.ShapeDtypeStruct((1,), jnp.int32),    # iters
        jax.ShapeDtypeStruct((1,), jnp.int32),    # status
    )
    call = jax.ffi.ffi_call(
        SOLVE_TARGET, out_types, vmap_method="sequential")
    return call(token, b, x0)


def amgx_release(token):
    """Free the AMGx plan behind `token`. Returns int32[1] status.

    Idempotent: releasing an unknown token returns 0 rather than erroring.
    """
    register_once()
    token = jnp.asarray(token, dtype=jnp.uint64)
    out_type = jax.ShapeDtypeStruct((1,), jnp.int32)
    call = jax.ffi.ffi_call(
        RELEASE_TARGET, out_type, vmap_method="sequential")
    return call(token)
