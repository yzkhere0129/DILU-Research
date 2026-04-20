"""jit-friendly wrapper around the JacobiResidual FFI target.

Design-doc §2.2 contract:
    inputs:  row_ptr int32[n+1], col_idx int32[nnz], values f64[nnz],
             diag f64[n], b f64[n], x f64[n]
    output:  y f64[n]  where  y[i] = diag[i] * (b[i] - (A @ x)[i])

`diag` must be precomputed as 1 / A[i, i] (NOT the raw diagonal).
"""
from __future__ import annotations

import jax
import jax.ffi
import jax.numpy as jnp

from .registration import target_name, register_once


def jacobi_residual(row_ptr, col_idx, values, diag, b, x):
    """One sweep of Jacobi-preconditioned residual: y = D^{-1}(b - A x).

    All inputs must be device-resident JAX arrays. `diag[i] == 1 / A[i,i]`.
    """
    register_once()

    row_ptr = jnp.asarray(row_ptr, dtype=jnp.int32)
    col_idx = jnp.asarray(col_idx, dtype=jnp.int32)
    values = jnp.asarray(values, dtype=jnp.float64)
    diag = jnp.asarray(diag, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)

    n = b.shape[0]
    out_type = jax.ShapeDtypeStruct((n,), jnp.float64)

    call = jax.ffi.ffi_call(
        target_name(),
        out_type,
        vmap_method="sequential",
    )
    return call(row_ptr, col_idx, values, diag, b, x)


def assemble_diag_inv(row_ptr, col_idx, values, n):
    """Extract the CSR diagonal and return its reciprocal as float64.

    Used by tests to build the precomputed D^{-1} that the kernel expects.
    """
    row_ptr = jnp.asarray(row_ptr, dtype=jnp.int32)
    col_idx = jnp.asarray(col_idx, dtype=jnp.int32)
    values = jnp.asarray(values, dtype=jnp.float64)

    def diag_of_row(i):
        start = row_ptr[i]
        end = row_ptr[i + 1]
        # Mask: which CSR entries in this row hit the diagonal?
        k = jnp.arange(values.shape[0], dtype=jnp.int32)
        in_row = (k >= start) & (k < end)
        is_diag = in_row & (col_idx == i)
        return jnp.sum(jnp.where(is_diag, values, 0.0))

    diag = jax.vmap(diag_of_row)(jnp.arange(n, dtype=jnp.int32))
    return 1.0 / diag


def jacobi_residual_reference(row_ptr, col_idx, values, diag, b, x):
    """Pure-JAX reference: y = diag * (b - A @ x), via a dense SpMV per row.

    Tolerance-target for the T1/T2/T3 verification suite. Deterministic
    left-to-right summation to match the kernel's reduction order.
    """
    n = b.shape[0]
    row_ptr = jnp.asarray(row_ptr, dtype=jnp.int32)
    col_idx = jnp.asarray(col_idx, dtype=jnp.int32)
    values = jnp.asarray(values, dtype=jnp.float64)
    diag = jnp.asarray(diag, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)

    # SpMV via segment_sum — the segment for row i is
    # values[row_ptr[i]:row_ptr[i+1]]. We emit one product per nnz entry and
    # scatter-add into per-row bins.
    nnz = values.shape[0]
    k = jnp.arange(nnz, dtype=jnp.int32)
    # row_of_entry[k] = i such that row_ptr[i] <= k < row_ptr[i+1]
    row_of_entry = jnp.searchsorted(row_ptr[1:], k, side="right")
    prods = values * x[col_idx]
    ax = jax.ops.segment_sum(prods, row_of_entry, num_segments=n)
    return diag * (b - ax)
