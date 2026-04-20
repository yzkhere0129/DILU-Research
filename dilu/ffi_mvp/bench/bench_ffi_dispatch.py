"""Baseline FFI-dispatch overhead measurement.

Measures per-call wall-clock latency at n=1k, 10k, 100k for the MVP kernel
under jax.jit, with inputs pre-placed on device. This is the baseline Phase 2
cuSPARSE level-scheduling work must beat.
"""
from __future__ import annotations

import os
import sys
import time

# VRAM safety rails — must precede any jax import on this 4 GB laptop.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np

# Fold dilu/ffi_mvp into import path without install.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "tests"))

from jax import config as _cfg
_cfg.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from dilu.ffi_mvp.python import jacobi_residual


def build_tridiag(n):
    """Build 1-D Laplacian CSR directly — never materialize the dense (n,n).

    Required for n >= 10^5; the dense roundtrip via laplacian_1d hits host-RAM
    OOM for large n (74 GiB at n=1e5).
    """
    # nnz = 3n - 2 (interior rows have 3 nnz, boundary rows have 2).
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    row_ptr[0] = 0
    row_ptr[1] = 2  # row 0: [0, 1]
    for i in range(1, n - 1):
        row_ptr[i + 1] = row_ptr[i] + 3
    row_ptr[n] = row_ptr[n - 1] + 2  # last row: [n-2, n-1]

    nnz = int(row_ptr[n])
    col_idx = np.empty(nnz, dtype=np.int32)
    values = np.empty(nnz, dtype=np.float64)

    k = 0
    # Row 0
    col_idx[k] = 0; values[k] = 2.0; k += 1
    col_idx[k] = 1; values[k] = -1.0; k += 1
    # Rows 1..n-2
    for i in range(1, n - 1):
        col_idx[k] = i - 1; values[k] = -1.0; k += 1
        col_idx[k] = i;     values[k] =  2.0; k += 1
        col_idx[k] = i + 1; values[k] = -1.0; k += 1
    # Row n-1
    col_idx[k] = n - 2; values[k] = -1.0; k += 1
    col_idx[k] = n - 1; values[k] =  2.0; k += 1
    assert k == nnz

    diag = np.full(n, 2.0, dtype=np.float64)
    return (
        np.ascontiguousarray(row_ptr),
        np.ascontiguousarray(col_idx),
        np.ascontiguousarray(values),
    ), diag


def measure(n: int, n_warmup: int = 200, n_iter: int = 2000):
    (row_ptr_h, col_idx_h, values_h), diag_h = build_tridiag(n)
    rng = np.random.default_rng(0)
    b_h = rng.standard_normal(n).astype(np.float64)
    x_h = rng.standard_normal(n).astype(np.float64)

    row_ptr = jax.device_put(jnp.asarray(row_ptr_h, dtype=jnp.int32))
    col_idx = jax.device_put(jnp.asarray(col_idx_h, dtype=jnp.int32))
    values = jax.device_put(jnp.asarray(values_h, dtype=jnp.float64))
    diag = jax.device_put(jnp.asarray(1.0 / diag_h, dtype=jnp.float64))
    b = jax.device_put(jnp.asarray(b_h, dtype=jnp.float64))
    x = jax.device_put(jnp.asarray(x_h, dtype=jnp.float64))

    @jax.jit
    def f(row_ptr, col_idx, values, diag, b, x):
        return jacobi_residual(row_ptr, col_idx, values, diag, b, x)

    # Warm up jit cache and any lazy-registration paths.
    for _ in range(n_warmup):
        f(row_ptr, col_idx, values, diag, b, x).block_until_ready()

    samples = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        t0 = time.perf_counter()
        y = f(row_ptr, col_idx, values, diag, b, x)
        y.block_until_ready()
        samples[i] = time.perf_counter() - t0

    return {
        "n": n,
        "n_iter": n_iter,
        "median_us": float(np.median(samples) * 1e6),
        "p95_us": float(np.percentile(samples, 95) * 1e6),
        "p99_us": float(np.percentile(samples, 99) * 1e6),
        "min_us": float(np.min(samples) * 1e6),
    }


def main():
    results = []
    for n in (1000, 10000, 100000):
        r = measure(n)
        print(
            f"n={r['n']:>7}  iters={r['n_iter']}  "
            f"median={r['median_us']:8.2f} us  "
            f"p95={r['p95_us']:8.2f} us  p99={r['p99_us']:8.2f} us  "
            f"min={r['min_us']:8.2f} us"
        )
        results.append(r)
    return results


if __name__ == "__main__":
    main()
