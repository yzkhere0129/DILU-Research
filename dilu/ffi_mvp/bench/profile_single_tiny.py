"""Single tiny `jax.profiler.trace` run — produces the baseline FFI dispatch
overhead number the Phase 1 brief asks for. Record and move on; no tuning.

Methodology:
  - Pre-place a tridiag n=1000 on device.
  - Warm up jit + FFI registration (100 calls, ignored).
  - Inside a `jax.profiler.trace(...)` block: run 100 iterations, each
    followed by `.block_until_ready()` so the trace captures per-call wall time.
  - Report median / min of the wall-clock samples from the Python side
    (cheap, enough for a baseline). The trace tarball in `tmp_trace/` is the
    ground-truth evidence if anyone wants to open it in Perfetto / TensorBoard.

This is deliberately separate from bench_ffi_dispatch.py so the profiler run
stays small (100 iter, not 2000) and the VRAM footprint is trivial.
"""
from __future__ import annotations

import os
import sys
import time

# VRAM rails BEFORE jax import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from jax import config as _cfg
_cfg.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp

from dilu.ffi_mvp.python import jacobi_residual
from bench_ffi_dispatch import build_tridiag


def main():
    n = 1000
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

    # Warm up jit cache + FFI registration.
    for _ in range(100):
        f(row_ptr, col_idx, values, diag, b, x).block_until_ready()

    trace_dir = os.path.join(_THIS_DIR, "tmp_trace")
    os.makedirs(trace_dir, exist_ok=True)

    n_iter = 100
    samples = np.empty(n_iter, dtype=np.float64)
    with jax.profiler.trace(trace_dir):
        for i in range(n_iter):
            t0 = time.perf_counter()
            y = f(row_ptr, col_idx, values, diag, b, x)
            y.block_until_ready()
            samples[i] = time.perf_counter() - t0

    print(
        f"PROFILE n={n}  iters={n_iter}  "
        f"median={float(np.median(samples)) * 1e6:.2f} us  "
        f"min={float(np.min(samples)) * 1e6:.2f} us  "
        f"p95={float(np.percentile(samples, 95)) * 1e6:.2f} us"
    )
    print(f"Profiler trace written under {trace_dir}/")


if __name__ == "__main__":
    main()
