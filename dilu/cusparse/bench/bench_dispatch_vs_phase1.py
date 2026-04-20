"""Phase 2 dispatch-overhead bench.

Two measurements:
  (A) `cusparse_dilu_apply` median latency at n=1k, 10k — compare against
      Phase 1's 267 µs / n=1k baseline on the same 3050.
  (B) Analysis amortization: T_analyze vs T_apply. Solve for the break-even
      iteration count N* = T_analyze / T_apply. Interpretation: if a PCG
      runs more than N* iterations on one plan, analysis cost is negligible.

Outputs a small report block to stdout — copy into phase2_cusparse_report.md.

Hardware rails (from Phase 1): XLA prealloc OFF, mem-fraction 0.5, platform
allocator; x64 on before any JAX import.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "tests"))

from jax import config as _cfg
_cfg.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from dilu.cusparse.python import (
    Plan, build_diag_offset, cusparse_dilu_analyze, cusparse_dilu_release,
)


def build_tridiag_csr(n: int):
    """1-D Laplacian CSR direct (no dense roundtrip)."""
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    col_idx_list = []
    values_list = []
    for i in range(n):
        if i > 0:
            col_idx_list.append(i - 1); values_list.append(-1.0)
        col_idx_list.append(i); values_list.append(2.0)
        if i < n - 1:
            col_idx_list.append(i + 1); values_list.append(-1.0)
        row_ptr[i + 1] = len(col_idx_list)
    col_idx = np.asarray(col_idx_list, dtype=np.int32)
    values = np.asarray(values_list, dtype=np.float64)
    diag_offset = build_diag_offset(row_ptr, col_idx)
    return (np.ascontiguousarray(row_ptr), np.ascontiguousarray(col_idx),
            np.ascontiguousarray(values), np.ascontiguousarray(diag_offset))


def measure_apply(n: int, n_warmup=200, n_iter=2000):
    row_ptr, col_idx, values, diag_offset = build_tridiag_csr(n)
    rng = np.random.default_rng(0)
    r_h = rng.standard_normal(n).astype(np.float64)

    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    do = jax.device_put(jnp.asarray(diag_offset))
    r_d = jax.device_put(jnp.asarray(r_h))

    with Plan(rp, ci, vv, do) as plan:
        d_star = plan.factor(vv); d_star.block_until_ready()

        @jax.jit
        def f(values, d_star, r):
            return plan.apply(values, d_star, r)

        # Warmup
        for _ in range(n_warmup):
            f(vv, d_star, r_d).block_until_ready()

        samples = np.empty(n_iter, dtype=np.float64)
        for i in range(n_iter):
            t0 = time.perf_counter()
            y = f(vv, d_star, r_d)
            y.block_until_ready()
            samples[i] = time.perf_counter() - t0

        return {
            "n": n,
            "median_us": float(np.median(samples) * 1e6),
            "p95_us":    float(np.percentile(samples, 95) * 1e6),
            "p99_us":    float(np.percentile(samples, 99) * 1e6),
            "min_us":    float(np.min(samples) * 1e6),
        }


def measure_analyze(n: int, n_repeat=10):
    """Time a cold analyze call. Pattern is fixed; we analyze, release,
    analyze, release. Returns median of n_repeat."""
    row_ptr, col_idx, values, diag_offset = build_tridiag_csr(n)
    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    do = jax.device_put(jnp.asarray(diag_offset))

    # Warm up JAX tracing/compile cache for analyze and release.
    tok = cusparse_dilu_analyze(rp, ci, vv, do); tok.block_until_ready()
    cusparse_dilu_release(tok).block_until_ready()

    samples = np.empty(n_repeat, dtype=np.float64)
    for i in range(n_repeat):
        t0 = time.perf_counter()
        tok = cusparse_dilu_analyze(rp, ci, vv, do)
        tok.block_until_ready()
        samples[i] = time.perf_counter() - t0
        cusparse_dilu_release(tok).block_until_ready()
    return {
        "n": n,
        "median_us": float(np.median(samples) * 1e6),
        "min_us":    float(np.min(samples) * 1e6),
    }


def fit_analyze_amortization(n: int, Ks=(10, 50, 200)):
    """total_time(K) = T_analyze + K * T_apply. Fit both constants from timings."""
    row_ptr, col_idx, values, diag_offset = build_tridiag_csr(n)
    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    do = jax.device_put(jnp.asarray(diag_offset))
    rng = np.random.default_rng(0)
    r_d = jax.device_put(jnp.asarray(rng.standard_normal(n).astype(np.float64)))

    totals = []
    for K in Ks:
        # One plan per K to fairly include analyze cost. Median of 5.
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            plan = Plan(rp, ci, vv, do)
            d_star = plan.factor(vv); d_star.block_until_ready()
            for _ in range(K):
                plan.apply(vv, d_star, r_d).block_until_ready()
            t1 = time.perf_counter()
            plan.release()
            ts.append(t1 - t0)
        totals.append(float(np.median(ts)))

    # Linear fit: [T_analyze, T_apply] via least squares on (K, total).
    Ks_arr = np.asarray(Ks, dtype=np.float64)
    totals_arr = np.asarray(totals, dtype=np.float64)
    A = np.vstack([np.ones_like(Ks_arr), Ks_arr]).T
    coefs, *_ = np.linalg.lstsq(A, totals_arr, rcond=None)
    T_ana, T_app = float(coefs[0]), float(coefs[1])
    return {
        "n": n, "Ks": list(Ks), "totals_us": [t * 1e6 for t in totals],
        "T_analyze_us": T_ana * 1e6, "T_apply_us": T_app * 1e6,
        "break_even_K": (T_ana / T_app) if T_app > 0 else float("inf"),
    }


def main():
    print("=== Phase 2 apply dispatch latency ===")
    for n in (1000, 10000):
        r = measure_apply(n, n_warmup=100, n_iter=500)
        print(f"  apply n={r['n']:>6}  median={r['median_us']:8.2f} us  "
              f"p95={r['p95_us']:8.2f} us  min={r['min_us']:8.2f} us")

    print("=== Phase 2 analyze single-call latency ===")
    for n in (1000, 10000):
        r = measure_analyze(n, n_repeat=5)
        print(f"  analyze n={r['n']:>6}  median={r['median_us']:8.2f} us  "
              f"min={r['min_us']:8.2f} us")

    print("=== Analysis amortization fit (Ks=10,50,200) ===")
    for n in (1000, 10000):
        r = fit_analyze_amortization(n)
        print(f"  n={r['n']:>6}  T_analyze={r['T_analyze_us']:8.1f} us  "
              f"T_apply={r['T_apply_us']:8.2f} us  break-even K={r['break_even_K']:.1f}")


if __name__ == "__main__":
    main()
