"""Headline bench — Phase 2 cuSPARSE DILU-PCG vs Phase 3 multicolor DILU-PCG.

Generates the table the brief (§10 Deliverable) asks for. Measures:
  - T_apply_median  (median over 1000 warmed-up calls)
  - T_apply_p95
  - iter_count (to tol=1e-8)
  - T_total_pcg = T_apply_median * iter_count (rough surrogate)

On three matrix sizes:
  1. n = 1000    (1-D tridiag)     — dispatch-dominated regime
  2. 16³ = 4096  (T7 stiff)        — Phase 2 T7 comparison point
  3. 32³ = 32768 (Test C 3-tier)   — physical-bench scale

This file is NOT a pytest; it is a bench driver. Run as:
    python bench_multicolor_vs_cusparse.py
Writes a markdown fragment to stdout that the final report copy-pastes in.
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TESTS_DIR = os.path.join(_REPO_ROOT, "dilu", "multicolor", "tests")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import time
import numpy as np
import jax
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dilu.cusparse.python import Plan as CusparsePlan, build_diag_offset
from dilu.multicolor.python import MulticolorPlan
from _harness import (  # noqa: E402
    laplacian_3d_7point, stiff_laplacian_3d,
)


def _tridiag_csr(n):
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    col_idx = []
    vals = []
    for i in range(n):
        if i > 0:
            col_idx.append(i - 1); vals.append(-1.0)
        col_idx.append(i); vals.append(2.0)
        if i < n - 1:
            col_idx.append(i + 1); vals.append(-1.0)
        row_ptr[i + 1] = len(col_idx)
    return (np.ascontiguousarray(row_ptr),
            np.ascontiguousarray(np.asarray(col_idx, dtype=np.int32)),
            np.ascontiguousarray(np.asarray(vals, dtype=np.float64)))


def _three_tier_density_csr(n, h):
    """Same 3-tier density stack as physical_benchmark_phase3.py Test C, but
    built here to keep this file self-contained.
    """
    xs = (np.arange(n) + 0.5) * h
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    eps = 1.5 * h
    z_sl = 0.40 - 0.10 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.12 ** 2))
    z_lg = 0.60
    cs = 0.5 * (1.0 - np.tanh((Z - z_sl) / eps))
    cg = 0.5 * (1.0 + np.tanh((Z - z_lg) / eps))
    cl = np.clip(1.0 - cs - cg, 0.0, 1.0)
    rho = 8000.0 * cs + 1000.0 * cl + 1.0 * cg
    rho = np.maximum(rho, 1.0)

    def idx(i, j, k):
        return (k * n + j) * n + i

    row_ptr = [0]; col_idx = []; vals = []; diag_off = []
    h2 = h * h
    fx = 2.0 / (rho[:-1, :, :] + rho[1:, :, :])
    fy = 2.0 / (rho[:, :-1, :] + rho[:, 1:, :])
    fz = 2.0 / (rho[:, :, :-1] + rho[:, :, 1:])
    pin = idx(0, 0, 0)
    for k in range(n):
        for j in range(n):
            for i in range(n):
                p = idx(i, j, k)
                if p == pin:
                    col_idx.append(p); vals.append(1.0)
                    diag_off.append(len(vals) - 1)
                    row_ptr.append(len(col_idx)); continue
                entries = []; dv = 0.0
                if i > 0:
                    b = fx[i-1, j, k] / h2; q = idx(i-1, j, k)
                    if q != pin: entries.append((q, -b))
                    dv += b
                if i < n-1:
                    b = fx[i, j, k] / h2; q = idx(i+1, j, k)
                    if q != pin: entries.append((q, -b))
                    dv += b
                if j > 0:
                    b = fy[i, j-1, k] / h2; q = idx(i, j-1, k)
                    if q != pin: entries.append((q, -b))
                    dv += b
                if j < n-1:
                    b = fy[i, j, k] / h2; q = idx(i, j+1, k)
                    if q != pin: entries.append((q, -b))
                    dv += b
                if k > 0:
                    b = fz[i, j, k-1] / h2; q = idx(i, j, k-1)
                    if q != pin: entries.append((q, -b))
                    dv += b
                if k < n-1:
                    b = fz[i, j, k] / h2; q = idx(i, j, k+1)
                    if q != pin: entries.append((q, -b))
                    dv += b
                entries.append((p, dv))
                entries.sort(key=lambda e: e[0])
                d_off = None; base = len(vals)
                for off, (c, v) in enumerate(entries):
                    col_idx.append(c); vals.append(v)
                    if c == p: d_off = base + off
                diag_off.append(d_off)
                row_ptr.append(len(col_idx))
    return (np.ascontiguousarray(np.asarray(row_ptr, dtype=np.int32)),
            np.ascontiguousarray(np.asarray(col_idx, dtype=np.int32)),
            np.ascontiguousarray(np.asarray(vals, dtype=np.float64)))


def _spmv_gpu(row_ptr, col_idx, values, x):
    n = x.shape[0]
    nnz = values.shape[0]
    k = jnp.arange(nnz, dtype=jnp.int32)
    row_of = jnp.searchsorted(row_ptr[1:], k, side="right")
    prods = values * x[col_idx]
    return jax.ops.segment_sum(prods, row_of, num_segments=n)


def _pcg(apply_M_inv, spmv, b, tol=1e-8, max_iter=2000):
    n = b.shape[0]
    x = jnp.zeros(n, dtype=jnp.float64)
    r = b - spmv(x); r.block_until_ready()
    b_norm = float(jnp.linalg.norm(b))
    z = apply_M_inv(r); p = z
    rz = float(jnp.dot(r, z))
    for it in range(max_iter):
        rnorm = float(jnp.linalg.norm(r))
        if rnorm / (b_norm + 1e-300) < tol:
            return x, it
        Ap = spmv(p)
        alpha = rz / float(jnp.dot(p, Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        z = apply_M_inv(r)
        rz_new = float(jnp.dot(r, z))
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    return x, max_iter


def _time_apply(apply_fn, r_d, warmup=20, reps=500):
    """Median / p95 of apply() wall time in microseconds.

    `apply_fn(r)` returns a JAX array; we block on it to measure real dispatch
    + execution time. Sub-millisecond precision via time.perf_counter.
    """
    # Warm up.
    for _ in range(warmup):
        z = apply_fn(r_d); z.block_until_ready()

    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        z = apply_fn(r_d)
        z.block_until_ready()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    median = ts[reps // 2]
    p95 = ts[int(reps * 0.95)]
    return median, p95


def bench_one(name, row_ptr, col_idx, values, b_h, grid_shape=None):
    n = int(row_ptr.shape[0]) - 1
    nnz = int(col_idx.shape[0])

    diag_offset = build_diag_offset(row_ptr, col_idx)
    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    do = jax.device_put(jnp.asarray(diag_offset))
    b_d = jax.device_put(jnp.asarray(b_h))

    def spmv(x):
        return _spmv_gpu(rp, ci, vv, x)

    # -------------------- Phase 2 (cuSPARSE) --------------------
    with CusparsePlan(rp, ci, vv, do) as plan2:
        d_star2 = plan2.factor(vv); d_star2.block_until_ready()
        def apply2(r):
            return plan2.apply(vv, d_star2, r)
        med2, p95_2 = _time_apply(apply2, b_d)
        _, it2 = _pcg(apply2, spmv, b_d, tol=1e-8, max_iter=2000)

    # -------------------- Phase 3 (multicolor) ------------------
    with MulticolorPlan(row_ptr, col_idx, values,
                        grid_shape=grid_shape) as plan3:
        n_colors = plan3.n_colors
        d_star3 = plan3.factor(vv); d_star3.block_until_ready()
        def apply3(r):
            return plan3.apply(vv, d_star3, r)
        med3, p95_3 = _time_apply(apply3, b_d)
        _, it3 = _pcg(apply3, spmv, b_d, tol=1e-8, max_iter=2000)

    total2 = med2 * it2  # µs
    total3 = med3 * it3  # µs
    ratio_apply = med3 / med2 if med2 > 0 else float("nan")
    ratio_total = total3 / total2 if total2 > 0 else float("nan")
    return {
        "name": name,
        "n": n,
        "nnz": nnz,
        "n_colors": n_colors,
        "P2_apply_med_us": med2,
        "P2_apply_p95_us": p95_2,
        "P2_iters": it2,
        "P2_total_us": total2,
        "P3_apply_med_us": med3,
        "P3_apply_p95_us": p95_3,
        "P3_iters": it3,
        "P3_total_us": total3,
        "ratio_apply": ratio_apply,
        "ratio_total": ratio_total,
    }


def main():
    results = []

    # 1. tridiag n=1000 (no grid shape — greedy fallback, but n=1000 bipartite
    #    chain still ends up 2-color).
    rp, ci, vv = _tridiag_csr(1000)
    rng = np.random.default_rng(0)
    b = rng.standard_normal(1000).astype(np.float64)
    results.append(bench_one("n=1000 tridiag", rp, ci, vv, b, grid_shape=None))

    # 2. 16^3 stiff (Phase 2 T7 matrix).
    rp, ci, vv = stiff_laplacian_3d(16, 16, 16, contrast=100.0)
    b = np.random.default_rng(0).standard_normal(4096).astype(np.float64)
    results.append(bench_one("16^3 stiff", rp, ci, vv, b, grid_shape=(16, 16, 16)))

    # 3. 32^3 three-tier (Test C matrix — the physical-bench scale).
    rp, ci, vv = _three_tier_density_csr(32, 1.0 / 32)
    b = np.random.default_rng(0).standard_normal(32768).astype(np.float64)
    results.append(bench_one("32^3 3-tier", rp, ci, vv, b, grid_shape=(32, 32, 32)))

    print()
    print("| Matrix | N | nnz | colors | P2 apply median/p95 (µs) | P2 iters "
          "| P3 apply median/p95 (µs) | P3 iters | apply speedup | total speedup |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['name']} | {r['n']} | {r['nnz']} | {r['n_colors']} | "
              f"{r['P2_apply_med_us']:.1f} / {r['P2_apply_p95_us']:.1f} | "
              f"{r['P2_iters']} | "
              f"{r['P3_apply_med_us']:.1f} / {r['P3_apply_p95_us']:.1f} | "
              f"{r['P3_iters']} | "
              f"{1.0/r['ratio_apply']:.2f}x | "
              f"{1.0/r['ratio_total']:.2f}x |")
    print()

    # JSON-ish raw dump for later reference.
    print("# raw results (microseconds):")
    for r in results:
        print(f"#  {r}")


if __name__ == "__main__":
    main()
