"""Regression oracle test: exact (iter, rel_resid) pair on fixed synthetic matrices.

These tests pin the specific iter counts and residuals observed on this hardware
(RTX 3050 4 GB) using reproducible synthetic matrices. A third party on the
same GPU architecture with the same AMGx build should get bit-identical results.

The matrices are generated from fixed seeds — no file I/O required, so this
test works in any clone without the LPBF dump.

Oracle values were measured on 2026-05-17 (AMGx 2.5.0, JAX 0.5.x, CUDA 12.4):
  - smoke_8cubed  (7-point Laplacian, n=512):   iters=13, relres~9.4e-11
  - stiff_16cubed (contrast=100, n=4096):       iters=15
  - stiff_128cubed (contrast=100, n=2097152):   iters=15

Run: pytest dilu/amgx/tests/test_regression_oracle.py -v
"""
import conftest  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from dilu.amgx.python import Plan, CLASSICAL_V_CYCLE, with_tolerance
from dilu.amgx.tests._harness import laplacian_3d_7point, stiff_laplacian_3d


# ---------------------------------------------------------------------------
# Iter-count regression: small matrix (fast, always runs)
# ---------------------------------------------------------------------------

def test_iter_regression_8cubed_laplacian():
    """8^3 7-point Laplacian: iter count must be exactly 13 at tol=1e-8.

    Oracle source: measured on 2026-05-17, AMGx 2.5.0, RTX 3050, seed=0.
    Allowed drift: ±2 iters to tolerate minor AMGx version differences.
    If this drifts by more than 2, the AMG hierarchy or config changed.
    """
    nx = ny = nz = 8
    row_ptr, col_idx, values = laplacian_3d_7point(nx, ny, nz)
    n = nx * ny * nz
    rng = np.random.default_rng(0)
    b = jax.device_put(jnp.asarray(rng.standard_normal(n).astype(np.float64)))
    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    cfg = with_tolerance(CLASSICAL_V_CYCLE, tol=1e-8, max_iters=100)

    with Plan(rp, ci, vv, cfg) as plan:
        x, iters_d, status_d = plan.solve(b)
        x.block_until_ready()
        iters = int(iters_d[0])
        status = int(status_d[0])

    ORACLE_ITERS = 13
    DRIFT_BUDGET = 2
    assert status == 0, f"status {status} != 0 (SUCCESS)"
    assert abs(iters - ORACLE_ITERS) <= DRIFT_BUDGET, (
        f"iter count {iters} deviates from oracle {ORACLE_ITERS} "
        f"by more than {DRIFT_BUDGET}. "
        f"Expected {ORACLE_ITERS - DRIFT_BUDGET}..{ORACLE_ITERS + DRIFT_BUDGET}."
    )


def test_relresid_regression_8cubed_laplacian():
    """8^3 Laplacian: rel residual must be in [0, 1e-8] — i.e. actually converged.

    This enforces the CONVERGENCE criterion, not just status==0.
    Oracle: relres~9.4e-11 observed; threshold is the solver tol (1e-8).
    """
    nx = ny = nz = 8
    row_ptr, col_idx, values = laplacian_3d_7point(nx, ny, nz)
    n = nx * ny * nz
    rng = np.random.default_rng(0)
    b_np = rng.standard_normal(n).astype(np.float64)
    b = jax.device_put(jnp.asarray(b_np))
    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    cfg = with_tolerance(CLASSICAL_V_CYCLE, tol=1e-8, max_iters=100)

    with Plan(rp, ci, vv, cfg) as plan:
        x_d, iters_d, status_d = plan.solve(b)
        x_d.block_until_ready()
        status = int(status_d[0])

    assert status == 0
    x_np = np.asarray(x_d)
    # Compute actual residual from CSR (not from AMGx's internal residual norm)
    from dilu.amgx.tests._harness import csr_spmv_np
    r = b_np - csr_spmv_np(row_ptr, col_idx, values, x_np)
    rel = float(np.linalg.norm(r)) / max(float(np.linalg.norm(b_np)), 1e-300)
    assert rel <= 1e-8, (
        f"rel_resid {rel:.3e} > solver tol 1e-8. "
        f"Reported status=0 (SUCCESS) but residual is above tol."
    )


def test_iter_regression_16cubed_stiff():
    """16^3 stiff Laplacian (contrast=100): iter count oracle at tol=1e-10.

    Oracle: 15 iters measured 2026-05-17 (seed=42). Drift budget ±3.
    """
    nx = ny = nz = 16
    row_ptr, col_idx, values = stiff_laplacian_3d(nx, ny, nz, contrast=100.0)
    n = nx * ny * nz
    rng = np.random.default_rng(42)
    b_np = rng.standard_normal(n).astype(np.float64)
    b = jax.device_put(jnp.asarray(b_np))
    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    cfg = with_tolerance(CLASSICAL_V_CYCLE, tol=1e-10, max_iters=200)

    with Plan(rp, ci, vv, cfg) as plan:
        x_d, iters_d, status_d = plan.solve(b)
        x_d.block_until_ready()
        iters = int(iters_d[0])
        status = int(status_d[0])

    ORACLE_ITERS = 15
    DRIFT_BUDGET = 3
    assert status == 0, f"status={status}"
    assert abs(iters - ORACLE_ITERS) <= DRIFT_BUDGET, (
        f"iter count {iters} deviates from oracle {ORACLE_ITERS} "
        f"by more than {DRIFT_BUDGET}."
    )


@pytest.mark.slow
def test_iter_regression_128cubed_stiff():
    """128^3 stiff Laplacian: iter count oracle at tol=1e-10.

    Oracle: 15 iters measured 2026-05-17. Drift budget ±5.
    This is marked slow because matrix CONSTRUCTION takes ~18 s in Python.
    """
    nx = ny = nz = 128
    row_ptr, col_idx, values = stiff_laplacian_3d(nx, ny, nz, contrast=100.0)
    n = nx * ny * nz
    rng = np.random.default_rng(42)
    b = jax.device_put(jnp.asarray(rng.standard_normal(n).astype(np.float64)))
    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    cfg = with_tolerance(CLASSICAL_V_CYCLE, tol=1e-10, max_iters=200)

    with Plan(rp, ci, vv, cfg) as plan:
        x_d, iters_d, status_d = plan.solve(b)
        x_d.block_until_ready()
        iters = int(iters_d[0])
        status = int(status_d[0])

    ORACLE_ITERS = 15
    DRIFT_BUDGET = 5
    assert status == 0, f"status={status}"
    assert abs(iters - ORACLE_ITERS) <= DRIFT_BUDGET, (
        f"iter count {iters} deviates from oracle {ORACLE_ITERS} "
        f"by more than {DRIFT_BUDGET}. "
        f"This likely means the AMG hierarchy or convergence criterion changed."
    )
