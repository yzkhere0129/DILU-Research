"""F2 acid test — Test C setup at 32^3 with AMG at fixed 15 iters.

The F2 signal from Phase 3 (design §6.4, carried into Phase 4 §6 C4):
    If `corr(|E_demean|, |∇log ρ|) > 0.5` at the 5-15 AMG iter window on the
    32^3 three-tier density setup, the preconditioner is producing a residual
    that traces the interface — an "interface halo". That is algorithmic
    death for AM-relevant problems.

This test DOES NOT attempt to tune config on failure. Per the Phase 4 brief
§4 fail rule: if corr > 0.5, HALT and write the failure note. Do NOT retry.

Also runs a reference solve at max_iters=200 to confirm AMG-PCG does converge
on this problem (eliminates "AMG just divergent" as a false alarm).
"""
import conftest  # noqa: F401

import os
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as sla
import jax
import jax.numpy as jnp

from dilu.amgx.python import Plan as AmgxPlan, CLASSICAL_V_CYCLE, with_tolerance
# Reuse Phase 2.5's problem builders directly — they are the gold spec.
from dilu.cusparse.tests.physical_benchmark import (
    _three_tier_density, build_variable_density_poisson_3d, csr_to_scipy,
    _cell_centres,
)

N_GRID = 32
H_GRID = 1.0 / N_GRID

FIXED_AMG_ITERS = 15


def _build_system_and_rhs(n, h):
    rho, cs, cl, cg = _three_tier_density(n, h)
    row_ptr, col_idx, values, diag_offset = build_variable_density_poisson_3d(
        rho, h, pin_cell=(0, 0, 0))

    # Same RHS as Phase 2.5's Test C: smooth laser-heating-like source,
    # mean-removed to match Neumann compatibility.
    xs = _cell_centres(n, h)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    src = (np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
           * np.exp(-((Z - 0.5) ** 2) / 0.05))
    src = src - src.mean()
    rhs = src.reshape(-1).copy()
    rhs[0] = 0.0
    return rho, row_ptr, col_idx, values, rhs


def _log_rho_grad_mag(rho, h):
    """|∇ log ρ| per cell (central differences, natural log)."""
    lr = np.log(rho)
    gx = np.zeros_like(lr)
    gy = np.zeros_like(lr)
    gz = np.zeros_like(lr)
    gx[1:-1, :, :] = (lr[2:, :, :] - lr[:-2, :, :]) / (2.0 * h)
    gx[0, :, :] = (lr[1, :, :] - lr[0, :, :]) / h
    gx[-1, :, :] = (lr[-1, :, :] - lr[-2, :, :]) / h
    gy[:, 1:-1, :] = (lr[:, 2:, :] - lr[:, :-2, :]) / (2.0 * h)
    gy[:, 0, :] = (lr[:, 1, :] - lr[:, 0, :]) / h
    gy[:, -1, :] = (lr[:, -1, :] - lr[:, -2, :]) / h
    gz[:, :, 1:-1] = (lr[:, :, 2:] - lr[:, :, :-2]) / (2.0 * h)
    gz[:, :, 0] = (lr[:, :, 1] - lr[:, :, 0]) / h
    gz[:, :, -1] = (lr[:, :, -1] - lr[:, :, -2]) / h
    return np.sqrt(gx * gx + gy * gy + gz * gz)


def test_f2_acid_32cubed():
    n = N_GRID
    h = H_GRID
    print(f"F2 acid: 32^3 three-tier density (gas/liquid/solid, "
          f"contrast 8000), AMG-PCG fixed {FIXED_AMG_ITERS} iters.")
    rho, row_ptr, col_idx, values, rhs = _build_system_and_rhs(n, h)
    n_total = n * n * n
    print(f"CSR: n={n_total}, nnz={int(values.size)}, "
          f"rho_min={rho.min():.2f}, rho_max={rho.max():.2f}, "
          f"contrast={rho.max()/rho.min():.2e}")

    # Gold standard: sparse LU direct solve.
    print("Running scipy spsolve (gold standard)...")
    A_scipy = csr_to_scipy(row_ptr, col_idx, values, n_total)
    p_exact = sla.spsolve(A_scipy, rhs, use_umfpack=False)
    p_exact = np.ascontiguousarray(p_exact, dtype=np.float64)
    res_exact = A_scipy.dot(p_exact) - rhs
    print(f"spsolve ||A p_exact - rhs||_inf = {np.max(np.abs(res_exact)):.3e}")

    # AMG-PCG at exactly `FIXED_AMG_ITERS` iterations. We achieve this by
    # setting max_iters=FIXED_AMG_ITERS and tolerance=0 (AMGx will iterate
    # until max_iters is hit).
    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    b_d = jax.device_put(jnp.asarray(rhs))

    cfg_fixed = with_tolerance(CLASSICAL_V_CYCLE, tol=0.0,
                               max_iters=FIXED_AMG_ITERS)

    with AmgxPlan(rp, ci, vv, cfg_fixed) as plan:
        p_amg_arr, iters_arr, status_arr = plan.solve(b_d)
        p_amg_arr.block_until_ready()
        p_amg = np.asarray(p_amg_arr)
        print(f"AMG-PCG fixed run: iters={int(iters_arr[0])}, "
              f"status={int(status_arr[0])} (expected 3=NOT_CONVERGED "
              "at this truncation)")

    E = p_amg - p_exact
    E_demean = E - np.mean(E)
    abs_E = np.abs(E_demean).reshape(n, n, n)
    log_rho_grad = _log_rho_grad_mag(rho, h)

    # Pearson correlation between |E_demean| and |∇ log ρ|.
    a = abs_E.ravel()
    b = log_rho_grad.ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    print(f"F2: max|E|={np.max(np.abs(E)):.3e}, "
          f"max|E_demean|={np.max(abs_E):.3e}, "
          f"corr(|E_demean|, |grad log rho|) = {corr:.4f}")

    # Also run a converged AMG-PCG to confirm the solver is HEALTHY on this
    # problem (F5 false alarm elimination).
    cfg_converged = with_tolerance(CLASSICAL_V_CYCLE, tol=1e-10, max_iters=200)
    with AmgxPlan(rp, ci, vv, cfg_converged) as plan2:
        p_c_arr, iters_c_arr, status_c_arr = plan2.solve(b_d)
        p_c_arr.block_until_ready()
        iters_c = int(iters_c_arr[0])
        status_c = int(status_c_arr[0])
    print(f"AMG-PCG converged run: iters={iters_c}, status={status_c} "
          f"(expected 0=SUCCESS)")
    assert status_c == 0, (
        f"AMG-PCG diverged on Test C at 32^3 (status={status_c}). "
        "This is an F5-class failure (halt, do not retune).")

    assert corr <= 0.5, (
        f"F2 TRIGGERED: corr(|E_demean|, |grad log rho|) = {corr:.4f} > 0.5. "
        "AMG produces an interface-traced residual. STOP per Phase 4 rule.")

    print(f"F2 acid PASS: corr={corr:.4f} <= 0.5")


if __name__ == "__main__":
    test_f2_acid_32cubed()
