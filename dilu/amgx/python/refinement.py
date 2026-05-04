"""Iterative refinement (IR) wrapper around AMGx for stagnant cases.

Wilkinson/Higham §12.1 style: solve A x_0 ≈ b → r = b - A x_0 (fp64) →
solve A δ = r → x ← x + δ. One step typically gains O(κ) factor in forward
error.

Used for cases where AMGx's PCG residual stagnates (cold-start step matrices
in LPBF: ddt term dominates with diag ~ 1/dt = O(1e12), AMG hierarchy
quality drops, residual reduction caps at ~1e-13 absolute regardless of tol).
"""
from __future__ import annotations

import time

import numpy as np

from jax import config as _jc
_jc.update("jax_enable_x64", True)   # AMGx FFI expects float64
import jax.numpy as jnp

from .plan import Plan
from .config import (
    CLASSICAL_V_DIAGSCALED, CLASSICAL_V_DIAGSCALED_BICGSTAB, with_tolerance,
)


def amgx_solve_with_refinement(
    A,                     # scipy.sparse.csr_matrix
    b: np.ndarray,
    x0: np.ndarray,
    eq_kind: str = "pd",   # "pd" → PCG, "T" → BiCGStab
    tol: float = 1e-12,
    max_iters: int = 500,
    n_refine: int = 2,     # IR steps after primary solve
    refine_tol: float = None,  # reserved; current impl always reuses `tol`
    verbose: bool = False,
) -> dict:
    """Solve A x = b with primary AMGx + n_refine IR passes.

    Returns dict with x, primary_iters, refine_iters, abs_residual,
    rel_residual, refinement_history (list of ‖r_k‖₂/‖b‖₂ per step), total_s.

    `refine_tol` is reserved for a future API where δ-solves use a different
    tolerance from the primary; current implementation reuses the primary
    Plan (which embeds `tol`) for all IR steps to avoid a second Plan setup.
    """
    cfg_base = (CLASSICAL_V_DIAGSCALED if eq_kind == "pd"
                else CLASSICAL_V_DIAGSCALED_BICGSTAB)
    cfg_primary = with_tolerance(cfg_base, tol, max_iters=max_iters)

    rp = jnp.asarray(A.indptr.astype(np.int32))
    ci = jnp.asarray(A.indices.astype(np.int32))
    vv = jnp.asarray(A.data.astype(np.float64))
    b_d = jnp.asarray(b)
    x0_d = jnp.asarray(x0)
    b_norm2 = float(np.linalg.norm(b))
    if b_norm2 == 0.0:
        b_norm2 = 1.0

    history = []
    refine_iters = []

    t0 = time.time()
    with Plan(rp, ci, vv, cfg_primary) as plan:
        x_jax, iters, status = plan.solve(b_d, x0_d)
        x_jax.block_until_ready()
        x = np.asarray(x_jax)
        primary_iters = int(iters[0])
        primary_status = int(status[0])
        primary_t = time.time() - t0

        r = A @ x - b
        r_norm = float(np.linalg.norm(r)) / b_norm2
        history.append(r_norm)
        if verbose:
            print(f"  primary: iters={primary_iters} status={primary_status} "
                  f"rel_resid={r_norm:.3e} t={primary_t:.3f}s")

        for k in range(n_refine):
            # r currently = A @ x - b ; flip sign for δ-solve RHS
            r_d = jnp.asarray(-r)
            zero = jnp.zeros_like(r_d)
            t1 = time.time()
            delta_jax, d_iters, _ = plan.solve(r_d, zero)
            delta_jax.block_until_ready()
            delta = np.asarray(delta_jax)
            refine_iters.append(int(d_iters[0]))
            x = x + delta
            r = A @ x - b
            r_norm = float(np.linalg.norm(r)) / b_norm2
            history.append(r_norm)
            if verbose:
                print(f"  IR {k+1}: δ-iters={int(d_iters[0])} "
                      f"rel_resid={r_norm:.3e} t={time.time()-t1:.3f}s")

    abs_residual = float(np.linalg.norm(r))   # final residual cached above
    return dict(
        x=x,
        primary_iters=primary_iters, primary_status=primary_status,
        refine_iters=refine_iters,
        abs_residual=abs_residual,
        rel_residual=abs_residual / max(b_norm2, 1e-300),
        refinement_history=history,
        total_s=time.time() - t0,
    )
