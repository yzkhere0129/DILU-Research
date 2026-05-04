"""PBiCG outer solver — direct port of PBiCG.C:69-253.

See audit §4.2 (main loop), §10.4 (convergence), §10.8 (init), §10.9 (singular),
§10.12 (preconditioner one-shot per solve), §10.18 (minIter).

Convergence test (lduMatrixSolver.C):
    converged = (r_final < tol) OR (relTol > small_ AND r_final < relTol * r_initial)
where small_ = 1e-20.

Singular test (PBiCG.C):
    if |wApT| / normFactor < VSMALL=1e-300: break with singular flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ldu import LDU, calc_losort
from .spmv import amul, tmul
from .dilu import calc_reciprocal_d, precondition, precondition_t
from .normfactor import norm_factor, NORM_FACTOR_SMALL
from .reductions import sum_abs as _lf_sum_abs, dot as _lf_dot

V_SMALL = 1e-300  # OpenFOAM scalar.H — used in singular check


@dataclass
class SolverPerformance:
    initial_residual: float
    final_residual: float
    n_iterations: int
    converged: bool
    singular: bool
    x: np.ndarray


def _converged(r_final: float, r_initial: float,
               tolerance: float, rel_tol: float) -> bool:
    if r_final < tolerance:
        return True
    if rel_tol > NORM_FACTOR_SMALL and r_final < rel_tol * r_initial:
        return True
    return False


def solve(ldu: LDU, b: np.ndarray, x0: np.ndarray,
          tolerance: float = 1e-12, rel_tol: float = 0.0,
          min_iter: int = 0, max_iter: int = 1000,
          sumA_override: Optional[np.ndarray] = None,
          nf_override: Optional[float] = None) -> SolverPerformance:
    """Solve A x = b with PBiCG-DILU starting from x0.

    Returns SolverPerformance with x in-place equivalent to OpenFOAM's
    psi after solve() returns.
    """
    psi = x0.astype(np.float64, copy=True)
    source = b.astype(np.float64, copy=False)

    # === DILU setup once per solve (audit §10.12) ===
    rD = calc_reciprocal_d(ldu)
    losort = calc_losort(ldu.neighbour)

    # === Initial residual (PBiCG.C:91-100) ===
    A_psi = amul(ldu, psi)
    rA = source - A_psi
    if nf_override is not None:
        nF = float(nf_override)
    else:
        nF = norm_factor(ldu, psi, source, A_psi, sumA_override=sumA_override)
    initial_residual = _lf_sum_abs(rA) / nF
    final_residual = initial_residual

    # === Initial transpose residual (PBiCG.C:118-125) ===
    A_T_psi = tmul(ldu, psi)
    rT = source - A_T_psi

    # Buffers
    pA = np.zeros_like(psi)
    pT = np.zeros_like(psi)
    wArT = 0.0  # placeholder; iter==0 special-case avoids using wArTold

    n_iter = 0
    converged = False
    singular = False

    # === Main loop (PBiCG.C:138-247) ===
    # do { ... } while (n_iter < max_iter && (n_iter < min_iter || not converged))
    # → at least one iteration runs if min_iter >= 1 OR not yet converged.
    while True:
        if n_iter >= max_iter:
            break
        if n_iter >= min_iter and converged:
            break

        wArTold = wArT

        wA = precondition(ldu, rD, rA)
        wT = precondition_t(ldu, rD, losort, rT)

        wArT = _lf_dot(wA, rT)

        if n_iter == 0:
            pA = wA.copy()
            pT = wT.copy()
        else:
            beta = wArT / wArTold
            pA = wA + beta * pA
            pT = wT + beta * pT

        wA = amul(ldu, pA)
        wT = tmul(ldu, pT)

        wApT = _lf_dot(wA, pT)

        if abs(wApT) / nF < V_SMALL:
            singular = True
            break

        alpha = wArT / wApT

        psi += alpha * pA
        rA -= alpha * wA
        rT -= alpha * wT

        final_residual = _lf_sum_abs(rA) / nF
        n_iter += 1
        converged = _converged(final_residual, initial_residual,
                                tolerance, rel_tol)

    return SolverPerformance(
        initial_residual=initial_residual,
        final_residual=final_residual,
        n_iterations=n_iter,
        converged=converged,
        singular=singular,
        x=psi,
    )
