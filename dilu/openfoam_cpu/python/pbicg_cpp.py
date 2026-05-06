"""PBiCG outer iteration that calls the C++ kernels directly.

Logic is identical to pbicg.py — only the kernel bodies are swapped for
the byte-exact C++ versions in `_kernels_cpp`. This module exists for the
P1 acceptance gate (single-core wall time vs OF) and as the template the
MPI version (P3) will extend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ldu import LDU, calc_losort
from .normfactor import NORM_FACTOR_SMALL
from . import _kernels_cpp as cpp


V_SMALL = 1e-300


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


def _norm_factor_cpp(ldu: LDU, psi: np.ndarray, source: np.ndarray,
                     A_psi: np.ndarray,
                     sumA_override: Optional[np.ndarray]) -> float:
    """OpenFOAM normFactor; uses C++ sum_a if no override.
    Mirror of normfactor.py:norm_factor."""
    if sumA_override is None:
        sa = cpp.sum_a(ldu.diag, ldu.lower, ldu.upper,
                       ldu.owner, ldu.neighbour)
    else:
        sa = sumA_override
    psi_mean = float(psi.sum() / psi.size)
    sa_psibar = sa * psi_mean
    nf = ((np.abs(A_psi - sa_psibar) + np.abs(source - sa_psibar)).sum()
          + NORM_FACTOR_SMALL)
    return float(nf)


def solve(ldu: LDU, b: np.ndarray, x0: np.ndarray,
          tolerance: float = 1e-12, rel_tol: float = 0.0,
          min_iter: int = 0, max_iter: int = 1000,
          sumA_override: Optional[np.ndarray] = None,
          nf_override: Optional[float] = None) -> SolverPerformance:
    """PBiCG-DILU using C++ kernels. Same control flow as pbicg.solve."""
    psi = x0.astype(np.float64, copy=True)
    source = b.astype(np.float64, copy=False)

    diag, lower, upper = ldu.diag, ldu.lower, ldu.upper
    owner, neighbour = ldu.owner, ldu.neighbour

    rD = cpp.calc_reciprocal_d(diag, lower, upper, owner, neighbour)
    losort = calc_losort(neighbour)

    A_psi = cpp.amul(diag, lower, upper, owner, neighbour, psi)
    rA = source - A_psi
    if nf_override is not None:
        nF = float(nf_override)
    else:
        nF = _norm_factor_cpp(ldu, psi, source, A_psi, sumA_override)
    initial_residual = cpp.sum_abs(rA) / nF
    final_residual = initial_residual

    A_T_psi = cpp.tmul(diag, lower, upper, owner, neighbour, psi)
    rT = source - A_T_psi

    pA = np.zeros_like(psi)
    pT = np.zeros_like(psi)
    wArT = 0.0

    n_iter = 0
    converged = False
    singular = False

    while True:
        if n_iter >= max_iter:
            break
        if n_iter >= min_iter and converged:
            break

        wArTold = wArT

        wA = cpp.precondition(diag, lower, upper, owner, neighbour, rD, rA)
        wT = cpp.precondition_t(diag, lower, upper, owner, neighbour,
                                 rD, losort, rT)

        wArT = cpp.dot(wA, rT)

        if n_iter == 0:
            pA = wA.copy()
            pT = wT.copy()
        else:
            beta = wArT / wArTold
            pA = wA + beta * pA
            pT = wT + beta * pT

        wA = cpp.amul(diag, lower, upper, owner, neighbour, pA)
        wT = cpp.tmul(diag, lower, upper, owner, neighbour, pT)

        wApT = cpp.dot(wA, pT)

        if abs(wApT) / nF < V_SMALL:
            singular = True
            break

        alpha = wArT / wApT

        psi += alpha * pA
        rA -= alpha * wA
        rT -= alpha * wT

        final_residual = cpp.sum_abs(rA) / nF
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
