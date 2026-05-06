"""PCG-DIC outer iteration over MPI domain decomposition.

DILU/DIC factorization in OpenFOAM is per-rank block-Jacobi: processor
boundary contributions are NOT folded into the local rD.  So we just call
the existing serial `cpp.calc_reciprocal_d` / `cpp.precondition` on each
rank's internal lduMatrix — no halo exchange needed inside the
preconditioner, only inside amul.

normFactor uses gAverage(psi) and gSumMag(...) over the GLOBAL field, so
we need MPI_Allreduce for the mean and the L1 norms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from .. import _kernels_cpp as cpp
from ..normfactor import NORM_FACTOR_SMALL
from .decompose import LocalLDU
from .spmv_mpi import amul_mpi, sum_a_mpi
from .reductions_mpi import gsum_mag, gsum_prod, gaverage


V_SMALL = 1e-300


@dataclass
class MpiSolverPerformance:
    initial_residual: float
    final_residual: float
    n_iterations: int
    converged: bool
    singular: bool
    x_local: np.ndarray   # local-rank slice of the solution


def _converged(rN, r0, tol, rel_tol):
    if rN < tol: return True
    if rel_tol > NORM_FACTOR_SMALL and rN < rel_tol * r0: return True
    return False


def _norm_factor_mpi(local: LocalLDU, psi_local, source_local, A_psi_local,
                     n_global: int, comm: MPI.Comm) -> float:
    """OF normFactor across all ranks.  See lduMatrixSolver.C:235-274."""
    sa_local = sum_a_mpi(local)
    psi_bar = gaverage(psi_local, n_global, comm)
    sa_psibar = sa_local * psi_bar
    # local L1 contributions; gSum across ranks.
    nf_local = cpp.sum_abs(np.ascontiguousarray(np.abs(A_psi_local - sa_psibar)
                                                 + np.abs(source_local - sa_psibar)))
    nf = comm.allreduce(nf_local, op=MPI.SUM) + NORM_FACTOR_SMALL
    return float(nf)


def solve_mpi(local: LocalLDU, source_local: np.ndarray, x0_local: np.ndarray,
              n_global: int, comm: MPI.Comm, *,
              tolerance: float = 1e-12, rel_tol: float = 0.0,
              min_iter: int = 0, max_iter: int = 1000
              ) -> MpiSolverPerformance:
    """Solve A·x = b with PCG-DIC across MPI ranks. byte-exact ↔ serial
    PCG up to ULP×N drift from MPI_Allreduce reduction-order."""
    psi = x0_local.astype(np.float64, copy=True)
    src = source_local.astype(np.float64, copy=False)

    # DILU = per-rank serial calc_reciprocal_d on local LDU
    rD = cpp.calc_reciprocal_d(local.diag, local.lower, local.upper,
                                 local.owner, local.neighbour)

    A_psi = amul_mpi(local, psi, comm)
    rA = src - A_psi
    nF = _norm_factor_mpi(local, psi, src, A_psi, n_global, comm)
    initial_residual = gsum_mag(rA, comm) / nF
    final_residual = initial_residual

    pA = np.zeros_like(psi)
    wArA = 0.0

    n_iter = 0
    converged = False
    singular = False

    while True:
        if n_iter >= max_iter: break
        if n_iter >= min_iter and converged: break

        wArAold = wArA
        wA = cpp.precondition(local.diag, local.lower, local.upper,
                                local.owner, local.neighbour, rD, rA)
        wArA = gsum_prod(wA, rA, comm)

        if n_iter == 0:
            pA = wA.copy()
        else:
            beta = wArA / wArAold
            pA = wA + beta * pA

        wA = amul_mpi(local, pA, comm)
        wApA = gsum_prod(wA, pA, comm)

        if abs(wApA) / nF < V_SMALL:
            singular = True
            break

        alpha = wArA / wApA
        psi += alpha * pA
        rA -= alpha * wA

        final_residual = gsum_mag(rA, comm) / nF
        n_iter += 1
        converged = _converged(final_residual, initial_residual,
                                tolerance, rel_tol)

    return MpiSolverPerformance(
        initial_residual=initial_residual,
        final_residual=final_residual,
        n_iterations=n_iter,
        converged=converged,
        singular=singular,
        x_local=psi,
    )
