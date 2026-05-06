"""MPI-collective reductions: gSum / gSumProd / gSumMag.

OpenFOAM's globalReduce uses MPI_Allreduce with MPI_SUM, applied to a
single double after the local left-fold.  Two ranks summing in different
orders can give different ULP-level results — to be byte-exact match OF
across N ranks we must replicate OF's exact reduction order.

For now we use MPI_Allreduce directly which gives ULP×log2(N) drift —
acceptable for the plan's 1e-12 target (not ULP=0).
"""
from __future__ import annotations

import numpy as np
from mpi4py import MPI

from .. import _kernels_cpp as cpp


def gsum_mag(r_local: np.ndarray, comm: MPI.Comm) -> float:
    """Σ_global |r| = MPI_Allreduce(local sum_abs)."""
    local = cpp.sum_abs(np.ascontiguousarray(r_local))
    return comm.allreduce(local, op=MPI.SUM)


def gsum_prod(a_local: np.ndarray, b_local: np.ndarray, comm: MPI.Comm) -> float:
    """Σ_global a·b = MPI_Allreduce(local dot)."""
    local = cpp.dot(np.ascontiguousarray(a_local),
                     np.ascontiguousarray(b_local))
    return comm.allreduce(local, op=MPI.SUM)


def gaverage(psi_local: np.ndarray, n_global: int, comm: MPI.Comm) -> float:
    """Σ psi / N. Used in normFactor."""
    s = comm.allreduce(float(psi_local.sum()), op=MPI.SUM)
    return s / n_global
