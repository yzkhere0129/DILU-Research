"""normFactor and L1 residual exactly as lduMatrixSolver.C:235-274.

OpenFOAM's normFactor is

    normFactor = Σ |A·ψ - sumA·ψ̄| + Σ |source - sumA·ψ̄| + small_

where ψ̄ = mean(ψ) and small_ = 1e-20 (NOT SMALL=1e-15 nor VSMALL=1e-37).
See audit §10.3.

PBiCG residual norm = sum(|r|) / normFactor.  L1 norm, not L2.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .ldu import LDU
from .spmv import sum_a

NORM_FACTOR_SMALL = 1e-20


def norm_factor(ldu: LDU, psi: np.ndarray, source: np.ndarray,
                A_psi: np.ndarray,
                sumA_override: Optional[np.ndarray] = None) -> float:
    """Compute OpenFOAM normFactor (lduMatrixSolver.C:235-274).

    sumA_override : if provided, use it instead of self-computed sum_a(ldu).
        Required for byte-exact match against OpenFOAM because OF's sumA
        uses raw_diag (NOT D() = diag + internalCoeffs) and subtracts coupled
        bouCoeffs — neither of which is reconstructible from the dumped
        (folded) LDU 5-tuple. dumper writes sumA.mm; pass it here.
    """
    sa = sum_a(ldu) if sumA_override is None else sumA_override
    psi_mean = float(psi.sum() / psi.size)  # gAverage = sum/N (FieldFunctions.C:647)
    sa_psibar = sa * psi_mean
    nf = ((np.abs(A_psi - sa_psibar) + np.abs(source - sa_psibar)).sum()
          + NORM_FACTOR_SMALL)
    return float(nf)


def residual_norm(r: np.ndarray, normFactor: float) -> float:
    return float(np.abs(r).sum() / normFactor)
