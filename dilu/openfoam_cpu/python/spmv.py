"""Amul / Tmul / sumA — direct port of lduMatrixATmul.C face-loop form.

Single-process; interface contributions skipped (single dump = no coupled
patches). See audit §5 and §10.13.
"""

from __future__ import annotations

import numpy as np

from .ldu import LDU


def amul(ldu: LDU, psi: np.ndarray) -> np.ndarray:
    """Apsi = A * psi. lduMatrixATmul.C face-loop branch."""
    out = ldu.diag * psi
    np.add.at(out, ldu.neighbour, ldu.lower * psi[ldu.owner])
    np.add.at(out, ldu.owner,     ldu.upper * psi[ldu.neighbour])
    return out


def tmul(ldu: LDU, psi: np.ndarray) -> np.ndarray:
    """Tpsi = A^T * psi. lduMatrixATmul.C:198-201, lower<->upper swap."""
    out = ldu.diag * psi
    np.add.at(out, ldu.neighbour, ldu.upper * psi[ldu.owner])
    np.add.at(out, ldu.owner,     ldu.lower * psi[ldu.neighbour])
    return out


def sum_a(ldu: LDU) -> np.ndarray:
    """sumA[c] = sum_j A[c, j]. lduMatrixATmul.C:219+. See audit §10.13."""
    out = ldu.diag.copy()
    np.add.at(out, ldu.neighbour, ldu.lower)
    np.add.at(out, ldu.owner,     ldu.upper)
    return out
