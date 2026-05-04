"""DILUPreconditioner.C direct port. Single-thread reference impl.

calcReciprocalD : DILUPreconditioner.C:64-92    (audit §3.2)
precondition    : DILUPreconditioner.C:95-136   (audit §3.3)
preconditionT   : DILUPreconditioner.C:139-184  (audit §3.4)

Hard rules from audit §10.1 / §10.15:
- All three loops are face-ordered, NOT cell-ordered
- calcReciprocalD MUST be single-threaded (face dependency)
- rD stores 1/D̃ (reciprocal), not D̃
- 1.0/rD has NO divide-by-zero guard, matching OpenFOAM bit-by-bit
- preconditionT is the only place losort is used

Performance: numba @njit makes the sequential face-loops native-speed
(~50× faster than pure Python). Falls back to Python if numba absent.
"""
from __future__ import annotations

import numpy as np

from .ldu import LDU


try:
    from numba import njit

    @njit(cache=True, fastmath=False)
    def _calc_rd_kernel(rD, upper, lower, owner, neighbour, n_faces):
        for f in range(n_faces):
            rD[neighbour[f]] -= upper[f] * lower[f] / rD[owner[f]]
        for i in range(rD.shape[0]):
            rD[i] = 1.0 / rD[i]
        return rD

    @njit(cache=True, fastmath=False)
    def _precondition_kernel(wA, rD, upper, lower, owner, neighbour, n_faces):
        for f in range(n_faces):
            wA[neighbour[f]] -= rD[neighbour[f]] * lower[f] * wA[owner[f]]
        for f in range(n_faces - 1, -1, -1):
            wA[owner[f]] -= rD[owner[f]] * upper[f] * wA[neighbour[f]]
        return wA

    @njit(cache=True, fastmath=False)
    def _precondition_t_kernel(wT, rD, upper, lower, owner, neighbour, losort, n_faces):
        for f in range(n_faces):
            wT[neighbour[f]] -= rD[neighbour[f]] * upper[f] * wT[owner[f]]
        for f in range(n_faces - 1, -1, -1):
            sf = losort[f]
            wT[owner[sf]] -= rD[owner[sf]] * lower[sf] * wT[neighbour[sf]]
        return wT

except ImportError:
    def _calc_rd_kernel(rD, upper, lower, owner, neighbour, n_faces):
        for f in range(n_faces):
            rD[neighbour[f]] -= upper[f] * lower[f] / rD[owner[f]]
        return 1.0 / rD

    def _precondition_kernel(wA, rD, upper, lower, owner, neighbour, n_faces):
        for f in range(n_faces):
            wA[neighbour[f]] -= rD[neighbour[f]] * lower[f] * wA[owner[f]]
        for f in range(n_faces - 1, -1, -1):
            wA[owner[f]] -= rD[owner[f]] * upper[f] * wA[neighbour[f]]
        return wA

    def _precondition_t_kernel(wT, rD, upper, lower, owner, neighbour, losort, n_faces):
        for f in range(n_faces):
            wT[neighbour[f]] -= rD[neighbour[f]] * upper[f] * wT[owner[f]]
        for f in range(n_faces - 1, -1, -1):
            sf = losort[f]
            wT[owner[sf]] -= rD[owner[sf]] * lower[sf] * wT[neighbour[sf]]
        return wT


def calc_reciprocal_d(ldu: LDU) -> np.ndarray:
    """Returns rD where rD[i] = 1/D̃_i, D̃ from D-ILU(0) factorization."""
    rD = ldu.diag.copy()
    return _calc_rd_kernel(rD, ldu.upper, ldu.lower,
                           ldu.owner, ldu.neighbour, ldu.upper.size)


def precondition(ldu: LDU, rD: np.ndarray, rA: np.ndarray) -> np.ndarray:
    """wA = M^{-1} rA where M = (D̃ + L) D̃^{-1} (D̃ + U)."""
    wA = rD * rA
    return _precondition_kernel(wA, rD, ldu.upper, ldu.lower,
                                ldu.owner, ldu.neighbour, ldu.upper.size)


def precondition_t(ldu: LDU, rD: np.ndarray, losort: np.ndarray,
                   rT: np.ndarray) -> np.ndarray:
    """wT = M^{-T} rT — uses losort for transpose backward sweep."""
    wT = rD * rT
    return _precondition_t_kernel(wT, rD, ldu.upper, ldu.lower,
                                  ldu.owner, ldu.neighbour, losort,
                                  ldu.upper.size)
