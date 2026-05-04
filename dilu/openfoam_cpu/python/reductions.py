"""Byte-exact replicas of OpenFOAM's reduction primitives.

OpenFOAM uses naive left-fold for sum, sumProd, sumMag (see audit §10.17,
FieldFunctions.C:478-491). NumPy's `np.dot`, `np.sum`, `(arr).sum()` use
pairwise reduction (and BLAS for dot), which gives DIFFERENT floating-point
results due to non-associativity.

For PBiCG to byte-match OpenFOAM's `initialResidual` and `finalResidual`,
we must replicate the exact reduction order. Numba @njit gives near-C speed
for these tight loops while preserving sequential left-fold semantics.

If numba is unavailable, falls back to pure Python loops (slow but correct).
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    @njit(cache=True, fastmath=False)
    def sum_abs(r: np.ndarray) -> float:
        """Σ_i |r[i]|, sequential left-fold (matches OF gSumMag)."""
        s = 0.0
        for i in range(r.shape[0]):
            s += abs(r[i])
        return s

    @njit(cache=True, fastmath=False)
    def dot(a: np.ndarray, b: np.ndarray) -> float:
        """Σ_i a[i]*b[i], sequential left-fold (matches OF gSumProd / sumProd)."""
        s = 0.0
        for i in range(a.shape[0]):
            s += a[i] * b[i]
        return s

except ImportError:
    def sum_abs(r: np.ndarray) -> float:
        s = 0.0
        for v in r:
            s += abs(float(v))
        return s

    def dot(a: np.ndarray, b: np.ndarray) -> float:
        s = 0.0
        for i in range(a.size):
            s += float(a[i]) * float(b[i])
        return s
