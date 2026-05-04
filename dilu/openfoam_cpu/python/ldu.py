"""LDU addressing helpers.

OpenFOAM stores a sparse matrix as five arrays:
    diag[i]            i = 0..nCells-1
    upper[f]   = A[owner[f],     neighbour[f]]      (strict upper)
    lower[f]   = A[neighbour[f], owner[f]]          (strict lower)
    owner[f]   < neighbour[f]                       (face index < cell index)
faces are sorted by (owner, neighbour) lexicographic order.

See audit §2 and §10.1 for source-line mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class LDU:
    diag: np.ndarray       # (nCells,) float64
    lower: np.ndarray      # (nFaces,) float64 = A[neighbour, owner]
    upper: np.ndarray      # (nFaces,) float64 = A[owner, neighbour]
    owner: np.ndarray      # (nFaces,) int32   = lowerAddr in OpenFOAM
    neighbour: np.ndarray  # (nFaces,) int32   = upperAddr in OpenFOAM

    @property
    def n_cells(self) -> int:
        return self.diag.size

    @property
    def n_faces(self) -> int:
        return self.upper.size


def csr_to_ldu(A: csr_matrix) -> LDU:
    """Reconstruct OpenFOAM LDU 5-tuple from a CSR matrix.

    OpenFOAM's face order is (owner, neighbour) lexicographic, owner < neighbour.
    This invariant lets us recover the original face order by stable lexsort.
    See audit §10.16.
    """
    A = A.tocoo()
    diag = np.zeros(A.shape[0], dtype=np.float64)
    diag_mask = (A.row == A.col)
    diag[A.row[diag_mask]] = A.data[diag_mask]

    upper_mask = A.row < A.col
    u_row = A.row[upper_mask].astype(np.int32)
    u_col = A.col[upper_mask].astype(np.int32)
    u_val = A.data[upper_mask].astype(np.float64)
    order_u = np.lexsort((u_col, u_row))  # primary owner=row, secondary nei=col
    owner = u_row[order_u]
    neighbour = u_col[order_u]
    upper = u_val[order_u]

    lower_mask = A.row > A.col
    l_row = A.row[lower_mask].astype(np.int32)  # = neighbour for that face
    l_col = A.col[lower_mask].astype(np.int32)  # = owner
    l_val = A.data[lower_mask].astype(np.float64)
    order_l = np.lexsort((l_row, l_col))  # primary owner=col, secondary nei=row
    lower = l_val[order_l]

    return LDU(diag=diag, lower=lower, upper=upper,
               owner=owner, neighbour=neighbour)


def calc_losort(neighbour: np.ndarray) -> np.ndarray:
    """OpenFOAM's losortAddr: face indices sorted by neighbour ascending,
    stable on ties. Implemented as counting sort in OpenFOAM
    (lduAddressing.C:34-91); we use NumPy stable mergesort which gives
    bit-identical face permutation. See audit §10.2.
    """
    return np.argsort(neighbour, kind='stable').astype(np.int32)
