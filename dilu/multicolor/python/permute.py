"""Host-side CSR permutation helpers.

Builds the reordered CSR `Ã = P A Pᵀ` used by Phase 3 kernels. Math §1.1,
Arch §3.1 step 2.

Contract:
  - `permute_csr(...)` reorders rows by `perm`, remaps column indices via
    `iperm`, and sorts each permuted row by ascending column index (cuSPARSE-
    compatible layout — needed for the `find_col_in_row` binary search in
    the verbatim-copied `dilu_factor_kernel.cu` from Phase 2).
  - `nnz_map[new_k]` is the original nnz position whose value ends up at
    `values_tilde[new_k]`. This lets the apply / refactor handlers gather
    fresh values per call without re-running the permutation pipeline.
  - `build_diag_offset_permuted(...)` locates the diagonal entry index in
    each permuted row.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def permute_csr(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    values: np.ndarray,
    perm: np.ndarray,
    iperm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (row_ptr_new, col_idx_new, values_new, nnz_map) for Ã = P A Pᵀ.

    `perm[new_idx] = old_idx` — new row `new_idx` equals old row `perm[new_idx]`.
    Column indices are remapped via `iperm[col_old] = col_new`.

    Rows are sorted by ascending column inside each row (required for the
    cuSPARSE-style binary search inside our verbatim `dilu_factor_kernel.cu`).

    Returns int32 row_ptr_new, int32 col_idx_new, float64 values_new, int32
    nnz_map. All C-contiguous.
    """
    row_ptr = np.ascontiguousarray(row_ptr, dtype=np.int32)
    col_idx = np.ascontiguousarray(col_idx, dtype=np.int32)
    values = np.ascontiguousarray(values, dtype=np.float64)
    perm = np.ascontiguousarray(perm, dtype=np.int32)
    iperm = np.ascontiguousarray(iperm, dtype=np.int32)

    N = int(row_ptr.shape[0]) - 1
    nnz = int(col_idx.shape[0])
    if values.shape[0] != nnz:
        raise ValueError("permute_csr: values / col_idx length mismatch")
    if perm.shape != (N,) or iperm.shape != (N,):
        raise ValueError(f"permute_csr: perm/iperm must have shape ({N},)")

    # Row lengths in the permuted matrix: length of new row `new_idx` equals
    # length of old row `perm[new_idx]`.
    old_row_lengths = np.diff(row_ptr).astype(np.int32)
    new_row_lengths = old_row_lengths[perm]
    row_ptr_new = np.empty(N + 1, dtype=np.int32)
    row_ptr_new[0] = 0
    np.cumsum(new_row_lengths, out=row_ptr_new[1:])
    if int(row_ptr_new[-1]) != nnz:
        raise ValueError(
            f"permute_csr: nnz mismatch after permutation "
            f"({int(row_ptr_new[-1])} vs {nnz})"
        )

    col_idx_new = np.empty(nnz, dtype=np.int32)
    values_new = np.empty(nnz, dtype=np.float64)
    nnz_map = np.empty(nnz, dtype=np.int32)

    for new_idx in range(N):
        old_idx = int(perm[new_idx])
        rs_old = int(row_ptr[old_idx])
        re_old = int(row_ptr[old_idx + 1])
        rs_new = int(row_ptr_new[new_idx])
        re_new = int(row_ptr_new[new_idx + 1])

        # Remap columns via iperm then sort within this row.
        cols_remapped = iperm[col_idx[rs_old:re_old]]
        order = np.argsort(cols_remapped, kind="stable")
        col_idx_new[rs_new:re_new] = cols_remapped[order]
        values_new[rs_new:re_new] = values[rs_old:re_old][order]
        # nnz_map[new_k] = original nnz index `rs_old + order[j]`.
        nnz_map[rs_new:re_new] = (rs_old + order).astype(np.int32)

    return (
        np.ascontiguousarray(row_ptr_new),
        np.ascontiguousarray(col_idx_new),
        np.ascontiguousarray(values_new),
        np.ascontiguousarray(nnz_map),
    )


def build_diag_offset_permuted(
    row_ptr_new: np.ndarray, col_idx_new: np.ndarray
) -> np.ndarray:
    """Locate the diagonal entry index in each permuted row.

    Mirrors `dilu.cusparse.python.build_diag_offset`. Column indices are
    ascending inside each row (enforced by `permute_csr`), so searchsorted
    works.
    """
    row_ptr_new = np.ascontiguousarray(row_ptr_new, dtype=np.int32)
    col_idx_new = np.ascontiguousarray(col_idx_new, dtype=np.int32)
    N = int(row_ptr_new.shape[0]) - 1
    out = np.empty(N, dtype=np.int32)
    for i in range(N):
        rs = int(row_ptr_new[i])
        re = int(row_ptr_new[i + 1])
        k = rs + int(np.searchsorted(col_idx_new[rs:re], i))
        if k >= re or int(col_idx_new[k]) != i:
            raise ValueError(
                f"build_diag_offset_permuted: row {i} has no diagonal entry"
            )
        out[i] = k
    return np.ascontiguousarray(out)


def inverse_permute_csr_values(
    values_new: np.ndarray, nnz_map: np.ndarray
) -> np.ndarray:
    """Given permuted values and the forward `nnz_map[new_k] = old_k`,
    produce `values_old` by scatter. Used for round-trip validation.
    """
    values_new = np.ascontiguousarray(values_new, dtype=np.float64)
    nnz_map = np.ascontiguousarray(nnz_map, dtype=np.int32)
    out = np.empty_like(values_new)
    out[nnz_map] = values_new
    return out
