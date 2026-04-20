"""Host-side graph coloring for multi-color DILU preconditioning.

Design doc references:
  - `docs/design/phase3_multicoloring_math.md` §2 (red-black bipartite,
    greedy fallback).
  - `docs/design/phase3_multicolor_ffi_architecture.md` §2.3 (output shape
    contract), §2.4 (red-black closed form), §2.5 (greedy first-fit).

Outputs (all int32, C-contiguous):
  - `perm[N]`     : permutation, perm[new_idx] = old_idx
  - `iperm[N]`    : inverse, iperm[old_idx] = new_idx
  - `colors_by_new_idx[N]` : color assigned to each row AFTER permutation
                             (contiguous by color)
  - `color_offsets[n_colors + 1]` : color c occupies new indices
                                    [color_offsets[c], color_offsets[c+1])

Zero external dependencies beyond NumPy. Arch §2.1.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def red_black_color(
    nx: int, ny: int, nz: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form 2-color (red-black) assignment for a 7-point stencil on a
    structured nx*ny*nz cubic grid.

    A 7-point stencil graph is bipartite under the parity map
    ``pi(i,j,k) = (i + j + k) mod 2`` — any edge connects cells of opposite
    parity. Permutation groups reds (color 0) before blacks (color 1).

    Row-major natural index: ``idx = (k*ny + j)*nx + i`` (matches `_harness`
    and `test_t7_pcg_stiff.py` helpers).
    """
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"red_black_color: invalid grid {nx}x{ny}x{nz}")
    N = nx * ny * nz
    ijk = np.arange(N, dtype=np.int64)
    i = ijk % nx
    j = (ijk // nx) % ny
    k = ijk // (nx * ny)
    parity = ((i + j + k) % 2).astype(np.int32)

    red_old = np.where(parity == 0)[0].astype(np.int32)
    black_old = np.where(parity == 1)[0].astype(np.int32)
    n_red = int(red_old.size)
    n_black = int(black_old.size)

    perm = np.concatenate([red_old, black_old]).astype(np.int32)
    iperm = np.empty(N, dtype=np.int32)
    iperm[perm] = np.arange(N, dtype=np.int32)

    colors_by_new_idx = np.concatenate(
        [np.zeros(n_red, dtype=np.int32),
         np.ones(n_black, dtype=np.int32)]
    )
    color_offsets = np.asarray([0, n_red, N], dtype=np.int32)

    return (np.ascontiguousarray(perm),
            np.ascontiguousarray(iperm),
            np.ascontiguousarray(colors_by_new_idx),
            np.ascontiguousarray(color_offsets))


def greedy_color_csr(
    row_ptr: np.ndarray, col_idx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Welsh-Powell-style first-fit greedy graph coloring on CSR adjacency.

    Walks vertices 0..N-1 in natural order; each vertex takes the smallest
    color not used by any already-colored neighbor. Deterministic; worst-case
    gives Delta+1 colors (Delta = max degree). For the 7-pt bipartite stencil
    this degenerates to 2 colors — matches `red_black_color` output (modulo
    within-color ordering).

    Returns the same 4 arrays as `red_black_color`. The permutation is a stable
    argsort by color, so the within-color ordering preserves natural indexing.
    """
    row_ptr = np.ascontiguousarray(row_ptr, dtype=np.int32)
    col_idx = np.ascontiguousarray(col_idx, dtype=np.int32)
    N = int(row_ptr.shape[0]) - 1
    color_of_old = np.full(N, -1, dtype=np.int32)

    # Small buffer for "colors seen by neighbors" — avoids Python set alloc
    # in the inner loop. Size bounded by max degree (~7 for our test matrices).
    # Worst case, grow dynamically via list when needed.
    for v in range(N):
        rs = int(row_ptr[v])
        re = int(row_ptr[v + 1])
        # Gather colored-neighbor colors.
        # (Inner loop in pure Python — fine for N ≤ 1e5; vectorize if needed.)
        seen = set()
        for p in range(rs, re):
            u = int(col_idx[p])
            if u == v:
                continue
            cu = int(color_of_old[u])
            if cu >= 0:
                seen.add(cu)
        c = 0
        while c in seen:
            c += 1
        color_of_old[v] = c

    n_colors = int(color_of_old.max()) + 1 if N > 0 else 0

    # Stable argsort so within-color ordering mirrors the natural order.
    perm = np.argsort(color_of_old, kind="stable").astype(np.int32)
    iperm = np.empty(N, dtype=np.int32)
    iperm[perm] = np.arange(N, dtype=np.int32)

    colors_by_new_idx = color_of_old[perm].astype(np.int32)

    # `color_offsets[c]` = first new_idx with color c. Since colors_by_new_idx is
    # already non-decreasing after stable-sort, use searchsorted.
    color_offsets = np.searchsorted(
        colors_by_new_idx, np.arange(n_colors + 1, dtype=np.int32),
        side="left",
    ).astype(np.int32)
    # Ensure terminal offset equals N (searchsorted returns N for value == n_colors).
    color_offsets[-1] = N

    return (np.ascontiguousarray(perm),
            np.ascontiguousarray(iperm),
            np.ascontiguousarray(colors_by_new_idx),
            np.ascontiguousarray(color_offsets))


def validate_coloring(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    perm: np.ndarray,
    iperm: np.ndarray,
    colors_by_new_idx: np.ndarray,
    color_offsets: np.ndarray,
) -> None:
    """Assert the coloring is valid for `A`'s adjacency graph.

    Contract (arch §2.3 invariant 3): for every CSR off-diagonal edge
    ``(row_old, col_old)`` in the ORIGINAL matrix, the two endpoints must have
    different colors. Checked by re-indexing row_old and col_old via `iperm`
    into new-index space and comparing `colors_by_new_idx`.

    Also checks permutation round-trip and color_offsets consistency.

    Raises ValueError on any violation. STOP #1 in arch §9.
    """
    row_ptr = np.asarray(row_ptr, dtype=np.int32)
    col_idx = np.asarray(col_idx, dtype=np.int32)
    perm = np.asarray(perm, dtype=np.int32)
    iperm = np.asarray(iperm, dtype=np.int32)
    colors_by_new_idx = np.asarray(colors_by_new_idx, dtype=np.int32)
    color_offsets = np.asarray(color_offsets, dtype=np.int32)

    N = int(row_ptr.shape[0]) - 1
    if perm.shape != (N,) or iperm.shape != (N,):
        raise ValueError(f"perm/iperm shape mismatch; expected ({N},)")
    if colors_by_new_idx.shape != (N,):
        raise ValueError(f"colors shape mismatch; expected ({N},)")

    # Permutation round-trip: iperm[perm[i]] == i.
    rt = iperm[perm]
    if not np.array_equal(rt, np.arange(N, dtype=np.int32)):
        bad = int(np.argmax(rt != np.arange(N, dtype=np.int32)))
        raise ValueError(
            f"perm/iperm not round-trip consistent at i={bad}: "
            f"iperm[perm[{bad}]]={int(rt[bad])}"
        )

    # color_offsets monotonic and terminal == N.
    if color_offsets[0] != 0:
        raise ValueError("color_offsets[0] != 0")
    if int(color_offsets[-1]) != N:
        raise ValueError(
            f"color_offsets[-1] = {int(color_offsets[-1])} != N = {N}"
        )
    if np.any(np.diff(color_offsets) < 0):
        raise ValueError("color_offsets not non-decreasing")

    # Rows within [color_offsets[c], color_offsets[c+1]) must all have color c.
    n_colors = int(color_offsets.shape[0]) - 1
    for c in range(n_colors):
        lo, hi = int(color_offsets[c]), int(color_offsets[c + 1])
        if lo == hi:
            continue
        seg = colors_by_new_idx[lo:hi]
        if not np.all(seg == c):
            bad = int(np.argmax(seg != c))
            raise ValueError(
                f"color block {c} has stray color "
                f"{int(seg[bad])} at new_idx {lo + bad}"
            )

    # CORE invariant (arch §2.3 #3): no two adjacent vertices share a color.
    # Vectorized over nnz: for each off-diagonal CSR entry, compare colors.
    # Build per-nnz row_of_entry via cumulative row lengths.
    nnz = int(col_idx.shape[0])
    if nnz > 0:
        row_lengths = np.diff(row_ptr).astype(np.int32)
        row_of_nnz = np.repeat(np.arange(N, dtype=np.int32), row_lengths)
        off_diag_mask = row_of_nnz != col_idx
        rows_old = row_of_nnz[off_diag_mask]
        cols_old = col_idx[off_diag_mask]
        rows_new = iperm[rows_old]
        cols_new = iperm[cols_old]
        c_row = colors_by_new_idx[rows_new]
        c_col = colors_by_new_idx[cols_new]
        bad_edges = np.where(c_row == c_col)[0]
        if bad_edges.size > 0:
            b0 = int(bad_edges[0])
            raise ValueError(
                f"coloring invalid: edge ({int(rows_old[b0])}, "
                f"{int(cols_old[b0])}) both have color {int(c_row[b0])} — "
                f"{bad_edges.size} violations total"
            )


def color_csr_auto(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    grid_shape: Tuple[int, int, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dispatcher: red-black fast path when `grid_shape` is supplied and row
    count matches, else greedy.

    Always runs `validate_coloring` before returning.
    """
    N = int(row_ptr.shape[0]) - 1
    if grid_shape is not None:
        nx, ny, nz = grid_shape
        if nx * ny * nz != N:
            raise ValueError(
                f"grid_shape {grid_shape} inconsistent with N={N}"
            )
        perm, iperm, colors, offsets = red_black_color(nx, ny, nz)
    else:
        perm, iperm, colors, offsets = greedy_color_csr(row_ptr, col_idx)

    validate_coloring(row_ptr, col_idx, perm, iperm, colors, offsets)
    return perm, iperm, colors, offsets
