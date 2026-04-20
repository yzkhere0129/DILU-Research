"""Step 1+2 acceptance tests (pure-Python; no CUDA).

Covers:
  - Red-black coloring validity on 7-pt stencils (T8-equiv).
  - Greedy coloring on the same stencils gives 2 colors (bipartite degeneracy).
  - Greedy coloring on a few random DD matrices produces a valid coloring.
  - `permute_csr` round-trip: permute then inverse-permute recovers A.
  - Permuted CSR has ascending col_idx within each row.
  - `build_diag_offset_permuted` locates each diagonal correctly.
"""
import conftest  # noqa: F401

import numpy as np

from dilu.multicolor.python.coloring import (
    red_black_color, greedy_color_csr, validate_coloring, color_csr_auto,
)
from dilu.multicolor.python.permute import (
    permute_csr, build_diag_offset_permuted, inverse_permute_csr_values,
)
from _harness import laplacian_3d_7point, dense_to_csr


def _random_dd_csr(n, seed):
    rng = np.random.default_rng(seed)
    density = min(5.0 / n, 1.0)
    mask = rng.random((n, n)) < density
    np.fill_diagonal(mask, False)
    offd = rng.uniform(-0.1, 0.1, size=(n, n))
    A = np.where(mask, offd, 0.0)
    A = 0.5 * (A + A.T)
    diag = rng.uniform(1.0, 10.0, size=n)
    row_abs_sum = np.sum(np.abs(A), axis=1)
    diag = np.maximum(diag, row_abs_sum + 1.0)
    np.fill_diagonal(A, diag)
    return dense_to_csr(A)


# ------------------- COLORING -------------------
def test_red_black_bipartite_small():
    for shape in [(4, 4, 4), (6, 5, 4), (8, 8, 8)]:
        nx, ny, nz = shape
        row_ptr, col_idx, _ = laplacian_3d_7point(nx, ny, nz)
        perm, iperm, colors, offsets = red_black_color(nx, ny, nz)
        validate_coloring(row_ptr, col_idx, perm, iperm, colors, offsets)
        N = nx * ny * nz
        n_red = int(offsets[1]); n_black = int(offsets[2]) - int(offsets[1])
        assert n_red + n_black == N
        # Red count differs from black by at most 1 for a cubic grid.
        assert abs(n_red - n_black) <= 1
        print(f"red-black OK at {shape}: n_red={n_red}, n_black={n_black}")


def test_greedy_2color_on_bipartite():
    # Bipartite graphs should get 2 colors from greedy first-fit.
    for shape in [(4, 4, 4), (6, 5, 4)]:
        nx, ny, nz = shape
        row_ptr, col_idx, _ = laplacian_3d_7point(nx, ny, nz)
        perm, iperm, colors, offsets = greedy_color_csr(row_ptr, col_idx)
        validate_coloring(row_ptr, col_idx, perm, iperm, colors, offsets)
        n_colors = int(offsets.shape[0]) - 1
        assert n_colors == 2, f"expected 2 colors for 7-pt stencil, got {n_colors}"
        print(f"greedy 2-color OK at {shape}")


def test_greedy_on_random_dd():
    for seed in range(5):
        row_ptr, col_idx, _ = _random_dd_csr(40, seed=seed)
        perm, iperm, colors, offsets = greedy_color_csr(row_ptr, col_idx)
        validate_coloring(row_ptr, col_idx, perm, iperm, colors, offsets)
        n_colors = int(offsets.shape[0]) - 1
        print(f"greedy random DD seed={seed}: n_colors={n_colors}")
        assert n_colors >= 1


def test_color_csr_auto_dispatcher():
    nx, ny, nz = 5, 4, 3
    row_ptr, col_idx, _ = laplacian_3d_7point(nx, ny, nz)
    # With grid_shape -> red-black.
    perm_rb, iperm_rb, c_rb, off_rb = color_csr_auto(
        row_ptr, col_idx, grid_shape=(nx, ny, nz))
    # Without -> greedy.
    perm_gd, iperm_gd, c_gd, off_gd = color_csr_auto(row_ptr, col_idx)
    # Both must be valid (validator already ran inside color_csr_auto).
    # They may differ in within-color ordering but both should give 2 colors.
    assert int(off_rb.shape[0]) == 3
    assert int(off_gd.shape[0]) == 3


# ------------------- PERMUTE -------------------
def _check_roundtrip(row_ptr, col_idx, values, perm, iperm):
    row_ptr_new, col_idx_new, values_new, nnz_map = permute_csr(
        row_ptr, col_idx, values, perm, iperm)

    N = int(row_ptr.shape[0]) - 1
    nnz = int(values.shape[0])

    # (a) nnz preserved
    assert int(row_ptr_new[-1]) == nnz

    # (b) within-row ascending columns
    for i in range(N):
        rs, re = int(row_ptr_new[i]), int(row_ptr_new[i + 1])
        if re - rs >= 2:
            assert np.all(np.diff(col_idx_new[rs:re]) > 0), \
                f"row {i} has non-ascending cols"

    # (c) values_new via nnz_map matches direct permute
    values_via_map = values[nnz_map]
    assert np.allclose(values_new, values_via_map, atol=0.0), \
        "values_new ≠ values[nnz_map]"

    # (d) structural round-trip: permute then inverse. We don't have an
    # inverse-permute-CSR primitive (not needed for Phase 3), but we can
    # verify by rebuilding A from the permuted CSR.
    A_orig = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for p in range(int(row_ptr[i]), int(row_ptr[i + 1])):
            A_orig[i, int(col_idx[p])] = float(values[p])
    A_tilde = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for p in range(int(row_ptr_new[i]), int(row_ptr_new[i + 1])):
            A_tilde[i, int(col_idx_new[p])] = float(values_new[p])

    # A_tilde should equal P A Pᵀ where P is the permutation (P x)[i] =
    # x[perm[i]] i.e. new <- old rows by `perm`.
    P = np.zeros((N, N), dtype=np.float64)
    for new_idx in range(N):
        P[new_idx, int(perm[new_idx])] = 1.0
    A_expected = P @ A_orig @ P.T
    assert np.allclose(A_tilde, A_expected, atol=1e-13), \
        "permuted CSR does not equal P A Pᵀ"

    # (e) values[nnz_map] via inverse_permute_csr_values recovers values.
    values_back = inverse_permute_csr_values(values_new, nnz_map)
    assert np.allclose(values_back, values, atol=0.0), \
        "inverse_permute_csr_values did not recover original"


def test_permute_csr_red_black_7pt():
    nx, ny, nz = 5, 4, 3
    row_ptr, col_idx, values = laplacian_3d_7point(nx, ny, nz)
    perm, iperm, _, _ = red_black_color(nx, ny, nz)
    _check_roundtrip(row_ptr, col_idx, values, perm, iperm)


def test_permute_csr_greedy_random():
    for seed in range(3):
        row_ptr, col_idx, values = _random_dd_csr(20, seed=seed)
        perm, iperm, _, _ = greedy_color_csr(row_ptr, col_idx)
        _check_roundtrip(row_ptr, col_idx, values, perm, iperm)


def test_diag_offset_permuted():
    for shape in [(5, 4, 3), (6, 6, 6)]:
        nx, ny, nz = shape
        row_ptr, col_idx, values = laplacian_3d_7point(nx, ny, nz)
        perm, iperm, _, _ = red_black_color(nx, ny, nz)
        row_ptr_new, col_idx_new, _, _ = permute_csr(
            row_ptr, col_idx, values, perm, iperm)
        do = build_diag_offset_permuted(row_ptr_new, col_idx_new)
        N = nx * ny * nz
        for i in range(N):
            assert int(col_idx_new[int(do[i])]) == i


# ------------------- invalid-coloring detection -------------------
def test_validator_catches_bad_coloring():
    nx, ny, nz = 4, 4, 4
    row_ptr, col_idx, _ = laplacian_3d_7point(nx, ny, nz)
    N = nx * ny * nz
    # Construct a degenerate "coloring" that gives every node the same color —
    # this is only valid for a graph with no edges, so it must fail.
    perm = np.arange(N, dtype=np.int32)
    iperm = np.arange(N, dtype=np.int32)
    colors = np.zeros(N, dtype=np.int32)
    offsets = np.asarray([0, N], dtype=np.int32)
    raised = False
    try:
        validate_coloring(row_ptr, col_idx, perm, iperm, colors, offsets)
    except ValueError as e:
        raised = True
        print(f"(expected failure) {e}")
    assert raised, "validator must reject monochromatic 'coloring' on 7-pt stencil"


if __name__ == "__main__":
    test_red_black_bipartite_small()
    test_greedy_2color_on_bipartite()
    test_greedy_on_random_dd()
    test_color_csr_auto_dispatcher()
    test_permute_csr_red_black_7pt()
    test_permute_csr_greedy_random()
    test_diag_offset_permuted()
    test_validator_catches_bad_coloring()
    print("Step 1+2 coloring/permute tests OK")
