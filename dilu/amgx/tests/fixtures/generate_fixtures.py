"""Synthesize canonical small matrices for self-contained tests.

Two fixtures are produced under `dilu/amgx/tests/fixtures/`:

  pd_tiny.npz   12x12x12 = 1728-cell 7-point Laplacian with mild variable
                density jump (mimics the LPBF pressure-correction SPD
                pattern; sign-convention matches OpenFOAM lduMatrix
                export — diagonal is NEGATIVE, so the AMGx wrapper's
                load helper must sign-flip).
  T_tiny.npz    12x12x12 = 1728-cell diffusion + small skew-symmetric
                advection (mimics the LPBF energy-equation near-symmetric
                pattern). Diagonal POSITIVE; asymmetry ~ 1e-4.

Each .npz contains:
  data, indices, indptr, shape  CSR triples for `A`
  b                              RHS (random, fixed seed)
  x_truth                        scipy.sparse.linalg.spsolve(A, b)
                                 (with sign-flip applied for pd before
                                  spsolve, so x_truth IS the SPD solution)

The arrays are deterministic: seeded via numpy default_rng(seed=20260517).

To regenerate (after a deliberate change to the spec):
    python dilu/amgx/tests/fixtures/generate_fixtures.py

Do NOT regenerate casually — the bit-exact tests in test_reproduce.py
hash the .npz contents.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


HERE = Path(__file__).parent
N = 12                # 12^3 = 1728 cells
SEED = 20260517


def _laplacian_csr_7point(n, density_jump_at=None):
    """7-point 3D Laplacian, unit spacing. Optional density jump (a coefficient
    that scales each face's contribution) localized to one octant. Returns
    CSR (indptr, indices, data) in NATURAL Laplacian sign (positive diagonal,
    negative off-diagonal). Caller flips sign for OpenFOAM convention if
    needed."""
    rng = np.random.default_rng(SEED)
    rho = np.ones((n, n, n), dtype=np.float64)
    if density_jump_at is not None:
        i0, j0, k0 = density_jump_at
        rho[i0:, j0:, k0:] = 1000.0  # 1000x density jump
    rho_flat = rho.reshape(-1)

    rows = []
    cols = []
    data = []
    def idx(i, j, k):
        return (i * n + j) * n + k

    for i in range(n):
        for j in range(n):
            for k in range(n):
                p = idx(i, j, k)
                diag = 0.0
                for di, dj, dk in [(-1, 0, 0), (1, 0, 0),
                                    (0, -1, 0), (0, 1, 0),
                                    (0, 0, -1), (0, 0, 1)]:
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < n and 0 <= jj < n and 0 <= kk < n:
                        q = idx(ii, jj, kk)
                        rho_face = 0.5 * (rho_flat[p] + rho_flat[q])
                        coef = 1.0 / rho_face
                        rows.append(p); cols.append(q); data.append(-coef)
                        diag += coef
                # Add small Helmholtz term to keep matrix nonsingular
                # (Neumann + pin would also work; this keeps the test pure)
                diag += 1e-3
                rows.append(p); cols.append(p); data.append(diag)
    A = sp.coo_matrix((data, (rows, cols)),
                       shape=(n**3, n**3)).tocsr()
    A.sum_duplicates()
    return A


def _diffusion_advection_csr(n):
    """7-point diffusion + small skew advection on 3D grid.
    Diagonal positive; off-diagonals slightly asymmetric (~1e-4).
    Mimics the LPBF T_corr0 matrix shape."""
    rng = np.random.default_rng(SEED + 1)
    rows, cols, data = [], [], []
    def idx(i, j, k):
        return (i * n + j) * n + k

    advect_eps = 1e-4  # asymmetry magnitude
    for i in range(n):
        for j in range(n):
            for k in range(n):
                p = idx(i, j, k)
                diag = 0.0
                # x-direction
                for sign, di in [(-1, -1), (+1, +1)]:
                    ii = i + di
                    if 0 <= ii < n:
                        q = idx(ii, j, k)
                        coef = 1.0 + sign * advect_eps   # upwind-ish asymmetry
                        rows.append(p); cols.append(q); data.append(-coef)
                        diag += coef
                for sign, dj in [(-1, -1), (+1, +1)]:
                    jj = j + dj
                    if 0 <= jj < n:
                        q = idx(i, jj, k)
                        coef = 1.0 + sign * advect_eps * 0.5
                        rows.append(p); cols.append(q); data.append(-coef)
                        diag += coef
                for dk in [-1, +1]:
                    kk = k + dk
                    if 0 <= kk < n:
                        q = idx(i, j, kk)
                        coef = 1.0
                        rows.append(p); cols.append(q); data.append(-coef)
                        diag += coef
                diag += 0.01  # reaction term for nonsingularity
                rows.append(p); cols.append(p); data.append(diag)
    A = sp.coo_matrix((data, (rows, cols)),
                       shape=(n**3, n**3)).tocsr()
    A.sum_duplicates()
    return A


def _make_pd_fixture(out_path):
    """pd_tiny: SPD Laplacian with density jump, sign-flipped to OpenFOAM
    negative-diagonal convention. Test loader must flip back before AMGx."""
    A_spd = _laplacian_csr_7point(N, density_jump_at=(N // 2, N // 2, N // 2))
    # SPD check
    diag = A_spd.diagonal()
    assert np.all(diag > 0), "pd fixture must be SPD before sign-flip"
    # Sign-flip to mimic OF export convention
    A_of = (-A_spd).tocsr()
    rng = np.random.default_rng(SEED + 100)
    b_of = rng.standard_normal(N**3).astype(np.float64)
    # Truth on the SPD form: A_spd x = b_spd, where b_spd = -b_of
    x_truth = spla.spsolve(A_spd.tocsc(), -b_of)
    np.savez_compressed(out_path,
                         data=A_of.data, indices=A_of.indices,
                         indptr=A_of.indptr, shape=np.array(A_of.shape),
                         b=b_of, x_truth=x_truth,
                         spd_after_flip=np.array([1], dtype=np.int8))
    print(f"wrote {out_path}: n={N**3} nnz={A_of.nnz} "
          f"diag_mean={A_of.diagonal().mean():.3e} (OF sign)")


def _make_T_fixture(out_path):
    """T_tiny: near-symmetric diffusion-advection, diagonal POSITIVE,
    no sign-flip needed."""
    A = _diffusion_advection_csr(N)
    asym = (A - A.T)
    asym_inf = float(np.abs(asym.data).max()) if asym.nnz else 0.0
    diag = A.diagonal()
    assert np.all(diag > 0), "T fixture should have positive diagonal"
    rng = np.random.default_rng(SEED + 200)
    b = rng.standard_normal(N**3).astype(np.float64)
    x_truth = spla.spsolve(A.tocsc(), b)
    np.savez_compressed(out_path,
                         data=A.data, indices=A.indices,
                         indptr=A.indptr, shape=np.array(A.shape),
                         b=b, x_truth=x_truth,
                         spd_after_flip=np.array([0], dtype=np.int8))
    print(f"wrote {out_path}: n={N**3} nnz={A.nnz} "
          f"diag_mean={diag.mean():.3e} asymmetry_max={asym_inf:.3e}")


def main():
    _make_pd_fixture(HERE / "pd_tiny.npz")
    _make_T_fixture(HERE / "T_tiny.npz")


if __name__ == "__main__":
    main()
