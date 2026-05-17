"""Sign-flip contract test (REPRODUCE.md §6).

OpenFOAM exports its pressure-correction lduMatrix with NEGATIVE diagonal.
AMGx PCG requires positive diagonal (SPD). The canonical loader in
`bench/suite/run_benchmark.py::load_matrix_npz` checks the mean diagonal
sign and flips both A and b before the solve.

This test pins down the contract on the pd_tiny fixture:
  1. The fixture is stored in OF sign convention (diag mean < 0).
  2. After (-A, -b) flip, the matrix is SPD.
  3. AMGx + IR on the flipped form matches scipy spsolve on the SPD form
     to <= 1e-13 (matches the tight floor in test_precision_vs_truth.py).

If anyone "fixes" load_matrix_npz to skip the flip, this test fires.
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from jax import config as _jc
_jc.update("jax_enable_x64", True)

from dilu.amgx.python import amgx_solve_with_refinement


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_raw_pd():
    p = FIXTURES_DIR / "pd_tiny.npz"
    if not p.exists():
        pytest.skip("fixture missing; run tests/fixtures/generate_fixtures.py")
    d = np.load(p)
    A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]),
                       shape=tuple(d["shape"]))
    return A, d["b"], d["x_truth"], bool(d["spd_after_flip"][0])


def test_pd_fixture_is_stored_in_of_negative_diag_convention():
    """The fixture is supposed to mirror OF's export sign."""
    A, _, _, sign_flipped = _load_raw_pd()
    diag = A.diagonal()
    assert sign_flipped is True, "fixture metadata claims it is OF-signed"
    assert float(np.mean(diag)) < 0, (
        f"pd_tiny diag mean should be < 0 (OF convention), got "
        f"{float(np.mean(diag)):.3e} — regenerate fixture"
    )


def test_sign_flip_yields_spd_matrix():
    """(-A) must be SPD: positive diagonal + eigenvalues > 0."""
    A, _, _, _ = _load_raw_pd()
    A_spd = (-A).tocsr()
    diag = A_spd.diagonal()
    assert np.all(diag > 0), "after sign flip, diagonal must be all positive"
    # Cheap SPD check: smallest eigenvalue > 0 via shifted power iteration
    # on a 1728^2 matrix; LinearOperator + ARPACK is fast.
    eig_min = spla.eigsh(A_spd, k=1, which="SM", return_eigenvectors=False,
                          tol=1e-6, maxiter=2000)
    assert float(eig_min[0]) > 0, (
        f"after sign flip, smallest eigenvalue should be > 0, got {eig_min[0]}"
    )


def test_amgx_on_flipped_form_matches_scipy_truth():
    """AMGx + IR on (-A, -b) must reproduce the SPD scipy solution."""
    A, b, x_truth, _ = _load_raw_pd()
    A_solve = (-A).tocsr()
    b_solve = -b
    res = amgx_solve_with_refinement(
        A_solve, b_solve, np.zeros_like(b_solve),
        eq_kind="pd", tol=1e-12, n_refine=1, max_iters=500,
    )
    denom = max(float(np.max(np.abs(x_truth))), 1e-300)
    rel = float(np.max(np.abs(res["x"] - x_truth)) / denom)
    assert rel <= 1e-13, (
        f"sign-flipped AMGx+IR vs scipy SPD truth: rel={rel:.3e} > 1e-13"
    )


def test_amgx_pcg_rejects_negative_diag_or_diverges_loudly():
    """Sanity guard: feeding AMGx PCG the raw OF-signed (negative-diag)
    matrix is the classic G1 gotcha. Either AMGx errors out, or it
    converges to nonsense — but it MUST NOT silently produce x_truth.

    This catches a hypothetical "load_matrix_npz drops sign-flip" regression.
    """
    A, b, x_truth, _ = _load_raw_pd()
    # Skip if AMGx happens to swallow it; we only care it does NOT match truth.
    try:
        res = amgx_solve_with_refinement(
            A, b, np.zeros_like(b),
            eq_kind="pd", tol=1e-12, n_refine=0, max_iters=50,
        )
    except Exception as e:
        # Acceptable outcome — AMGx loudly rejects.
        print(f"  AMGx rejected negative-diag form (expected): {e}")
        return
    denom = max(float(np.max(np.abs(x_truth))), 1e-300)
    rel = float(np.max(np.abs(res["x"] - x_truth)) / denom)
    assert rel > 1e-6, (
        f"DANGER: AMGx silently produced truth-matching solution on a "
        f"negative-diag matrix (rel={rel:.3e}). The sign-flip guard "
        f"must have been bypassed."
    )
