"""Regression test: AMGx + IR achieves machine precision vs scipy spsolve truth.

Two modes:

  1. Fixture mode (always runs): uses self-contained 12x12x12 synthetic
     matrices under `tests/fixtures/{pd,T}_tiny.npz`. No external data
     needed. Acceptance: rel_vs_truth <= 1e-10 (headline ~1e-15).

  2. LaserbeamFoam mode (env-gated): runs against real LPBF dumps when
     `DILU_AMGX_LMF_ROOT` is set, e.g.
         export DILU_AMGX_LMF_ROOT=/path/to/LaserbeamFoam
     and the cross-phase reader `dilu.benchmark.openfoam_crosscheck.reader`
     is importable. Tests are silently skipped otherwise.

Run: pytest dilu/amgx/tests/test_precision_vs_truth.py -v
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
from scipy.sparse.linalg import spsolve

from jax import config as _jc
_jc.update("jax_enable_x64", True)

from dilu.amgx.python import amgx_solve_with_refinement


REL_VS_TRUTH_THRESHOLD = 1e-10  # user spec; actual achieved is ~1e-15
REL_VS_TRUTH_TIGHT     = 1e-13  # regression guard; catches drops from ~1e-15

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Mode 1: self-contained fixture tests (always run)
# ---------------------------------------------------------------------------


def _load_fixture(name):
    """Load a tests/fixtures/{name}_tiny.npz fixture.

    Returns (A_csr, b, x_truth, sign_flipped_bool). For pd_tiny the
    matrix is stored in OpenFOAM negative-diagonal convention, so the
    AMGx wrapper's load helper must sign-flip — we do the same here
    inside `_solve_and_check` to keep this test self-contained.
    """
    p = FIXTURES_DIR / f"{name}_tiny.npz"
    if not p.exists():
        pytest.skip(f"fixture not generated: {p}; "
                     f"run dilu/amgx/tests/fixtures/generate_fixtures.py")
    d = np.load(p)
    A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]),
                       shape=tuple(d["shape"]))
    b = d["b"]
    x_truth = d["x_truth"]
    sign_flipped = bool(d["spd_after_flip"][0])
    return A, b, x_truth, sign_flipped


def _solve_and_check(name, eq_kind, threshold, n_refine):
    A, b, x_truth, sign_flipped = _load_fixture(name)
    if sign_flipped:
        # Mimic the production loader (bench/suite/run_benchmark.py).
        A_solve = (-A).tocsr()
        b_solve = -b
    else:
        A_solve, b_solve = A, b
    x0 = np.zeros_like(b_solve)
    res = amgx_solve_with_refinement(
        A_solve, b_solve, x0,
        eq_kind=eq_kind, tol=1e-12, n_refine=n_refine, max_iters=500,
    )
    denom = max(float(np.max(np.abs(x_truth))), 1e-300)
    rel = float(np.max(np.abs(res["x"] - x_truth)) / denom)
    print(f"  fixture {name} ({eq_kind}): rel_vs_truth={rel:.3e}, "
          f"primary_iters={res['primary_iters']}, refine_iters={res['refine_iters']}")
    assert rel <= threshold, f"{name} {eq_kind} fixture: rel={rel:.3e} > {threshold}"


def test_fixture_pd_amgx_ir_meets_user_spec():
    """pd fixture: AMGx + 1 IR step reaches user-spec 1e-10 vs scipy."""
    _solve_and_check("pd", "pd", REL_VS_TRUTH_THRESHOLD, n_refine=1)


def test_fixture_T_amgx_ir_meets_user_spec():
    """T fixture: AMGx + 1 IR step reaches user-spec 1e-10 vs scipy."""
    _solve_and_check("T", "T", REL_VS_TRUTH_THRESHOLD, n_refine=1)


def test_fixture_pd_amgx_ir_meets_tight_floor():
    """pd fixture tight: regression guard at 1e-13."""
    _solve_and_check("pd", "pd", REL_VS_TRUTH_TIGHT, n_refine=1)


def test_fixture_T_amgx_ir_meets_tight_floor():
    """T fixture tight: regression guard at 1e-13."""
    _solve_and_check("T", "T", REL_VS_TRUTH_TIGHT, n_refine=1)


# ---------------------------------------------------------------------------
# Mode 2: LaserbeamFoam-backed tests (env-gated)
# ---------------------------------------------------------------------------

_LMF_ROOT = os.environ.get("DILU_AMGX_LMF_ROOT")


def _lmf_case(case_dir: str, ts: str, eq: str) -> Path:
    """Resolve a LaserbeamFoam matrix dump path under $DILU_AMGX_LMF_ROOT."""
    if _LMF_ROOT is None:
        return Path("/__unset__/__skip__")
    return Path(_LMF_ROOT) / "tutorials" / "laserMeltFoam" / case_dir / \
        "postProcessing" / "matrices" / ts / f"{eq}_corr0"


# Try to import the cross-phase reader; if unavailable, the fixture tests
# above still run and the LMF-mode tests will skip individually at runtime.
# This is intentionally NOT pytest.importorskip — that would skip the whole
# module including the self-contained fixture tests.
try:
    from dilu.benchmark.openfoam_crosscheck.reader import load_ofmm, normalize_sign  # noqa
    _HAS_LMF_READER = True
except ImportError:
    _HAS_LMF_READER = False
    load_ofmm = normalize_sign = None  # placeholders for type checkers


LMF_CASES = [
    ("pd", _lmf_case("dumper_pipeline_test", "2.64e-12",         "pd")),
    ("T",  _lmf_case("dumper_pipeline_test", "2.64e-12",         "T")),
    ("pd", _lmf_case("LPBF_sanity",          "2.636507509e-12",  "pd")),
    ("T",  _lmf_case("LPBF_sanity",          "2.636507509e-12",  "T")),
]


@pytest.mark.parametrize("eq_kind,matrix_dir", LMF_CASES)
def test_amgx_with_refinement_reaches_machine_precision(eq_kind, matrix_dir):
    """AMGx + 1 IR step <= 1e-10 vs scipy spsolve truth (typically ~1e-15)."""
    if not _HAS_LMF_READER:
        pytest.skip("dilu.benchmark reader unavailable (standalone packaging)")
    if _LMF_ROOT is None or not matrix_dir.exists():
        pytest.skip(f"DILU_AMGX_LMF_ROOT unset or dump absent: {matrix_dir}")

    bundle = load_ofmm(matrix_dir)
    bundle = normalize_sign(bundle)
    A, b, x0 = bundle.A, bundle.b, bundle.x0

    x_truth = spsolve(A.tocsc(), b)
    denom = max(float(np.max(np.abs(x_truth))), 1e-300)

    res = amgx_solve_with_refinement(
        A, b, x0,
        eq_kind=eq_kind, tol=1e-12, n_refine=1, max_iters=500,
    )

    rel_vs_truth = float(np.max(np.abs(res["x"] - x_truth)) / denom)
    print(f"  [{eq_kind}] {matrix_dir.parent.name} N={A.shape[0]}: "
          f"rel_vs_truth={rel_vs_truth:.3e}, primary_iters={res['primary_iters']}, "
          f"refine_iters={res['refine_iters']}")

    assert rel_vs_truth <= REL_VS_TRUTH_THRESHOLD, (
        f"AMGx+IR failed precision target: rel_vs_truth={rel_vs_truth:.3e} "
        f"> threshold {REL_VS_TRUTH_THRESHOLD}"
    )


@pytest.mark.parametrize("eq_kind,matrix_dir", LMF_CASES)
def test_amgx_with_refinement_meets_tight_floor(eq_kind, matrix_dir):
    """Tight regression guard: AMGx + 1 IR step <= 1e-13 vs scipy truth.

    Separate from the user-spec 1e-10 gate so that a regression from the
    achieved ~1e-15 down to ~1e-12 (still within user spec but worse) is
    caught in CI.  Same underlying solve as the previous test, only the
    threshold differs.
    """
    if not _HAS_LMF_READER:
        pytest.skip("dilu.benchmark reader unavailable (standalone packaging)")
    if _LMF_ROOT is None or not matrix_dir.exists():
        pytest.skip(f"DILU_AMGX_LMF_ROOT unset or dump absent: {matrix_dir}")

    bundle = load_ofmm(matrix_dir)
    bundle = normalize_sign(bundle)
    A, b, x0 = bundle.A, bundle.b, bundle.x0

    x_truth = spsolve(A.tocsc(), b)
    denom = max(float(np.max(np.abs(x_truth))), 1e-300)

    res = amgx_solve_with_refinement(
        A, b, x0,
        eq_kind=eq_kind, tol=1e-12, n_refine=1, max_iters=500,
    )

    rel_vs_truth = float(np.max(np.abs(res["x"] - x_truth)) / denom)
    print(f"  [{eq_kind}] {matrix_dir.parent.name}: tight rel_vs_truth={rel_vs_truth:.3e}")
    assert rel_vs_truth <= REL_VS_TRUTH_TIGHT, (
        f"REGRESSION: rel_vs_truth={rel_vs_truth:.3e} "
        f"> tight floor {REL_VS_TRUTH_TIGHT} "
        f"(headline expected ~1e-15)"
    )


def test_no_refinement_still_meets_1e10_on_easy_cases():
    """Even without IR, AMGx tol=1e-12 should reach 1e-10 on most cases."""
    p = _lmf_case("dumper_pipeline_test", "2.64e-12", "T")
    if not _HAS_LMF_READER:
        pytest.skip("dilu.benchmark reader unavailable (standalone packaging)")
    if _LMF_ROOT is None or not p.exists():
        pytest.skip(f"DILU_AMGX_LMF_ROOT unset or dump absent: {p}")

    bundle = load_ofmm(p)
    bundle = normalize_sign(bundle)
    A, b, x0 = bundle.A, bundle.b, bundle.x0

    x_truth = spsolve(A.tocsc(), b)
    denom = max(float(np.max(np.abs(x_truth))), 1e-300)

    res = amgx_solve_with_refinement(
        A, b, x0,
        eq_kind="T", tol=1e-12, n_refine=0, max_iters=500,
    )
    rel = float(np.max(np.abs(res["x"] - x_truth)) / denom)
    assert rel <= REL_VS_TRUTH_THRESHOLD, (
        f"T without IR failed: {rel:.3e}"
    )


# Lazy resolution: re-evaluate when the test runs, not at collection
def _sanity_pd():
    return _lmf_case("LPBF_sanity", "2.636507509e-12", "pd")


def _sanity_T():
    return _lmf_case("LPBF_sanity", "2.636507509e-12", "T")


def test_n_refine_5_does_not_regress_accuracy():
    """n_refine=5 should not make accuracy worse than n_refine=1."""
    p = _sanity_pd()
    if not _HAS_LMF_READER:
        pytest.skip("dilu.benchmark reader unavailable (standalone packaging)")
    if _LMF_ROOT is None or not p.exists():
        pytest.skip(f"DILU_AMGX_LMF_ROOT unset or dump absent: {p}")

    bundle = load_ofmm(p)
    bundle = normalize_sign(bundle)
    A, b, x0 = bundle.A, bundle.b, bundle.x0

    x_truth = spsolve(A.tocsc(), b)
    denom = max(float(np.max(np.abs(x_truth))), 1e-300)

    res = amgx_solve_with_refinement(
        A, b, x0,
        eq_kind="pd", tol=1e-12, n_refine=5, max_iters=500,
    )

    assert len(res["refine_iters"]) == 5, (
        f"Expected 5 refine_iters entries, got {len(res['refine_iters'])}"
    )
    rel = float(np.max(np.abs(res["x"] - x_truth)) / denom)
    print(f"  n_refine=5 rel_vs_truth={rel:.3e}")
    assert rel <= REL_VS_TRUTH_THRESHOLD, (
        f"n_refine=5 regressed accuracy: rel={rel:.3e}"
    )


def test_zero_rhs_returns_zero_solution():
    """When b=0 the exact solution is x=0; AMGx+IR must return near-zero."""
    p = _sanity_pd()
    if not _HAS_LMF_READER:
        pytest.skip("dilu.benchmark reader unavailable (standalone packaging)")
    if _LMF_ROOT is None or not p.exists():
        pytest.skip(f"DILU_AMGX_LMF_ROOT unset or dump absent: {p}")

    bundle = load_ofmm(p)
    bundle = normalize_sign(bundle)
    A = bundle.A
    n = A.shape[0]
    b_zero = np.zeros(n)
    x0_zero = np.zeros(n)

    res = amgx_solve_with_refinement(
        A, b_zero, x0_zero,
        eq_kind="pd", tol=1e-12, n_refine=1, max_iters=500,
    )

    abs_err = float(np.max(np.abs(res["x"])))
    A_scale = float(np.abs(A.data).max())
    print(f"  zero-rhs abs_err={abs_err:.3e}, A_max={A_scale:.3e}")
    assert abs_err <= 1e-10 * A_scale, (
        f"zero-rhs returned non-zero solution: abs_err={abs_err:.3e}"
    )


def test_eq_kind_T_config_selects_bicgstab():
    """eq_kind='T' must select BICGSTAB config, not PCG.

    Pure-Python config-routing test; no external data, no GPU. Always runs.
    """
    import json
    from dilu.amgx.python.config import (
        CLASSICAL_V_DIAGSCALED,
        CLASSICAL_V_DIAGSCALED_BICGSTAB,
        with_tolerance,
    )

    cfg_pd = json.loads(with_tolerance(CLASSICAL_V_DIAGSCALED, 1e-12))
    cfg_T  = json.loads(with_tolerance(CLASSICAL_V_DIAGSCALED_BICGSTAB, 1e-12))

    assert cfg_pd["solver"]["solver"] == "PCG", (
        f"pd config should use PCG, got {cfg_pd['solver']['solver']!r}"
    )
    assert cfg_T["solver"]["solver"] == "BICGSTAB", (
        f"T config should use BICGSTAB, got {cfg_T['solver']['solver']!r}"
    )


def test_plan_released_after_with_block():
    """Plan._released must be True after the with-block exits (no leak)."""
    p = _sanity_T()
    if not _HAS_LMF_READER:
        pytest.skip("dilu.benchmark reader unavailable (standalone packaging)")
    if _LMF_ROOT is None or not p.exists():
        pytest.skip(f"DILU_AMGX_LMF_ROOT unset or dump absent: {p}")

    import jax.numpy as jnp
    from dilu.amgx.python.plan import Plan
    from dilu.amgx.python.config import CLASSICAL_V_DIAGSCALED_BICGSTAB, with_tolerance

    bundle = load_ofmm(p)
    bundle = normalize_sign(bundle)
    A = bundle.A

    rp = jnp.asarray(A.indptr.astype(np.int32))
    ci = jnp.asarray(A.indices.astype(np.int32))
    vv = jnp.asarray(A.data.astype(np.float64))
    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED_BICGSTAB, 1e-12)

    plan_ref = None
    with Plan(rp, ci, vv, cfg) as plan:
        plan_ref = plan
        assert not plan._released, "Plan should not be released inside with-block"

    assert plan_ref._released, (
        "Plan._released must be True after __exit__; context manager leak detected"
    )
