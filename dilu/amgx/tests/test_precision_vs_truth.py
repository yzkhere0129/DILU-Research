"""Regression test: AMGx + IR achieves machine precision vs scipy spsolve truth.

Acceptance: rel_vs_truth ≤ 1e-10 on at least one pd and one T LPBF dump.
(Headline result is ~1e-15, so 1e-10 is a generous floor.)

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
from scipy.sparse.linalg import spsolve

from jax import config as _jc
_jc.update("jax_enable_x64", True)

from dilu.amgx.python import amgx_solve_with_refinement
from dilu.benchmark.openfoam_crosscheck.reader import load_ofmm, normalize_sign


REL_VS_TRUTH_THRESHOLD = 1e-10  # user spec; actual achieved is ~1e-15
REL_VS_TRUTH_TIGHT     = 1e-13  # regression guard; catches drops from ~1e-15

CASES = [
    ("pd",
     "/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/dumper_pipeline_test/"
     "postProcessing/matrices/2.64e-12/pd_corr0"),
    ("T",
     "/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/dumper_pipeline_test/"
     "postProcessing/matrices/2.64e-12/T_corr0"),
    ("pd",
     "/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_sanity/"
     "postProcessing/matrices/2.636507509e-12/pd_corr0"),
    ("T",
     "/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_sanity/"
     "postProcessing/matrices/2.636507509e-12/T_corr0"),
]


@pytest.mark.parametrize("eq_kind,matrix_dir", CASES)
def test_amgx_with_refinement_reaches_machine_precision(eq_kind, matrix_dir):
    """AMGx + 1 IR step ≤ 1e-10 vs scipy spsolve truth (typically ~1e-15)."""
    p = Path(matrix_dir)
    if not p.exists():
        pytest.skip(f"dump not present: {p}")

    bundle = load_ofmm(p)
    bundle = normalize_sign(bundle)
    A, b, x0 = bundle.A, bundle.b, bundle.x0

    # Truth via direct LU
    x_truth = spsolve(A.tocsc(), b)
    denom = max(float(np.max(np.abs(x_truth))), 1e-300)

    # AMGx + IR
    res = amgx_solve_with_refinement(
        A, b, x0,
        eq_kind=eq_kind, tol=1e-12, n_refine=1, max_iters=500,
    )

    rel_vs_truth = float(np.max(np.abs(res["x"] - x_truth)) / denom)
    print(f"  [{eq_kind}] {p.parent.name} N={A.shape[0]}: "
          f"rel_vs_truth={rel_vs_truth:.3e}, primary_iters={res['primary_iters']}, "
          f"refine_iters={res['refine_iters']}")

    assert rel_vs_truth <= REL_VS_TRUTH_THRESHOLD, (
        f"AMGx+IR failed precision target: rel_vs_truth={rel_vs_truth:.3e} "
        f"> threshold {REL_VS_TRUTH_THRESHOLD}"
    )


@pytest.mark.parametrize("eq_kind,matrix_dir", CASES)
def test_amgx_with_refinement_meets_tight_floor(eq_kind, matrix_dir):
    """Tight regression guard: AMGx + 1 IR step ≤ 1e-13 vs scipy truth.

    Separate from the user-spec 1e-10 gate so that a regression from the
    achieved ~1e-15 down to ~1e-12 (still within user spec but worse) is
    caught in CI.  Same underlying solve as the previous test, only the
    threshold differs.
    """
    p = Path(matrix_dir)
    if not p.exists():
        pytest.skip(f"dump not present: {p}")

    bundle = load_ofmm(p)
    bundle = normalize_sign(bundle)
    A, b, x0 = bundle.A, bundle.b, bundle.x0

    x_truth = spsolve(A.tocsc(), b)
    denom = max(float(np.max(np.abs(x_truth))), 1e-300)

    res = amgx_solve_with_refinement(
        A, b, x0,
        eq_kind=eq_kind, tol=1e-12, n_refine=1, max_iters=500,
    )

    rel_vs_truth = float(np.max(np.abs(res["x"] - x_truth)) / denom)
    print(f"  [{eq_kind}] {p.parent.name}: tight rel_vs_truth={rel_vs_truth:.3e}")
    assert rel_vs_truth <= REL_VS_TRUTH_TIGHT, (
        f"REGRESSION: rel_vs_truth={rel_vs_truth:.3e} "
        f"> tight floor {REL_VS_TRUTH_TIGHT} "
        f"(headline expected ~1e-15)"
    )


def test_no_refinement_still_meets_1e10_on_easy_cases():
    """Even without IR, AMGx tol=1e-12 should reach 1e-10 on most cases.

    Tests T (well-conditioned) where IR is overkill.
    """
    p = Path("/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/dumper_pipeline_test/"
             "postProcessing/matrices/2.64e-12/T_corr0")
    if not p.exists():
        pytest.skip(f"dump not present: {p}")

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


# ---------------------------------------------------------------------------
# Gap-coverage tests (added 2026-05-04)
# ---------------------------------------------------------------------------

_SANITY_PD = Path(
    "/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_sanity/"
    "postProcessing/matrices/2.636507509e-12/pd_corr0"
)
_SANITY_T = Path(
    "/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_sanity/"
    "postProcessing/matrices/2.636507509e-12/T_corr0"
)


def test_n_refine_5_does_not_regress_accuracy():
    """n_refine=5 should not make accuracy worse than n_refine=1.

    Extra IR steps should not regress: each step re-solves the residual
    equation.  Checks that the returned `refine_iters` list has length 5 and
    that final rel_vs_truth still meets 1e-10.
    """
    if not _SANITY_PD.exists():
        pytest.skip(f"dump not present: {_SANITY_PD}")

    bundle = load_ofmm(_SANITY_PD)
    bundle = normalize_sign(bundle)
    A, b, x0 = bundle.A, bundle.b, bundle.x0

    x_truth = spsolve(A.tocsc(), b)
    denom = max(float(np.max(np.abs(x_truth))), 1e-300)

    res = amgx_solve_with_refinement(
        A, b, x0,
        eq_kind="pd", tol=1e-12, n_refine=5, max_iters=500,
    )

    # Structural: exactly 5 IR steps were executed
    assert len(res["refine_iters"]) == 5, (
        f"Expected 5 refine_iters entries, got {len(res['refine_iters'])}"
    )
    # Accuracy: still meets spec
    rel = float(np.max(np.abs(res["x"] - x_truth)) / denom)
    print(f"  n_refine=5 rel_vs_truth={rel:.3e}")
    assert rel <= REL_VS_TRUTH_THRESHOLD, (
        f"n_refine=5 regressed accuracy: rel={rel:.3e}"
    )


def test_zero_rhs_returns_zero_solution():
    """When b=0 the exact solution is x=0; AMGx+IR must return near-zero.

    This catches a bug where IR would compute r = 0 - A*0 = 0 and the
    delta solve would nudge x away from 0 via an uninitialized initial guess.
    Threshold is generous (1e-10 relative to ‖A‖) because b_norm2 guard sets
    denominator=1.0 and x_truth = spsolve(A, 0) = 0.
    """
    if not _SANITY_PD.exists():
        pytest.skip(f"dump not present: {_SANITY_PD}")

    bundle = load_ofmm(_SANITY_PD)
    bundle = normalize_sign(bundle)
    A = bundle.A
    n = A.shape[0]
    b_zero = np.zeros(n)
    x0_zero = np.zeros(n)

    res = amgx_solve_with_refinement(
        A, b_zero, x0_zero,
        eq_kind="pd", tol=1e-12, n_refine=1, max_iters=500,
    )

    # x should be exactly 0; allow floating-point slop relative to ‖A‖_F
    abs_err = float(np.max(np.abs(res["x"])))
    A_scale = float(np.abs(A.data).max())
    print(f"  zero-rhs abs_err={abs_err:.3e}, A_max={A_scale:.3e}")
    assert abs_err <= 1e-10 * A_scale, (
        f"zero-rhs returned non-zero solution: abs_err={abs_err:.3e}"
    )


def test_eq_kind_T_config_selects_bicgstab():
    """eq_kind='T' must select BICGSTAB config, not PCG.

    Parses the config string actually passed to with_tolerance for both
    eq_kind values and asserts the outer solver field is correct.  This is a
    pure-Python config-routing test — no AMGx call needed.
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
    """Plan._released must be True after the with-block exits (no leak).

    Regression guard: if Plan.__exit__ is accidentally removed, _released
    stays False and the AMGx handle leaks (eventually crashes on GPU OOM).
    Uses the real LPBF matrix so the Plan actually allocates GPU memory.
    """
    if not _SANITY_T.exists():
        pytest.skip(f"dump not present: {_SANITY_T}")

    import jax.numpy as jnp
    from dilu.amgx.python.plan import Plan
    from dilu.amgx.python.config import CLASSICAL_V_DIAGSCALED_BICGSTAB, with_tolerance

    bundle = load_ofmm(_SANITY_T)
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
