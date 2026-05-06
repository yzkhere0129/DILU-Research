"""Byte-exact unit test: C++ kernel outputs == numba @njit Python outputs.

For each senior pd matrix (initial + evaporation, 78 total), compare
  amul, tmul, sum_a, calc_reciprocal_d, precondition, precondition_t,
  sum_abs, dot
output ULP=0 against the python reference. ULP=0 here means
np.array_equal — every byte identical.

Acceptance: every kernel passes on every matrix. Plan §1.4 gate.

Run:
    /home/yzk/jax-env/bin/python -m dilu.openfoam_cpu.tests.test_cpp_byte_exact
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from dilu.openfoam_cpu.python import _kernels_cpp as cpp
from dilu.openfoam_cpu.python.dilu import (
    calc_reciprocal_d as py_calc_rd,
    precondition as py_precondition,
    precondition_t as py_precondition_t,
)
from dilu.openfoam_cpu.python.ldu import LDU, csr_to_ldu, calc_losort
from dilu.openfoam_cpu.python.spmv import amul as py_amul, tmul as py_tmul, sum_a as py_sum_a
from dilu.openfoam_cpu.python.reductions import sum_abs as py_sum_abs, dot as py_dot

from dilu.amgx.bench.senior_data_loader import set_dataset, list_available, load_step


def _check(name: str, expected: np.ndarray, got: np.ndarray, results: dict):
    """Compare two arrays and record byte-exact + tolerance metrics."""
    expected = np.asarray(expected, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    if expected.shape != got.shape:
        results[name] = dict(byte_exact=False,
                             max_abs=float("inf"), max_rel=float("inf"),
                             error=f"shape mismatch {expected.shape} vs {got.shape}")
        return
    byte_exact = bool(np.array_equal(expected.tobytes(), got.tobytes()))
    diff = np.abs(expected - got)
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = max(float(np.abs(expected).max()), 1e-300)
    max_rel = max_abs / denom
    results[name] = dict(byte_exact=byte_exact,
                         max_abs=max_abs, max_rel=max_rel)


def _check_scalar(name: str, expected: float, got: float, results: dict):
    bytes_eq = (np.float64(expected).tobytes() == np.float64(got).tobytes())
    abs_diff = abs(expected - got)
    rel_diff = abs_diff / max(abs(expected), 1e-300)
    results[name] = dict(byte_exact=bytes_eq,
                         max_abs=abs_diff, max_rel=rel_diff,
                         expected=expected, got=got)


def run_one(ldu: LDU, label: str) -> dict:
    """Run all 8 kernels, return per-kernel byte-exact results."""
    rng = np.random.default_rng(0xDEADBEEF + ldu.n_cells)
    psi = rng.standard_normal(ldu.n_cells)
    rA = rng.standard_normal(ldu.n_cells)
    rT = rng.standard_normal(ldu.n_cells)
    losort = calc_losort(ldu.neighbour)

    res = {}

    # amul
    expected = py_amul(ldu, psi)
    got = cpp.amul(ldu.diag, ldu.lower, ldu.upper, ldu.owner, ldu.neighbour, psi)
    _check("amul", expected, got, res)

    # tmul
    expected = py_tmul(ldu, psi)
    got = cpp.tmul(ldu.diag, ldu.lower, ldu.upper, ldu.owner, ldu.neighbour, psi)
    _check("tmul", expected, got, res)

    # sum_a
    expected = py_sum_a(ldu)
    got = cpp.sum_a(ldu.diag, ldu.lower, ldu.upper, ldu.owner, ldu.neighbour)
    _check("sum_a", expected, got, res)

    # calc_reciprocal_d
    expected = py_calc_rd(ldu)
    got = cpp.calc_reciprocal_d(ldu.diag, ldu.lower, ldu.upper, ldu.owner, ldu.neighbour)
    _check("calc_rD", expected, got, res)
    rD_py = expected  # use the trusted output downstream

    # precondition
    expected = py_precondition(ldu, rD_py, rA)
    got = cpp.precondition(ldu.diag, ldu.lower, ldu.upper, ldu.owner, ldu.neighbour,
                            rD_py, rA)
    _check("precondition", expected, got, res)

    # precondition_t
    expected = py_precondition_t(ldu, rD_py, losort, rT)
    got = cpp.precondition_t(ldu.diag, ldu.lower, ldu.upper, ldu.owner, ldu.neighbour,
                              rD_py, losort, rT)
    _check("precondition_t", expected, got, res)

    # sum_abs / dot
    _check_scalar("sum_abs", py_sum_abs(rA), cpp.sum_abs(rA), res)
    _check_scalar("dot",     py_dot(rA, rT), cpp.dot(rA, rT), res)

    return res


def main() -> int:
    print("Byte-exact comparison: C++ kernels vs numba @njit Python\n")

    cases = []
    for ds in ("initial", "evaporation"):
        set_dataset(ds)
        for step, corr in list_available():
            cases.append((ds, step, corr))
    print(f"Loaded {len(cases)} cases (initial + evaporation)\n")

    kernels = ["amul", "tmul", "sum_a", "calc_rD",
               "precondition", "precondition_t", "sum_abs", "dot"]

    fail_total = 0
    n_tested = 0
    summary = {k: dict(pass_=0, fail_=0,
                       max_abs=0.0, max_rel=0.0,
                       worst_case=None) for k in kernels}

    t0 = time.time()
    for ds, step, corr in cases:
        set_dataset(ds)
        bundle = load_step(step, corr)
        ldu = csr_to_ldu(bundle.A)
        try:
            res = run_one(ldu, f"{ds}/{step}/{corr}")
        except Exception as e:
            print(f"  {ds}/{step}/{corr}: ERROR {e}")
            fail_total += 1
            continue
        n_tested += 1
        case_label = f"{ds}/{step:>3d}/{corr}"
        any_fail = False
        for k in kernels:
            r = res[k]
            if r["byte_exact"]:
                summary[k]["pass_"] += 1
            else:
                summary[k]["fail_"] += 1
                fail_total += 1
                any_fail = True
                if r["max_abs"] > summary[k]["max_abs"]:
                    summary[k]["max_abs"] = r["max_abs"]
                    summary[k]["max_rel"] = r["max_rel"]
                    summary[k]["worst_case"] = case_label
        if any_fail:
            parts = []
            for k in kernels:
                r = res[k]
                tag = "OK" if r["byte_exact"] else f"rel={r['max_rel']:.2e}"
                parts.append(f"{k}:{tag}")
            print(f"  {case_label} — " + "  ".join(parts))

    elapsed = time.time() - t0

    print()
    print("=" * 78)
    print(f"{'kernel':16s} {'pass':>6s} {'fail':>6s} {'max_abs':>12s} {'max_rel':>12s}  worst case")
    print("-" * 78)
    for k in kernels:
        s = summary[k]
        wc = s["worst_case"] or "-"
        print(f"{k:16s} {s['pass_']:>6d} {s['fail_']:>6d} "
              f"{s['max_abs']:>12.3e} {s['max_rel']:>12.3e}  {wc}")
    print("=" * 78)
    print(f"Total: {n_tested} cases tested in {elapsed:.1f}s, {fail_total} kernel failures")

    return 0 if fail_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
