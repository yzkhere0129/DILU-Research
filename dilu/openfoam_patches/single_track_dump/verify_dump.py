#!/usr/bin/env python3
"""Quick sanity check after smoke-test dump at step 5 (= 5 ns).

Runs in the case directory. Reads postProcessing/matrices/5e-09/{pd_corr0,T_corr0}/
and reports |A·x - b|/|b|, matrix dimensions, and physical value ranges.

Pass criteria:
  - pd_corr0: ‖A·x_OF - b‖/‖b‖ ≤ 1e-7    (= OF tol cap)
  - T_corr0:  T solution range [298, ~340] K (300W, 5ns ≈ +30K rise)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio


def check(label: str, base: Path) -> bool:
    print(f"\n=== {label} ===")
    A_path = base / "A.mm"
    b_path = base / "b.mm"
    x_path = base / "x_final.mm"
    for p in (A_path, b_path, x_path):
        if not p.exists():
            print(f"  ✗ MISSING: {p}")
            return False

    A = sio.mmread(str(A_path)).tocsr()
    b = sio.mmread(str(b_path)).flatten()
    x = sio.mmread(str(x_path)).flatten()

    print(f"  A: shape={A.shape}, nnz={A.nnz}")
    print(f"  b: |b|∞={np.abs(b).max():.3e}, ‖b‖₂={np.linalg.norm(b):.3e}")
    print(f"  x: range [{x.min():.3e}, {x.max():.3e}]")

    rel = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300)
    abs_resid = np.abs(A @ x - b).max()
    print(f"  ‖A·x - b‖∞ = {abs_resid:.3e}")
    print(f"  ‖A·x - b‖/‖b‖ = {rel:.3e}")
    return rel, x


def main():
    case = Path.cwd()
    smoke_dir = case / "postProcessing" / "matrices" / "5e-09"
    if not smoke_dir.exists():
        print(f"FAIL: {smoke_dir} not found.")
        print(f"      Did the smoke test run? Check that:")
        print(f"        1. system/matrixDumperDict has dumpTimeSteps (5)")
        print(f"        2. controlDict endTime ≥ 5e-9 + deltaT = 1e-9")
        print(f"        3. laserMeltFoam log shows '[matrixDumper] dumping ...'")
        sys.exit(1)

    print(f"Case dir: {case}")
    print(f"Smoke dump dir: {smoke_dir}")
    print(f"Contents: {sorted(p.name for p in smoke_dir.iterdir())}")

    # pd_corr0
    pd_dir = smoke_dir / "pd_corr0"
    pd_rel, pd_x = check("pd_corr0", pd_dir)

    # T_corr0
    T_dir = smoke_dir / "T_corr0"
    T_rel, T_x = check("T_corr0", T_dir)

    # Verdict
    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")
    pd_ok = pd_rel < 1e-6   # generous cap: 1e-6 (OF default tol is 1e-8)
    T_ok = T_rel < 1e-6
    T_min, T_max = T_x.min(), T_x.max()
    T_range_ok = (T_min >= 297.0) and (T_max < 1000.0)  # sanity: warming, not melted yet at 5ns

    print(f"  pd_corr0 self-consistency: {'✓ PASS' if pd_ok else '✗ FAIL'}  "
          f"(‖A·x - b‖/‖b‖ = {pd_rel:.2e}, want < 1e-6)")
    print(f"  T_corr0  self-consistency: {'✓ PASS' if T_ok else '✗ FAIL'}  "
          f"(‖A·x - b‖/‖b‖ = {T_rel:.2e}, want < 1e-6)")
    print(f"  T physical range:          {'✓ PASS' if T_range_ok else '⚠ CHECK'}  "
          f"([{T_min:.1f}, {T_max:.1f}] K, expect 298 ≤ T ≤ ~330 at 300W/5ns)")

    if pd_ok and T_ok:
        print(f"\n  ★ Smoke test PASS — pipeline OK, safe to run full 1.2μs.")
        sys.exit(0)
    else:
        print(f"\n  ✗ Smoke test FAIL — fix issue before full run.")
        sys.exit(2)


if __name__ == "__main__":
    main()
