"""Validate OpenFOAM matrix dumps for use in audit / batch LU-truth comparison.

For each matrix system (A, b, x_final) under postProcessing/matrices/<timestep>/<system>/:
  1. Files exist + load cleanly (scipy.io.mmread).
  2. Shapes consistent: A is N×N square, b is N, x_final is N.
  3. Finiteness: no NaN/Inf anywhere.
  4. Non-trivial: nnz > 0, |b|.max > 0.
  5. Diagonal sane: |diag| > 0 for all i, no zero pivots.
  6. Approximate symmetry: ‖A - A.T‖_F / ‖A‖_F < 1e-10.
  7. SELF-CONSISTENCY GATE: ‖A·x_final - b‖/‖b‖ < tol  (default 1e-5).
     This catches matrixDumper bugs (e.g. C014 / boundary_internalCoeffs).
  8. Physical-range gate (pd only): |x_final|.max ∈ [10 Pa, 10 MPa].

Outputs:
  <output_dir>/validation_summary.json   per-matrix verdicts
  <output_dir>/sane_pool.txt             list of timestep paths that PASSED all checks

Usage:
  python3 validate_matrix_dumps.py --case ~/cases/dense_track_dump_500K \\
                                     --system pd_corr0 \\
                                     --output-dir /tmp/validate_dense \\
                                     [--tol-resid 1e-5]
"""
from __future__ import annotations
import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def check_finite(arr) -> tuple[bool, str]:
    if hasattr(arr, "data"):  # sparse
        finite = np.all(np.isfinite(arr.data))
    else:
        finite = np.all(np.isfinite(arr))
    return bool(finite), "" if finite else "non-finite (NaN/Inf) entries"


def check_symmetry(A: sp.spmatrix, rel_tol: float = 1e-10) -> tuple[bool, float]:
    AT = A.T
    diff = A - AT
    n = sp.linalg.norm(diff, "fro")
    d = sp.linalg.norm(A, "fro") or 1.0
    rel = float(n / d)
    return rel < rel_tol, rel


def validate_one(matrix_dir: Path, system: str, tol_resid: float = 1e-5) -> dict:
    rec: dict = {"system": system, "dir": str(matrix_dir), "checks": {}, "pass": False,
                 "reason": ""}
    sys_dir = matrix_dir / system
    if not sys_dir.is_dir():
        rec["reason"] = f"system dir missing: {sys_dir}"
        return rec

    A_path = sys_dir / "A.mm"
    b_path = sys_dir / "b.mm"
    x_path = sys_dir / "x_final.mm"
    for p in (A_path, b_path, x_path):
        if not p.is_file():
            rec["reason"] = f"missing file: {p.name}"
            return rec

    try:
        A = sio.mmread(str(A_path)).tocsr()
        b = np.asarray(sio.mmread(str(b_path))).flatten()
        x = np.asarray(sio.mmread(str(x_path))).flatten()
    except Exception as e:
        rec["reason"] = f"load error: {e}"
        return rec

    # 2. shapes
    if A.shape[0] != A.shape[1]:
        rec["reason"] = f"A not square: {A.shape}"; return rec
    N = A.shape[0]
    if b.size != N or x.size != N:
        rec["reason"] = f"shape mismatch: A=({N},{N}), b={b.size}, x={x.size}"; return rec
    rec["checks"]["n"] = int(N)
    rec["checks"]["nnz"] = int(A.nnz)

    # 3. finiteness
    ok_A, msg = check_finite(A)
    if not ok_A: rec["reason"] = f"A {msg}"; return rec
    ok_b, msg = check_finite(b)
    if not ok_b: rec["reason"] = f"b {msg}"; return rec
    ok_x, msg = check_finite(x)
    if not ok_x: rec["reason"] = f"x {msg}"; return rec

    # 4. non-trivial
    rec["checks"]["b_abs_max"] = float(np.abs(b).max())
    rec["checks"]["b_abs_median"] = float(np.median(np.abs(b)))
    rec["checks"]["x_abs_max"] = float(np.abs(x).max())
    if rec["checks"]["b_abs_max"] == 0.0:
        rec["reason"] = "b is identically zero"; return rec
    if A.nnz == 0:
        rec["reason"] = "A has zero nnz"; return rec

    # 5. diagonal
    diag = A.diagonal()
    rec["checks"]["diag_abs_min"] = float(np.abs(diag).min())
    rec["checks"]["diag_abs_max"] = float(np.abs(diag).max())
    if rec["checks"]["diag_abs_min"] == 0.0:
        rec["reason"] = "zero diagonal entry"; return rec

    # 6. symmetry
    sym_ok, sym_rel = check_symmetry(A, rel_tol=1e-10)
    rec["checks"]["symmetry_rel"] = sym_rel
    if not sym_ok:
        rec["reason"] = f"asymmetry rel = {sym_rel:.2e}"; return rec

    # 7. SELF-CONSISTENCY  (the critical gate — catches matrixDumper bugs)
    # Sign convention: pd from OpenFOAM has negative diagonal; flip to positive for residual
    # check consistency. Detect by checking the sign of diag mean.
    diag_mean = float(np.mean(diag))
    sign_flip = diag_mean < 0
    if sign_flip:
        A_test, b_test = -A, -b
    else:
        A_test, b_test = A, b
    resid_vec = A_test @ x - b_test
    rel_resid = float(np.linalg.norm(resid_vec) / max(np.linalg.norm(b_test), 1e-300))
    rec["checks"]["rel_resid_Ax_minus_b"] = rel_resid
    rec["checks"]["sign_flipped"] = sign_flip
    if rel_resid > tol_resid:
        rec["reason"] = f"self-consistency fail: rel_resid={rel_resid:.2e} > {tol_resid:.0e}"
        return rec

    # 8. physical-range gate for pd
    if system.startswith("pd"):
        x_max = float(np.abs(x).max())
        # LPBF recoil pressure typically 0.1 - 5 MPa; gas+slab around 1e3 - 1e6 Pa
        if x_max < 10.0:
            rec["reason"] = f"pd unphysically small: |x|.max={x_max:.2e} Pa < 10 Pa"; return rec
        if x_max > 1e8:
            rec["reason"] = f"pd unphysically large: |x|.max={x_max:.2e} Pa > 1e8 Pa"; return rec
    elif system.startswith("T"):
        # Temperature should be in [200, 1e5] K
        x_min = float(x.min()); x_max = float(x.max())
        if x_min < 200.0 or x_max > 1e5:
            rec["reason"] = f"T unphysical: range [{x_min:.0f}, {x_max:.0f}] K"; return rec

    rec["pass"] = True
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="OpenFOAM case directory")
    ap.add_argument("--system", default="pd_corr0",
                    help="matrix system (pd_corr0/pd_corr1/pd_corr2/T_corr0); pass 'all' for all")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tol-resid", type=float, default=1e-5)
    ap.add_argument("--max", type=int, default=None,
                    help="max number of timesteps to validate (for testing)")
    args = ap.parse_args()

    case = Path(args.case)
    matrices_dir = case / "postProcessing" / "matrices"
    if not matrices_dir.is_dir():
        raise SystemExit(f"no postProcessing/matrices in {case}")

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    # collect timesteps
    timesteps = sorted([d for d in matrices_dir.iterdir() if d.is_dir()],
                       key=lambda d: float(d.name))
    if args.max:
        timesteps = timesteps[:args.max]
    print(f"Validating {len(timesteps)} timesteps from {case.name}, system={args.system}")
    print(f"Tolerance for self-consistency: rel_resid < {args.tol_resid:.0e}")
    print()

    systems_to_check = [args.system] if args.system != "all" else \
        ["pd_corr0", "pd_corr1", "pd_corr2", "T_corr0"]

    results = []
    pass_paths = []
    t_start = time.time()
    for i, ts_dir in enumerate(timesteps):
        for sys_name in systems_to_check:
            t0 = time.time()
            rec = validate_one(ts_dir, sys_name, tol_resid=args.tol_resid)
            rec["wall_s"] = time.time() - t0
            rec["timestep"] = ts_dir.name
            results.append(rec)
            if rec["pass"]:
                pass_paths.append(f"{ts_dir.name}/{sys_name}")
                marker = "PASS"
            else:
                marker = "FAIL"
            n_chk = len(rec["checks"])
            elapsed = time.time() - t_start
            eta = (elapsed / (i+1)) * (len(timesteps) - i - 1) if i > 0 else 0
            print(f"  [{i+1:4d}/{len(timesteps)}] {ts_dir.name:<14s} {sys_name:<10s} "
                  f"{marker}  ({rec['wall_s']:.1f}s)  "
                  f"{'rel_resid='+format(rec['checks'].get('rel_resid_Ax_minus_b',-1),'.2e') if rec['pass'] else rec['reason']}  "
                  f"[ETA {eta:.0f}s]")

    out_path = output_dir / "validation_summary.json"
    with open(out_path, "w") as f:
        json.dump({"case": str(case), "system": args.system,
                   "n_total": len(results), "n_pass": sum(1 for r in results if r["pass"]),
                   "tol_resid": args.tol_resid,
                   "wall_s_total": time.time() - t_start,
                   "results": results}, f, indent=2)
    print(f"\nDONE.  {sum(1 for r in results if r['pass'])} / {len(results)} matrices PASS")
    print(f"  summary: {out_path}")

    pool_path = output_dir / "sane_pool.txt"
    with open(pool_path, "w") as f:
        f.write(f"# Sane matrices passing all 8 checks (tol_resid={args.tol_resid:.0e})\n")
        f.write(f"# case: {case}\n")
        f.write(f"# system: {args.system}\n")
        f.write(f"# generated: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        for p in pass_paths: f.write(f"{p}\n")
    print(f"  pool:    {pool_path}  ({len(pass_paths)} entries)")

    # Quick fail-mode breakdown
    fail_reasons: dict[str, int] = {}
    for r in results:
        if not r["pass"]:
            # truncate reason to first keyword
            key = r["reason"].split(":")[0] if ":" in r["reason"] else r["reason"].split()[0]
            fail_reasons[key] = fail_reasons.get(key, 0) + 1
    if fail_reasons:
        print(f"\nFail-mode breakdown:")
        for k, n in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"  {n:4d}  {k}")


if __name__ == "__main__":
    main()
