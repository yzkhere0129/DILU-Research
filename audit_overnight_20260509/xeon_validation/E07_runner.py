"""E07 — AMGx + 1 IR vs CHOLMOD LU per-timestep diff (settles C004 / S3).

For each of 6 timesteps:
  - LU truth via CHOLMOD
  - x_AMGx_e12 from npz (already computed)
  - diff: max|x_AMGx_e12 - x_LU|, ‖diff‖₂, plus relative
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    record_environment, write_result_json,
    normalize_sign, load_raw_matrix, find_npz, load_npz_with_solutions,
)

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(args.npz_dir)

    # sksparse 0.5.0: cholesky() returns (R, p) tuple; use cho_factor + .solve.
    try:
        from sksparse.cholmod import cho_factor as _sks_factor_fn
        def _sks_factor(Acsc): return _sks_factor_fn(Acsc)
        def _sks_solve(f, b): return f.solve(b)
        _have_sksparse = True
    except ImportError:
        try:
            from sksparse.cholmod import cholesky as _sks_legacy
            def _sks_factor(Acsc): return _sks_legacy(Acsc)
            def _sks_solve(f, b): return f(b)
            _have_sksparse = True
        except ImportError:
            from scipy.sparse.linalg import spsolve
            _have_sksparse = False

    case_dir_candidates = [
        Path("/home/yzk/cases/single_track_dump"),
        Path("/home/yzk/single_track_dump"),
    ]
    case_dir = next((c for c in case_dir_candidates
                     if (c / "postProcessing/matrices").exists()), None)
    if case_dir is None:
        sys.exit(f"FAIL: no case dir found")

    timesteps = find_npz(npz_dir)
    env = record_environment()
    per_timestep = []

    for time_str, phase, npz_path in timesteps:
        rep_dir = output_dir / f"01_{time_str}"
        result_path = rep_dir / "result.json"
        if args.resume and result_path.exists():
            print(f"{phase} t={time_str}: SKIP"); continue

        # Load AMGx_e12 from npz
        npz_data = load_npz_with_solutions(npz_path)
        x_AMGx_e12 = npz_data["x_truth"]
        x_AMGx_e8 = npz_data["x_AMGx_e8"]
        x_OF = npz_data["x_OF"]

        # Load raw A, b
        A, b_raw, _ = load_raw_matrix(case_dir, time_str)
        A_pos, b_pos, sf = normalize_sign(A, b_raw)

        # LU truth
        t0 = time.perf_counter_ns()
        if _have_sksparse:
            factor = _sks_factor(A_pos.tocsc())
            x_LU = _sks_solve(factor, b_pos)
            method = "CHOLMOD"
        else:
            from scipy.sparse.linalg import spsolve
            x_LU = spsolve(A_pos.tocsc(), b_pos)
            method = "SuperLU"
        lu_wall_s = (time.perf_counter_ns() - t0) / 1e9

        res_LU = float(np.linalg.norm(A_pos @ x_LU - b_pos) /
                        max(np.linalg.norm(b_pos), 1e-300))
        if res_LU > 1e-12:
            print(f"WARN: {phase} t={time_str}: LU residual {res_LU:.2e} > 1e-12")

        x_LU_inf = float(np.abs(x_LU).max())
        x_LU_2 = float(np.linalg.norm(x_LU))

        # Compute diffs (note: x_OF, x_AMGx are stored in original sign convention;
        # x_LU was computed from sign-flipped form. They may need to match conventions —
        # the npz x_truth came from amgx_solve(A_pos, b_pos), so it's in same sign as x_LU.
        # x_OF in npz: was loaded directly from x_final.mm (negative-diag convention).
        # Need to verify: x_LU should equal -? Actually no, x doesn't change with sign flip;
        # only A and b do. So x_LU and x_OF are in same units.

        results_per_solver = {}
        for name, x in [("OF", x_OF), ("AMGx_e8", x_AMGx_e8), ("AMGx_e12_IR", x_AMGx_e12)]:
            d = np.abs(x - x_LU)
            results_per_solver[name] = {
                "max_diff_Pa": float(d.max()),
                "rel_max": float(d.max() / max(x_LU_inf, 1e-300)),
                "L2_diff_Pa": float(np.linalg.norm(d)),
                "rel_L2": float(np.linalg.norm(d) / max(x_LU_2, 1e-300)),
                "median_diff_Pa": float(np.median(d)),
                "cells_above_100Pa": int((d > 100).sum()),
                "cells_above_1kPa": int((d > 1000).sum()),
            }

        record = {
            "expt_id": "E07",
            "timestep": time_str,
            "phase": phase,
            "lu_method": method,
            "lu_wall_s": lu_wall_s,
            "lu_rel_resid": res_LU,
            "x_LU_inf_norm": x_LU_inf,
            "x_LU_L2_norm": x_LU_2,
            "comparisons_vs_LU": results_per_solver,
            "env": env,
        }
        write_result_json(result_path, record)
        per_timestep.append(record)

        print(f"\n{phase} t={time_str}:")
        print(f"  LU ({method}): {lu_wall_s:.1f}s  res={res_LU:.2e}  ‖x_LU‖∞={x_LU_inf:.3e}")
        for n, r in results_per_solver.items():
            print(f"  {n:<12s}: max|diff|={r['max_diff_Pa']:.3e} Pa  "
                  f"rel={r['rel_max']:.2e}  >100Pa={r['cells_above_100Pa']}")

    # Aggregate
    aggr_path = output_dir / "aggregate.json"
    aggr = {
        "expt_id": "E07",
        "n_timesteps": len(per_timestep),
        "per_timestep": per_timestep,
    }
    if per_timestep:
        for sname in ["OF", "AMGx_e8", "AMGx_e12_IR"]:
            rels = [r["comparisons_vs_LU"][sname]["rel_max"] for r in per_timestep]
            aggr[f"{sname}_max_rel_max"] = max(rels)
            aggr[f"{sname}_min_rel_max"] = min(rels)
            aggr[f"{sname}_median_rel_max"] = float(np.median(rels))
    write_result_json(aggr_path, aggr)
    print(f"\n[E07] aggregate → {aggr_path}")


if __name__ == "__main__":
    main()
