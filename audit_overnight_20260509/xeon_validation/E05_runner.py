"""E05 — CHOLMOD fresh refactor wall, ≥5 reps × 6 timesteps.

Per-step: cholesky() full factor + solve. No reuse.

Closeout fixes:
  A009 gc.collect between reps
  A019 method + version recorded in result.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    record_environment, file_md5, write_result_json, TimedSection,
    normalize_sign, load_raw_matrix, find_npz, cold_cache,
)

import numpy as np
from scipy.sparse.linalg import spsolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.smoke: args.reps = 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(args.npz_dir)

    try:
        from sksparse.cholmod import cholesky
        method = "CHOLMOD"
    except ImportError:
        method = "SuperLU"
        cholesky = None
        print("sksparse not available; falling back to scipy SuperLU. wall numbers will not match expected CHOLMOD speed.")

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
    print(f"E05 method: {method}")

    for time_str, phase, npz_path in timesteps:
        A, b_raw, x_OF = load_raw_matrix(case_dir, time_str)
        A_pos, b_pos, sf = normalize_sign(A, b_raw)
        b_md5 = file_md5(case_dir / "postProcessing/matrices" / time_str / "pd_corr0/b.mm")
        A_md5 = file_md5(case_dir / "postProcessing/matrices" / time_str / "pd_corr0/A.mm")
        print(f"\n=== {phase} t={time_str} ===")

        for rep in range(1, args.reps + 1):
            rep_dir = output_dir / f"{rep:02d}_{time_str}"
            result_path = rep_dir / "result.json"
            if args.resume and result_path.exists():
                print(f"  rep{rep:02d}: SKIP"); continue

            gc.collect()  # A009
            cold_cache()
            with TimedSection() as outer:
                t0 = time.perf_counter_ns()
                if cholesky is not None:
                    factor = cholesky(A_pos.tocsc())
                    factor_ns = time.perf_counter_ns() - t0
                    t1 = time.perf_counter_ns()
                    x = factor(b_pos)
                    solve_ns = time.perf_counter_ns() - t1
                else:
                    x = spsolve(A_pos.tocsc(), b_pos)
                    factor_ns = time.perf_counter_ns() - t0
                    solve_ns = 0  # SuperLU bundles factor+solve

            res = float(np.linalg.norm(A_pos @ x - b_pos) /
                        max(np.linalg.norm(b_pos), 1e-300))

            payload = {
                "expt_id": "E05",
                "rep": rep,
                "method": method,
                "timestep": time_str,
                "phase": phase,
                "wall_seconds": outer["wall_seconds"],
                "wall_seconds_method": outer["wall_seconds_method"],
                "mem_peak_kb": outer["mem_peak_kb"],
                "factor_ns": int(factor_ns),
                "solve_ns": int(solve_ns),
                "factor_ms": factor_ns / 1e6,
                "solve_ms": solve_ns / 1e6,
                "rel_resid_actual": res,
                "input_md5s": {"A.mm": A_md5, "b.mm": b_md5},
                "n": int(A.shape[0]),
                "nnz": int(A.nnz),
                "sign_flipped": sf,
                "env": env,
                "config_full_str": f"method={method},mode=fresh_factor",  # A019
            }
            write_result_json(result_path, payload)
            print(f"  rep{rep:02d}: factor={factor_ns/1e6:.0f}ms solve={solve_ns/1e6:.0f}ms "
                  f"resid={res:.2e}")

    print(f"\n[E05] complete.")


if __name__ == "__main__":
    main()
