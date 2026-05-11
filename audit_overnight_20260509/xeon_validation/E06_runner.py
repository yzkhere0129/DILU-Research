"""E06 — CHOLMOD symbolic-reuse: analyze() once + cholesky_inplace() per step.

Sequence-of-6 approach (mirroring E03/E04). 5 reps each.
Watch for cholesky_inplace() failures (numerical issues) — record fallback to full.

Closeout: A009 gc, A019 method recorded.
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
    record_environment, write_result_json,
    normalize_sign, load_raw_matrix, find_npz, cold_cache,
)

import numpy as np


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

    # sksparse 0.5.0: cholesky() returns (R, p) tuple; use cho_factor.
    # symbolic-reuse path:  CholeskyFactor.factorize(A_new) (was cholesky_inplace pre-0.5.0).
    _API = None
    try:
        from sksparse.cholmod import cho_factor as _sks_factor_fn
        def _sks_factor(Acsc): return _sks_factor_fn(Acsc)
        def _sks_solve(f, b): return f.solve(b)
        def _sks_refactor(f, Acsc): f.factorize(Acsc); return f
        _API = "0.5.0"
    except ImportError:
        try:
            from sksparse.cholmod import cholesky as _sks_legacy
            def _sks_factor(Acsc): return _sks_legacy(Acsc)
            def _sks_solve(f, b): return f(b)
            def _sks_refactor(f, Acsc): f.cholesky_inplace(Acsc); return f
            _API = "pre-0.5.0"
        except ImportError:
            sys.exit("FAIL: sksparse not available. E06 requires CHOLMOD. Install scikit-sparse.")

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

    print("Pre-loading 6 matrices...")
    matrices = []
    for t, phase, npz_path in timesteps:
        A, b_raw, x_OF = load_raw_matrix(case_dir, t)
        A_pos, b_pos, sf = normalize_sign(A, b_raw)
        matrices.append({"t": t, "phase": phase, "A_csc": A_pos.tocsc(), "b": b_pos, "A_csr": A_pos})

    for rep in range(1, args.reps + 1):
        rep_dir = output_dir / f"rep_{rep:02d}"
        result_path = rep_dir / "result.json"
        if args.resume and result_path.exists():
            print(f"\nrep{rep:02d}: SKIP"); continue

        gc.collect()  # A009
        cold_cache()
        per_step = []
        factor = None
        warnings = []
        total_t0 = time.perf_counter_ns()

        for i, m in enumerate(matrices):
            t0 = time.perf_counter_ns()
            if factor is None:
                factor = _sks_factor(m["A_csc"])
                factor_ns = time.perf_counter_ns() - t0
                refact_ns = 0
                method_used = "full_factor"
            else:
                # Try numeric-reuse refactor; on failure fall back to full new factor.
                try:
                    factor = _sks_refactor(factor, m["A_csc"])
                    refact_ns = time.perf_counter_ns() - t0
                    factor_ns = 0
                    method_used = "symbolic_reuse_refactor"
                except Exception as e:
                    warnings.append(f"step {i}: refactor fail: {e}")
                    factor = _sks_factor(m["A_csc"])
                    factor_ns = time.perf_counter_ns() - t0
                    refact_ns = 0
                    method_used = "fallback_full"

            t1 = time.perf_counter_ns()
            x = _sks_solve(factor, m["b"])
            solve_ns = time.perf_counter_ns() - t1
            res = float(np.linalg.norm(m["A_csr"] @ x - m["b"]) /
                        max(np.linalg.norm(m["b"]), 1e-300))

            per_step.append({
                "step": i, "timestep": m["t"], "phase": m["phase"],
                "factor_ms": factor_ns / 1e6,
                "refact_ms": refact_ns / 1e6,
                "solve_ms": solve_ns / 1e6,
                "total_ms": (factor_ns + refact_ns + solve_ns) / 1e6,
                "rel_resid": res,
                "method_used": method_used,
            })
            print(f"  rep{rep:02d} step{i} ({m['phase']:<11s}): "
                  f"{method_used} {(factor_ns+refact_ns)/1e6:>5.0f}ms "
                  f"solve={solve_ns/1e6:>4.0f}ms resid={res:.2e}")

        total_wall_s = (time.perf_counter_ns() - total_t0) / 1e9

        payload = {
            "expt_id": "E06",
            "rep": rep,
            "wall_seconds": total_wall_s,
            "wall_seconds_method": "MEASURED:perf_counter_ns",
            "n_steps": len(matrices),
            "method": "CHOLMOD_symbolic_reuse",
            "sksparse_api": _API,
            "config_full_str": f"method=CHOLMOD,mode=cho_factor_then_factorize_per_step,api={_API}",  # A019
            "per_step": per_step,
            "warnings": warnings,
            "env": env,
        }
        write_result_json(result_path, payload)
        print(f"  rep{rep:02d} TOTAL: {total_wall_s:.2f} s")

    print(f"\n[E06] complete.")


if __name__ == "__main__":
    main()
