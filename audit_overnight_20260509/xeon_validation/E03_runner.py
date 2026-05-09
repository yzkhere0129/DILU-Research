"""E03 — AMGx amortized (Plan once + N×update_coefficients) wall.

Runs the entire 6-step sequence per rep:
  Plan(A_0) — record setup_ms_first
  solve(b_0, x0=0) — record solve_ms[0]
  for i in 1..5:
    update_coefficients(A_i.data) — record update_ms[i]
    solve(b_i, x0=0) — record solve_ms[i]    # COLD INIT for E03 baseline

Repeat ≥5 times. Sum across step gives realistic amortized total wall.
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
    record_environment, file_md5, write_result_json,
    TimedSection, normalize_sign, load_raw_matrix, find_npz, cold_cache,
)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

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

    try:
        from jax import config as _jc
        _jc.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from dilu.amgx.python import Plan, CLASSICAL_V_DIAGSCALED, with_tolerance
    except ImportError as e:
        sys.exit(f"FAIL: AMGx wrapper not importable: {e}")

    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, 1e-8, max_iters=2000)

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

    # Pre-load all matrices (eager)
    print("Pre-loading 6 matrices...")
    matrices = []
    for t, phase, npz_path in timesteps:
        A, b_raw, x_OF = load_raw_matrix(case_dir, t)
        A_pos, b_pos, sf = normalize_sign(A, b_raw)
        matrices.append({"t": t, "phase": phase, "A": A_pos, "b": b_pos, "x_OF": x_OF, "sf": sf})
        print(f"  {phase} t={t}: N={A_pos.shape[0]} nnz={A_pos.nnz}")

    for rep in range(1, args.reps + 1):
        rep_dir = output_dir / f"rep_{rep:02d}"
        result_path = rep_dir / "result.json"
        if args.resume and result_path.exists():
            print(f"\nrep {rep:02d}: SKIP (resume)"); continue

        cold_cache()
        per_step = []
        plan = None
        total_t0 = time.perf_counter_ns()

        for i, m in enumerate(matrices):
            A_pos, b_pos = m["A"], m["b"]
            rp = jnp.asarray(A_pos.indptr.astype(np.int32))
            ci = jnp.asarray(A_pos.indices.astype(np.int32))
            vv = jnp.asarray(A_pos.data.astype(np.float64))
            b_d = jnp.asarray(b_pos.astype(np.float64))
            x0_d = jnp.zeros_like(b_d)

            t0 = time.perf_counter_ns()
            if plan is None:
                # First step: full setup
                plan = Plan(rp, ci, vv, cfg)
                t1 = time.perf_counter_ns()
                setup_ns = t1 - t0
                update_ns = 0
                # solve
                x, iters, status = plan.solve(b_d, x0_d)
                x.block_until_ready()
                solve_ns = time.perf_counter_ns() - t1
            else:
                # Subsequent: update_coefficients
                plan.update_coefficients(vv)
                t1 = time.perf_counter_ns()
                update_ns = t1 - t0
                setup_ns = 0
                x, iters, status = plan.solve(b_d, x0_d)
                x.block_until_ready()
                solve_ns = time.perf_counter_ns() - t1

            x_h = np.asarray(x)
            res = float(np.linalg.norm(A_pos @ x_h - b_pos) /
                        max(np.linalg.norm(b_pos), 1e-300))
            step_total_ns = setup_ns + update_ns + solve_ns
            per_step.append({
                "step": i, "timestep": m["t"], "phase": m["phase"],
                "setup_ms": setup_ns / 1e6,
                "update_ms": update_ns / 1e6,
                "solve_ms": solve_ns / 1e6,
                "total_ms": step_total_ns / 1e6,
                "iter_count": int(iters[0]),
                "rel_resid": res,
            })
            print(f"  rep{rep:02d} step{i} ({m['phase']:<11s} t={m['t']}): "
                  f"setup={setup_ns/1e6:>5.0f}ms update={update_ns/1e6:>5.0f}ms "
                  f"solve={solve_ns/1e6:>5.0f}ms iter={int(iters[0]):>4d} resid={res:.2e}")

        if plan is not None:
            plan.__exit__(None, None, None)
        total_wall_s = (time.perf_counter_ns() - total_t0) / 1e9

        payload = {
            "expt_id": "E03",
            "rep": rep,
            "wall_seconds": total_wall_s,
            "wall_seconds_method": "MEASURED:perf_counter_ns",
            "n_steps": len(matrices),
            "tol_requested": 1e-8,
            "config": {"name": "CLASSICAL_V_DIAGSCALED", "max_iters": 2000,
                        "warm_start": False},
            "per_step": per_step,
            "env": env,
        }
        write_result_json(result_path, payload)
        print(f"  rep{rep:02d} TOTAL: {total_wall_s:.2f} s")

    print(f"\n[E03] complete. Results in {output_dir}/")


if __name__ == "__main__":
    main()
