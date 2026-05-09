"""E02 — AMGx PCG cold-start single-shot wall.

For each of 6 timesteps × N reps:
  - cold cache
  - Plan(A) fresh setup
  - plan.solve(b, x0=0)
  - record setup_s, solve_s, iter, rel_resid
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow import of _common
sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    record_environment, file_md5, write_result_json,
    TimedSection, sanity_check_matrix, normalize_sign,
    load_raw_matrix, find_npz, cold_cache,
)

# JAX/AMGx setup
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.reps = 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(args.npz_dir)

    # Lazy import AMGx — fail loud if missing
    try:
        from jax import config as _jc
        _jc.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from dilu.amgx.python import Plan, CLASSICAL_V_DIAGSCALED, with_tolerance
    except ImportError as e:
        sys.exit(f"FAIL: AMGx wrapper not importable: {e}")

    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, 1e-8, max_iters=2000)

    # Discover npzs (use them to load A, b — they have the matrix? actually no, they have x_OF only)
    # We need raw A, b — must read from postProcessing/matrices.
    # Workaround: npz has b, but A is huge. Let's load from npz and reconstruct? npz doesn't have A either.
    # Plan: use the npz x_OF + b, and reconstruct A from npz fields.
    # Actually the npz files don't store A. We need the raw .mm files.
    # The dense_track or single_track case dir is needed for A.
    # → assume case dir is at /home/yzk/cases/single_track_dump (lab Xeon) or /home/yzk/single_track_dump (dev)

    case_dir_candidates = [
        Path("/home/yzk/cases/single_track_dump"),
        Path("/home/yzk/single_track_dump"),
    ]
    case_dir = None
    for cd in case_dir_candidates:
        if (cd / "postProcessing/matrices").exists():
            case_dir = cd; break
    if case_dir is None:
        sys.exit(f"FAIL: no case dir found in {case_dir_candidates}")
    print(f"Using case dir: {case_dir}")

    timesteps = find_npz(npz_dir)

    env = record_environment()
    print(f"Environment: {json.dumps(env, indent=2)}")
    print(f"reps per timestep: {args.reps}")
    print(f"npz_dir: {npz_dir}")
    print(f"case_dir: {case_dir}")

    for time_str, phase, npz_path in timesteps:
        # Load raw A, b (large but needed for AMGx)
        print(f"\n=== {phase} t={time_str} ===")
        A, b_raw, x_OF = load_raw_matrix(case_dir, time_str)
        A_pos, b_pos, sign_flipped = normalize_sign(A, b_raw)
        b_md5 = file_md5(case_dir / "postProcessing/matrices" / time_str / "pd_corr0/b.mm")
        A_md5 = file_md5(case_dir / "postProcessing/matrices" / time_str / "pd_corr0/A.mm")
        print(f"  N={A.shape[0]}, nnz={A.nnz}, sign_flipped={sign_flipped}")

        for rep in range(1, args.reps + 1):
            rep_dir = output_dir / f"{rep:02d}_{time_str}"
            result_path = rep_dir / "result.json"
            if args.resume and result_path.exists():
                print(f"  rep {rep:02d}: SKIP (resume — exists)")
                continue
            cold_cache()

            # Move data to GPU
            rp = jnp.asarray(A_pos.indptr.astype(np.int32))
            ci = jnp.asarray(A_pos.indices.astype(np.int32))
            vv = jnp.asarray(A_pos.data.astype(np.float64))
            b_d = jnp.asarray(b_pos.astype(np.float64))
            x0_d = jnp.zeros_like(b_d)

            with TimedSection() as outer:
                t0 = time.perf_counter_ns()
                with Plan(rp, ci, vv, cfg) as plan:
                    setup_ns = time.perf_counter_ns() - t0
                    t1 = time.perf_counter_ns()
                    x, iters, status = plan.solve(b_d, x0_d)
                    x.block_until_ready()
                    solve_ns = time.perf_counter_ns() - t1
                    x_h = np.asarray(x)

            res = float(np.linalg.norm(A_pos @ x_h - b_pos)
                        / max(np.linalg.norm(b_pos), 1e-300))

            payload = {
                "expt_id": "E02",
                "rep": rep,
                "timestep": time_str,
                "phase": phase,
                "started_at": "(captured by TimedSection)",
                "wall_seconds": outer["wall_seconds"],
                "wall_seconds_method": outer["wall_seconds_method"],
                "mem_peak_kb": outer["mem_peak_kb"],
                "mem_peak_method": outer["mem_peak_method"],
                "setup_ns": int(setup_ns),
                "solve_ns": int(solve_ns),
                "setup_ms": setup_ns / 1e6,
                "solve_ms": solve_ns / 1e6,
                "iter_count": int(iters[0]),
                "status": int(status[0]),
                "rel_resid_actual": res,
                "tol_requested": 1e-8,
                "config": {"name": "CLASSICAL_V_DIAGSCALED", "max_iters": 2000},
                "input_files": {
                    "A.mm": str(case_dir / "postProcessing/matrices" / time_str / "pd_corr0/A.mm"),
                    "b.mm": str(case_dir / "postProcessing/matrices" / time_str / "pd_corr0/b.mm"),
                },
                "input_md5s": {"A.mm": A_md5, "b.mm": b_md5},
                "n": int(A.shape[0]),
                "nnz": int(A.nnz),
                "sign_flipped": sign_flipped,
                "env": env,
                "warnings": [],
            }
            write_result_json(result_path, payload)
            print(f"  rep {rep:02d}: setup={setup_ns/1e6:.0f}ms solve={solve_ns/1e6:.0f}ms "
                  f"iter={int(iters[0])} resid={res:.2e}")

    print(f"\n[E02] complete. Results in {output_dir}/")


if __name__ == "__main__":
    main()
