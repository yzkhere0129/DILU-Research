"""Phase 4 — AMG-PCG vs Phase 2/3 DILU-PCG headline scaling benchmark.

Protocol (per Phase 4 brief §5):
  - Scan grid sizes {16, 32, 64, 128}^3 with stiff-Laplacian (contrast 100).
  - Run AMG-PCG (classical V-cycle, D2 interpolator, BLOCK_JACOBI smoother)
    to tol=1e-10 on each. Record:
      * iters (AMGx-reported)
      * setup wall (plan construction; includes AMGx matrix upload +
        coarsening hierarchy build)
      * solve wall (plan.solve call, including b upload and x download)
      * total wall = setup + solve
      * apply-dispatch median/p95 (AMG "apply" = one full PCG iter by our
        metric — we sample median per-iter by dividing solve wall by iters;
        there is no separate "preconditioner apply" to time under AMG's
        V-cycle-inside-outer-PCG structure, unlike DILU)
  - Compare against Phase 2 / Phase 3 iter-count and total-wall numbers
    quoted from docs/benchmark/phase3_scaling_64_128.md (those are the
    canonical baselines; we do NOT re-run them).

Outputs:
  - Markdown table printed to stdout (copy into the Phase 4 report).
  - docs/benchmark/phase4_amgx_scaling.json (machine-readable).

Hardware safety:
  - 128^3 AGGRESSIVE_COARSENING by default (per T13 finding).
  - `--cfg classical` flag forces the CLASSICAL config even at 128^3 (may
    run tight on a 4 GB card).
  - Each grid point is run in the same Python process but `jax.clear_caches`
    is called between grids to release VRAM.
"""
from __future__ import annotations

# Safety rails before jax import.
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import gc
import json
import time
import subprocess

import numpy as np
import jax
from jax import config as _jc
_jc.update("jax_enable_x64", True)
import jax.numpy as jnp

from dilu.amgx.python import (
    Plan as AmgxPlan, CLASSICAL_V_CYCLE, AGGRESSIVE_COARSENING, with_tolerance)
from dilu.amgx.tests._harness import stiff_laplacian_3d


# Phase 2 / Phase 3 baselines from docs/benchmark/phase3_scaling_64_128.md §2.
# (iters, total PCG wall [s])  — '*' rows are 16^3 / 32^3 surrogate numbers.
P2_BASELINE = {
    16:  {"iters": 24,  "total_wall_s": 0.0445},
    32:  {"iters": 128, "total_wall_s": 0.5558},
    64:  {"iters": 99,  "total_wall_s": 6.490},
    128: {"iters": 186, "total_wall_s": 28.470},
}
P3_BASELINE = {
    16:  {"iters": 36,  "total_wall_s": 0.0343},
    32:  {"iters": 197, "total_wall_s": 0.2041},
    64:  {"iters": 158, "total_wall_s": 8.082},
    128: {"iters": 296, "total_wall_s": 34.997},
}


def _vram_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True).strip()
        return int(out.split("\n")[0])
    except Exception:
        return -1


def run_one(grid: int, cfg_json: str, cfg_label: str):
    contrast = 100.0
    print(f"\n=== grid={grid}^3, cfg={cfg_label} ===")
    t0 = time.time()
    row_ptr, col_idx, values = stiff_laplacian_3d(grid, grid, grid, contrast)
    n = grid ** 3
    nnz = int(values.shape[0])
    build_wall = time.time() - t0
    print(f"  csr: n={n}, nnz={nnz}, build_wall={build_wall:.1f}s")

    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    rng = np.random.default_rng(42)
    b = jax.device_put(jnp.asarray(rng.standard_normal(n)))

    vram_pre = _vram_mib()

    t_plan = time.time()
    with AmgxPlan(rp, ci, vv, cfg_json) as plan:
        t_setup_end = time.time()
        setup_wall = t_setup_end - t_plan
        vram_post_setup = _vram_mib()
        # Warm-up: first solve has a one-shot JIT/dispatch tax; we measure
        # a second solve on the same plan as the "steady-state" per-solve
        # wall. Both solves use the same b so they must produce the same x.
        x1, iters1, st1 = plan.solve(b)
        x1.block_until_ready()
        t_solve_start = time.time()
        x2, iters2, st2 = plan.solve(b)
        x2.block_until_ready()
        solve_wall = time.time() - t_solve_start
        iters = int(iters2[0])
        status = int(st2[0])
        total_wall_fresh = (t_setup_end - t_plan) + (t_solve_start
                                                       - t_setup_end + solve_wall)
        # 'Fresh' = setup + 1st solve (same as a production one-shot pattern).
        first_solve_wall = t_solve_start - t_setup_end
    vram_post = _vram_mib()

    result = {
        "grid": grid,
        "n": n,
        "nnz": nnz,
        "cfg_label": cfg_label,
        "build_csr_wall_s": build_wall,
        "setup_wall_s": setup_wall,
        "first_solve_wall_s": first_solve_wall,
        "steady_solve_wall_s": solve_wall,
        "total_fresh_wall_s": setup_wall + first_solve_wall,
        "iters": iters,
        "status": status,
        "vram_pre_mib": vram_pre,
        "vram_post_setup_mib": vram_post_setup,
        "vram_post_mib": vram_post,
        "per_iter_median_ms": (solve_wall / max(iters, 1)) * 1000.0,
    }
    print(f"  setup={setup_wall:.3f}s, solve(first)={first_solve_wall:.3f}s, "
          f"solve(steady)={solve_wall:.3f}s, iters={iters}, status={status}")
    print(f"  vram: pre={vram_pre} MiB, post_setup={vram_post_setup} MiB, "
          f"post={vram_post} MiB")
    return result


def print_table(results, outfile: str | None = None):
    header = (
        "| grid | N | nnz | P2 iters | P2 wall (s) | P3 iters | P3 wall (s) | "
        "P4 iters | P4 setup (s) | P4 solve (s) | P4 total (s) | "
        "P4/P2 iter ratio | P4/P2 wall speedup |")
    sep = "|" + "|".join(["---"] * 13) + "|"
    print("\n" + header)
    print(sep)
    for r in results:
        g = r["grid"]
        p2 = P2_BASELINE.get(g, {})
        p3 = P3_BASELINE.get(g, {})
        if p2 and p2.get("iters"):
            iter_ratio = r["iters"] / p2["iters"]
            wall_speedup = p2["total_wall_s"] / r["total_fresh_wall_s"]
        else:
            iter_ratio = float("nan")
            wall_speedup = float("nan")
        print(f"| {g}^3 | {r['n']} | {r['nnz']} | "
              f"{p2.get('iters', '—')} | {p2.get('total_wall_s', '—'):.3f} | "
              f"{p3.get('iters', '—')} | {p3.get('total_wall_s', '—'):.3f} | "
              f"{r['iters']} | {r['setup_wall_s']:.3f} | "
              f"{r['steady_solve_wall_s']:.3f} | "
              f"{r['total_fresh_wall_s']:.3f} | "
              f"{iter_ratio:.3f} | {wall_speedup:.2f}x |")
    if outfile:
        with open(outfile, "w") as f:
            json.dump({"p2": P2_BASELINE, "p3": P3_BASELINE,
                       "p4": results}, f, indent=2)
        print(f"\nWrote {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", nargs="+", type=int,
                        default=[16, 32, 64, 128])
    parser.add_argument("--cfg", choices=["classical", "aggressive", "auto"],
                        default="auto",
                        help="AMG config (auto: aggressive for 128, classical otherwise)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = []
    for g in args.grids:
        if args.cfg == "classical":
            cfg = with_tolerance(CLASSICAL_V_CYCLE, tol=1e-10, max_iters=200)
            cfg_label = "CLASSICAL"
        elif args.cfg == "aggressive":
            cfg = with_tolerance(AGGRESSIVE_COARSENING, tol=1e-10, max_iters=200)
            cfg_label = "AGGRESSIVE"
        else:  # auto
            if g >= 128:
                cfg = with_tolerance(AGGRESSIVE_COARSENING, tol=1e-10, max_iters=200)
                cfg_label = "AGGRESSIVE"
            else:
                cfg = with_tolerance(CLASSICAL_V_CYCLE, tol=1e-10, max_iters=200)
                cfg_label = "CLASSICAL"

        try:
            r = run_one(g, cfg, cfg_label)
            results.append(r)
        except Exception as e:
            print(f"FAILED at grid={g}: {e}")
            results.append({
                "grid": g, "error": str(e),
                "cfg_label": cfg_label,
            })
        # Release VRAM between grids.
        jax.clear_caches()
        gc.collect()

    print_table(results, args.out)


if __name__ == "__main__":
    main()
