"""Precision improvement experiment: test tight AMGx config on existing data.

Runs CLASSICAL_V_DIAGSCALED_TIGHT (tol=1e-14, presweeps/postsweeps=2) on the
existing spot_melt_npz matrices and compares rel_vs_OF against the baseline
CLASSICAL_V_DIAGSCALED (tol=1e-10, presweeps/postsweeps=1).

Also saves x vectors for global norm computation.

Usage:
    python3 precision_experiment.py <root>
    python3 precision_experiment.py <root> --eq pd
    python3 precision_experiment.py <root> --max-matrices 10  # quick test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Precision experiment")
    ap.add_argument("root", type=Path, help="Root of spot_melt_npz")
    ap.add_argument("--eq", choices=["pd", "T"], default="pd")
    ap.add_argument("--max-matrices", type=int, default=0,
                    help="Limit number of matrices (0 = all)")
    ap.add_argument("--compare-only", action="store_true",
                    help="Only compare existing results, don't re-solve")
    args = ap.parse_args()

    # Import AMGx
    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

    from dilu.amgx.python import Plan as AmgxPlan, CLASSICAL_V_DIAGSCALED, CLASSICAL_V_DIAGSCALED_TIGHT
    from dilu.benchmark.openfoam_crosscheck.reader import load_ofmm, normalize_sign

    # Find all matrices for the given equation
    dirs = sorted(args.root.glob(f"*/*{args.eq}_corr*"))
    if args.max_matrices > 0:
        dirs = dirs[:args.max_matrices]

    print(f"Testing {len(dirs)} {args.eq} matrices")
    print(f"Baseline: CLASSICAL_V_DIAGSCALED (tol=1e-10, sweeps=1)")
    print(f"Tight:    CLASSICAL_V_DIAGSCALED_TIGHT (tol=1e-14, sweeps=2)")
    print()

    results = []

    for i, mdir in enumerate(dirs):
        ts = mdir.parent.parent.name
        eq_dir = mdir.name

        # Load matrix
        bundle = load_ofmm(mdir)
        bundle = normalize_sign(bundle)

        A, b, x0 = bundle.A, bundle.b, bundle.x0
        n = A.shape[0]
        row_ptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        col_idx = np.ascontiguousarray(A.indices, dtype=np.int32)
        values = np.ascontiguousarray(A.data, dtype=np.float64)

        import jax
        import jax.numpy as jnp
        rp = jax.device_put(jnp.asarray(row_ptr))
        ci = jax.device_put(jnp.asarray(col_idx))
        vv = jax.device_put(jnp.asarray(values))
        b_d = jax.device_put(jnp.asarray(b))
        x0_d = jax.device_put(jnp.asarray(x0))

        row = {"ts": ts, "eq": eq_dir, "n": n, "nnz": values.size}

        for label, cfg, tol in [
            ("baseline", CLASSICAL_V_DIAGSCALED, 1e-10),
            ("tight", CLASSICAL_V_DIAGSCALED_TIGHT, 1e-14),
        ]:
            try:
                plan = AmgxPlan(rp, ci, vv, cfg)
                t0 = time.time()
                x, iters, status = plan.solve(b_d, x0_d)
                x.block_until_ready()
                solve_s = time.time() - t0

                x_h = np.asarray(x)
                r = A @ x_h - b
                selfres = float(np.linalg.norm(r) / max(np.linalg.norm(b), 1e-300))

                rel_vs_OF = None
                if bundle.x_final is not None:
                    x_OF = bundle.x_final
                    denom = max(float(np.max(np.abs(x_OF))), 1e-300)
                    rel_vs_OF = float(np.max(np.abs(x_h - x_OF))) / denom

                row[f"{label}_iters"] = int(iters[0])
                row[f"{label}_solve_s"] = solve_s
                row[f"{label}_selfres"] = selfres
                row[f"{label}_rel_vs_OF"] = rel_vs_OF
                row[f"{label}_status"] = "ok" if int(status[0]) == 0 else f"err={status[0]}"

                # Save x vector for tight config
                if label == "tight":
                    out_dir = mdir / "results"
                    out_dir.mkdir(exist_ok=True)
                    np.save(out_dir / "amgx_classical_v_diagscaled_tight_x.npy", x_h)

                plan.release()

            except Exception as e:
                row[f"{label}_status"] = f"error:{e}"

        results.append(row)

        # Progress
        bl_of = row.get("baseline_rel_vs_OF", "?")
        t_of = row.get("tight_rel_vs_OF", "?")
        bl_str = f"{bl_of:.2e}" if isinstance(bl_of, float) else str(bl_of)
        t_str = f"{t_of:.2e}" if isinstance(t_of, float) else str(t_of)
        print(f"[{i+1}/{len(dirs)}] {ts}: baseline={bl_str}, tight={t_str}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    bl_rvs = [r["baseline_rel_vs_OF"] for r in results if r.get("baseline_rel_vs_OF") is not None]
    t_rvs = [r["tight_rel_vs_OF"] for r in results if r.get("tight_rel_vs_OF") is not None]

    if bl_rvs and t_rvs:
        bl_arr = np.array(bl_rvs)
        t_arr = np.array(t_rvs)
        print(f"\nBaseline (tol=1e-10, sweeps=1):")
        print(f"  N={len(bl_arr)}, median={np.median(bl_arr):.2e}, "
              f"P95={np.percentile(bl_arr,95):.2e}, max={np.max(bl_arr):.2e}")
        print(f"\nTight (tol=1e-14, sweeps=2):")
        print(f"  N={len(t_arr)}, median={np.median(t_arr):.2e}, "
              f"P95={np.percentile(t_arr,95):.2e}, max={np.max(t_arr):.2e}")
        print(f"\nImprovement ratio (median): {np.median(bl_arr)/np.median(t_arr):.1f}×")
        print(f"Improvement ratio (max):    {np.max(bl_arr)/np.max(t_arr):.1f}×")

        # Iteration comparison
        bl_iters = [r["baseline_iters"] for r in results if r.get("baseline_iters") is not None]
        t_iters = [r["tight_iters"] for r in results if r.get("tight_iters") is not None]
        if bl_iters and t_iters:
            print(f"\nIterations: baseline median={np.median(bl_iters):.0f}, "
                  f"tight median={np.median(t_iters):.0f} "
                  f"(+{(np.median(t_iters)/np.median(bl_iters)-1)*100:.0f}%)")

    # Save results
    out_file = args.root / "precision_experiment.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # Save CSV for easy comparison
    import csv
    csv_file = args.root / "precision_experiment.csv"
    if results:
        with open(csv_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"CSV saved to {csv_file}")


if __name__ == "__main__":
    main()
