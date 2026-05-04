"""Run AMGx + IR on senior's pd matrix CSV dataset; report precision.

Three metrics per (step, corr):
  rel_vs_xref  = ||x_amgx - x_ref||_inf / ||x_ref||_inf
                 (x_ref = OF PCG-DIC tol=1e-8 stop, our v3.2-style apparent gap)
  rel_vs_truth = ||x_amgx - x_truth||_inf / ||x_truth||_inf
                 (x_truth = scipy.sparse.linalg.spsolve direct LU; gold standard)
  ref_vs_truth = ||x_ref - x_truth||_inf / ||x_truth||_inf
                 (OF's own ceiling)

Outputs JSON + console table.
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

from jax import config as _jc
_jc.update("jax_enable_x64", True)

from dilu.amgx.python import amgx_solve_with_refinement
from dilu.amgx.bench.senior_data_loader import (
    load_step, list_available, MESH_N,
)


def normalize_sign(A, b):
    """If diag is all-negative (OF Laplacian), flip A,b → SPD with positive diag.
    Returns (A_new, b_new, negated_bool).
    """
    diag = A.diagonal()
    if np.all(diag <= 0) and np.any(diag < 0):
        return -A, -b, True
    return A, b, False


def run(do_truth: bool, n_refine: int, tol: float, out_path: Path):
    avail = list_available()
    print(f"[senior sweep] {len(avail)} (step,corr) bundles, "
          f"tol={tol}, n_refine={n_refine}, do_truth={do_truth}")
    print()
    print(f"{'step':>4s} {'corr':>4s} {'amgx_iter':>9s} {'amgx_resid':>11s} "
          f"{'rel_vs_xref':>13s} {'rel_vs_truth':>14s} {'ref_vs_truth':>14s} "
          f"{'wall':>5s}")
    print("-" * 110)

    rows = []
    for step, corr in avail:
        try:
            bundle = load_step(step, corr)
            A_raw, b_raw, x_ref = bundle.A, bundle.b, bundle.x_ref
            A, b, negated = normalize_sign(A_raw, b_raw)

            # AMGx + IR
            x0 = np.zeros_like(b)        # cold start (no prev iterate from senior)
            t0 = time.time()
            res = amgx_solve_with_refinement(
                A, b, x0,
                eq_kind="pd", tol=tol, n_refine=n_refine, max_iters=500,
            )
            wall = time.time() - t0
            x_amgx = res["x"]

            denom_xref = max(float(np.max(np.abs(x_ref))), 1e-300)
            rel_vs_xref = float(np.max(np.abs(x_amgx - x_ref)) / denom_xref)

            row = dict(
                step=step, corr=corr, n=bundle.n, nnz=int(A_raw.nnz),
                negated=negated,
                amgx_iters=res["primary_iters"],
                amgx_status=res["primary_status"],
                refine_iters=res["refine_iters"],
                amgx_resid_rel2=res["rel_residual"],
                rel_vs_xref=rel_vs_xref,
                wall_s=wall,
            )

            # Truth via scipy spsolve (slow on 512K)
            if do_truth:
                t1 = time.time()
                x_truth = spsolve(A.tocsc(), b)
                tt = time.time() - t1
                denom_truth = max(float(np.max(np.abs(x_truth))), 1e-300)
                row["truth_solve_s"] = tt
                row["rel_vs_truth"] = float(np.max(np.abs(x_amgx - x_truth)) / denom_truth)
                row["ref_vs_truth"] = float(np.max(np.abs(x_ref - x_truth)) / denom_truth)
                row["truth_resid_rel2"] = float(
                    np.linalg.norm(A @ x_truth - b) / max(np.linalg.norm(b), 1e-300))

            rows.append(row)
            avt = row.get("rel_vs_truth")
            rvt = row.get("ref_vs_truth")
            print(f"{step:>4d} {corr:>4d} {row['amgx_iters']:>9d} "
                  f"{row['amgx_resid_rel2']:>11.2e} {row['rel_vs_xref']:>13.2e} "
                  f"{(f'{avt:.2e}' if avt is not None else '—'):>14s} "
                  f"{(f'{rvt:.2e}' if rvt is not None else '—'):>14s} "
                  f"{wall:>5.2f}")
        except Exception as e:
            print(f"{step}/{corr}: ERROR {type(e).__name__}: {e}")
            rows.append(dict(step=step, corr=corr, error=str(e)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(dict(date=datetime.now().isoformat(),
                       tol=tol, n_refine=n_refine, do_truth=do_truth,
                       n_bundles=len(rows), rows=rows), fh, indent=2)
    print(f"\nWrote {out_path}")

    # Aggregate
    valid = [r for r in rows if "rel_vs_xref" in r]
    if valid:
        v = [r["rel_vs_xref"] for r in valid]
        print(f"\nrel_vs_xref  N={len(v)}: max={max(v):.2e} median={sorted(v)[len(v)//2]:.2e} min={min(v):.2e}")
    valid_t = [r for r in rows if "rel_vs_truth" in r]
    if valid_t:
        v = [r["rel_vs_truth"] for r in valid_t]
        print(f"rel_vs_truth N={len(v)}: max={max(v):.2e} median={sorted(v)[len(v)//2]:.2e} min={min(v):.2e}")
        v = [r["ref_vs_truth"] for r in valid_t]
        print(f"ref_vs_truth N={len(v)}: max={max(v):.2e} median={sorted(v)[len(v)//2]:.2e} min={min(v):.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--n-refine", type=int, default=1)
    ap.add_argument("--no-truth", action="store_true",
                    help="skip scipy spsolve (much faster)")
    ap.add_argument("--out", type=Path,
                    default=Path("/home/yzk/DILU-Research/dilu/amgx/bench/senior_precision_results.json"))
    args = ap.parse_args()
    run(do_truth=not args.no_truth, n_refine=args.n_refine, tol=args.tol,
        out_path=args.out)
