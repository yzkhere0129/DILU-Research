"""AMGx precision validator vs scipy truth + OpenFOAM dump.

For each dumped (A, b, x0, x_final, sumA, normFactor):
  1. Compute x_truth = scipy.spsolve(A, b)        ← gold standard (direct LU)
  2. Run AMGx at tol=1e-12 (PCG for pd, BiCGStab for T)
  3. Report:
        rel_vs_truth = ||x_AMGx − x_truth||∞ / ||x_truth||∞
        rel_vs_OF    = ||x_AMGx − x_OF||∞    / ||x_OF||∞
        OF_vs_truth  = ||x_OF    − x_truth||∞ / ||x_truth||∞    (shows OF ceiling)

The headline metric for the precision claim is rel_vs_truth (truth = direct
solve). rel_vs_OF reflects OpenFOAM's own per-equation tol (pd: 1e-8 loose,
T: 1e-12 tight).

Outputs to dilu/amgx/bench/precision_results_<date>.{json,md}.
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

import jax
from jax import config as _jc
_jc.update("jax_enable_x64", True)
import jax.numpy as jnp

from dilu.amgx.python import (
    Plan as AmgxPlan,
    CLASSICAL_V_DIAGSCALED,
    CLASSICAL_V_DIAGSCALED_BICGSTAB,
    with_tolerance,
)
from dilu.amgx.python.refinement import amgx_solve_with_refinement
from dilu.benchmark.openfoam_crosscheck.reader import load_ofmm, normalize_sign


CASES = [
    ("LPBF_sanity",
     "/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_sanity/postProcessing/matrices"),
    ("dumper_pipeline",
     "/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/dumper_pipeline_test/postProcessing/matrices"),
]


def list_dumps():
    """Yield (case_label, ts, eq_dir, matrix_dir) for every dump."""
    for label, root in CASES:
        for ts in sorted(os.listdir(root)):
            for eq in sorted(os.listdir(f"{root}/{ts}")):
                d = Path(f"{root}/{ts}/{eq}")
                if (d / "A.mm").exists() or (d / "data.npz").exists():
                    yield label, ts, eq, d


def solve_with_amgx(A, b, x0, eq_name: str, tol: float = 1e-12,
                    max_iters: int = 500) -> dict:
    """Run AMGx; pick PCG (pd, symmetric) or BiCGStab (T, asymmetric)."""
    base = (CLASSICAL_V_DIAGSCALED if eq_name.lower().startswith("pd")
            else CLASSICAL_V_DIAGSCALED_BICGSTAB)
    cfg = with_tolerance(base, tol, max_iters=max_iters)

    rp = jnp.asarray(A.indptr.astype(np.int32))
    ci = jnp.asarray(A.indices.astype(np.int32))
    vv = jnp.asarray(A.data.astype(np.float64))
    b_d = jnp.asarray(b)
    x0_d = jnp.asarray(x0)

    t0 = time.time()
    with AmgxPlan(rp, ci, vv, cfg) as plan:
        setup_t = time.time() - t0
        t1 = time.time()
        x, iters, status = plan.solve(b_d, x0_d)
        x.block_until_ready()
        solve_t = time.time() - t1
        x_h = np.asarray(x)
    return dict(x=x_h,
                iters=int(iters[0]), status=int(status[0]),
                setup_s=setup_t, solve_s=solve_t,
                cfg_kind="PCG" if eq_name.lower().startswith("pd") else "BICGSTAB")


def compare_one(matrix_dir: Path, label: str, ts: str, eq: str,
                tol: float, do_truth: bool, n_refine: int = 0) -> dict:
    """Load matrix, solve, compute the 3 distance metrics. Returns row dict."""
    bundle = load_ofmm(matrix_dir)
    bundle = normalize_sign(bundle)
    A, b, x0, xf = bundle.A, bundle.b, bundle.x0, bundle.x_final
    n = A.shape[0]

    row = dict(case=label, ts=ts, eq=eq, n=n, nnz=int(A.nnz),
               sign_negated=bool(bundle.negated),
               of_iter=bundle.meta.get("solver_openfoam", {}).get("iterations"),
               of_init_res=bundle.meta.get("solver_openfoam", {}).get("initial_residual"),
               of_final_res=bundle.meta.get("solver_openfoam", {}).get("final_residual"),
               amgx_tol=tol, n_refine=n_refine)

    # AMGx (with optional iterative refinement)
    if n_refine > 0:
        eq_kind = "pd" if eq.lower().startswith("pd") else "T"
        ref = amgx_solve_with_refinement(A, b, x0, eq_kind=eq_kind, tol=tol,
                                          n_refine=n_refine, verbose=False)
        x_amgx = ref["x"]
        row["amgx_iters"] = ref["primary_iters"]
        row["amgx_status"] = ref["primary_status"]
        row["amgx_solve_s"] = ref["total_s"]
        row["amgx_setup_s"] = None
        row["amgx_cfg"] = "PCG" if eq_kind == "pd" else "BICGSTAB"
        row["refine_iters"] = ref["refine_iters"]
        row["refine_history"] = ref["refinement_history"]
    else:
        res = solve_with_amgx(A, b, x0, eq, tol=tol)
        row.update(dict(amgx_iters=res["iters"], amgx_status=res["status"],
                        amgx_setup_s=res["setup_s"], amgx_solve_s=res["solve_s"],
                        amgx_cfg=res["cfg_kind"]))
        x_amgx = res["x"]
    row["amgx_resid_rel2"] = float(np.linalg.norm(A @ x_amgx - b)
                                    / max(np.linalg.norm(b), 1e-300))

    # Truth (only if N small enough; else skip)
    if do_truth and n <= 64000:
        try:
            t0 = time.time()
            x_truth = spsolve(A.tocsc(), b)
            row["truth_solve_s"] = time.time() - t0
            denom_t = max(float(np.max(np.abs(x_truth))), 1e-300)
            row["amgx_vs_truth"] = float(np.max(np.abs(x_amgx - x_truth)) / denom_t)
            if xf is not None:
                row["of_vs_truth"] = float(np.max(np.abs(xf - x_truth)) / denom_t)
            row["truth_resid_rel2"] = float(np.linalg.norm(A @ x_truth - b)
                                            / max(np.linalg.norm(b), 1e-300))
        except Exception as e:
            row["truth_error"] = f"{type(e).__name__}:{e}"

    # AMGx vs OF (always)
    if xf is not None:
        denom_OF = max(float(np.max(np.abs(xf))), 1e-300)
        row["amgx_vs_OF"] = float(np.max(np.abs(x_amgx - xf)) / denom_OF)

    return row


def run(tol: float, do_truth: bool, out_stem: Path, n_refine: int = 0) -> None:
    rows = []
    print(f"[sweep] tol={tol}, truth={do_truth}, n_refine={n_refine}, "
          f"output={out_stem}")
    print()
    print(f"{'case':17s} {'ts':18s} {'eq':10s} {'N':>5s} "
          f"{'a_iter':>6s} {'a_resid':>9s} {'a_vs_truth':>11s} "
          f"{'a_vs_OF':>9s} {'OF_vs_truth':>12s} {'wall':>5s}")
    print("-" * 130)

    for label, ts, eq, mdir in list_dumps():
        try:
            r = compare_one(mdir, label, ts, eq, tol=tol, do_truth=do_truth,
                            n_refine=n_refine)
            rows.append(r)

            avt = r.get("amgx_vs_truth")
            avo = r.get("amgx_vs_OF")
            ovt = r.get("of_vs_truth")
            avt_s = f"{avt:.2e}" if avt is not None else "—"
            avo_s = f"{avo:.2e}" if avo is not None else "—"
            ovt_s = f"{ovt:.2e}" if ovt is not None else "—"
            print(f"{label:17s} {ts:18s} {eq:10s} {r['n']:>5d} "
                  f"{r['amgx_iters']:>6d} {r['amgx_resid_rel2']:>9.2e} "
                  f"{avt_s:>11s} {avo_s:>9s} {ovt_s:>12s} {r['amgx_solve_s']:>5.2f}")
        except Exception as e:
            print(f"{label}/{ts}/{eq}  ERROR  {type(e).__name__}: {e}")
            rows.append(dict(case=label, ts=ts, eq=eq, error=str(e)))

    # JSON dump
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    with open(out_stem.with_suffix(".json"), "w") as fh:
        json.dump(dict(date=datetime.now().isoformat(),
                       tol=tol, do_truth=do_truth,
                       rows=rows), fh, indent=2)

    # Aggregate stats
    valid = [r for r in rows if "amgx_vs_truth" in r]
    if valid:
        vt = [r["amgx_vs_truth"] for r in valid]
        print()
        print(f"AMGx vs truth — N={len(vt)}: "
              f"max={max(vt):.2e}  median={sorted(vt)[len(vt)//2]:.2e}  min={min(vt):.2e}")

    valid_OF = [r for r in rows if "of_vs_truth" in r]
    if valid_OF:
        vo = [r["of_vs_truth"] for r in valid_OF]
        print(f"OF vs truth   — N={len(vo)}: "
              f"max={max(vo):.2e}  median={sorted(vo)[len(vo)//2]:.2e}  min={min(vo):.2e}")
    print()
    print(f"Wrote {out_stem.with_suffix('.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--no-truth", action="store_true")
    ap.add_argument("--n-refine", type=int, default=0,
                    help="iterative refinement steps after primary AMGx solve")
    ap.add_argument("--out", type=Path,
                    default=Path("/home/yzk/DILU-Research/dilu/amgx/bench/precision_results"))
    args = ap.parse_args()
    run(args.tol, do_truth=not args.no_truth, out_stem=args.out,
        n_refine=args.n_refine)
