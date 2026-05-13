"""5060 AMGx config-selection sanity check — pick fastest stable config.

Runs 5 candidate AMGx configs × 2 tolerances on K=5 sane matrices.
Reports per-config mean wall + iter + final residual.
Use this BEFORE the 384-step replay to confirm CLASSICAL_V_DIAGSCALED
is still optimal on 5060 (it was settled in v3.2 audit on 3050 dev).

Usage:
    ~/jax-env/bin/python3 -u dilu/amgx/bench/bench_5060_config_select.py \\
        --case ~/cases/dense_track_dump_500K \\
        --pool validate_dense/sane_pool.txt \\
        --k 5 \\
        --out audit_overnight_20260509/lab_5060_replay/config_select.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np
import scipy.io as sio

from jax import config as _jc
_jc.update("jax_enable_x64", True)
import jax.numpy as jnp

from dilu.amgx.python import (
    Plan,
    CLASSICAL_V_DIAGSCALED,
    CLASSICAL_V_CYCLE,
    CLASSICAL_GS_PCG,
    AGGREGATION_PCG,
    AGGRESSIVE_COARSENING,
    with_tolerance,
)


CANDIDATES = {
    "DIAGSCALED":           CLASSICAL_V_DIAGSCALED,
    "CLASSICAL_V":          CLASSICAL_V_CYCLE,
    "CLASSICAL_GS_PCG":     CLASSICAL_GS_PCG,
    "AGGREGATION_PCG":      AGGREGATION_PCG,
    "AGGRESSIVE_COARSEN":   AGGRESSIVE_COARSENING,
}


def load_one(case: Path, rel: str):
    sys_dir = case / "postProcessing" / "matrices" / rel
    A = sio.mmread(str(sys_dir / "A.mm")).tocsr()
    b = np.asarray(sio.mmread(str(sys_dir / "b.mm"))).flatten()
    if float(np.mean(A.diagonal())) < 0:
        A = -A; b = -b
    return A, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--k", type=int, default=5, help="sample K matrices, evenly spaced")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-iters", type=int, default=2000)
    args = ap.parse_args()

    case = Path(args.case).expanduser()
    pool = Path(args.pool).expanduser()
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    rels = [l.strip() for l in pool.read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    rels.sort(key=lambda s: float(s.split("/")[0]))
    idx = np.linspace(0, len(rels)-1, args.k, dtype=int)
    samples = [rels[i] for i in idx]
    print(f"# {len(rels)} pool, sampling {args.k} indices: {idx.tolist()}")

    matrices = []
    for rel in samples:
        A, b = load_one(case, rel)
        matrices.append((rel.split("/")[0], A, b))
        print(f"  loaded {rel}  N={A.shape[0]} nnz={A.nnz}")

    results = {}
    for name, base_cfg in CANDIDATES.items():
        for tol in (1e-8, 1e-12):
            tag = f"{name}__tol{tol:g}"
            print(f"\n=== {tag} ===")
            cfg_json = with_tolerance(base_cfg, tol, max_iters=args.max_iters)
            rows = []
            for ts, A, b in matrices:
                rp = jnp.asarray(A.indptr.astype(np.int32))
                ci = jnp.asarray(A.indices.astype(np.int32))
                vv = jnp.asarray(A.data.astype(np.float64))
                b_d = jnp.asarray(b)
                try:
                    t0 = time.time()
                    plan = Plan(rp, ci, vv, cfg_json)
                    t_setup = time.time() - t0

                    t0 = time.time()
                    x_d, it_d, st_d = plan.solve(b_d, jnp.zeros_like(b_d))
                    x_d.block_until_ready()
                    t_solve = time.time() - t0
                    iters = int(np.asarray(it_d)[0])
                    status = int(np.asarray(st_d)[0])
                    x = np.asarray(x_d)
                    rel_resid = float(np.linalg.norm(A @ x - b) /
                                       max(np.linalg.norm(b), 1e-300))
                    plan.release()
                except Exception as e:
                    print(f"  {ts}: FAILED {e!r}")
                    rows.append({"ts": ts, "ok": False, "err": repr(e)})
                    continue
                rows.append({
                    "ts": ts, "ok": True,
                    "setup_s": t_setup, "solve_s": t_solve,
                    "total_s": t_setup + t_solve,
                    "iters": iters, "status": status, "rel_resid": rel_resid,
                })
                print(f"  {ts}: setup={t_setup*1000:.0f}ms solve={t_solve*1000:.0f}ms "
                      f"iter={iters} st={status} resid={rel_resid:.2e}")
            ok = [r for r in rows if r.get("ok")]
            agg = None
            if ok:
                agg = {
                    "n_ok": len(ok),
                    "setup_s_mean": float(np.mean([r["setup_s"] for r in ok])),
                    "solve_s_mean": float(np.mean([r["solve_s"] for r in ok])),
                    "total_s_mean": float(np.mean([r["total_s"] for r in ok])),
                    "iter_mean":   float(np.mean([r["iters"]   for r in ok])),
                    "rel_resid_max": float(max(r["rel_resid"] for r in ok)),
                    "n_status_nonzero": sum(1 for r in ok if r["status"] != 0),
                }
                print(f"  → mean total={agg['total_s_mean']*1000:.0f}ms  "
                      f"iter_mean={agg['iter_mean']:.0f}  "
                      f"resid_max={agg['rel_resid_max']:.2e}  "
                      f"failed_status={agg['n_status_nonzero']}")
            results[tag] = {"rows": rows, "agg": agg}

    # Pick best per tol (lowest total_s_mean with n_status_nonzero==0)
    print("\n========== WINNERS ==========")
    for tol_str in ("1e-08", "1e-12"):
        cands = [(tag, r["agg"]) for tag, r in results.items()
                 if tag.endswith(f"__tol{tol_str}") and r["agg"]
                 and r["agg"]["n_status_nonzero"] == 0]
        if not cands:
            print(f"  tol={tol_str}: NO CONVERGED CONFIG")
            continue
        cands.sort(key=lambda t: t[1]["total_s_mean"])
        print(f"  tol={tol_str}:")
        for tag, agg in cands:
            print(f"    {tag}: total_mean={agg['total_s_mean']*1000:.0f}ms "
                  f"iter_mean={agg['iter_mean']:.0f} resid_max={agg['rel_resid_max']:.2e}")
        results[f"winner_tol{tol_str}"] = cands[0][0]

    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
