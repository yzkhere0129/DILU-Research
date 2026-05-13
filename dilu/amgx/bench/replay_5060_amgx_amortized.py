"""5060 AMGx amortized 384-step replay — prove or refute the amortized hypothesis.

Hypothesis (PROJECT_STATUS_REPORT §1.3):
    AMGx update_coefficients + warm-start should be 100-1000× faster than
    LU refactor per step on real CFD time-march.

Run 4 protocols over N ≤ 384 sane_pool matrices from dense_track_dump_500K:
    fresh_e8        each step: new Plan(tol=1e-8) + solve(x0=0)
    amortized_e8    setup once, per step: update_coefficients + solve(x0=x_prev)
    fresh_e12_IR    each step: new Plan(tol=1e-12) + solve + 1 IR
    amortized_e12_IR setup once, per step: update + solve(x0=x_prev) + 1 IR

For each step record:
    setup_s, update_s, solve_s, ir_s, total_s
    iters, status, rel_resid
    sign_flipped, n, nnz

Output:
    <out>/replay_<protocol>.npz       per-step arrays
    <out>/summary.json                 aggregate stats
    <out>/run_meta.json                env + GPU info

Usage (lab 5060):
    ~/jax-env/bin/python3 -u dilu/amgx/bench/replay_5060_amgx_amortized.py \\
        --case ~/cases/dense_track_dump_500K \\
        --pool validate_dense/sane_pool.txt \\
        --output-dir audit_overnight_20260509/lab_5060_replay \\
        --max 384 \\
        --protocols fresh_e8,amortized_e8,fresh_e12_IR,amortized_e12_IR
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# JAX env knobs — set BEFORE jax import
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from jax import config as _jc
_jc.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp

from dilu.amgx.python import (
    Plan, CLASSICAL_V_DIAGSCALED, with_tolerance,
)


# ----------------------------------------------------------------------------
# Matrix loading & sign normalization
# ----------------------------------------------------------------------------

def load_matrix(case: Path, ts_rel: str):
    """Load A, b for one timestep. Returns (A_csr_SPD, b_SPD, sign_flipped)."""
    sys_dir = case / "postProcessing" / "matrices" / ts_rel
    A = sio.mmread(str(sys_dir / "A.mm")).tocsr()
    b = np.asarray(sio.mmread(str(sys_dir / "b.mm"))).flatten()
    diag_mean = float(np.mean(A.diagonal()))
    sign_flipped = diag_mean < 0
    if sign_flipped:
        A = -A
        b = -b
    return A, b, sign_flipped


def read_pool(pool_path: Path) -> list[str]:
    rels = []
    for line in pool_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rels.append(line)
    # Sort by timestep (numeric, parsed from first dir)
    def key(rel):
        ts = rel.split("/")[0]
        try:
            return float(ts)
        except ValueError:
            return float("inf")
    rels.sort(key=key)
    return rels


# ----------------------------------------------------------------------------
# Protocol runners
# ----------------------------------------------------------------------------

PROTOCOL_CONFIGS = {
    "fresh_e8":          {"tol": 1e-8,  "n_ir": 0, "amortized": False},
    "amortized_e8":      {"tol": 1e-8,  "n_ir": 0, "amortized": True},
    "fresh_e12_IR":      {"tol": 1e-12, "n_ir": 1, "amortized": False},
    "amortized_e12_IR":  {"tol": 1e-12, "n_ir": 1, "amortized": True},
}


def run_protocol(name: str, bundles, base_config: str, max_iters: int):
    """Run one protocol over the bundle list. Returns dict of per-step arrays."""
    cfg = PROTOCOL_CONFIGS[name]
    tol = cfg["tol"]
    n_ir = cfg["n_ir"]
    amortized = cfg["amortized"]
    cfg_json = with_tolerance(base_config, tol, max_iters=max_iters)

    n_steps = len(bundles)
    # per-step records
    rec = {
        "step": np.arange(n_steps, dtype=np.int32),
        "ts": np.array([b["ts"] for b in bundles], dtype=object),
        "n": np.zeros(n_steps, dtype=np.int64),
        "nnz": np.zeros(n_steps, dtype=np.int64),
        "sign_flipped": np.zeros(n_steps, dtype=bool),
        "setup_s": np.zeros(n_steps),
        "update_s": np.zeros(n_steps),
        "solve_s": np.zeros(n_steps),
        "ir_s": np.zeros(n_steps),
        "total_s": np.zeros(n_steps),
        "iters": np.zeros(n_steps, dtype=np.int32),
        "status": np.zeros(n_steps, dtype=np.int32),
        "rel_resid": np.zeros(n_steps),
    }

    plan = None
    x_prev_np = None  # for warm-start (amortized only)

    print(f"\n=== protocol: {name}  tol={tol}  n_ir={n_ir}  amortized={amortized} ===")
    t_proto_start = time.time()

    for k, bun in enumerate(bundles):
        A = bun["A"]
        b = bun["b"]
        sign_flipped = bun["sign_flipped"]
        n = A.shape[0]
        nnz = A.nnz
        rec["n"][k] = n
        rec["nnz"][k] = nnz
        rec["sign_flipped"][k] = sign_flipped

        rp = jnp.asarray(A.indptr.astype(np.int32))
        ci = jnp.asarray(A.indices.astype(np.int32))
        vv = jnp.asarray(A.data.astype(np.float64))
        b_d = jnp.asarray(b)

        # ---- setup / update ----
        if amortized:
            if plan is None:
                t0 = time.time()
                plan = Plan(rp, ci, vv, cfg_json)
                rec["setup_s"][k] = time.time() - t0
            else:
                t0 = time.time()
                plan.update_coefficients(vv)
                rec["update_s"][k] = time.time() - t0
            this_plan = plan
        else:
            # fresh per step
            if plan is not None:
                plan.release(); plan = None
            t0 = time.time()
            this_plan = Plan(rp, ci, vv, cfg_json)
            rec["setup_s"][k] = time.time() - t0

        # ---- warm-start x0 ----
        if amortized and x_prev_np is not None and x_prev_np.shape == (n,):
            x0 = jnp.asarray(x_prev_np)
        else:
            x0 = jnp.zeros((n,), dtype=jnp.float64)

        # ---- solve ----
        t0 = time.time()
        x_jax, iters_dev, status_dev = this_plan.solve(b_d, x0)
        x_jax.block_until_ready()
        iters_dev.block_until_ready()
        status_dev.block_until_ready()
        rec["solve_s"][k] = time.time() - t0
        rec["iters"][k] = int(np.asarray(iters_dev)[0])
        rec["status"][k] = int(np.asarray(status_dev)[0])
        x = np.asarray(x_jax)

        # ---- iterative refinement ----
        t_ir = 0.0
        for _ in range(n_ir):
            r_np = b - A @ x
            zero = jnp.zeros((n,), dtype=jnp.float64)
            t0 = time.time()
            d_jax, _, _ = this_plan.solve(jnp.asarray(r_np), zero)
            d_jax.block_until_ready()
            t_ir += time.time() - t0
            x = x + np.asarray(d_jax)
        rec["ir_s"][k] = t_ir

        # ---- precision check ----
        rec["rel_resid"][k] = float(
            np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300)
        )
        rec["total_s"][k] = (rec["setup_s"][k] + rec["update_s"][k]
                             + rec["solve_s"][k] + rec["ir_s"][k])

        if not amortized:
            this_plan.release(); this_plan = None

        x_prev_np = x  # for amortized warm-start

        if k % 10 == 0 or k == n_steps - 1:
            cum = time.time() - t_proto_start
            print(f"  [{k+1}/{n_steps}] ts={bun['ts']:>12}  "
                  f"setup={rec['setup_s'][k]*1000:6.0f}ms  "
                  f"upd={rec['update_s'][k]*1000:6.0f}ms  "
                  f"solve={rec['solve_s'][k]*1000:6.0f}ms  "
                  f"ir={rec['ir_s'][k]*1000:5.0f}ms  "
                  f"iter={rec['iters'][k]:4d}  "
                  f"resid={rec['rel_resid'][k]:.2e}  "
                  f"cum={cum/60:.1f}min")

        # free per-step GPU buffers
        del rp, ci, vv, b_d, x0, x_jax, iters_dev, status_dev
        if k % 20 == 0:
            gc.collect()

    if plan is not None:
        plan.release()

    rec["protocol_wall_s"] = time.time() - t_proto_start
    return rec


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="OF case dir containing postProcessing/matrices/")
    ap.add_argument("--pool", required=True, help="sane_pool.txt path")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max", type=int, default=384, help="cap number of matrices")
    ap.add_argument("--protocols", default="fresh_e8,amortized_e8,fresh_e12_IR,amortized_e12_IR")
    ap.add_argument("--config", choices=["DIAGSCALED"], default="DIAGSCALED",
                    help="base AMGx config (DIAGSCALED = CLASSICAL_V_DIAGSCALED)")
    ap.add_argument("--max-iters", type=int, default=2000,
                    help="AMGx max outer PCG iters (default 2000 to give tol=1e-12 room)")
    args = ap.parse_args()

    case = Path(args.case).expanduser()
    pool = Path(args.pool).expanduser()
    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    rels = read_pool(pool)
    if len(rels) > args.max:
        rels = rels[:args.max]
    print(f"# Loading {len(rels)} matrices from {case} ...")

    # Pre-load ALL matrices once to avoid I/O noise dominating timing
    # (500K LPBF: ~3.4M nnz, ~80MB ASCII → ~25MB in memory each; 384 × 25MB = ~10GB.
    #  If RAM is tight, switch to lazy load.)
    bundles = []
    t0 = time.time()
    for i, rel in enumerate(rels):
        A, b, sf = load_matrix(case, rel)
        bundles.append({"ts": rel.split("/")[0], "rel": rel,
                        "A": A, "b": b, "sign_flipped": sf})
        if i % 20 == 0:
            print(f"  loaded {i+1}/{len(rels)}  cum={time.time()-t0:.1f}s")
    print(f"  loaded all {len(rels)} matrices in {time.time()-t0:.1f}s")

    base_config = CLASSICAL_V_DIAGSCALED

    # Env meta
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True).strip()
    except Exception as e:
        smi = f"nvidia-smi err: {e}"
    run_meta = {
        "case": str(case),
        "pool": str(pool),
        "n_matrices": len(rels),
        "n_per_matrix": int(bundles[0]["A"].shape[0]) if bundles else None,
        "nnz_per_matrix": int(bundles[0]["A"].nnz) if bundles else None,
        "protocols": args.protocols.split(","),
        "config": args.config,
        "max_iters": args.max_iters,
        "gpu": smi,
        "python": sys.version,
        "jax": jax.__version__,
        "wall_started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    print(f"\nGPU: {smi}")

    # Run each protocol
    summary = {}
    for proto in args.protocols.split(","):
        proto = proto.strip()
        if proto not in PROTOCOL_CONFIGS:
            print(f"  skip unknown protocol: {proto}"); continue
        rec = run_protocol(proto, bundles, base_config, args.max_iters)

        # Save raw npz
        npz_path = out / f"replay_{proto}.npz"
        save_kwargs = {k: v for k, v in rec.items() if k != "protocol_wall_s"}
        save_kwargs["protocol_wall_s"] = np.array([rec["protocol_wall_s"]])
        # ts is object dtype list of strings; convert to fixed unicode
        save_kwargs["ts"] = np.array(save_kwargs["ts"], dtype="U16")
        np.savez_compressed(npz_path, **save_kwargs)
        print(f"  → {npz_path}")

        # Aggregate
        agg = {
            "n_steps": int(len(rec["step"])),
            "wall_total_s": float(rec["protocol_wall_s"]),
            "setup_s_total": float(np.sum(rec["setup_s"])),
            "update_s_total": float(np.sum(rec["update_s"])),
            "solve_s_total": float(np.sum(rec["solve_s"])),
            "ir_s_total": float(np.sum(rec["ir_s"])),
            "iter_mean": float(np.mean(rec["iters"])),
            "iter_median": float(np.median(rec["iters"])),
            "iter_min": int(np.min(rec["iters"])),
            "iter_max": int(np.max(rec["iters"])),
            "rel_resid_max": float(np.max(rec["rel_resid"])),
            "rel_resid_median": float(np.median(rec["rel_resid"])),
            "n_status_nonzero": int(np.sum(rec["status"] != 0)),
        }
        summary[proto] = agg
        print(f"  {proto}: total={agg['wall_total_s']/60:.2f}min  "
              f"iter_med={agg['iter_median']:.0f}  "
              f"resid_max={agg['rel_resid_max']:.2e}  "
              f"failed={agg['n_status_nonzero']}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWritten summary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
