"""Single-core dev driver for laserMeltFoam matrix dump → AMGx truth → plot npz.

Differs from prepare_lab32_plot_data.py: NO reconstruct step (single-rank
dump is already a global matrix at postProcessing/matrices/<t>/<eq>/A.mm).

Pipeline per (timestep, equation):
  1. Read A.mm, b.mm, x_final.mm directly from dump
  2. Sign-flip → positive-diag for AMGx
  3. AMGx tol=1e-12 (truth)  + AMGx tol=1e-8
  4. Save compact npz with timing + iter + actual residual stats

Usage on dev (after extracting tar from lab Xeon):
    cd ~/DILU-Research
    source ~/jax-env/bin/activate
    python3 -u -m dilu.amgx.bench.prepare_single_core_plot_data \\
        --case ~/single_track_dump \\
        --time 3.8e-07 --eq pd_corr0 --phase melting

Output: dilu/amgx/bench/single_<phase>_<eq>_<time>.npz
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix

from jax import config as _jc
_jc.update("jax_enable_x64", True)
import jax.numpy as jnp

from dilu.amgx.python import Plan, CLASSICAL_V_DIAGSCALED, with_tolerance
from dilu.amgx.bench.sweep_amgx_senior_data import normalize_sign


def amgx_solve(A_csr: csr_matrix, b: np.ndarray, *, tol: float, max_iter: int = 500):
    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, tol, max_iters=max_iter)
    rp = jnp.asarray(A_csr.indptr.astype(np.int32))
    ci = jnp.asarray(A_csr.indices.astype(np.int32))
    vv = jnp.asarray(A_csr.data.astype(np.float64))
    b_d = jnp.asarray(b.astype(np.float64))
    x0_d = jnp.zeros_like(b_d)
    t0 = time.time()
    with Plan(rp, ci, vv, cfg) as plan:
        t_setup = time.time() - t0
        t1 = time.time()
        x, iters, status = plan.solve(b_d, x0_d)
        x.block_until_ready()
        t_solve = time.time() - t1
        x_h = np.asarray(x)
    rel = float(np.linalg.norm(A_csr @ x_h - b) / max(np.linalg.norm(b), 1e-300))
    return dict(x=x_h, iters=int(iters[0]), status=int(status[0]),
                t_setup=t_setup, t_solve=t_solve,
                rel_resid_actual=rel, tol_requested=tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="path to extracted single-track case")
    ap.add_argument("--time", required=True, help="OF timeName, e.g. 3.8e-07")
    ap.add_argument("--eq", default="pd_corr0", help="pd_corr{0,1,2} or T_corr0")
    ap.add_argument("--phase", required=True, help="tag for output filename")
    ap.add_argument("--nx", type=int, default=50)
    ap.add_argument("--ny", type=int, default=200)
    ap.add_argument("--nz", type=int, default=50)
    ap.add_argument("--dx", type=float, default=4e-6)
    ap.add_argument("--out-dir", default="dilu/amgx/bench")
    args = ap.parse_args()

    case = Path(args.case).expanduser()
    eq_dir = case / "postProcessing" / "matrices" / args.time / args.eq
    if not eq_dir.exists():
        raise SystemExit(f"Not found: {eq_dir}")

    timings = {}

    # ---------- 1. read matrices directly ----------
    print(f"\n[1/3] Reading {eq_dir} ...", flush=True)
    t0 = time.time()
    A = sio.mmread(str(eq_dir / "A.mm")).tocsr()
    b = sio.mmread(str(eq_dir / "b.mm")).flatten()
    x_OF = sio.mmread(str(eq_dir / "x_final.mm")).flatten()
    timings["load_s"] = time.time() - t0
    n = A.shape[0]
    print(f"  N = {n}, nnz = {A.nnz}", flush=True)

    rel_OF = float(np.linalg.norm(A @ x_OF - b) / max(np.linalg.norm(b), 1e-300))
    print(f"  ‖A·x_OF - b‖/‖b‖ = {rel_OF:.3e}", flush=True)

    A_pos, b_pos, flipped = normalize_sign(A, b)
    print(f"  sign-flipped: {flipped}", flush=True)

    # ---------- 2. AMGx truth ----------
    print(f"\n[2/3] AMGx tol=1e-12 (truth) ...", flush=True)
    res_truth = amgx_solve(A_pos, b_pos, tol=1e-12)
    timings["amgx_truth_setup_s"] = res_truth["t_setup"]
    timings["amgx_truth_solve_s"] = res_truth["t_solve"]
    print(f"  iter={res_truth['iters']}  setup={res_truth['t_setup']*1e3:.1f}ms  "
          f"solve={res_truth['t_solve']*1e3:.1f}ms  "
          f"rel_resid={res_truth['rel_resid_actual']:.2e}", flush=True)

    # ---------- 2b. AMGx 1e-8 ----------
    print(f"\n[2b] AMGx tol=1e-8 ...", flush=True)
    res_e8 = amgx_solve(A_pos, b_pos, tol=1e-8)
    timings["amgx_e8_setup_s"] = res_e8["t_setup"]
    timings["amgx_e8_solve_s"] = res_e8["t_solve"]
    print(f"  iter={res_e8['iters']}  setup={res_e8['t_setup']*1e3:.1f}ms  "
          f"solve={res_e8['t_solve']*1e3:.1f}ms  "
          f"rel_resid={res_e8['rel_resid_actual']:.2e}", flush=True)

    # ---------- 3. pack ----------
    print(f"\n[3/3] Packing npz ...", flush=True)
    expected_n = args.nx * args.ny * args.nz
    if n != expected_n:
        print(f"  ⚠ N={n} != {args.nx}*{args.ny}*{args.nz}={expected_n}; "
              f"falling back to flat 1-D index", flush=True)
        i_idx = np.arange(n, dtype=np.int32)
        j_idx = np.zeros(n, dtype=np.int32)
        k_idx = np.zeros(n, dtype=np.int32)
    else:
        cid = np.arange(n, dtype=np.int64)
        i_idx = (cid % args.nx).astype(np.int32)
        j_idx = ((cid // args.nx) % args.ny).astype(np.int32)
        k_idx = (cid // (args.nx * args.ny)).astype(np.int32)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"single_{args.phase}_{args.eq}_{args.time}.npz"
    np.savez_compressed(
        out,
        x_OF=x_OF.astype(np.float64),
        x_AMGx_e8=res_e8["x"].astype(np.float64),
        x_truth=res_truth["x"].astype(np.float64),
        b=b.astype(np.float64),
        i=i_idx, j=j_idx, k=k_idx,
        n=np.array([n], dtype=np.int64),
        nx=np.array([args.nx], dtype=np.int32),
        ny=np.array([args.ny], dtype=np.int32),
        nz=np.array([args.nz], dtype=np.int32),
        dx=np.array([args.dx], dtype=np.float64),
        meta=np.array([json.dumps(dict(
            phase=args.phase, time=args.time, eq=args.eq,
            n_global=n, nnz=int(A.nnz),
            rel_OF_consistency=rel_OF,
            sign_flipped=flipped,
            amgx_truth=dict(
                iters=res_truth["iters"],
                rel_resid_actual=res_truth["rel_resid_actual"],
                t_setup_s=res_truth["t_setup"],
                t_solve_s=res_truth["t_solve"],
                tol_requested=res_truth["tol_requested"],
            ),
            amgx_e8=dict(
                iters=res_e8["iters"],
                rel_resid_actual=res_e8["rel_resid_actual"],
                t_setup_s=res_e8["t_setup"],
                t_solve_s=res_e8["t_solve"],
                tol_requested=res_e8["tol_requested"],
            ),
            timings_s=timings,
        ))], dtype=object),
    )
    print(f"\n→ {out}  ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
