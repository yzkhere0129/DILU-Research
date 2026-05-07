"""Lab 5060 driver: 32-rank lab Xeon dump → AMGx truth → compact npz for 3D plot.

Per-phase pipeline (one timestep per phase: melting + evaporation):
  1. reconstruct GLOBAL (A, b, x_OF) from 32 processor*/postProcessing/matrices/
     (calls dilu.amgx.bench.reconstruct_lab32_to_global)
  2. sign-flip → positive-diag AMGx convention
  3. AMGx tol=1e-12  → x_truth   (timed: setup + solve, iter, rel_resid)
  4. AMGx tol=1e-8   → x_AMGx_e8 (timed: setup + solve, iter, rel_resid)
  5. compute structured-mesh cell coords (i,j,k → x,y,z)  -- skips reading C
  6. save compact npz: x_OF, x_AMGx_e8, x_truth, b, ijk, timings

Output (per phase): /tmp/lab32_plot_<phase>_<eq>_<time>.npz
                     (x_*, b, i,j,k arrays + timing dict + meta)

Run on lab 5060 (needs CUDA + AMGx):
    cd ~/DILU-Research && git pull origin main
    # Make sure dump is local (rsync from HR54WV2 first if needed):
    #   rsync -avz manyxu@HR54WV2:~/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_lab32_dump/ \
    #         ~/lab32_dump/
    python3 -u -m dilu.amgx.bench.prepare_lab32_plot_data \\
        --case ~/lab32_dump \\
        --time 3.8e-07 --eq pd_corr0 --phase melting

    python3 -u -m dilu.amgx.bench.prepare_lab32_plot_data \\
        --case ~/lab32_dump \\
        --time 9.0e-07 --eq pd_corr0 --phase evaporation

Then scp the small npz to dev:
    scp /tmp/lab32_plot_*.npz dev:~/DILU-Research/dilu/amgx/bench/
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from jax import config as _jc
_jc.update("jax_enable_x64", True)
import jax.numpy as jnp

from dilu.amgx.python import Plan, CLASSICAL_V_DIAGSCALED, with_tolerance
from dilu.amgx.bench.sweep_amgx_senior_data import normalize_sign

# Default mesh geometry (matches LPBF_crosscheck which lab32 was copied from).
# Override via CLI if mesh changed.
NX, NY, NZ = 80, 320, 80
DX = 2.5e-6  # 2.5 μm cells


def amgx_solve(A_csr: csr_matrix, b: np.ndarray, *, tol: float,
                max_iter: int = 500, n_refine: int = 0):
    """Run AMGx CLASSICAL_V_DIAGSCALED at given tol.

    Returns dict with x, iters, status, t_setup, t_solve, rel_resid_actual.
    Times include H2D/D2H + .block_until_ready() (true wall).
    """
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

    rel_resid = float(np.linalg.norm(A_csr @ x_h - b)
                      / max(np.linalg.norm(b), 1e-300))
    return dict(
        x=x_h,
        iters=int(iters[0]),
        status=int(status[0]),
        t_setup=t_setup,
        t_solve=t_solve,
        rel_resid_actual=rel_resid,
        tol_requested=tol,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True,
                     help="Path to LPBF_lab32_dump (contains processor*/)")
    ap.add_argument("--time", required=True,
                     help="OF time directory name, e.g. 3.8e-07")
    ap.add_argument("--eq", default="pd_corr0",
                     help="Equation label, default pd_corr0")
    ap.add_argument("--phase", required=True,
                     help="Phase tag for output filename (melting/evaporation)")
    ap.add_argument("--nx", type=int, default=NX)
    ap.add_argument("--ny", type=int, default=NY)
    ap.add_argument("--nz", type=int, default=NZ)
    ap.add_argument("--dx", type=float, default=DX)
    ap.add_argument("--reconstruct-out", default="/tmp",
                     help="dir for reconstruct intermediate npz")
    ap.add_argument("--out", default="/tmp",
                     help="dir for final compact plot npz")
    args = ap.parse_args()

    case = Path(args.case).expanduser()
    intermed = Path(args.reconstruct_out) / f"global_{args.eq}_{args.time}.npz"

    timings = {}

    # ---------- 1. reconstruct ----------
    print(f"\n[1/4] Reconstructing GLOBAL matrix from 32 procs ...", flush=True)
    t0 = time.time()
    cmd = [sys.executable, "-u", "-m",
            "dilu.amgx.bench.reconstruct_lab32_to_global",
            "--case", str(case),
            "--time", args.time,
            "--eq", args.eq,
            "--out", args.reconstruct_out]
    print(f"  $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, check=True)
    timings["reconstruct_s"] = time.time() - t0
    print(f"  reconstruct wall = {timings['reconstruct_s']:.2f} s", flush=True)

    # ---------- 2. load + sign-flip ----------
    print(f"\n[2/4] Loading + sign-flipping ...", flush=True)
    t0 = time.time()
    z = np.load(intermed)
    n = int(z["n_global"][0])
    A = csr_matrix((z["A_data"], z["A_indices"], z["A_indptr"]),
                    shape=(n, n))
    b = z["b"]
    x_OF = z["x_final"]
    rel_OF = float(np.linalg.norm(A @ x_OF - b)
                    / max(np.linalg.norm(b), 1e-300))
    print(f"  N = {n}  nnz = {A.nnz}", flush=True)
    print(f"  ‖A·x_OF - b‖/‖b‖ = {rel_OF:.3e}  "
          f"(should be ≤ ~1e-8 for clean dump)", flush=True)

    A_pos, b_pos, flipped = normalize_sign(A, b)
    print(f"  sign-flipped: {flipped}", flush=True)
    timings["load_signflip_s"] = time.time() - t0

    # ---------- 3. AMGx tol=1e-12 (truth) ----------
    print(f"\n[3/4] AMGx tol=1e-12 (truth proxy) ...", flush=True)
    res_truth = amgx_solve(A_pos, b_pos, tol=1e-12, max_iter=500)
    timings["amgx_truth_setup_s"]  = res_truth["t_setup"]
    timings["amgx_truth_solve_s"]  = res_truth["t_solve"]
    timings["amgx_truth_total_s"]  = res_truth["t_setup"] + res_truth["t_solve"]
    print(f"  tol=1e-12: iter={res_truth['iters']}  "
          f"setup={res_truth['t_setup']*1e3:.1f} ms  "
          f"solve={res_truth['t_solve']*1e3:.1f} ms  "
          f"rel_resid={res_truth['rel_resid_actual']:.2e}", flush=True)

    # ---------- 3b. AMGx tol=1e-8 ----------
    print(f"\n[3b/4] AMGx tol=1e-8 ...", flush=True)
    res_e8 = amgx_solve(A_pos, b_pos, tol=1e-8, max_iter=500)
    timings["amgx_e8_setup_s"]  = res_e8["t_setup"]
    timings["amgx_e8_solve_s"]  = res_e8["t_solve"]
    timings["amgx_e8_total_s"]  = res_e8["t_setup"] + res_e8["t_solve"]
    print(f"  tol=1e-8:  iter={res_e8['iters']}  "
          f"setup={res_e8['t_setup']*1e3:.1f} ms  "
          f"solve={res_e8['t_solve']*1e3:.1f} ms  "
          f"rel_resid={res_e8['rel_resid_actual']:.2e}", flush=True)

    # The sign-flip means A_pos·x_pos = b_pos. Original system was -A·x = -b.
    # Solutions x_pos and x_orig coincide (both solve A x = b, just sign convention
    # of A & b is flipped together). So x_truth from AMGx == x for original.
    x_truth = res_truth["x"]
    x_AMGx_e8 = res_e8["x"]

    # ---------- 4. structured-mesh coords + pack ----------
    print(f"\n[4/4] Computing cell coords + packing ...", flush=True)
    expected_n = args.nx * args.ny * args.nz
    if n != expected_n:
        print(f"  ⚠ N={n} != nx*ny*nz={expected_n}; mesh shape arg mismatch.",
              flush=True)
        print(f"  ⚠ Falling back to flat 1-D index (no spatial layout)", flush=True)
        i_idx = np.arange(n, dtype=np.int32)
        j_idx = np.zeros(n, dtype=np.int32)
        k_idx = np.zeros(n, dtype=np.int32)
    else:
        cid = np.arange(n, dtype=np.int64)
        i_idx = (cid % args.nx).astype(np.int32)
        j_idx = ((cid // args.nx) % args.ny).astype(np.int32)
        k_idx = (cid // (args.nx * args.ny)).astype(np.int32)

    # ---------- save ----------
    out = Path(args.out) / f"lab32_plot_{args.phase}_{args.eq}_{args.time}.npz"
    np.savez_compressed(
        out,
        x_OF=x_OF.astype(np.float64),
        x_AMGx_e8=x_AMGx_e8.astype(np.float64),
        x_truth=x_truth.astype(np.float64),
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
    print(f"\n→ {out}", flush=True)

    # ---------- console summary ----------
    print(f"\n{'='*70}\nSUMMARY  ({args.phase} @ t={args.time}, {args.eq})\n{'='*70}")
    print(f"  N = {n}, nnz = {A.nnz}")
    print(f"  reconstruct                    : {timings['reconstruct_s']*1e3:>9.1f} ms")
    print(f"  load + sign-flip               : {timings['load_signflip_s']*1e3:>9.1f} ms")
    print(f"  AMGx tol=1e-12 (truth) setup   : {timings['amgx_truth_setup_s']*1e3:>9.1f} ms")
    print(f"  AMGx tol=1e-12 (truth) solve   : {timings['amgx_truth_solve_s']*1e3:>9.1f} ms  "
          f"({res_truth['iters']} iter, rel_resid={res_truth['rel_resid_actual']:.2e})")
    print(f"  AMGx tol=1e-8         setup   : {timings['amgx_e8_setup_s']*1e3:>9.1f} ms")
    print(f"  AMGx tol=1e-8         solve   : {timings['amgx_e8_solve_s']*1e3:>9.1f} ms  "
          f"({res_e8['iters']} iter, rel_resid={res_e8['rel_resid_actual']:.2e})")
    print(f"\n  OF dump consistency: ‖A·x_OF - b‖/‖b‖ = {rel_OF:.2e}")
    print(f"\n  Done. scp {out} to dev:~/DILU-Research/dilu/amgx/bench/")


if __name__ == "__main__":
    main()
