"""Lab Xeon — pure matrix-calculator benchmark with our C++ PCG (DIC).

Reads raw CSV directly from:
  ~/DILU-Research/dilu/benchmark/Melting/Melting/      (33 matrices)
  ~/DILU-Research/dilu/benchmark/Evaporation/Evaporation/ (57 matrices)

For each matrix bundle:
  1. normalize_sign  — flip A,b so diag becomes positive (DIC requires +diag)
  2. b ← b - mean(b) — project to col(A) for Neumann compatibility
  3. C++ PCG-DIC, x0=0, tol=1e-10, max_iter=500
  4. record iter / wall / converged / rel_resid_actual = ‖A·x - b‖₂/‖b‖₂

Pre-loads all matrices upfront so timing measures only solver wall, not I/O.
Solver-internal numpy arrays go through our pybind11 bindings → numba @njit
fallback if the C++ .so isn't built (still byte-exact, just ~10× slower).

Output: /tmp/bench_pcg_xeon.{json,log}

Run on lab Xeon (HR54WV2):
    cd ~/DILU-Research && git pull origin main
    cd dilu/openfoam_cpu/cpp && ./build.sh         # one-time
    cd ~/DILU-Research
    python3 -u -m dilu.amgx.bench.bench_pcg_xeon \\
        > /tmp/bench_pcg_xeon.log 2>&1 &
    tail -f /tmp/bench_pcg_xeon.log
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from dilu.amgx.bench.senior_data_loader import (
    set_dataset, list_available, load_step,
)
from dilu.amgx.bench.sweep_amgx_senior_data import normalize_sign
from dilu.openfoam_cpu.python.ldu import csr_to_ldu
from dilu.openfoam_cpu.python import pcg


def project_col_space(b: np.ndarray) -> np.ndarray:
    return b - b.mean()


def preload(name: str) -> list:
    """Load all bundles for one dataset into memory; pre-apply sign-flip + b-proj."""
    set_dataset(name)
    avail = list_available()
    print(f"\n[{name}] preloading {len(avail)} bundles ...", flush=True)
    out = []
    t0 = time.time()
    for i, (s, c) in enumerate(avail, 1):
        bundle = load_step(s, c)
        if bundle is None:
            print(f"  {s}/{c}: missing, skip"); continue
        A_pos, b_pos, flipped = normalize_sign(bundle.A, bundle.b)
        b_proj = project_col_space(b_pos)
        out.append(dict(
            step=bundle.step, corr=bundle.corr,
            t=bundle.t_seconds, n=bundle.n,
            A=A_pos, b=b_proj,
            x_ref=bundle.x_ref, x0_pre=bundle.x0_pre,
            sign_flipped=flipped,
        ))
        if i % 10 == 0 or i == len(avail):
            print(f"  loaded {i}/{len(avail)}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  done — {len(out)} bundles, {time.time()-t0:.1f}s total", flush=True)
    return out


def run_one(it: dict, *, tol: float, max_iter: int) -> dict:
    A, b = it["A"], it["b"]
    ldu = csr_to_ldu(A)

    x0 = np.zeros_like(b)
    t0 = time.time()
    res = pcg.solve(ldu, b, x0, tolerance=tol, min_iter=1, max_iter=max_iter)
    wall = time.time() - t0

    rel_resid = float(np.linalg.norm(A @ res.x - b)
                       / max(np.linalg.norm(b), 1e-300))
    return dict(
        step=it["step"], corr=it["corr"], t=it["t"], n=it["n"],
        iters=res.n_iterations, converged=bool(res.converged),
        singular=bool(res.singular),
        wall_ms=wall * 1e3,
        r_init=res.initial_residual,
        r_final=res.final_residual,
        rel_resid_actual=rel_resid,
        sign_flipped=it["sign_flipped"],
    )


def summarize(name: str, rows: list[dict]):
    n = len(rows)
    n_conv = sum(1 for r in rows if r["converged"])
    walls = np.array([r["wall_ms"] for r in rows])
    iters = np.array([r["iters"]   for r in rows])
    resids = np.array([r["rel_resid_actual"] for r in rows])
    walls_c = np.array([r["wall_ms"] for r in rows if r["converged"]] or [0])
    iters_c = np.array([r["iters"]   for r in rows if r["converged"]] or [1])
    print()
    print(f"  Summary [{name}]: {n_conv}/{n} converged")
    print(f"    wall_ms median (all)    {np.median(walls):.1f}")
    print(f"    wall_ms median (conv)   {np.median(walls_c):.1f}")
    print(f"    iter    median (all)    {int(np.median(iters))}")
    print(f"    iter    median (conv)   {int(np.median(iters_c))}")
    print(f"    per-iter (conv) ms/it   {np.median(walls_c / iters_c):.2f}")
    print(f"    rel_resid max / median  {resids.max():.2e} / {np.median(resids):.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--datasets", default="melting,evaporation")
    ap.add_argument("--out", default="/tmp/bench_pcg_xeon.json")
    args = ap.parse_args()

    print(f"bench_pcg_xeon — C++ PCG (DIC) as matrix calculator")
    print(f"  tol={args.tol:.0e}, max_iter={args.max_iter}")
    print(f"  datasets: {args.datasets}", flush=True)

    out = dict(date=datetime.now().isoformat(),
               machine="xeon-cpu",
               solver="C++ PCG-DIC (single core)",
               tol=args.tol, max_iter=args.max_iter,
               datasets={})

    for name in [s.strip() for s in args.datasets.split(",")]:
        bundles = preload(name)
        if not bundles:
            continue
        print(f"\n[{name}] running PCG on {len(bundles)} bundles ...", flush=True)
        print(f"  {'case':>10s} {'iter':>5s} {'conv':>4s} {'wall_ms':>9s} "
              f"{'r_final':>10s} {'rel_resid':>10s}", flush=True)
        rows = []
        for it in bundles:
            r = run_one(it, tol=args.tol, max_iter=args.max_iter)
            rows.append(r)
            flag = ("S" if r["singular"] else "") + ("" if r["converged"] else "X")
            label = f"{r['step']:>3d}/{r['corr']}"
            yn = "Y" if r["converged"] else "N"
            print(f"  {label:>10s} {r['iters']:>5d} {yn:>4s} "
                  f"{r['wall_ms']:>9.1f} {r['r_final']:>10.2e} "
                  f"{r['rel_resid_actual']:>10.2e} {flag}", flush=True)
        summarize(name, rows)
        out["datasets"][name] = rows

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
