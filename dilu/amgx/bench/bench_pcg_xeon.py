"""Lab Xeon — run our C++ PCG (DIC) on senior pd matrices as a pure
"matrix calculator" benchmark.  No OpenFOAM, no CFD context.

Inputs: 21 initial + 57 evaporation pd matrices (already in git as npz).
Per case we apply:
  1. normalize_sign     — flip A,b so diag becomes +. (DIC needs pos diag.)
  2. project to col(A)  — b ← b - mean(b)  to satisfy Neumann compatibility.
For each case we measure:
  - iter count, converged?
  - solve wall (ms)
  - rel_resid_actual = ||A·x - b||₂ / ||b||₂  (the only honest precision metric
    when x_xref is loose / setReference-mismatched)

Cases that don't converge in `--max-iter` are still reported (iter=max, residual=...).

Output: /tmp/bench_pcg_xeon.{json,log}

Run on lab Xeon (HR54WV2):
    cd ~/DILU-Research && git pull origin main
    cd dilu/openfoam_cpu/cpp && ./build.sh   # one-time, builds .so via cmake
    cd ~/DILU-Research && python3 -u -m dilu.amgx.bench.bench_pcg_xeon \\
        > /tmp/bench_pcg_xeon.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from dilu.amgx.bench.senior_data_loader import set_dataset, list_available, load_step
from dilu.amgx.bench.sweep_amgx_senior_data import normalize_sign
from dilu.openfoam_cpu.python.ldu import csr_to_ldu
from dilu.openfoam_cpu.python import pcg


def project_col_space(b: np.ndarray) -> np.ndarray:
    """For Neumann pure-Laplacian, ker(A^T)=span(1). Project b ⊥ 1."""
    return b - b.mean()


def run_one(bundle, *, tol: float, max_iter: int) -> dict:
    A_pos, b_pos, sign_flipped = normalize_sign(bundle.A, bundle.b)
    b_proj = project_col_space(b_pos)

    ldu = csr_to_ldu(A_pos)

    t0 = time.time()
    res = pcg.solve(ldu, b_proj, np.zeros_like(b_proj),
                    tolerance=tol, min_iter=1, max_iter=max_iter)
    wall = time.time() - t0

    Ax = A_pos @ res.x
    rel_resid_actual = float(np.linalg.norm(Ax - b_proj)
                              / max(np.linalg.norm(b_proj), 1e-300))

    return dict(
        step=bundle.step, corr=bundle.corr, n=bundle.n,
        iters=res.n_iterations, converged=res.converged,
        singular=res.singular,
        wall_ms=wall * 1e3,
        r_init=res.initial_residual,
        r_final=res.final_residual,
        rel_resid_actual=rel_resid_actual,
        sign_flipped=sign_flipped,
    )


def run_dataset(name: str, *, tol: float, max_iter: int) -> list[dict]:
    print(f"\n{'='*78}")
    print(f"Dataset: {name}")
    print(f"{'='*78}")
    set_dataset(name)
    avail = list_available()
    print(f"  {len(avail)} bundles available")
    print()
    print(f"  {'case':>14s} {'iter':>5s} {'conv':>5s} {'wall_ms':>10s} "
          f"{'r_final':>10s} {'rel_resid':>10s}")
    print("  " + "-" * 70)

    rows = []
    for step, corr in avail:
        bundle = load_step(step, corr)
        if bundle is None:
            continue
        r = run_one(bundle, tol=tol, max_iter=max_iter)
        rows.append(r)
        flag = ""
        if r["singular"]: flag += "S"
        if not r["converged"]: flag += "X"
        print(f"  {f'{step:>3d}/{corr}':>14s} {r['iters']:>5d} "
              f"{'Y' if r['converged'] else 'N':>5s} "
              f"{r['wall_ms']:>10.1f} "
              f"{r['r_final']:>10.2e} "
              f"{r['rel_resid_actual']:>10.2e} {flag}")
    return rows


def summarize(name: str, rows: list[dict]):
    n = len(rows)
    n_conv = sum(1 for r in rows if r["converged"])
    walls = np.array([r["wall_ms"] for r in rows])
    iters = np.array([r["iters"]   for r in rows])
    resids = np.array([r["rel_resid_actual"] for r in rows])
    walls_conv = np.array([r["wall_ms"] for r in rows if r["converged"]])
    iters_conv = np.array([r["iters"] for r in rows if r["converged"]])
    print()
    print(f"  Summary [{name}]:  {n_conv}/{n} converged")
    print(f"    wall   median (all):     {np.median(walls):.1f} ms")
    if walls_conv.size:
        print(f"    wall   median (conv):    {np.median(walls_conv):.1f} ms")
    print(f"    iter   median (all):     {int(np.median(iters))}")
    if iters_conv.size:
        print(f"    iter   median (conv):    {int(np.median(iters_conv))}")
    if iters_conv.size:
        per_iter = walls_conv / iters_conv
        print(f"    per-iter (median, conv): {np.median(per_iter):.2f} ms/iter")
    print(f"    rel_resid_actual max:    {resids.max():.2e}")
    print(f"    rel_resid_actual med:    {np.median(resids):.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--datasets", default="initial,evaporation")
    ap.add_argument("--out", default="/tmp/bench_pcg_xeon.json")
    args = ap.parse_args()

    print(f"bench_pcg_xeon — C++ PCG (DIC) as matrix calculator")
    print(f"  tol={args.tol:.0e}, max_iter={args.max_iter}")
    print(f"  datasets: {args.datasets}")
    print(f"  python: PCG outer + C++ kernels for amul / DIC / reductions")

    out = dict(date=datetime.now().isoformat(),
               tol=args.tol, max_iter=args.max_iter,
               datasets={})
    for name in args.datasets.split(","):
        rows = run_dataset(name.strip(), tol=args.tol, max_iter=args.max_iter)
        summarize(name.strip(), rows)
        out["datasets"][name.strip()] = rows

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
