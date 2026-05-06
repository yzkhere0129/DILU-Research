"""Reconstruct GLOBAL (A, b, x_final) from 32 processor*/ matrixDumper outputs.

Each processor wrote postProcessing/matrices/<time>/<eq>/{A.mm, b.mm,
x_final.mm, x0.mm, metadata.json}. The dump is the EFFECTIVE local
lduMatrix (already with addBoundaryDiag merged in).

To get GLOBAL (A, b, x):
  1. Read OF's cellProcAddressing & faceProcAddressing from the
     decomposed processor*/constant/polyMesh/ directories — these tell us
     which global cell/face each local entry came from.
  2. Concatenate per-processor diag/source/x by global cell ID.
  3. Concatenate per-processor lower/upper by global face ID, plus the
     processor-patch face contributions which become NEW entries in the
     global lduMatrix (formerly between two ranks now between two cells).

Per-processor validation FIRST (simpler, doesn't need addressing):
  for each processor: ‖A_local · x_local - b_local‖ / ‖b_local‖ ≈ 1e-8
  → if every processor passes, our dump pipeline is consistent.

Run:
  python3 -u -m dilu.amgx.bench.reconstruct_lab32_dump \\
      --case ~/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_lab32_dump \\
      --time 3.2e-07 --eq pd_corr0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix


def load_processor_dump(proc_dir: Path, time_name: str, eq_name: str) -> dict:
    """Load one processor's (A_local, b_local, x_final_local, metadata)."""
    base = proc_dir / "postProcessing" / "matrices" / time_name / eq_name
    A = sio.mmread(str(base / "A.mm")).tocsr()
    b = sio.mmread(str(base / "b.mm")).flatten()
    x_final = sio.mmread(str(base / "x_final.mm")).flatten()
    x0 = sio.mmread(str(base / "x0.mm")).flatten()
    meta = json.load((base / "metadata.json").open())
    return dict(A=A, b=b, x_final=x_final, x0=x0, meta=meta,
                proc=int(proc_dir.name.replace("processor", "")))


def per_processor_consistency(case_dir: Path, time_name: str, eq_name: str):
    """Validate: each processor's local (A, b, x_final) satisfies A·x ≈ b."""
    procs = sorted(case_dir.glob("processor*"))
    print(f"\nFound {len(procs)} processor directories")
    print(f"\nPer-processor consistency for {time_name} / {eq_name}:")
    print(f"  {'proc':>5s}  {'N':>8s}  {'nnz':>10s}  {'iter':>5s}  "
          f"{'rN':>10s}  {'‖A·x_final - b‖/‖b‖':>22s}")
    print("  " + "-" * 75)
    rows = []
    for pd in procs:
        d = load_processor_dump(pd, time_name, eq_name)
        denom = max(np.linalg.norm(d['b']), 1e-300)
        rel = float(np.linalg.norm(d['A'] @ d['x_final'] - d['b']) / denom)
        of_iter = d['meta']['solver_openfoam']['iterations']
        of_rN = d['meta']['solver_openfoam']['final_residual']
        print(f"  {d['proc']:>5d}  {d['A'].shape[0]:>8d}  {d['A'].nnz:>10d}  "
              f"{of_iter:>5d}  {of_rN:>10.2e}  {rel:>22.3e}")
        rows.append(dict(proc=d['proc'], n=d['A'].shape[0], nnz=d['A'].nnz,
                          rel=rel, of_iter=of_iter, of_rN=of_rN))

    rels = np.array([r["rel"] for r in rows])
    n_total = sum(r["n"] for r in rows)
    print(f"\n  TOTAL: N_global={n_total}, "
          f"max rel = {rels.max():.3e}, "
          f"median rel = {np.median(rels):.3e}")
    if rels.max() < 1e-3:
        print(f"  ✅ All processors consistent within 1e-3 — dump pipeline OK")
    else:
        print(f"  ❌ Some processor has large residual; check matrixDumper")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True,
                     help="Path to OF case dir with processor0..N subdirs")
    ap.add_argument("--time", required=True, help="time name e.g. 3.2e-07")
    ap.add_argument("--eq", required=True,
                     help="eq name e.g. pd_corr0 / T_corr0")
    args = ap.parse_args()

    case_dir = Path(args.case).expanduser()
    if not case_dir.exists():
        print(f"case dir not found: {case_dir}"); sys.exit(1)

    per_processor_consistency(case_dir, args.time, args.eq)


if __name__ == "__main__":
    main()
