"""Run scipy SuperLU on senior's dumped (A, b) — gold-standard truth.

Purpose: prove the dumped matrix-source pair from senior's CSVs does NOT
have x_xref as its solution.  LU is direct (no iterative tol noise) so
‖A·x_LU - b‖ should be ~1e-15 (machine ε).  If x_LU disagrees with senior's
x_xref by more than ~1e-12 relative, then the dumped (A, b) is missing
something that OF added between the dump point and the actual solve.

Lab Xeon has plenty of RAM (~256 GB?), so SuperLU at 512K cells is trivial.
2M cells (LPBF_crosscheck) might also work with 30+ GB free.

Run on lab Xeon (HR54WV2):
    cd ~/DILU-Research && git pull origin main
    python3 -u -m dilu.amgx.bench.lab_LU_truth \\
        > /tmp/lab_LU_truth.log 2>&1
    # Result: /tmp/lab_LU_truth.npz  (x_LU + comparison numbers)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

OUT_NPZ = Path("/tmp/lab_LU_truth.npz")


def load_value_csv(p: Path) -> np.ndarray:
    rows = []
    with p.open() as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5: continue
            try: rows.append((int(parts[0]), float(parts[4])))
            except ValueError: continue
    arr = np.array(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]
    return arr[:, 1]


def load_matrix_csv(p: Path, n: int) -> csr_matrix:
    rs, cs, vs = [], [], []
    with p.open() as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 9: continue
            try:
                rs.append(int(parts[0])); cs.append(int(parts[4]))
                vs.append(float(parts[8]))
            except ValueError: continue
    rs = np.asarray(rs, dtype=np.int32)
    cs = np.asarray(cs, dtype=np.int32)
    vs = np.asarray(vs, dtype=np.float64)
    off = rs != cs
    return csr_matrix(
        (np.concatenate([vs, vs[off]]),
         (np.concatenate([rs, cs[off]]), np.concatenate([cs, rs[off]]))),
        shape=(n, n)).tocsr()


def lu_solve_one(label: str, base: Path, step: int, corr: int, piso_glob: str):
    """Load (A, b, x_solution, x_PISO) for one bundle; LU; report."""
    print(f"\n{'='*70}")
    print(f"=== {label}  step={step} corr={corr}")
    print(f"{'='*70}")

    pre = base / "Pre_Solving"
    after = base / "After_Solving"
    mp = pre / f"matrix_pd_{step}_{corr}.csv"
    sp = pre / f"source_pd_{step}_{corr}.csv"
    x0p = pre / f"solution_pd_{step}_{corr}.csv"
    pp_matches = list(after.glob(piso_glob))
    if not (mp.exists() and sp.exists() and x0p.exists() and pp_matches):
        print(f"  files missing — skip"); return None
    pp = pp_matches[0]

    print(f"  loading CSVs ...", flush=True)
    t0 = time.time()
    b = load_value_csv(sp)
    x_solution = load_value_csv(x0p)
    x_PISO = load_value_csv(pp)
    n = b.size
    A = load_matrix_csv(mp, n)
    print(f"  load_t = {time.time()-t0:.1f}s, N={n}, nnz={A.nnz}", flush=True)

    print(f"  factorizing SuperLU (COLAMD) ...", flush=True)
    t0 = time.time()
    lu = splu(A.tocsc(), permc_spec="COLAMD")
    print(f"  factorize_t = {time.time()-t0:.1f}s, "
          f"L+U nnz = {lu.L.nnz + lu.U.nnz}", flush=True)

    print(f"  triangular solve ...", flush=True)
    t0 = time.time()
    x_LU = lu.solve(b)
    print(f"  solve_t = {time.time()-t0:.1f}s", flush=True)

    denom_b = max(np.linalg.norm(b), 1e-300)
    denom_LU = max(float(np.abs(x_LU).max()), 1e-300)

    out = dict(
        label=label, step=step, corr=corr, n=int(n),
        # Consistency: how well does each x satisfy A·x = b?
        rel_resid_LU=float(np.linalg.norm(A @ x_LU - b) / denom_b),
        rel_resid_solution=float(np.linalg.norm(A @ x_solution - b) / denom_b),
        rel_resid_PISO=float(np.linalg.norm(A @ x_PISO - b) / denom_b),
        # Match: how close is each candidate x to LU truth?
        rel_solution_vs_LU=float(np.abs(x_solution - x_LU).max() / denom_LU),
        rel_PISO_vs_LU=float(np.abs(x_PISO - x_LU).max() / denom_LU),
        # Scales
        x_LU_min=float(x_LU.min()), x_LU_max=float(x_LU.max()),
        x_solution_min=float(x_solution.min()), x_solution_max=float(x_solution.max()),
        x_PISO_min=float(x_PISO.min()), x_PISO_max=float(x_PISO.max()),
        b_norm=float(np.linalg.norm(b)),
        b_sum=float(b.sum()), b_abssum=float(np.abs(b).sum()),
    )
    print(f"\n  CONSISTENCY  ‖A·x - b‖₂ / ‖b‖₂  (LU should be ~1e-15):")
    print(f"    LU       : {out['rel_resid_LU']:.3e}    ← gold standard")
    print(f"    solution : {out['rel_resid_solution']:.3e}")
    print(f"    PISO     : {out['rel_resid_PISO']:.3e}")
    print(f"  MATCH       ‖x - x_LU‖_∞ / ‖x_LU‖_∞  (should be ~1e-12 if consistent):")
    print(f"    solution : {out['rel_solution_vs_LU']:.3e}")
    print(f"    PISO     : {out['rel_PISO_vs_LU']:.3e}")
    print(f"  SCALES")
    print(f"    x_LU       : [{out['x_LU_min']:.3e}, {out['x_LU_max']:.3e}]")
    print(f"    x_solution : [{out['x_solution_min']:.3e}, {out['x_solution_max']:.3e}]")
    print(f"    x_PISO     : [{out['x_PISO_min']:.3e}, {out['x_PISO_max']:.3e}]")

    # Save x_LU + the senior x's for later plotting
    np.savez_compressed(
        OUT_NPZ.parent / f"lab_LU_{label}_{step}_{corr}.npz",
        x_LU=x_LU, x_solution=x_solution, x_PISO=x_PISO,
        b=b, n=np.array([n], dtype=np.int64),
    )
    return out


def main():
    base = Path(__file__).resolve().parents[2] / "benchmark"

    results = []
    # Senior melting (512K cells)
    melt_dir = base / "Melting" / "Melting"
    if melt_dir.exists():
        for step in (65, 70, 75):
            r = lu_solve_one("melting", melt_dir, step, 1, f"pd_PISO_{step}_1_t*.csv")
            if r: results.append(r)

    # Senior evaporation (also 512K)
    evap_dir = base / "Evaporation" / "Evaporation"
    if evap_dir.exists():
        for step in (84, 95, 102):
            r = lu_solve_one("evaporation", evap_dir, step, 1, f"pd_PISO_{step}_1_t*.csv")
            if r: results.append(r)
    else:
        print(f"\n  (skip evaporation — directory missing)")

    json_out = base.parent / "amgx" / "bench" / "lab_LU_truth_results.json"
    json_out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {json_out}")
    print("Per-bundle x_LU saved to /tmp/lab_LU_*.npz")


if __name__ == "__main__":
    main()
