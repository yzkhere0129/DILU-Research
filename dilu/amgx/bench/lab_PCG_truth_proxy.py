"""Backup/alternative to LU truth: PCG to machine precision (tol=1e-14).

If SuperLU on 512K 3D Laplacian is too slow on lab, this gives a near-
machine-precision proxy for x_truth in 3-10 min total. Method:
  1. PCG-DIC at tol=1e-10 from x0=0 (gets us into the right basin)
  2. Warm-start PCG-DIC at tol=1e-14 (push to machine precision)
  3. Verify ‖A·x - b‖_2 / ‖b‖_2 ≤ 1e-13

Comparison numbers reported per case:
  rel_resid_LU      = ‖A·x_LU - b‖ / ‖b‖  (= 1e-13 by construction)
  rel_resid_solution = same, using senior's Pre_Solving/solution as x
  rel_resid_PISO     = same, using senior's After_Solving/pd_PISO as x
  rel_solution_vs_LU = ‖x_solution - x_LU‖_∞ / ‖x_LU‖_∞
  rel_PISO_vs_LU     = ‖x_PISO     - x_LU‖_∞ / ‖x_LU‖_∞

If senior's dump is consistent with x_xref (his claim), all three
rel_resid_* should be ~1e-8 and the rel_*_vs_LU should be ~1e-12.
If our diagnosis is right, only rel_resid_LU is small; rel_resid_solution
≈ rel_resid_PISO ≈ 18.

Run on lab Xeon (no LU needed, just numpy + scipy, lightweight RAM):
    cd ~/DILU-Research && git pull origin main
    python3 -u -m dilu.amgx.bench.lab_PCG_truth_proxy \\
        > /tmp/lab_PCG_truth.log 2>&1
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

# Don't import senior_data_loader (which uses different paths); inline-load.


def load_value(p: Path) -> np.ndarray:
    rows = []
    with p.open() as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5: continue
            try: rows.append((int(parts[0]), float(parts[4])))
            except ValueError: continue
    arr = np.array(rows, dtype=np.float64)
    return arr[np.argsort(arr[:, 0])][:, 1]


def load_matrix(p: Path, n: int) -> csr_matrix:
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
    rs = np.asarray(rs, np.int32); cs = np.asarray(cs, np.int32)
    vs = np.asarray(vs, np.float64)
    off = rs != cs
    return csr_matrix(
        (np.concatenate([vs, vs[off]]),
         (np.concatenate([rs, cs[off]]), np.concatenate([cs, rs[off]]))),
        shape=(n, n)).tocsr()


def pcg_truth(A_csr: csr_matrix, b: np.ndarray, *,
              tol_loose=1e-10, tol_tight=1e-14, max_iter=5000):
    """PCG to machine precision via 2-stage warm-start, using scipy.cg.

    No dependency on our compiled C++ kernels — only scipy + numpy.
    Uses Jacobi preconditioner (diag^{-1}). For SPD A this is fine; for
    senior's (negative-diagonal) Laplacian we sign-flip A and b first.
    """
    from scipy.sparse.linalg import cg as scipy_cg, LinearOperator

    diag = A_csr.diagonal()
    if diag.min() < 0 and diag.max() <= 0:
        A_pos = -A_csr; b_pos = -b
        was_flipped = True
    else:
        A_pos = A_csr; b_pos = b
        was_flipped = False
    diag_pos = A_pos.diagonal()
    diag_safe = np.where(np.abs(diag_pos) > 1e-30, diag_pos, 1.0)
    M = LinearOperator(A_pos.shape, matvec=lambda v: v / diag_safe,
                        dtype=np.float64)

    x0 = np.zeros_like(b_pos)
    t0 = time.time()
    x_loose, info = scipy_cg(A_pos, b_pos, M=M, x0=x0,
                              atol=tol_loose, rtol=tol_loose,
                              maxiter=max_iter)
    t_l = time.time() - t0
    rN_l = float(np.linalg.norm(A_pos @ x_loose - b_pos)
                  / max(np.linalg.norm(b_pos), 1e-300))
    print(f"    loose tol={tol_loose:.0e}: info={info}, wall={t_l:.1f}s, "
          f"rel_resid={rN_l:.2e}", flush=True)

    t0 = time.time()
    x_tight, info2 = scipy_cg(A_pos, b_pos, M=M, x0=x_loose.copy(),
                               atol=tol_tight, rtol=tol_tight,
                               maxiter=max_iter)
    t_t = time.time() - t0
    rN_t = float(np.linalg.norm(A_pos @ x_tight - b_pos)
                  / max(np.linalg.norm(b_pos), 1e-300))
    print(f"    tight tol={tol_tight:.0e}: info={info2}, wall={t_t:.1f}s, "
          f"rel_resid={rN_t:.2e}", flush=True)

    actual_rel = float(np.linalg.norm((-A_csr if was_flipped else A_csr) @ x_tight
                                       - b) / max(np.linalg.norm(b), 1e-300))
    return x_tight, dict(
        loose_info=int(info), tight_info=int(info2),
        loose_t=t_l, tight_t=t_t,
        loose_rel_resid=rN_l, tight_rel_resid=rN_t,
        actual_resid=actual_rel,
    )


def check_one(label: str, base: Path, step: int, corr: int, piso_glob: str):
    pre = base / "Pre_Solving"
    after = base / "After_Solving"
    mp = pre / f"matrix_pd_{step}_{corr}.csv"
    sp = pre / f"source_pd_{step}_{corr}.csv"
    x0p = pre / f"solution_pd_{step}_{corr}.csv"
    pp_matches = [p for p in after.glob(piso_glob)
                   if not p.name.endswith("Zone.Identifier")]
    if not (mp.exists() and sp.exists() and x0p.exists() and pp_matches):
        print(f"  {label} {step}/{corr}: missing files, skip")
        return None
    pp = pp_matches[0]

    print(f"\n=== {label} step={step} corr={corr} ===", flush=True)
    print(f"  loading CSVs ...", flush=True)
    t0 = time.time()
    b = load_value(sp)
    x_solution = load_value(x0p)
    x_PISO = load_value(pp)
    n = b.size
    A = load_matrix(mp, n)
    print(f"  load_t={time.time()-t0:.1f}s, N={n}", flush=True)

    print(f"  running PCG to machine precision (LU proxy) ...", flush=True)
    x_truth, meta = pcg_truth(A, b)
    if meta["actual_resid"] > 1e-10:
        print(f"  ⚠️  truth proxy did not converge (rel_resid={meta['actual_resid']:.2e})")

    denom_b = max(np.linalg.norm(b), 1e-300)
    denom_truth = max(float(np.abs(x_truth).max()), 1e-300)
    out = dict(
        label=label, step=step, corr=corr, n=int(n),
        truth_meta=meta,
        rel_resid_truth=float(np.linalg.norm(A @ x_truth - b) / denom_b),
        rel_resid_solution=float(np.linalg.norm(A @ x_solution - b) / denom_b),
        rel_resid_PISO=float(np.linalg.norm(A @ x_PISO - b) / denom_b),
        rel_solution_vs_truth=float(np.abs(x_solution - x_truth).max() / denom_truth),
        rel_PISO_vs_truth=float(np.abs(x_PISO - x_truth).max() / denom_truth),
        x_truth_min=float(x_truth.min()), x_truth_max=float(x_truth.max()),
        x_solution_min=float(x_solution.min()), x_solution_max=float(x_solution.max()),
        x_PISO_min=float(x_PISO.min()), x_PISO_max=float(x_PISO.max()),
        b_norm=float(np.linalg.norm(b)),
        b_sum_over_abssum=float(b.sum() / max(np.abs(b).sum(), 1e-300)),
    )

    print(f"\n  CONSISTENCY  ‖A·x - b‖₂ / ‖b‖₂:")
    print(f"    truth (PCG@1e-14)  : {out['rel_resid_truth']:.3e}     ← machine ε")
    print(f"    senior solution    : {out['rel_resid_solution']:.3e}")
    print(f"    senior pd_PISO     : {out['rel_resid_PISO']:.3e}")
    print(f"  MATCH AGAINST TRUTH  ‖x - x_truth‖_∞ / ‖x_truth‖_∞:")
    print(f"    senior solution    : {out['rel_solution_vs_truth']:.3e}")
    print(f"    senior pd_PISO     : {out['rel_PISO_vs_truth']:.3e}")
    print(f"  SCALES")
    print(f"    truth      : [{out['x_truth_min']:.3e}, {out['x_truth_max']:.3e}]")
    print(f"    solution   : [{out['x_solution_min']:.3e}, {out['x_solution_max']:.3e}]")
    print(f"    pd_PISO    : [{out['x_PISO_min']:.3e}, {out['x_PISO_max']:.3e}]")

    np.savez_compressed(
        Path("/tmp") / f"lab_PCGtruth_{label}_{step}_{corr}.npz",
        x_truth=x_truth, x_solution=x_solution, x_PISO=x_PISO, b=b,
        n=np.array([n], dtype=np.int64))
    return out


def main():
    base = Path(__file__).resolve().parents[2] / "benchmark"
    results = []

    melt = base / "Melting" / "Melting"
    if melt.exists():
        for step in (65, 70, 75):
            r = check_one("melting", melt, step, 1, f"pd_PISO_{step}_1_t*.csv")
            if r: results.append(r)

    evap = base / "Evaporation" / "Evaporation"
    if evap.exists():
        for step in (84, 95, 102):
            r = check_one("evaporation", evap, step, 1, f"pd_PISO_{step}_1_t*.csv")
            if r: results.append(r)

    out_json = base.parent / "amgx" / "bench" / "lab_PCG_truth_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_json}")
    print("Per-case x_truth saved to /tmp/lab_PCGtruth_*.npz")


if __name__ == "__main__":
    main()
