"""Robust LU truth on lab Xeon — three-way fallback: SuperLU → CHOLMOD → MUMPS.

Goal: get x_LU (algebraic exact solution to senior's dumped (A, b)),
no matter what. Then ‖x_LU - x_xref‖/‖x_LU‖ becomes the IRREFUTABLE
proof of dump inconsistency.

Strategy:
  Tier 1 — SuperLU + COLAMD ordering (scipy default; 10-25 GB)
  Tier 2 — CHOLMOD (Cholesky for SPD; 3-8 GB; needs scikit-sparse)
  Tier 3 — MUMPS (multifrontal; 2-5 GB; needs pymumps)
  Tier 4 — fallback to LSMR (we already have)

Senior's matrix: needs sign-flip first (OF stores Laplacian with negative
diag). Also has near-zero diag on ~50% of cells (gas phase) so we add
tiny regularization ε·I to make it invertible without changing physics
(ε ~ 1e-15 × max|diag|).

Run on lab Xeon (NOT dev, dev only has 9.7 GB):
    cd ~/DILU-Research && git pull origin main
    python3 -u -m dilu.amgx.bench.lab_LU_robust \\
        --dataset melting --steps 65,70,75 --corrs 1 \\
        > /tmp/lab_LU_robust.log 2>&1 &
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, eye as speye

TIER_ORDER = "cholmod"  # default; main() overrides via --tier-order


# -----------------------------------------------------------------------
# CSV loaders (no senior_data_loader dependency, fully self-contained)
# -----------------------------------------------------------------------
def load_value(p: Path):
    rows = []
    with p.open() as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5: continue
            try: rows.append((int(parts[0]), float(parts[4])))
            except ValueError: continue
    a = np.array(rows, dtype=np.float64)
    return a[np.argsort(a[:, 0])][:, 1]


def load_matrix(p: Path, n: int):
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


# -----------------------------------------------------------------------
# Three LU tiers with try/except
# -----------------------------------------------------------------------
def try_superlu(A, b):
    """Tier 1: scipy SuperLU. Tries COLAMD, MMD_AT_PLUS_A, MMD_ATA orderings."""
    from scipy.sparse.linalg import splu
    for permc in ("MMD_AT_PLUS_A", "COLAMD", "MMD_ATA"):  # MMD often better for 3D
        try:
            print(f"    [tier1 SuperLU] trying permc_spec={permc} ...", flush=True)
            t0 = time.time()
            lu = splu(A.tocsc(), permc_spec=permc)
            t_fact = time.time() - t0
            print(f"      factorize_t = {t_fact:.1f}s, "
                  f"L+U nnz = {lu.L.nnz + lu.U.nnz}", flush=True)
            t0 = time.time()
            x = lu.solve(b)
            print(f"      solve_t = {time.time()-t0:.1f}s", flush=True)
            return x, f"SuperLU/{permc}"
        except (MemoryError, RuntimeError) as e:
            print(f"      FAILED: {type(e).__name__}: {str(e)[:100]}", flush=True)
            gc.collect()
            continue
    return None, None


def try_cholmod(A, b):
    """Tier 2: CHOLMOD for SPD. Tries Cholesky with progressively larger ε
    regularization, then falls back to LDL^T (which handles indefinite/rank-
    deficient matrices). Senior's matrix has near-zero diag (~50% gas-phase
    cells) AND tiny-negative eigenvalues from numerical noise, so strict
    Cholesky often fails — LDL^T usually saves us.

    The ε we add is RELATIVE to ‖A‖ (using max|diag| as proxy):
      tries ε = 1e-15·‖A‖, 1e-12·‖A‖, 1e-10·‖A‖, 1e-8·‖A‖
    The introduced perturbation in x is bounded by ε·κ(A)·‖x‖, so for
    ε=1e-10 and κ~1e3 we get ~1e-7 relative perturbation in x_LU —
    still 8 orders better than senior's 1e-1 gap.
    """
    try:
        from sksparse.cholmod import cholesky, cholesky_AAt, analyze
    except ImportError:
        print(f"    [tier2 CHOLMOD] scikit-sparse not installed; skip", flush=True)
        return None, None

    diag = A.diagonal()
    A_norm = max(abs(diag).max(), 1e-30)
    print(f"    [tier2 CHOLMOD] A_norm (max|diag|) = {A_norm:.2e}, "
          f"diag<1e-15 cells = {(np.abs(diag) < 1e-15).sum()}",
          flush=True)

    # Try Cholesky with progressively larger ε
    for eps_rel in (1e-15, 1e-12, 1e-10, 1e-8, 1e-6):
        eps = eps_rel * A_norm
        print(f"      trying Cholesky with ε={eps:.2e} (= {eps_rel:.0e}·A_norm) ...",
              flush=True)
        A_reg = (A + eps * speye(A.shape[0])).tocsc()
        try:
            t0 = time.time()
            f = cholesky(A_reg)
            print(f"        factorize_t = {time.time()-t0:.1f}s ✓", flush=True)
            t0 = time.time()
            x = f(b)
            print(f"        solve_t = {time.time()-t0:.1f}s", flush=True)
            return x, f"CHOLMOD/Chol(ε={eps_rel:.0e})"
        except Exception as e:
            print(f"        FAILED: {str(e)[:80]}", flush=True)

    # Last resort: try LDL^T (handles indefinite matrices natively)
    print(f"      all Cholesky attempts failed; trying LDL^T ...", flush=True)
    try:
        from sksparse.cholmod import cholesky as cholmod_factor
        eps = 1e-10 * A_norm
        A_reg = (A + eps * speye(A.shape[0])).tocsc()
        t0 = time.time()
        f = cholmod_factor(A_reg, mode="auto")  # auto picks LDL^T if needed
        print(f"        factorize_t = {time.time()-t0:.1f}s", flush=True)
        x = f(b)
        return x, "CHOLMOD/LDLT"
    except Exception as e:
        print(f"        FAILED: {str(e)[:100]}", flush=True)
        return None, None


def try_mumps(A, b):
    """Tier 3: MUMPS via pymumps."""
    try:
        import mumps
    except ImportError:
        print(f"    [tier3 MUMPS] pymumps not installed; skip", flush=True)
        return None, None
    try:
        ctx = mumps.DMumpsContext(par=1, sym=1)  # SPD = sym=1
        ctx.set_centralized_sparse(A.tocoo())
        ctx.set_rhs(b.copy())
        ctx.run(job=6)  # full factor + solve
        x = ctx.id.rhs.copy()
        ctx.destroy()
        return x, "MUMPS"
    except Exception as e:
        print(f"      FAILED: {type(e).__name__}: {str(e)[:100]}", flush=True)
        return None, None


def try_lsmr(A, b):
    """Tier 4 (fallback): LSMR — always converges, gives best least-squares fit."""
    from scipy.sparse.linalg import lsmr
    print(f"    [tier4 LSMR] all direct solves failed; using LSMR proxy", flush=True)
    res = lsmr(A, b, atol=1e-12, btol=1e-12, conlim=0, maxiter=10000, show=False)
    return res[0], f"LSMR/{res[2]}iter"


# -----------------------------------------------------------------------
# Main: per-bundle solve + comparison
# -----------------------------------------------------------------------
def solve_one(label: str, base: Path, step: int, corr: int):
    pre = base / "Pre_Solving"
    after = base / "After_Solving"
    mp = pre / f"matrix_pd_{step}_{corr}.csv"
    sp = pre / f"source_pd_{step}_{corr}.csv"
    x0p = pre / f"solution_pd_{step}_{corr}.csv"
    pp_matches = list(after.glob(f"pd_PISO_{step}_{corr}_t*.csv"))
    pp_matches = [p for p in pp_matches if not p.name.endswith("Zone.Identifier")]
    if not (mp.exists() and sp.exists() and x0p.exists() and pp_matches):
        print(f"  {label} {step}/{corr}: missing files, skip"); return None
    pp = pp_matches[0]

    print(f"\n{'='*70}\n=== {label} step={step} corr={corr} ===\n{'='*70}", flush=True)
    t0 = time.time()
    b_raw = load_value(sp)
    x_pre = load_value(x0p)
    x_PISO = load_value(pp)
    n = b_raw.size
    A_raw = load_matrix(mp, n)
    print(f"  load_t={time.time()-t0:.1f}s, N={n}", flush=True)

    # Sign flip if OF Laplacian with negative diag
    if A_raw.diagonal().max() <= 0:
        A = -A_raw; b = -b_raw; flipped = True
    else:
        A = A_raw; b = b_raw; flipped = False
    print(f"  sign_flipped = {flipped}", flush=True)

    # Try tiers — CHOLMOD first (5× less memory than SuperLU on SPD).
    # If you want to force SuperLU (e.g., to verify CHOLMOD result), pass
    # --tier-order=superlu.
    if TIER_ORDER == "cholmod":
        tiers = (try_cholmod, try_superlu, try_mumps, try_lsmr)
    elif TIER_ORDER == "superlu":
        tiers = (try_superlu, try_cholmod, try_mumps, try_lsmr)
    else:
        raise ValueError(f"unknown tier_order {TIER_ORDER}")
    x = method = None
    for tier in tiers:
        x, method = tier(A, b)
        if x is not None: break

    if x is None:
        print(f"  ❌ ALL TIERS FAILED"); return None

    # Compare against senior's x_xref
    denom_b = max(np.linalg.norm(b_raw), 1e-300)
    denom_x = max(float(np.abs(x).max()), 1e-300)
    rel_resid = float(np.linalg.norm(A_raw @ x - b_raw) / denom_b)
    rel_solution_vs_LU = float(np.abs(x_pre - x).max() / denom_x)
    rel_PISO_vs_LU = float(np.abs(x_PISO - x).max() / denom_x)

    print(f"\n  RESULT (method = {method}):")
    print(f"    ‖A·x_LU - b‖₂ / ‖b‖₂        = {rel_resid:.3e}     "
          f"{'← machine ε ✓' if rel_resid < 1e-10 else '← ⚠️ not algebraic exact, see method'}")
    print(f"    ‖x_solution - x_LU‖_∞ / ‖x_LU‖_∞ = {rel_solution_vs_LU:.3e}")
    print(f"    ‖x_PISO     - x_LU‖_∞ / ‖x_LU‖_∞ = {rel_PISO_vs_LU:.3e}")
    print(f"    x_LU range:       [{x.min():.3e}, {x.max():.3e}]")
    print(f"    x_solution range: [{x_pre.min():.3e}, {x_pre.max():.3e}]")
    print(f"    x_PISO range:     [{x_PISO.min():.3e}, {x_PISO.max():.3e}]")

    # Save
    out_dir = Path("/tmp/lab_LU_robust")
    out_dir.mkdir(exist_ok=True)
    np.savez_compressed(out_dir / f"{label}_{step}_{corr}.npz",
                          x_LU=x, x_solution=x_pre, x_PISO=x_PISO, b=b_raw,
                          n=np.array([n], dtype=np.int64))

    return dict(label=label, step=step, corr=corr, method=method,
                rel_resid_LU=rel_resid,
                rel_solution_vs_LU=rel_solution_vs_LU,
                rel_PISO_vs_LU=rel_PISO_vs_LU)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="melting", choices=["melting", "evaporation"])
    ap.add_argument("--steps", default="65,70,75",
                     help="comma-separated step numbers")
    ap.add_argument("--corrs", default="1", help="comma-separated corr numbers")
    ap.add_argument("--tier-order", default="cholmod",
                     choices=["cholmod", "superlu"],
                     help="cholmod (default, low memory) or superlu first")
    args = ap.parse_args()
    global TIER_ORDER
    TIER_ORDER = args.tier_order

    repo = Path(__file__).resolve().parents[2]
    base_dir = repo / "benchmark" / args.dataset.capitalize() / args.dataset.capitalize()
    if not base_dir.exists():
        print(f"❌ {base_dir} missing"); return

    steps = [int(s) for s in args.steps.split(",")]
    corrs = [int(c) for c in args.corrs.split(",")]

    rows = []
    for step in steps:
        for corr in corrs:
            r = solve_one(args.dataset, base_dir, step, corr)
            if r: rows.append(r)

    out_json = repo / "amgx" / "bench" / f"lab_LU_robust_{args.dataset}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_json}")
    print(f"Per-bundle x_LU saved to /tmp/lab_LU_robust/")


if __name__ == "__main__":
    main()
