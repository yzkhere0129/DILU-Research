"""Synthetic 3D Poisson Laplacian — solver scaling test (find LU OOM threshold).

Generates structured-grid 3D Laplacian (7-point stencil) at multiple sizes:
  64K, 256K, 500K, 1M, 2M, 5M (memory permitting).

For each size, runs OF-style PCG iterative + AMGx + CHOLMOD LU and records
wall, peak memory, max iter. Identifies where LU OOMs and where AMGx GPU
becomes faster than CPU iterative.

This is **synthetic** — not the full LPBF physics matrix, but Laplacian-like
structure is a good first-order proxy. Results inform when to choose direct
vs iterative on real LPBF problems.

Usage:
    python -m dilu.experiments.solver_production_comparison.synthetic_scaling \\
        --sizes 32 50 64 80 100 \\
        --skip-lu-above 1000000     # 跳过 LU 当 N > 1M (avoid OOM)
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, eye as sp_eye, diags
from scipy.sparse.linalg import cg, spsolve

try:
    from sksparse.cholmod import cholesky
    HAVE_CHOLMOD = True
except ImportError:
    HAVE_CHOLMOD = False

try:
    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    from jax import config as _jc
    _jc.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from dilu.amgx.python import Plan, CLASSICAL_V_DIAGSCALED, with_tolerance
    HAVE_AMGX = True
except ImportError:
    HAVE_AMGX = False


REPO = Path(__file__).resolve().parents[3]


def build_laplacian_3d(nx: int, ny: int = None, nz: int = None) -> csr_matrix:
    """7-point 3D Laplacian on uniform structured grid, Dirichlet on all boundaries.

    Returns symmetric positive definite CSR (with diag pinned for invertibility).
    """
    ny = ny or nx
    nz = nz or nx
    n = nx * ny * nz
    print(f"  building {nx}×{ny}×{nz} = {n:,} cell Laplacian ...")

    # Use kron product of 1D Laplacians for efficiency
    def lap1d(n):
        d = np.full(n, 2.0)
        d[0] = 1.0; d[-1] = 1.0   # Neumann on ends → tweak; use Dirichlet pin instead
        e = np.full(n - 1, -1.0)
        return diags([e, d, e], [-1, 0, 1], shape=(n, n), format="csr")

    Lx = lap1d(nx); Ly = lap1d(ny); Lz = lap1d(nz)
    Ix = sp_eye(nx); Iy = sp_eye(ny); Iz = sp_eye(nz)

    from scipy.sparse import kron
    A = kron(kron(Iz, Iy), Lx) + kron(kron(Iz, Ly), Ix) + kron(kron(Lz, Iy), Ix)
    A = A.tocsr()

    # Pin one cell for SPD (avoid singularity)
    A[0, 0] += 1.0   # Dirichlet on cell 0

    # Random rhs (representative of typical CFD residual)
    rng = np.random.default_rng(42)
    b = rng.standard_normal(n) * 1e-3
    return A, b


def peak_rss_mb():
    """Get current process peak RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB → MB


def run_amgx(A: csr_matrix, b: np.ndarray, tol=1e-8, max_iter=2000):
    """AMGx fresh setup."""
    if not HAVE_AMGX: return None
    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, tol, max_iters=max_iter)
    rp = jnp.asarray(A.indptr.astype(np.int32))
    ci = jnp.asarray(A.indices.astype(np.int32))
    vv = jnp.asarray(A.data.astype(np.float64))
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

    res = float(np.linalg.norm(A @ x_h - b) / max(np.linalg.norm(b), 1e-300))
    return dict(method="AMGx", iters=int(iters[0]),
                setup_s=t_setup, solve_s=t_solve, total_s=t_setup+t_solve,
                rel_resid=res)


def run_cholmod(A: csr_matrix, b: np.ndarray):
    """CHOLMOD direct LU (well, Cholesky for SPD)."""
    if not HAVE_CHOLMOD: return None
    rss_before = peak_rss_mb()
    t0 = time.time()
    factor = cholesky(A.tocsc())
    t_factor = time.time() - t0
    t1 = time.time()
    x = factor(b)
    t_solve = time.time() - t1
    rss_peak = peak_rss_mb()

    res = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))
    return dict(method="CHOLMOD", iters="direct",
                setup_s=t_factor, solve_s=t_solve, total_s=t_factor+t_solve,
                rel_resid=res, peak_rss_mb=rss_peak,
                rss_growth_mb=rss_peak - rss_before)


def run_scipy_cg(A: csr_matrix, b: np.ndarray, tol=1e-8, max_iter=10000):
    """scipy CG (no preconditioner) — proxy for what raw iterative looks like."""
    t0 = time.time()
    x, info = cg(A, b, rtol=tol, atol=0, maxiter=max_iter)
    t_solve = time.time() - t0
    res = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))
    return dict(method="scipy_CG", iters=info if info > 0 else "converged",
                setup_s=0, solve_s=t_solve, total_s=t_solve,
                rel_resid=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int,
                     default=[32, 50, 64, 80, 100],
                     help="cube root of mesh sizes (e.g., 32 → 32^3 = 32K cells)")
    ap.add_argument("--skip-lu-above", type=int, default=2_000_000,
                     help="skip CHOLMOD LU above this N to avoid OOM")
    ap.add_argument("--skip-cg-above", type=int, default=2_000_000,
                     help="skip scipy CG above this N (slow)")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--out", default="dilu/experiments/solver_production_comparison/synthetic_scaling.json")
    args = ap.parse_args()

    print(f"AMGx avail: {HAVE_AMGX}, CHOLMOD avail: {HAVE_CHOLMOD}")
    results = []
    for nx in args.sizes:
        n = nx ** 3
        print(f"\n{'='*70}")
        print(f"N = {nx}^3 = {n:,} cells")
        print(f"{'='*70}")
        gc.collect()
        A, b = build_laplacian_3d(nx, nx, nx)
        print(f"  A shape={A.shape}, nnz={A.nnz}")

        case_results = dict(N=n, nx=nx, nnz=int(A.nnz), methods={})

        # AMGx
        if HAVE_AMGX:
            print(f"  Running AMGx ...")
            try:
                r = run_amgx(A, b, tol=args.tol)
                case_results["methods"]["amgx"] = r
                print(f"    AMGx: iter={r['iters']}, setup={r['setup_s']:.2f}s, "
                      f"solve={r['solve_s']:.2f}s, resid={r['rel_resid']:.2e}")
            except Exception as e:
                print(f"    AMGx FAIL: {e}")
                case_results["methods"]["amgx"] = dict(error=str(e))

        # CHOLMOD LU
        if HAVE_CHOLMOD and n <= args.skip_lu_above:
            print(f"  Running CHOLMOD ...")
            try:
                r = run_cholmod(A, b)
                case_results["methods"]["cholmod"] = r
                print(f"    CHOLMOD: factor={r['setup_s']:.2f}s, solve={r['solve_s']:.2f}s, "
                      f"peak_RSS={r['peak_rss_mb']:.0f} MB, resid={r['rel_resid']:.2e}")
            except Exception as e:
                print(f"    CHOLMOD FAIL: {e}")
                case_results["methods"]["cholmod"] = dict(error=str(e))
        else:
            print(f"  CHOLMOD skipped (N > {args.skip_lu_above:,})")
            case_results["methods"]["cholmod"] = dict(skipped="N too large or no scikit-sparse")

        # scipy CG (no preconditioner) — only at small/medium for proxy
        if n <= args.skip_cg_above:
            print(f"  Running scipy CG (no precond) ...")
            try:
                r = run_scipy_cg(A, b, tol=args.tol)
                case_results["methods"]["scipy_cg"] = r
                print(f"    scipy CG: iter={r['iters']}, solve={r['solve_s']:.2f}s, "
                      f"resid={r['rel_resid']:.2e}")
            except Exception as e:
                print(f"    scipy CG FAIL: {e}")
                case_results["methods"]["scipy_cg"] = dict(error=str(e))

        results.append(case_results)
        del A, b
        gc.collect()

    # Save
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n→ {out_path}")

    # Summary table
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")
    print(f"{'N':>10s} {'nx':>4s} {'AMGx (s)':>10s} {'AMGx iter':>10s} "
          f"{'CHOLMOD (s)':>13s} {'CHOLMOD RSS':>13s} {'scipy CG (s)':>13s}")
    print("-"*82)
    for r in results:
        a = r["methods"].get("amgx", {})
        c = r["methods"].get("cholmod", {})
        s = r["methods"].get("scipy_cg", {})
        a_t = f"{a.get('total_s', '—'):.2f}" if "total_s" in a else "—"
        a_i = f"{a.get('iters', '—')}"
        c_t = f"{c.get('total_s', '—'):.2f}" if "total_s" in c else "skip"
        c_r = f"{c.get('peak_rss_mb', 0):.0f}MB" if "peak_rss_mb" in c else "—"
        s_t = f"{s.get('total_s', '—'):.2f}" if "total_s" in s else "—"
        print(f"{r['N']:>10,d} {r['nx']:>4d} {a_t:>10s} {a_i:>10s} "
              f"{c_t:>13s} {c_r:>13s} {s_t:>13s}")


if __name__ == "__main__":
    main()
