"""Production-style replay: feed N matrix sequence through 5 solver modes.

Validates the hypothesis from solver_comparison_500K.md key takeaways:
   "AMGx amortized + warm-start beats LU 100× in real CFD time-stepping
    because LU must refactor every step while AMGx reuses AMG hierarchy."

5 solver modes per replay:

  1. AMGx_fresh        : new Plan() every step       (worst case for AMGx)
  2. AMGx_amortized    : Plan() once, update_coefficients() every subsequent step
  3. AMGx_amortized_warm: same as 2 + previous x_solution as init guess
  4. CHOLMOD_factor    : cholesky() full factor every step  (worst case for LU)
  5. CHOLMOD_symbolic  : analyze() once, .cholesky_inplace(A_new) each step
                         (reuses symbolic structure; only numeric factor recomputed)

Each mode reports per-timestep:
  - iters (for iterative solvers)
  - solve_wall_ms
  - setup/update/factor_wall_ms
  - rel_resid actual

Plus aggregate:
  - total wall (the production-relevant number)
  - mean per-step wall
  - all-timesteps max rel_resid

Usage:
    # Test on existing 6 timesteps:
    python -m dilu.experiments.solver_production_comparison.replay_amortized \\
        --case ~/single_track_dump --tol 1e-8

    # When 384-step dense dump arrives:
    python -m dilu.experiments.solver_production_comparison.replay_amortized \\
        --case ~/dense_track_dump_500K --tol 1e-8 --max-timesteps 384
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Optional CHOLMOD
try:
    from sksparse.cholmod import cholesky, analyze
    HAVE_CHOLMOD = True
except ImportError:
    HAVE_CHOLMOD = False
    print("[warn] scikit-sparse not installed — CHOLMOD modes disabled (will use SuperLU)")

# AMGx (only if available)
try:
    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    from jax import config as _jc
    _jc.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from dilu.amgx.python import Plan, CLASSICAL_V_DIAGSCALED, with_tolerance
    HAVE_AMGX = True
except ImportError as e:
    HAVE_AMGX = False
    print(f"[warn] AMGx not importable ({e}) — skipping AMGx modes")


REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Matrix loader
# ---------------------------------------------------------------------------
def discover_matrices(case_dir: Path, eq_name: str = "pd_corr0",
                       max_timesteps: int = None):
    """Find all dumped matrix dirs sorted by physical time.

    Yields (time_str, time_float, A_csr, b_array, x_OF_array)
    """
    base = case_dir / "postProcessing" / "matrices"
    if not base.exists():
        raise SystemExit(f"matrices dir not found: {base}")

    times = sorted(p.name for p in base.iterdir() if p.is_dir())
    times.sort(key=lambda s: float(s))   # numerical sort
    if max_timesteps:
        times = times[:max_timesteps]

    for t_str in times:
        eq_dir = base / t_str / eq_name
        if not eq_dir.exists():
            continue
        A = sio.mmread(str(eq_dir / "A.mm")).tocsr()
        b = sio.mmread(str(eq_dir / "b.mm")).flatten()
        x_OF = sio.mmread(str(eq_dir / "x_final.mm")).flatten()
        yield t_str, float(t_str), A, b, x_OF


def normalize_sign(A: csr_matrix, b: np.ndarray):
    """OF Laplacian has negative diag — flip to positive for solver convention."""
    diag = A.diagonal()
    if np.all(diag <= 0) and np.any(diag < 0):
        return -A, -b, True
    return A, b, False


# ---------------------------------------------------------------------------
# AMGx modes
# ---------------------------------------------------------------------------
def run_amgx_fresh(matrices, tol=1e-8, max_iter=2000):
    """Mode 1: new Plan every step (full setup overhead)."""
    if not HAVE_AMGX: return []
    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, tol, max_iters=max_iter)
    rows = []
    for i, (t_str, t, A, b, x_OF) in enumerate(matrices):
        A_p, b_p, _ = normalize_sign(A, b)
        rp = jnp.asarray(A_p.indptr.astype(np.int32))
        ci = jnp.asarray(A_p.indices.astype(np.int32))
        vv = jnp.asarray(A_p.data.astype(np.float64))
        b_d = jnp.asarray(b_p.astype(np.float64))
        x0_d = jnp.zeros_like(b_d)

        t0 = time.time()
        with Plan(rp, ci, vv, cfg) as plan:
            t_setup = time.time() - t0
            t1 = time.time()
            x, iters, status = plan.solve(b_d, x0_d)
            x.block_until_ready()
            t_solve = time.time() - t1
            x_h = np.asarray(x)
        res = float(np.linalg.norm(A_p @ x_h - b_p)
                    / max(np.linalg.norm(b_p), 1e-300))
        rows.append(dict(step=i, time=t, mode="amgx_fresh",
                          iters=int(iters[0]),
                          setup_ms=t_setup*1e3, update_ms=0,
                          solve_ms=t_solve*1e3,
                          total_ms=(t_setup+t_solve)*1e3,
                          rel_resid=res))
        print(f"  [fresh    ] step={i:3d} t={t_str:>10s} iter={int(iters[0]):>4d} "
              f"setup={t_setup*1e3:>7.0f}ms solve={t_solve*1e3:>7.0f}ms "
              f"resid={res:.2e}")
        gc.collect()
    return rows


def run_amgx_amortized(matrices, tol=1e-8, max_iter=2000, warm_start=False):
    """Mode 2/3: Plan once, update_coefficients per step. Optional warm-start."""
    if not HAVE_AMGX: return []
    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, tol, max_iters=max_iter)
    rows = []
    plan = None
    x_prev = None
    label = "amgx_amortized" + ("_warm" if warm_start else "")

    for i, (t_str, t, A, b, x_OF) in enumerate(matrices):
        A_p, b_p, _ = normalize_sign(A, b)
        rp = jnp.asarray(A_p.indptr.astype(np.int32))
        ci = jnp.asarray(A_p.indices.astype(np.int32))
        vv = jnp.asarray(A_p.data.astype(np.float64))
        b_d = jnp.asarray(b_p.astype(np.float64))

        if plan is None:
            t0 = time.time()
            plan = Plan(rp, ci, vv, cfg)
            t_setup = time.time() - t0
            t_update = 0.0
        else:
            t_setup = 0.0
            t0 = time.time()
            plan.update_coefficients(vv)
            t_update = time.time() - t0

        # Init guess
        if warm_start and x_prev is not None:
            x0_d = jnp.asarray(x_prev.astype(np.float64))
        else:
            x0_d = jnp.zeros_like(b_d)

        t1 = time.time()
        x, iters, status = plan.solve(b_d, x0_d)
        x.block_until_ready()
        t_solve = time.time() - t1
        x_h = np.asarray(x)
        x_prev = x_h
        res = float(np.linalg.norm(A_p @ x_h - b_p)
                    / max(np.linalg.norm(b_p), 1e-300))
        rows.append(dict(step=i, time=t, mode=label,
                          iters=int(iters[0]),
                          setup_ms=t_setup*1e3, update_ms=t_update*1e3,
                          solve_ms=t_solve*1e3,
                          total_ms=(t_setup+t_update+t_solve)*1e3,
                          rel_resid=res))
        print(f"  [{label:<18s}] step={i:3d} t={t_str:>10s} iter={int(iters[0]):>4d} "
              f"setup={t_setup*1e3:>5.0f}ms upd={t_update*1e3:>5.0f}ms "
              f"solve={t_solve*1e3:>6.0f}ms resid={res:.2e}")

    if plan is not None:
        plan.__exit__(None, None, None)   # release resources
    gc.collect()
    return rows


# ---------------------------------------------------------------------------
# LU modes
# ---------------------------------------------------------------------------
def run_lu_fresh(matrices, use_cholmod=True):
    """Mode 4: full factor every step (worst-case LU)."""
    rows = []
    for i, (t_str, t, A, b, x_OF) in enumerate(matrices):
        A_p, b_p, _ = normalize_sign(A, b)
        if use_cholmod and HAVE_CHOLMOD:
            t0 = time.time()
            factor = cholesky(A_p.tocsc())
            t_factor = time.time() - t0
            t1 = time.time()
            x = factor(b_p)
            t_solve = time.time() - t1
            method = "CHOLMOD"
        else:
            t0 = time.time()
            x = spsolve(A_p.tocsc(), b_p)
            t_factor = 0.0
            t_solve = time.time() - t0
            method = "SuperLU"
        res = float(np.linalg.norm(A_p @ x - b_p)
                    / max(np.linalg.norm(b_p), 1e-300))
        rows.append(dict(step=i, time=t, mode=f"lu_fresh_{method}",
                          iters="direct",
                          setup_ms=0, update_ms=t_factor*1e3,
                          solve_ms=t_solve*1e3,
                          total_ms=(t_factor+t_solve)*1e3,
                          rel_resid=res))
        print(f"  [lu_fresh ] step={i:3d} t={t_str:>10s} factor={t_factor*1e3:>7.0f}ms "
              f"solve={t_solve*1e3:>5.0f}ms total={(t_factor+t_solve)*1e3:.0f}ms")
    return rows


def run_lu_symbolic_reuse(matrices):
    """Mode 5: CHOLMOD analyze() once, .cholesky_inplace(A_new) each step.

    Reuses symbolic factorization (sparsity pattern reordering) since CFD
    matrices keep the same nonzero pattern step-to-step.
    """
    if not HAVE_CHOLMOD:
        return []
    rows = []
    factor = None
    for i, (t_str, t, A, b, x_OF) in enumerate(matrices):
        A_p, b_p, _ = normalize_sign(A, b)
        if factor is None:
            t0 = time.time()
            factor = cholesky(A_p.tocsc())   # first time: full factor
            t_setup = time.time() - t0
            t_update = 0.0
        else:
            t_setup = 0.0
            t0 = time.time()
            factor.cholesky_inplace(A_p.tocsc())   # reuses symbolic
            t_update = time.time() - t0
        t1 = time.time()
        x = factor(b_p)
        t_solve = time.time() - t1
        res = float(np.linalg.norm(A_p @ x - b_p)
                    / max(np.linalg.norm(b_p), 1e-300))
        rows.append(dict(step=i, time=t, mode="lu_symbolic_reuse",
                          iters="direct",
                          setup_ms=t_setup*1e3, update_ms=t_update*1e3,
                          solve_ms=t_solve*1e3,
                          total_ms=(t_setup+t_update+t_solve)*1e3,
                          rel_resid=res))
        print(f"  [lu_symb  ] step={i:3d} t={t_str:>10s} setup={t_setup*1e3:>6.0f}ms "
              f"refact={t_update*1e3:>5.0f}ms solve={t_solve*1e3:>5.0f}ms")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def aggregate(rows, mode):
    sub = [r for r in rows if r.get("mode") == mode or
           (mode == "lu_fresh" and r.get("mode", "").startswith("lu_fresh"))]
    if not sub: return None
    walls = [r["total_ms"] for r in sub]
    setup = sub[0]["setup_ms"] if len(sub) > 0 else 0
    return dict(
        mode=mode, n_steps=len(sub),
        total_wall_s=sum(walls)/1000,
        mean_per_step_ms=np.mean(walls),
        median_per_step_ms=np.median(walls),
        first_step_ms=walls[0],
        rest_mean_ms=np.mean(walls[1:]) if len(walls) > 1 else None,
        max_rel_resid=max(r["rel_resid"] for r in sub),
        setup_first_ms=setup,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="OF case dir with postProcessing/matrices")
    ap.add_argument("--eq", default="pd_corr0")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-timesteps", type=int, default=None)
    ap.add_argument("--modes", nargs="+",
                     default=["amgx_fresh", "amgx_amortized", "amgx_amortized_warm",
                              "lu_fresh", "lu_symbolic_reuse"],
                     help="which modes to run")
    ap.add_argument("--out",
                     default="dilu/experiments/solver_production_comparison/replay_results.json")
    args = ap.parse_args()

    case = Path(args.case).expanduser()
    print(f"Case dir : {case}")
    print(f"Equation : {args.eq}")
    print(f"Tol      : {args.tol}")
    print(f"AMGx avail: {HAVE_AMGX}, CHOLMOD avail: {HAVE_CHOLMOD}")
    print()

    # Load all matrices (eager — small enough for 6-384 timesteps)
    print("Discovering + loading matrices ...")
    matrices = list(discover_matrices(case, args.eq, args.max_timesteps))
    print(f"  loaded {len(matrices)} timesteps")
    if not matrices:
        raise SystemExit("no matrices found — check --case")
    print()

    all_rows = []
    aggregates = {}

    for mode in args.modes:
        print(f"\n{'='*70}")
        print(f"MODE: {mode}")
        print(f"{'='*70}")
        if mode == "amgx_fresh":
            rows = run_amgx_fresh(matrices, tol=args.tol)
        elif mode == "amgx_amortized":
            rows = run_amgx_amortized(matrices, tol=args.tol, warm_start=False)
        elif mode == "amgx_amortized_warm":
            rows = run_amgx_amortized(matrices, tol=args.tol, warm_start=True)
        elif mode == "lu_fresh":
            rows = run_lu_fresh(matrices, use_cholmod=True)
        elif mode == "lu_symbolic_reuse":
            rows = run_lu_symbolic_reuse(matrices)
        else:
            print(f"  [skip] unknown mode: {mode}")
            continue
        all_rows.extend(rows)
        aggr = aggregate(rows, mode)
        aggregates[mode] = aggr
        if aggr:
            print(f"\n  AGGREGATE: total {aggr['total_wall_s']:.2f} s, "
                  f"mean per step {aggr['mean_per_step_ms']:.0f} ms, "
                  f"first step {aggr['first_step_ms']:.0f} ms, "
                  f"rest mean {aggr.get('rest_mean_ms', 'n/a') and f'{aggr['rest_mean_ms']:.0f}'} ms")

    # Save
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "case": str(case),
        "n_timesteps": len(matrices),
        "tol": args.tol,
        "modes_run": args.modes,
        "aggregates": aggregates,
        "per_step": all_rows,
    }, indent=2))
    print(f"\n→ {out}")

    # Console summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY ({len(matrices)} timesteps)")
    print(f"{'='*80}")
    print(f"{'mode':<25s} {'total wall (s)':>14s} {'mean/step (ms)':>14s} "
          f"{'rest mean (ms)':>14s} {'max rel_resid':>13s}")
    print("-"*82)
    for mode, a in aggregates.items():
        if a is None: continue
        rest = f"{a['rest_mean_ms']:.0f}" if a.get('rest_mean_ms') is not None else "n/a"
        print(f"{mode:<25s} {a['total_wall_s']:>14.2f} "
              f"{a['mean_per_step_ms']:>14.0f} "
              f"{rest:>14s} {a['max_rel_resid']:>13.2e}")


if __name__ == "__main__":
    main()
