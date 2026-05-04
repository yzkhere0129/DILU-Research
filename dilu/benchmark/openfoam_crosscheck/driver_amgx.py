"""AMGx driver (both SPD and non-symmetric supported via different configs).

For pd (SPD): CLASSICAL_V_CYCLE + AGGRESSIVE_COARSENING
For T (asymmetric): AMGx PCG is SPD-only; for T we use the BiCGStab-flavoured
config if one exists, else we skip. Our existing configs are PCG-based, so
we skip T by default here — AMGx covering T would require a separate config
preset not yet written. (Flagged as TODO in the benchmark report.)

Per-matrix outputs:
    results/amgx_<label>.json
    results/amgx_<label>_x.npy  (optional)
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import jax
from jax import config as _jc
_jc.update("jax_enable_x64", True)
import jax.numpy as jnp

from dilu.amgx.python import (
    Plan as AmgxPlan,
    CLASSICAL_V_CYCLE,
    AGGRESSIVE_COARSENING,
    CLASSICAL_V_DIAGSCALED,
    CLASSICAL_V_DIAGSCALED_TIGHT,
    CLASSICAL_V_DIAGSCALED_BICGSTAB,
    CLASSICAL_GS_PCG,
    CLASSICAL_GS_BICGSTAB,
    AGGREGATION_PCG,
    with_tolerance,
)
from .reader import OFMatrixBundle, load_ofmm, normalize_sign, regularize


CONFIGS = {
    "classical_v":          CLASSICAL_V_CYCLE,
    "aggressive":           AGGRESSIVE_COARSENING,
    "classical_v_diagscaled": CLASSICAL_V_DIAGSCALED,
    "classical_v_diagscaled_tight": CLASSICAL_V_DIAGSCALED_TIGHT,
    "classical_v_diagscaled_bicgstab": CLASSICAL_V_DIAGSCALED_BICGSTAB,
    "classical_gs_pcg":     CLASSICAL_GS_PCG,
    "classical_gs_bicgstab":CLASSICAL_GS_BICGSTAB,
    "aggregation_pcg":      AGGREGATION_PCG,
}


def amgx_solve_one(bundle: OFMatrixBundle,
                   cfg_label: str,
                   tol: float = 1e-10,
                   reg_eps_rel: float = 0.0) -> dict:
    bundle = normalize_sign(bundle)
    if reg_eps_rel > 0.0:
        bundle = regularize(bundle, eps_rel=reg_eps_rel)
    A, b, x0 = bundle.A, bundle.b, bundle.x0
    n = A.shape[0]
    row_ptr = np.ascontiguousarray(A.indptr,  dtype=np.int32)
    col_idx = np.ascontiguousarray(A.indices, dtype=np.int32)
    values  = np.ascontiguousarray(A.data,    dtype=np.float64)

    cfg_json = with_tolerance(CONFIGS[cfg_label], tol)

    meta = {
        "solver": f"amgx_{cfg_label}",
        "n": int(n), "nnz": int(values.size),
        "cfg_label": cfg_label, "tol": tol,
        "setup_s": None, "solve_s": None,
        "iters": None, "status": "unknown",
        "abs_residual": None, "rel_residual": None,
        "final_residual_of": bundle.meta.get("solver_openfoam", {}).get("final_residual"),
        "sign_negated": bool(bundle.negated),
        "reg_eps_rel": float(reg_eps_rel),
        "reg_eps_added": float(bundle.eps_reg),
    }

    try:
        rp = jax.device_put(jnp.asarray(row_ptr))
        ci = jax.device_put(jnp.asarray(col_idx))
        vv = jax.device_put(jnp.asarray(values))
        b_d  = jax.device_put(jnp.asarray(b))
        x0_d = jax.device_put(jnp.asarray(x0))

        t0 = time.time()
        with AmgxPlan(rp, ci, vv, cfg_json) as plan:
            t_setup_end = time.time()
            meta["setup_s"] = t_setup_end - t0

            t_s = time.time()
            x, iters, status = plan.solve(b_d, x0_d)
            x.block_until_ready()
            meta["solve_s"] = time.time() - t_s
            meta["iters"] = int(iters[0])
            meta["status_code"] = int(status[0])

        x_h = np.asarray(x)
        r = A @ x_h - b
        meta["abs_residual"] = float(np.linalg.norm(r))
        meta["rel_residual"] = float(np.linalg.norm(r) / max(np.linalg.norm(b), 1e-300))

        # vs-OpenFOAM accuracy. NOTE: when reg_eps_rel > 0, the matrix has
        # been modified, so x_h solves a different system than OpenFOAM did.
        # rel_vs_OF will reflect that drift.
        if bundle.x_final is not None:
            x_OF = bundle.x_final
            denom = max(float(np.max(np.abs(x_OF))), 1e-300)
            meta["rel_vs_OF"] = float(np.max(np.abs(x_h - x_OF))) / denom
        else:
            meta["rel_vs_OF"] = None

        meta["status"] = "ok" if meta["status_code"] == 0 else f"amgx_status={meta['status_code']}"
        return {"meta": meta, "x": x_h}
    except Exception as e:
        meta["status"] = f"error:{type(e).__name__}:{e}"
        return {"meta": meta, "x": None}


def _dump_one(matrix_dir: Path,
              cfg_labels: list[str],
              save_x: bool,
              tol: float,
              reg_eps_rel: float = 0.0) -> None:
    bundle = load_ofmm(matrix_dir)
    eq = bundle.meta.get("equation", {}).get("name", "?")
    out_dir = matrix_dir / "results"
    out_dir.mkdir(exist_ok=True)

    if eq.lower() == "t":
        # AMGx PCG cannot solve non-symmetric T. Record a skip entry.
        meta = {"solver": "amgx", "status": "skip:eq_not_spd",
                "equation": eq}
        with open(out_dir / "amgx_all.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"[amgx] {matrix_dir.name}: SKIP (eq=T, not SPD)")
        del bundle
        gc.collect()
        return

    for label in cfg_labels:
        result = amgx_solve_one(bundle, label, tol=tol, reg_eps_rel=reg_eps_rel)
        meta = result["meta"]
        suffix = f"_reg{reg_eps_rel:.0e}" if reg_eps_rel > 0 else ""
        with open(out_dir / f"amgx_{label}{suffix}.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        if save_x and result["x"] is not None:
            np.save(out_dir / f"amgx_{label}{suffix}_x.npy", result["x"])

        rel_str = f"{meta['rel_residual']:.2e}" if meta['rel_residual'] is not None else "—"
        print(f"[amgx:{label}{suffix}] {matrix_dir.name}: "
              f"iters={meta['iters']}, setup={meta.get('setup_s')}, "
              f"solve={meta.get('solve_s')}, rel={rel_str}, "
              f"status={meta['status']}")
        del result
        gc.collect()

    del bundle
    jax.clear_caches()
    gc.collect()


def amortized_run(matrix_dirs: list, cfg_label: str, tol: float,
                  save_x: bool, reg_eps_rel: float = 0.0,
                  allow_t_eq: bool = False) -> None:
    """Process all matrix dirs reusing a SINGLE AMGx Plan across them.

    Setup happens once on the first matrix (slow ~1.5s); subsequent matrices
    use `plan.update_coefficients(new_values)` (fast, ms-level) before solve.
    Total wallclock = 1× setup + N× solve, instead of N× (setup + solve).

    Requires: all matrices share identical sparsity pattern (row_ptr/col_idx).
    For LPBF case-level dump that's true (same mesh).
    """
    if not matrix_dirs:
        print("[amortized] no matrices")
        return

    cfg_json = with_tolerance(CONFIGS[cfg_label], tol)
    suffix = f"_reg{reg_eps_rel:.0e}" if reg_eps_rel > 0 else ""

    plan = None
    pattern_token = None  # (n, nnz, indptr_hash, indices_hash) sanity guard

    for i, mdir in enumerate(matrix_dirs):
        bundle = load_ofmm(mdir)
        eq = bundle.meta.get("equation", {}).get("name", "?")

        if eq.lower() == "t" and not allow_t_eq:
            print(f"[amortized:{cfg_label}] {mdir.name}: SKIP (eq=T)")
            del bundle; gc.collect(); continue

        bundle = normalize_sign(bundle)
        if reg_eps_rel > 0.0:
            bundle = regularize(bundle, eps_rel=reg_eps_rel)

        A, b, x0 = bundle.A, bundle.b, bundle.x0
        n = A.shape[0]
        row_ptr = np.ascontiguousarray(A.indptr,  dtype=np.int32)
        col_idx = np.ascontiguousarray(A.indices, dtype=np.int32)
        values  = np.ascontiguousarray(A.data,    dtype=np.float64)

        # Sanity: pattern unchanged
        cur_token = (n, values.size, hash(row_ptr.tobytes()),
                     hash(col_idx.tobytes()))
        if pattern_token is None:
            pattern_token = cur_token
        elif pattern_token != cur_token:
            print(f"[amortized] WARN pattern changed at {mdir.name}, "
                  f"recreating plan")
            if plan is not None:
                plan.release(); plan = None
            pattern_token = cur_token

        meta = {
            "solver": f"amgx_{cfg_label}{suffix}",
            "n": int(n), "nnz": int(values.size),
            "cfg_label": cfg_label, "tol": tol,
            "amortized": True,
            "is_first_in_batch": plan is None,
            "setup_s": None, "update_s": None, "solve_s": None,
            "iters": None, "status": "unknown",
            "abs_residual": None, "rel_residual": None,
            "rel_vs_OF": None,
            "final_residual_of": bundle.meta.get("solver_openfoam", {}).get("final_residual"),
            "sign_negated": bool(bundle.negated),
            "reg_eps_rel": float(reg_eps_rel),
            "reg_eps_added": float(bundle.eps_reg),
        }

        try:
            rp = jax.device_put(jnp.asarray(row_ptr))
            ci = jax.device_put(jnp.asarray(col_idx))
            vv = jax.device_put(jnp.asarray(values))
            b_d  = jax.device_put(jnp.asarray(b))
            x0_d = jax.device_put(jnp.asarray(x0))

            if plan is None:
                # First matrix: full setup
                t0 = time.time()
                plan = AmgxPlan(rp, ci, vv, cfg_json)
                meta["setup_s"] = time.time() - t0
            else:
                # Subsequent: refresh values only
                t0 = time.time()
                plan.update_coefficients(vv)
                meta["update_s"] = time.time() - t0

            t_s = time.time()
            x, iters, status = plan.solve(b_d, x0_d)
            x.block_until_ready()
            meta["solve_s"] = time.time() - t_s
            meta["iters"] = int(iters[0])
            meta["status_code"] = int(status[0])

            x_h = np.asarray(x)
            r = A @ x_h - b
            meta["abs_residual"] = float(np.linalg.norm(r))
            meta["rel_residual"] = float(np.linalg.norm(r) / max(np.linalg.norm(b), 1e-300))
            if bundle.x_final is not None:
                x_OF = bundle.x_final
                denom = max(float(np.max(np.abs(x_OF))), 1e-300)
                meta["rel_vs_OF"] = float(np.max(np.abs(x_h - x_OF))) / denom
            meta["status"] = "ok" if meta["status_code"] == 0 else f"amgx_status={meta['status_code']}"

            out_dir = mdir / "results"
            out_dir.mkdir(exist_ok=True)
            with open(out_dir / f"amgx_{cfg_label}{suffix}_amortized.json", "w") as fh:
                json.dump(meta, fh, indent=2)
            if save_x:
                np.save(out_dir / f"amgx_{cfg_label}{suffix}_amortized_x.npy", x_h)

            rel_str = f"{meta['rel_residual']:.2e}" if meta['rel_residual'] is not None else "—"
            of_str = f"{meta['rel_vs_OF']:.2e}" if meta['rel_vs_OF'] is not None else "—"
            tag = "SETUP" if meta['is_first_in_batch'] else "AMORT"
            t_extra = f"setup={meta['setup_s']:.2f}s" if meta['setup_s'] else f"update={meta['update_s']:.3f}s"
            print(f"[amgx:{cfg_label}{suffix} {tag}] {mdir.name}: "
                  f"iters={meta['iters']}, {t_extra}, solve={meta['solve_s']:.2f}s, "
                  f"rel={rel_str}, rel_vs_OF={of_str}")
        except Exception as e:
            meta["status"] = f"error:{type(e).__name__}:{e}"
            print(f"[amgx] {mdir.name}: ERROR {meta['status']}")

        del bundle, A, b
        gc.collect()

    if plan is not None:
        plan.release()
    print("[amortized] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pattern", default="*/*_corr*")
    ap.add_argument("--cfg", nargs="+", default=["classical_v", "aggressive"],
                    choices=list(CONFIGS.keys()))
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--reg", type=float, default=0.0,
                    help="Add eps*I regularization with eps = reg * max(|diag|).")
    ap.add_argument("--amortize", action="store_true",
                    help="Reuse plan across matrices (1× setup + N× solve). "
                         "Requires --cfg to be exactly one config.")
    ap.add_argument("--allow-t-eq", action="store_true",
                    help="Don't skip T (non-symmetric) matrices. Use with "
                         "BiCGStab-based config.")
    ap.add_argument("--no-save-x", action="store_true")
    args = ap.parse_args()

    dirs = sorted(args.root.glob(args.pattern))
    print(f"amgx driver: {len(dirs)} matrices under {args.root}, "
          f"cfgs={args.cfg}, reg={args.reg}, amortize={args.amortize}")

    if args.amortize:
        if len(args.cfg) != 1:
            raise SystemExit("--amortize requires exactly one --cfg")
        amortized_run(dirs, args.cfg[0], args.tol,
                      save_x=not args.no_save_x,
                      reg_eps_rel=args.reg,
                      allow_t_eq=args.allow_t_eq)
    else:
        for d in dirs:
            _dump_one(d, cfg_labels=args.cfg,
                      save_x=not args.no_save_x, tol=args.tol,
                      reg_eps_rel=args.reg)


if __name__ == "__main__":
    main()
