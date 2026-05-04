"""Multicolor DILU-BiCGStab driver — for non-symmetric matrices (T equation).

Same BiCGStab loop as driver_cusparse_bicgstab, just uses MulticolorPlan
preconditioner (red-black or greedy color permutation + parallel triangular
solve) instead of cuSPARSE level scheduling.
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

from dilu.multicolor.python import MulticolorPlan
from dilu.multicolor.python.wrapper import multicolor_apply
from .reader import OFMatrixBundle, load_ofmm, normalize_sign


def _spmv(rp, ci, vv, x, n, row_of):
    prods = vv * x[ci]
    return jax.ops.segment_sum(prods, row_of, num_segments=n)


def _make_bicgstab_jit_mc(n: int, max_iter: int, tol: float):
    """JIT'd preconditioned BiCGStab using Multicolor DILU."""
    def bicgstab(rp, ci, vv, b_d, x0_d, token, d_star, row_of):
        def spmv(x):
            return _spmv(rp, ci, vv, x, n, row_of)

        def precond(r):
            return multicolor_apply(token, vv, d_star, r)

        b_norm = jnp.maximum(jnp.linalg.norm(b_d), 1e-300)
        x0 = x0_d
        r0 = b_d - spmv(x0)
        r_hat = r0
        rho0 = jnp.float64(1.0)
        alpha0 = jnp.float64(1.0)
        omega0 = jnp.float64(1.0)
        v0 = jnp.zeros(n, dtype=jnp.float64)
        p0 = jnp.zeros(n, dtype=jnp.float64)
        rnorm0 = jnp.linalg.norm(r0)

        init = (jnp.int32(0), x0, r0, r_hat, p0, v0, rho0, alpha0, omega0, rnorm0)

        def cond_fn(state):
            it = state[0]; rnorm = state[9]
            return jnp.logical_and(it < max_iter, rnorm / b_norm >= tol)

        def body_fn(state):
            it, x, r, r_hat, p, v, rho, alpha, omega, _rnorm = state
            rho_new = jnp.dot(r_hat, r)
            beta = (rho_new / jnp.where(jnp.abs(rho) > 1e-300, rho, 1.0)) \
                 * (alpha / jnp.where(jnp.abs(omega) > 1e-300, omega, 1.0))
            p_new = r + beta * (p - omega * v)
            y = precond(p_new)
            v_new = spmv(y)
            rhv = jnp.dot(r_hat, v_new)
            alpha_new = rho_new / jnp.where(jnp.abs(rhv) > 1e-300, rhv, 1.0)
            s = r - alpha_new * v_new
            z = precond(s)
            t = spmv(z)
            tt = jnp.dot(t, t)
            omega_new = jnp.dot(t, s) / jnp.where(jnp.abs(tt) > 1e-300, tt, 1.0)
            x_new = x + alpha_new * y + omega_new * z
            r_new = s - omega_new * t
            rnorm_new = jnp.linalg.norm(r_new)
            return (it + 1, x_new, r_new, r_hat, p_new, v_new,
                    rho_new, alpha_new, omega_new, rnorm_new)

        final = jax.lax.while_loop(cond_fn, body_fn, init)
        return final[1], final[0], final[9], b_norm

    return jax.jit(bicgstab)


def multicolor_bicgstab_of(bundle: OFMatrixBundle,
                          tol: float = 1e-10,
                          max_iter: int = 500,
                          grid_shape: tuple | None = None) -> dict:
    bundle = normalize_sign(bundle)
    A, b, x0_host = bundle.A, bundle.b, bundle.x0
    n = A.shape[0]
    row_ptr = np.ascontiguousarray(A.indptr,  dtype=np.int32)
    col_idx = np.ascontiguousarray(A.indices, dtype=np.int32)
    values  = np.ascontiguousarray(A.data,    dtype=np.float64)

    meta = {
        "solver": "multicolor_dilu_bicgstab",
        "n": int(n), "nnz": int(values.size),
        "tol": tol, "max_iter": max_iter,
        "setup_s": None, "factor_s": None,
        "compile_s": None, "solve_s": None,
        "iters": None, "converged": None,
        "abs_residual": None, "rel_residual": None,
        "rel_vs_OF": None,
        "final_residual_of": bundle.meta.get("solver_openfoam", {}).get("final_residual"),
        "status": "unknown",
        "sign_negated": bool(bundle.negated),
        "grid_shape": list(grid_shape) if grid_shape else None,
    }

    try:
        rp = jax.device_put(jnp.asarray(row_ptr))
        ci = jax.device_put(jnp.asarray(col_idx))
        vv = jax.device_put(jnp.asarray(values))
        b_d  = jax.device_put(jnp.asarray(b))
        x0_d = jax.device_put(jnp.asarray(x0_host))

        row_of_np = np.repeat(np.arange(n, dtype=np.int32),
                              np.diff(row_ptr).astype(np.int32))
        row_of_d = jax.device_put(jnp.asarray(row_of_np))

        t0 = time.time()
        with MulticolorPlan(rp, ci, vv, grid_shape=grid_shape) as plan:
            t_setup_end = time.time()
            meta["setup_s"] = t_setup_end - t0

            t_f = time.time()
            d_star = plan.factor(vv)
            d_star.block_until_ready()
            meta["factor_s"] = time.time() - t_f

            t_c = time.time()
            bicg_fn = _make_bicgstab_jit_mc(n, max_iter, tol)
            meta["compile_s"] = time.time() - t_c

            t_s = time.time()
            x_d, iters_d, rnorm_d, b_norm_d = bicg_fn(
                rp, ci, vv, b_d, x0_d, plan._token, d_star, row_of_d
            )
            x_d.block_until_ready()
            meta["solve_s"] = time.time() - t_s
            iters = int(iters_d)
            rnorm = float(rnorm_d)
            b_norm = float(b_norm_d)
            converged = (rnorm / max(b_norm, 1e-300)) < tol
            meta["iters"] = iters
            meta["converged"] = bool(converged)
            x_h = np.asarray(x_d)

        r_h = A @ x_h - b
        meta["abs_residual"] = float(np.linalg.norm(r_h))
        meta["rel_residual"] = float(np.linalg.norm(r_h) / max(np.linalg.norm(b), 1e-300))

        if bundle.x_final is not None:
            x_OF = bundle.x_final
            denom = max(float(np.max(np.abs(x_OF))), 1e-300)
            meta["rel_vs_OF"] = float(np.max(np.abs(x_h - x_OF))) / denom

        meta["status"] = "ok" if converged else "not_converged"
        return {"meta": meta, "x": x_h}
    except Exception as e:
        meta["status"] = f"error:{type(e).__name__}:{e}"
        return {"meta": meta, "x": None}


def _dump_one(matrix_dir: Path, save_x: bool = True, tol: float = 1e-10,
              max_iter: int = 500, grid_shape: tuple | None = None) -> None:
    bundle = load_ofmm(matrix_dir)
    eq = bundle.meta.get("equation", {}).get("name", "?")
    out_dir = matrix_dir / "results"
    out_dir.mkdir(exist_ok=True)

    result = multicolor_bicgstab_of(bundle, tol=tol, max_iter=max_iter,
                                    grid_shape=grid_shape)
    meta = result["meta"]
    with open(out_dir / "multicolor_bicgstab.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    if save_x and result["x"] is not None:
        np.save(out_dir / "multicolor_bicgstab_x.npy", result["x"])

    rel_OF = meta.get("rel_vs_OF")
    rel_OF_str = f"{rel_OF:.2e}" if rel_OF is not None else "—"
    print(f"[multicolor_bicgstab] {matrix_dir.name} (eq={eq}): "
          f"iters={meta['iters']}, solve={meta['solve_s']:.2f}s, "
          f"rel_res={meta['rel_residual']:.2e}, "
          f"rel_vs_OF={rel_OF_str}, status={meta['status']}")
    del bundle, result
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pattern", default="*/*_corr*")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--grid", nargs=3, type=int, default=None,
                    metavar=("NX", "NY", "NZ"))
    ap.add_argument("--no-save-x", action="store_true")
    args = ap.parse_args()
    grid = tuple(args.grid) if args.grid else None

    dirs = sorted(args.root.glob(args.pattern))
    print(f"multicolor_bicgstab driver: {len(dirs)} matrices, grid={grid}")
    for d in dirs:
        _dump_one(d, save_x=not args.no_save_x, tol=args.tol,
                  max_iter=args.max_iter, grid_shape=grid)


if __name__ == "__main__":
    main()
