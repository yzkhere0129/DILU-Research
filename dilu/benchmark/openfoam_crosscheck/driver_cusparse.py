"""cuSPARSE DILU-PCG driver (for SPD matrices only — pd equation).

Reads the CSR from an OpenFOAM-dumped matrix dir, feeds it to our Phase 2
cuSPARSE DILU plan + a JAX PCG loop, records iter/time/residual, and
writes results/cusparse.json (+ optional cusparse_x.npy).

Uses the exact same DILU-PCG recipe as dilu/cusparse/tests/physical_benchmark.py
so results are comparable to the canonical-case benchmark.
"""
from __future__ import annotations

# Safety rails BEFORE jax import.
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

from dilu.cusparse.python import Plan as CuPlan, build_diag_offset
from dilu.cusparse.python.wrapper import cusparse_dilu_apply
from .reader import OFMatrixBundle, load_ofmm, normalize_sign

# Pluggable SpMV backend. Set via env or CLI:
#   "seg"        : original segment_sum + searchsorted (re-compute row_of every call)
#   "seg_static" : segment_sum with row_of precomputed once, passed in
#   "csr"        : jax.experimental.sparse.csr_matvec (CSR-native; XLA chooses backend)
SPMV_BACKEND = os.environ.get("DILU_SPMV", "seg")


def _spmv_seg(rp, ci, vv, x, n, row_of=None):
    """CSR SpMV via segment_sum.

    `row_of` may be precomputed and passed in (cheaper); else recomputed
    via searchsorted on every call.
    """
    if row_of is None:
        nnz = vv.shape[0]
        k = jnp.arange(nnz, dtype=jnp.int32)
        row_of = jnp.searchsorted(rp[1:], k, side="right")
    prods = vv * x[ci]
    return jax.ops.segment_sum(prods, row_of, num_segments=n)


def _spmv_csr(rp, ci, vv, x, n):
    """CSR SpMV via jax.experimental.sparse.csr_matvec."""
    from jax.experimental import sparse as jsparse
    mat = jsparse.CSR((vv, ci, rp), shape=(n, n))
    return jsparse.csr_matvec(mat, x)


def _make_pcg_jit(n: int, max_iter: int, tol: float, spmv_backend: str = "seg"):
    """Build a JIT-compiled PCG loop. `spmv_backend` ∈ {"seg","seg_static","csr"}."""
    def pcg(rp, ci, vv, do, d_star, b_d, x0_d, token, row_of):
        if spmv_backend == "csr":
            def spmv(x):
                return _spmv_csr(rp, ci, vv, x, n)
        elif spmv_backend == "seg_static":
            def spmv(x):
                return _spmv_seg(rp, ci, vv, x, n, row_of=row_of)
        else:  # "seg"
            def spmv(x):
                return _spmv_seg(rp, ci, vv, x, n, row_of=None)

        def precond(r):
            # FFI call; jit-compatible. Plan owns row_ptr/col_idx/diag_offset
            # after analyze, so we only pass token + values + d_star + r.
            return cusparse_dilu_apply(token, vv, d_star, r)

        b_norm = jnp.maximum(jnp.linalg.norm(b_d), 1e-300)
        x0 = x0_d
        r0 = b_d - spmv(x0)
        z0 = precond(r0)
        rz0 = jnp.dot(r0, z0)
        rnorm0 = jnp.linalg.norm(r0)

        # State: (it, x, r, p, z, rz, rnorm)
        init = (jnp.int32(0), x0, r0, z0, z0, rz0, rnorm0)

        def cond_fn(state):
            it, _, _, _, _, _, rnorm = state
            return jnp.logical_and(it < max_iter, rnorm / b_norm >= tol)

        def body_fn(state):
            it, x, r, p, z, rz, _rnorm = state
            Ap = spmv(p)
            pap = jnp.dot(p, Ap)
            # safety: avoid division by zero on degenerate pap
            alpha = rz / jnp.where(jnp.abs(pap) > 1e-300, pap, 1.0)
            x_new = x + alpha * p
            r_new = r - alpha * Ap
            z_new = precond(r_new)
            rz_new = jnp.dot(r_new, z_new)
            beta = rz_new / jnp.where(jnp.abs(rz) > 1e-300, rz, 1.0)
            p_new = z_new + beta * p
            rnorm_new = jnp.linalg.norm(r_new)
            return (it + 1, x_new, r_new, p_new, z_new, rz_new, rnorm_new)

        final = jax.lax.while_loop(cond_fn, body_fn, init)
        final_it, final_x, _, _, _, _, final_rnorm = final
        return final_x, final_it, final_rnorm, b_norm

    return jax.jit(pcg)


def dilu_pcg_of(bundle: OFMatrixBundle,
                tol: float = 1e-10,
                max_iter: int = 500) -> dict:
    """Run DILU-PCG on a loaded bundle. Returns meta dict with x attached.

    Implementation: builds a JIT-compiled while_loop PCG and dispatches once.
    """
    bundle = normalize_sign(bundle)
    A, b, x0_host = bundle.A, bundle.b, bundle.x0
    n = A.shape[0]
    row_ptr = np.ascontiguousarray(A.indptr,  dtype=np.int32)
    col_idx = np.ascontiguousarray(A.indices, dtype=np.int32)
    values  = np.ascontiguousarray(A.data,    dtype=np.float64)

    meta = {
        "solver": "cusparse_dilu_pcg",
        "n": int(n), "nnz": int(values.size),
        "tol": tol, "max_iter": max_iter,
        "spmv_backend": SPMV_BACKEND,
        "analyze_s": None, "factor_s": None,
        "compile_s": None, "solve_s": None,
        "iters": None, "converged": None,
        "abs_residual": None, "rel_residual": None,
        "final_residual_of": bundle.meta.get("solver_openfoam", {}).get("final_residual"),
        "status": "unknown",
        "sign_negated": bool(bundle.negated),
    }

    try:
        diag_off = build_diag_offset(row_ptr, col_idx)

        rp = jax.device_put(jnp.asarray(row_ptr))
        ci = jax.device_put(jnp.asarray(col_idx))
        vv = jax.device_put(jnp.asarray(values))
        do = jax.device_put(jnp.asarray(diag_off))
        b_d  = jax.device_put(jnp.asarray(b))
        x0_d = jax.device_put(jnp.asarray(x0_host))

        # Precompute row_of for "seg_static" path (np.repeat is fast on host).
        row_of_np = np.repeat(np.arange(n, dtype=np.int32),
                              np.diff(row_ptr).astype(np.int32))
        row_of_d = jax.device_put(jnp.asarray(row_of_np))

        t_an = time.time()
        with CuPlan(rp, ci, vv, do) as plan:
            t_an_end = time.time()
            meta["analyze_s"] = t_an_end - t_an

            t_f = time.time()
            d_star = plan.factor(vv)
            d_star.block_until_ready()
            meta["factor_s"] = time.time() - t_f

            # Build & compile JIT'd PCG (first call traces; cached after)
            t_c = time.time()
            pcg_fn = _make_pcg_jit(n, max_iter, tol, spmv_backend=SPMV_BACKEND)
            meta["compile_s"] = time.time() - t_c  # ~0 if cached

            t_s = time.time()
            x_d, iters_d, rnorm_d, b_norm_d = pcg_fn(
                rp, ci, vv, do, d_star, b_d, x0_d, plan.token, row_of_d
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

        # Self-consistency residual on original (sign-normalized) A
        r_h = A @ x_h - b
        meta["abs_residual"] = float(np.linalg.norm(r_h))
        meta["rel_residual"] = float(np.linalg.norm(r_h) / max(np.linalg.norm(b), 1e-300))

        # vs-OpenFOAM accuracy: ‖x_solver - x_OF‖_∞ / ‖x_OF‖_∞
        # This is the BENCHMARK metric (does our solver reproduce OF's answer?).
        if bundle.x_final is not None:
            x_OF = bundle.x_final
            denom = max(float(np.max(np.abs(x_OF))), 1e-300)
            meta["rel_vs_OF"] = float(np.max(np.abs(x_h - x_OF))) / denom
        else:
            meta["rel_vs_OF"] = None

        meta["status"] = "ok" if converged else "not_converged"
        return {"meta": meta, "x": x_h}
    except Exception as e:
        meta["status"] = f"error:{type(e).__name__}:{e}"
        return {"meta": meta, "x": None}


def _dump_one(matrix_dir: Path, save_x: bool = True,
              tol: float = 1e-10, max_iter: int = 500) -> None:
    bundle = load_ofmm(matrix_dir)
    eq = bundle.meta.get("equation", {}).get("name", "?")

    # cuSPARSE DILU-PCG requires SPD — skip non-symmetric equations (T)
    if eq.lower() == "t":
        print(f"[cusparse] {matrix_dir.name}: SKIP (eq={eq}, not SPD)")
        return

    out_dir = matrix_dir / "results"
    out_dir.mkdir(exist_ok=True)

    result = dilu_pcg_of(bundle, tol=tol, max_iter=max_iter)
    meta = result["meta"]
    with open(out_dir / "cusparse.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    if save_x and result["x"] is not None:
        np.save(out_dir / "cusparse_x.npy", result["x"])

    rel_OF = meta.get("rel_vs_OF")
    rel_OF_str = f"{rel_OF:.2e}" if rel_OF is not None else "—"
    print(f"[cusparse] {matrix_dir.name}: "
          f"iters={meta['iters']}, solve={meta['solve_s']:.2f}s, "
          f"rel_res={meta['rel_residual']:.2e}, "
          f"rel_vs_OF={rel_OF_str}, status={meta['status']}")
    del bundle, result
    gc.collect()
    # NOTE: NOT calling jax.clear_caches() — that would force JIT
    # recompilation per matrix (~5s overhead each). Same shape across
    # all our matrices means trace cache is reusable.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pattern", default="*/*_corr*")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--no-save-x", action="store_true")
    args = ap.parse_args()

    dirs = sorted(args.root.glob(args.pattern))
    print(f"cusparse driver: {len(dirs)} matrices under {args.root}")
    for d in dirs:
        _dump_one(d, save_x=not args.no_save_x,
                  tol=args.tol, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
