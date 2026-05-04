"""Reference solver: scipy.sparse.linalg.spsolve (direct) + gmres (iterative).

Produces the L2 reference x_ref. Direct solve OOMs above ~200K unknowns, so we
skip it for larger matrices and fall back to GMRES with tight tol.

Per-matrix output: <matrix_dir>/results/scipy.json
Output x is saved as <matrix_dir>/results/scipy_x.npy (binary, ~16 MB @ 2M) to
serve as L2 reference for other drivers.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve, gmres, LinearOperator

from .reader import OFMatrixBundle, load_ofmm


# Guard: direct solve is O(N^2) or worse for fill-in and will OOM on > ~200K
DIRECT_N_MAX = 200_000


def solve_scipy(bundle: OFMatrixBundle, *, force_gmres: bool = False) -> dict:
    A, b = bundle.A, bundle.b
    n = A.shape[0]
    meta = {
        "solver": "scipy",
        "n": int(n),
        "nnz": int(A.nnz),
        "method": None,
        "setup_s": 0.0,
        "solve_s": 0.0,
        "iters": None,
        "abs_residual": None,
        "rel_residual": None,
        "final_residual_of": bundle.meta.get("solver_openfoam", {}).get("final_residual"),
        "status": "unknown",
    }

    try:
        if n <= DIRECT_N_MAX and not force_gmres:
            meta["method"] = "spsolve"
            t0 = time.time()
            x = spsolve(A, b)
            meta["solve_s"] = time.time() - t0
            meta["iters"] = 0
        else:
            # GMRES(restart=50) with tight tol; returns after OpenFOAM tol.
            meta["method"] = "gmres"
            # Precompute diag for diagonal precond (cheap and helps a lot
            # on these tiny-diag stiff matrices).
            diag = A.diagonal()
            diag_safe = np.where(np.abs(diag) > 1e-300, diag, 1.0)
            M = LinearOperator(A.shape, matvec=lambda r: r / diag_safe)
            t0 = time.time()
            iters_holder = [0]
            def _cb(_x):
                iters_holder[0] += 1
            x, info = gmres(A, b, M=M, rtol=1e-12, atol=0.0,
                            restart=50, maxiter=5000, callback=_cb)
            meta["solve_s"] = time.time() - t0
            meta["iters"] = iters_holder[0]
            meta["status"] = f"gmres_info={info}"

        r = A @ x - b
        meta["abs_residual"] = float(np.linalg.norm(r))
        meta["rel_residual"] = float(np.linalg.norm(r) / max(np.linalg.norm(b), 1e-300))
        if meta.get("status", "unknown") == "unknown":
            meta["status"] = "ok"
        return {"meta": meta, "x": x}
    except MemoryError as e:
        meta["status"] = f"oom:{e}"
        return {"meta": meta, "x": None}
    except Exception as e:
        meta["status"] = f"error:{type(e).__name__}:{e}"
        return {"meta": meta, "x": None}


def _dump_one(matrix_dir: Path, save_x: bool = True, force_gmres: bool = False) -> None:
    bundle = load_ofmm(matrix_dir)
    out_dir = matrix_dir / "results"
    out_dir.mkdir(exist_ok=True)

    result = solve_scipy(bundle, force_gmres=force_gmres)
    meta = result["meta"]
    x = result["x"]

    with open(out_dir / "scipy.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    if save_x and x is not None:
        np.save(out_dir / "scipy_x.npy", x)

    print(f"[scipy] {matrix_dir.name}: method={meta['method']}, "
          f"solve={meta['solve_s']:.2f}s, iters={meta['iters']}, "
          f"rel={meta['rel_residual']:.2e}, status={meta['status']}")
    del bundle, x
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path,
                    help="postProcessing/matrices root with <time>/<eq>_corr* subdirs")
    ap.add_argument("--pattern", default="*/*_corr*",
                    help="glob relative to root (default all).")
    ap.add_argument("--no-save-x", action="store_true")
    ap.add_argument("--force-gmres", action="store_true")
    args = ap.parse_args()

    dirs = sorted(args.root.glob(args.pattern))
    print(f"scipy driver: {len(dirs)} matrices under {args.root}")
    for d in dirs:
        _dump_one(d, save_x=not args.no_save_x, force_gmres=args.force_gmres)


if __name__ == "__main__":
    main()
