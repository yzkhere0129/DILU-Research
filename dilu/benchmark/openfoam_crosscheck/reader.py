"""MatrixMarket + metadata reader for OpenFOAM-dumped linear systems.

A dumped solve lives in a directory with:
    A.mm, b.mm, x0.mm, x_final.mm, metadata.json

Usage:
    bundle = load_ofmm(Path("postProcessing/matrices/<time>/<eq>_corr<k>"))
    A, b, x0, x_final, meta = (
        bundle.A, bundle.b, bundle.x0, bundle.x_final, bundle.meta
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix


@dataclass
class OFMatrixBundle:
    A: csr_matrix
    b: np.ndarray
    x0: np.ndarray
    x_final: np.ndarray | None
    meta: dict
    path: Path
    negated: bool = False   # True if A,b were flipped (see normalize_sign)
    eps_reg: float = 0.0    # ε added to diag (see regularize); 0 = no reg


def regularize(bundle: "OFMatrixBundle", eps_rel: float = 1e-6) -> "OFMatrixBundle":
    """Add a small mass term ε·I to A's diagonal to make A strictly SPD.

    Required for classical AMG solvers when A is near-singular. OpenFOAM's
    pd matrix from a pure-Neumann Laplacian only differs from singular by
    `setReference()`'s diagonal doubling on one cell — yielding λ_min ~ 1e-13
    relative to ‖A‖. AMG coarsening fails on such matrices.

    After regularization (A + ε·I) x = b, where ε = eps_rel × max(diag(A)),
    the matrix is strictly SPD and AMG is stable. The resulting x drifts
    from the true solution by O(eps_rel) in relative terms.

    Idempotent (rerunning compounds the eps; tracker on `eps_reg`).
    """
    from scipy.sparse import eye as sparse_eye
    diag = bundle.A.diagonal()
    diag_max = float(np.abs(diag).max())
    eps = float(eps_rel) * diag_max
    A_reg = (bundle.A + eps * sparse_eye(bundle.A.shape[0])).tocsr()
    return OFMatrixBundle(
        A=A_reg,
        b=bundle.b,
        x0=bundle.x0,
        x_final=bundle.x_final,
        meta=bundle.meta,
        path=bundle.path,
        negated=bundle.negated,
        eps_reg=bundle.eps_reg + eps,
    )


def normalize_sign(bundle: "OFMatrixBundle") -> "OFMatrixBundle":
    """OpenFOAM's Laplacian convention produces A with negative diag and
    positive off-diag — symmetric but NSD (the solve is for Δφ directly,
    and OpenFOAM's DIC/PCG handles either sign).

    Our Phase 2 cuSPARSE DILU and Phase 4 AMGx PCG both assume SPD with
    positive diag. The equivalent SPD system is (-A) x = (-b), yielding
    the same x.

    This helper flips sign IFF all diag entries are ≤ 0. Idempotent, and
    `negated=True` is recorded on the returned bundle so later code can
    report honestly.
    """
    diag = bundle.A.diagonal()
    if np.all(diag <= 0) and np.any(diag < 0):
        return OFMatrixBundle(
            A=-bundle.A,
            b=-bundle.b,
            x0=bundle.x0,       # x unchanged (solution is the same)
            x_final=bundle.x_final,
            meta=bundle.meta,
            path=bundle.path,
            negated=True,
            eps_reg=bundle.eps_reg,
        )
    return bundle


def _read_mm_array(path: Path) -> np.ndarray:
    with open(path) as fh:
        header = fh.readline()
        assert header.startswith("%%MatrixMarket matrix array real general"), \
            f"unexpected MM header: {header}"
        # Skip any further comment lines starting with '%'
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"truncated array MM: {path}")
            if not line.startswith("%"):
                dims = line.split()
                break
        n_rows = int(dims[0])
        values = np.fromfile(fh, sep="\n", count=n_rows, dtype=np.float64)
        if values.size < n_rows:
            # Fallback for windows line endings / whitespace issues
            fh.seek(0)
            all_lines = [
                L for L in fh.read().splitlines()
                if L and not L.startswith("%")
            ][1:]  # skip dimensions
            values = np.array([float(L) for L in all_lines], dtype=np.float64)
    assert values.size == n_rows, f"{path}: got {values.size} expected {n_rows}"
    return values


def _load_npz(directory: Path) -> OFMatrixBundle:
    """Load from compact npz form produced by deploy/shrink_dump.py.

    File `data.npz` contains: A_data, A_indices, A_indptr, n_rows, b, x_final.
    metadata.json sits beside it.
    """
    z = np.load(directory / "data.npz")
    n = int(z["n_rows"])
    A = csr_matrix(
        (z["A_data"], z["A_indices"], z["A_indptr"]), shape=(n, n)
    )
    b = z["b"]
    xf = z["x_final"]
    if xf.size == 0:
        xf = None
    with open(directory / "metadata.json") as fh:
        meta = json.load(fh)
    return OFMatrixBundle(
        A=A, b=b, x0=np.zeros(n), x_final=xf, meta=meta, path=directory
    )


def load_ofmm(directory: Path) -> OFMatrixBundle:
    """Load (A, b, x0, x_final, meta) from a matrix dir.

    Auto-detects format:
      - data.npz present → compact binary (from shrink_dump.py)
      - else A.mm/b.mm/x0.mm/x_final.mm → MatrixMarket ASCII (from C++ dumper)
    """
    directory = Path(directory)
    if (directory / "data.npz").exists():
        return _load_npz(directory)
    A = sio.mmread(directory / "A.mm").tocsr()
    b = _read_mm_array(directory / "b.mm")
    x0 = _read_mm_array(directory / "x0.mm")
    x_final = None
    xp = directory / "x_final.mm"
    if xp.exists():
        x_final = _read_mm_array(xp)
    with open(directory / "metadata.json") as fh:
        meta = json.load(fh)
    return OFMatrixBundle(
        A=A, b=b, x0=x0, x_final=x_final, meta=meta, path=directory
    )


def sanity_check(bundle: OFMatrixBundle) -> dict:
    """Verify A @ x_final ≈ b at OpenFOAM's reported residual level."""
    A, b, xF = bundle.A, bundle.b, bundle.x_final
    assert xF is not None, "sanity_check needs x_final"
    r = A @ xF - b
    abs_r = np.linalg.norm(r)
    norm_b = np.linalg.norm(b)
    # OpenFOAM's residual uses a different normFactor; we report several metrics.
    rel_r = abs_r / max(norm_b, 1e-300)
    of_final = bundle.meta.get("solver_openfoam", {}).get("final_residual")
    return {
        "abs_residual": float(abs_r),
        "rel_residual": float(rel_r),
        "norm_b": float(norm_b),
        "norm_Ax": float(np.linalg.norm(A @ xF)),
        "norm_r": float(abs_r),
        "of_reported_final_residual": of_final,
    }


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    dirs = sorted(root.glob("*/*_corr*"))
    if not dirs:
        print(f"No matrix dirs under {root}")
        sys.exit(1)

    print(f"{'dir':<60s} {'N':>6s} {'nnz':>8s} {'|Ax-b|':>12s} {'rel':>10s} {'ofRes':>10s}")
    print("-" * 110)
    for d in dirs:
        try:
            b = load_ofmm(d)
            s = sanity_check(b)
            print(
                f"{str(d.relative_to(root)):<60s} "
                f"{b.A.shape[0]:>6d} {b.A.nnz:>8d} "
                f"{s['abs_residual']:>12.3e} "
                f"{s['rel_residual']:>10.3e} "
                f"{s['of_reported_final_residual']:>10.3e}"
            )
        except Exception as e:
            print(f"{d}: FAIL — {e}")
