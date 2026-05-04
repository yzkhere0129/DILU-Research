"""Quick post-run check: do the dumped A @ x_final ≈ b?

Usage:
    python3 sanity.py <case_dir>/postProcessing/matrices

Only requires numpy + scipy (not jax).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio


def _read_mm_array(path: Path) -> np.ndarray:
    with open(path) as fh:
        header = fh.readline()
        assert header.startswith("%%MatrixMarket matrix array real general"), \
            f"unexpected MM header in {path}: {header}"
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"truncated array MM: {path}")
            if not line.startswith("%"):
                n_rows = int(line.split()[0])
                break
        values = np.array([float(L) for L in fh
                           if L.strip() and not L.startswith("%")],
                          dtype=np.float64)
    assert values.size == n_rows, f"{path}: got {values.size} expected {n_rows}"
    return values


def check_one(d: Path) -> dict:
    A = sio.mmread(d / "A.mm").tocsr()
    b = _read_mm_array(d / "b.mm")
    xf_path = d / "x_final.mm"
    if not xf_path.exists():
        return {"status": "no x_final", "abs_res": None, "rel_res": None}
    xF = _read_mm_array(xf_path)
    r = A @ xF - b
    with open(d / "metadata.json") as fh:
        meta = json.load(fh)
    return {
        "step":     meta["time"]["step_index"],
        "time":     meta["time"]["time_value"],
        "eq":       meta["equation"]["name"],
        "n":        A.shape[0],
        "nnz":      A.nnz,
        "abs_res":  float(np.linalg.norm(r)),
        "rel_res":  float(np.linalg.norm(r) / max(np.linalg.norm(b), 1e-300)),
        "of_res":   meta.get("solver_openfoam", {}).get("final_residual"),
        "status":   "ok",
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 sanity.py <postProcessing/matrices>")
        sys.exit(1)
    root = Path(sys.argv[1])
    dirs = sorted(root.glob("*/*_corr*"))
    if not dirs:
        print(f"No matrix dirs under {root}"); sys.exit(1)

    print(f"{'dir':<50s}{'eq':>3s}{'step':>5s}{'N':>9s}{'nnz':>11s}"
          f"{'|Ax-b|':>12s}{'rel':>10s}{'of_res':>10s}")
    print("-" * 110)
    fails = 0
    for d in dirs:
        try:
            r = check_one(d)
            if r["status"] != "ok":
                print(f"{str(d.relative_to(root)):<50s}  {r['status']}")
                fails += 1
                continue
            # PASS criterion: our rel_res within 100x of OF's reported final_residual
            of_r = r["of_res"] or 0
            pass_ = (r["rel_res"] < max(of_r * 100, 1e-6))
            mark = "" if pass_ else " ✗"
            print(f"{str(d.relative_to(root)):<50s}{r['eq']:>3s}{r['step']:>5d}"
                  f"{r['n']:>9d}{r['nnz']:>11d}"
                  f"{r['abs_res']:>12.3e}{r['rel_res']:>10.3e}"
                  f"{(r['of_res'] or 0):>10.3e}{mark}")
            if not pass_:
                fails += 1
        except Exception as e:
            print(f"{d}: FAIL — {e}")
            fails += 1

    print(f"\nTotal: {len(dirs)} matrices, {len(dirs)-fails} pass, {fails} fail")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
