"""Convert senior's NEW Evaporation dataset CSV → compact npz.

Source layout (under DICPCG_Benchmark_Data-20260506.../DICPCG_Benchmark_Data/Evaporation):
  Pre_Solving/matrix_pd_<step>_<corr>.csv     row,row_x,row_y,row_z,col,col_x,col_y,col_z,value
                                              (UPPER triangle only + diagonals)
  Pre_Solving/source_pd_<step>_<corr>.csv     cellID,x,y,z,value
  Pre_Solving/solution_pd_<step>_<corr>.csv   cellID,x,y,z,value (initial guess)
  After_Solving/pd_PISO_<step>_<corr>_t<time>.csv  cellID,x,y,z,value (OF final x)

Matrix is stored as upper-triangle COO. Diagonal entries appear with row==col.
We rebuild full symmetric matrix by mirroring upper to lower.

Output: dilu/benchmark/DICPCG_Benchmark_Data_Evaporation_npz/bundle_pd_<step>_<corr>.npz
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


SRC_BASE = Path(
    "/home/yzk/DILU-Research/dilu/benchmark/"
    "DICPCG_Benchmark_Data-20260506T064717Z-3-002/"
    "DICPCG_Benchmark_Data/Evaporation"
)
OUT_DIR = Path(
    "/home/yzk/DILU-Research/dilu/benchmark/DICPCG_Benchmark_Data_Evaporation_npz"
)


def list_steps() -> list[tuple[int, int, float]]:
    """Returns [(step, corr, time_value)] for all PISO output files."""
    out = []
    for p in sorted((SRC_BASE / "After_Solving").glob("pd_PISO_*_t*.csv")):
        m = re.match(r"pd_PISO_(\d+)_(\d+)_t([0-9.e+-]+)\.csv", p.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
    return sorted(out)


def load_value_csv(p: Path) -> tuple[np.ndarray, np.ndarray]:
    """(value, coords) from cellID,x,y,z,value CSV. coords[i] = (x,y,z) for cellID=i."""
    arr = np.loadtxt(p, delimiter=",", skiprows=1, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]
    return arr[:, 4].astype(np.float64), arr[:, 1:4].astype(np.float64)


def load_matrix_csv(p: Path, n: int) -> csr_matrix:
    """Load upper-triangle COO + diagonal, mirror to make symmetric, return CSR.

    Some CSVs have malformed trailing lines (incomplete columns). We parse
    line-by-line with a strict 9-column requirement.
    """
    rows_l, cols_l, vals_l = [], [], []
    with p.open() as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 9:
                continue
            try:
                rows_l.append(int(parts[0]))
                cols_l.append(int(parts[4]))
                vals_l.append(float(parts[8]))
            except ValueError:
                continue
    rows = np.asarray(rows_l, dtype=np.int32)
    cols = np.asarray(cols_l, dtype=np.int32)
    vals = np.asarray(vals_l, dtype=np.float64)

    # Mirror: for entries with row != col, also add (col, row, value) [symmetric]
    off_mask = rows != cols
    rows_full = np.concatenate([rows, cols[off_mask]])
    cols_full = np.concatenate([cols, rows[off_mask]])
    vals_full = np.concatenate([vals, vals[off_mask]])

    A = csr_matrix((vals_full, (rows_full, cols_full)), shape=(n, n))
    return A.tocsr()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    avail = list_steps()
    print(f"Converting {len(avail)} bundles to npz ...")

    total = 0
    coords_cached = None
    for step, corr, t in avail:
        mp = SRC_BASE / "Pre_Solving" / f"matrix_pd_{step}_{corr}.csv"
        sp = SRC_BASE / "Pre_Solving" / f"source_pd_{step}_{corr}.csv"
        # PISO file uses _t<value> suffix
        pp = SRC_BASE / "After_Solving" / f"pd_PISO_{step}_{corr}_t{t:g}.csv"
        if not pp.exists():
            # try alt formatting
            pp = next((SRC_BASE / "After_Solving").glob(f"pd_PISO_{step}_{corr}_t*.csv"), None)
        if not (mp.exists() and sp.exists() and pp and pp.exists()):
            print(f"  step {step}/{corr}: SKIP (missing files)")
            continue

        b, _coords = load_value_csv(sp)
        x_ref, _ = load_value_csv(pp)
        n = b.size
        if coords_cached is None:
            coords_cached = _coords
        A = load_matrix_csv(mp, n)

        out = OUT_DIR / f"bundle_pd_{step:03d}_{corr}.npz"
        np.savez_compressed(
            out,
            A_data=A.data.astype(np.float64),
            A_indices=A.indices.astype(np.int32),
            A_indptr=A.indptr.astype(np.int32),
            b=b.astype(np.float64),
            x_ref=x_ref.astype(np.float64),
            n=np.array([n], dtype=np.int32),
            t=np.array([t], dtype=np.float64),
            step=np.array([step], dtype=np.int32),
            corr=np.array([corr], dtype=np.int32),
        )
        sz = out.stat().st_size
        total += sz
        print(f"  step {step:>3d}/{corr}: t={t:.2e}s  N={n}  nnz={A.nnz}  → {out.name} ({sz/1e6:.1f} MB)",
              flush=True)

    # Save shared coordinates once (mesh is structured 80³, identical across all steps)
    if coords_cached is not None:
        np.savez_compressed(OUT_DIR / "cell_coords.npz",
                            coords=coords_cached.astype(np.float64))
        print(f"\nWrote cell_coords.npz ({coords_cached.shape})")

    print(f"\nTotal: {total/1e6:.1f} MB across {len(avail)} bundles")


if __name__ == "__main__":
    main()
