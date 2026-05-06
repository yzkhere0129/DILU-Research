"""Load senior's pd matrix benchmark CSVs.

Layout (under DICPCG_Benchmark_Data/Initial_Period/Solving/):
    matrix_pd_<step>_<corr>.csv     row,col,value (sparse COO, OF LDU values)
    source_pd_<step>_<corr>.csv     cellID,x,y,z,value  (b vector; coords are 0)
    solution_pd_<step>_<corr>.csv   cellID,x,y,z,value  (OF PCG-DIC final x; ref)
    cell_centers.csv                cellID,x,y,z         (broken — all zeros)

Mesh is structured 80×80×80 = 512000 cells (verified by N=512000).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


# Two datasets, each with its own npz / CSV path:
#   "initial"     — Initial_Period (cold start, T=298K, 21 matrices)
#   "evaporation" — Evaporation phase (700ns to 1.06μs, melt + recoil onset, 57 matrices)
# Resolve relative to repo root (= 4 dirs above this file: dilu/amgx/bench/<this>)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCH_BASE = _REPO_ROOT / "dilu" / "benchmark"

DATASETS = {
    "initial": {
        "npz_dir": _BENCH_BASE / "DICPCG_Benchmark_Data_npz",
        "csv_dir": _BENCH_BASE / "DICPCG_Benchmark_Data" / "Initial_Period",
    },
    "evaporation": {
        "npz_dir": _BENCH_BASE / "DICPCG_Benchmark_Data_Evaporation_npz",
        "csv_dir": (_BENCH_BASE
                    / "DICPCG_Benchmark_Data-20260506T064717Z-3-002"
                    / "DICPCG_Benchmark_Data" / "Evaporation"),
    },
}

# Default to "initial" for backwards compat
SENIOR_NPZ_BASE = DATASETS["initial"]["npz_dir"]
SENIOR_CSV_BASE = DATASETS["initial"]["csv_dir"]
SENIOR_BASE = SENIOR_CSV_BASE  # legacy alias

MESH_NX, MESH_NY, MESH_NZ = 80, 80, 80
MESH_N = MESH_NX * MESH_NY * MESH_NZ  # = 512000


def set_dataset(name: str) -> None:
    """Switch between 'initial' and 'evaporation' datasets globally."""
    global SENIOR_NPZ_BASE, SENIOR_CSV_BASE, SENIOR_BASE
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}, choose from {list(DATASETS)}")
    SENIOR_NPZ_BASE = DATASETS[name]["npz_dir"]
    SENIOR_CSV_BASE = DATASETS[name]["csv_dir"]
    SENIOR_BASE = SENIOR_CSV_BASE


@dataclass
class SeniorBundle:
    A: csr_matrix          # (N,N), assembled from COO; symmetric for pd
    b: np.ndarray          # (N,)
    x_ref: np.ndarray      # (N,) — OpenFOAM PCG-DIC solution (loose ref, tol 1e-8)
    step: int
    corr: int
    n: int


def _load_value_csv(path: Path) -> np.ndarray:
    """Load (cellID,x,y,z,value) CSV → return value column ordered by cellID."""
    arr = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 4))
    arr = arr[np.argsort(arr[:, 0])]
    return arr[:, 1].astype(np.float64)


def _load_matrix_csv(path: Path, n: int) -> csr_matrix:
    """Load (row,col,value) COO CSV → CSR matrix. Symmetric for pd."""
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    rows = arr[:, 0].astype(np.int32)
    cols = arr[:, 1].astype(np.int32)
    vals = arr[:, 2]
    A = csr_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    # OpenFOAM dumps only one triangle for symmetric matrices? Verify.
    # If not symmetric, we'd need to symmetrize.
    return A


def _load_step_npz(step: int, corr: int) -> SeniorBundle | None:
    """Load from compact npz if present (preferred path).
    Tries both 2-digit (initial) and 3-digit (evaporation) step padding.
    """
    for fname in (f"bundle_pd_{step:02d}_{corr}.npz",
                  f"bundle_pd_{step:03d}_{corr}.npz"):
        p = SENIOR_NPZ_BASE / fname
        if p.exists():
            z = np.load(p)
            n = int(z["n"][0])
            A = csr_matrix(
                (z["A_data"], z["A_indices"], z["A_indptr"]), shape=(n, n)
            )
            return SeniorBundle(A=A, b=z["b"], x_ref=z["x_ref"],
                                step=step, corr=corr, n=n)
    return None


def _load_step_csv(step: int, corr: int) -> SeniorBundle | None:
    """Load from raw CSV (legacy / source-of-truth path)."""
    base = SENIOR_CSV_BASE / "Solving"
    mp = base / f"matrix_pd_{step}_{corr}.csv"
    sp = base / f"source_pd_{step}_{corr}.csv"
    xp = base / f"solution_pd_{step}_{corr}.csv"
    if not (mp.exists() and sp.exists() and xp.exists()):
        return None
    b = _load_value_csv(sp)
    x_ref = _load_value_csv(xp)
    n = b.size
    A = _load_matrix_csv(mp, n)
    return SeniorBundle(A=A, b=b, x_ref=x_ref, step=step, corr=corr, n=n)


def load_step(step: int, corr: int) -> SeniorBundle | None:
    """Load (matrix, source, solution). Tries npz first (fast, git-tracked),
    falls back to CSV (slower, gitignored 6.4 GB raw)."""
    bundle = _load_step_npz(step, corr)
    if bundle is not None:
        return bundle
    return _load_step_csv(step, corr)


def list_available() -> list[tuple[int, int]]:
    """Return list of (step, corr) tuples available (in npz or CSV)."""
    out = []
    if SENIOR_NPZ_BASE.exists():
        for p in sorted(SENIOR_NPZ_BASE.glob("bundle_pd_*.npz")):
            stem = p.stem  # bundle_pd_01_2
            parts = stem.split("_")
            try:
                step = int(parts[2]); corr = int(parts[3])
                out.append((step, corr))
            except (IndexError, ValueError):
                continue
        if out:
            return out
    # Fallback: scan CSV
    for step in range(1, 12):
        for corr in (1, 2, 3):
            base = SENIOR_CSV_BASE / "Solving"
            if all((base / f"{p}_pd_{step}_{corr}.csv").exists()
                   for p in ("matrix", "source", "solution")):
                out.append((step, corr))
    return out


def reshape_to_grid(values: np.ndarray) -> np.ndarray:
    """Reshape flat (N,) → (NX, NY, NZ). Assumes OpenFOAM blockMesh convention
    where cellID = i + j*NX + k*NX*NY (i fastest, k slowest)."""
    assert values.size == MESH_N, f"expected {MESH_N} cells, got {values.size}"
    return values.reshape((MESH_NZ, MESH_NY, MESH_NX)).transpose(2, 1, 0)
