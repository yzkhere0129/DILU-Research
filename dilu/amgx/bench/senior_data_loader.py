"""Load senior's pd matrix benchmark CSVs — direct from raw, no npz.

Datasets and on-disk layout (under dilu/benchmark/, identical on dev box
and both lab machines after manual scp):

    Melting/Melting/        — t=320…380 ns,  steps 65-75, ~33 matrices
    Evaporation/Evaporation/ — t=700ns…1.06μs, steps 84-102, ~57 matrices

Each phase directory has:
    Pre_Solving/matrix_pd_<step>_<corr>.csv
        9-column upper-triangle COO with cell coords:
        row,row_x,row_y,row_z,col,col_x,col_y,col_z,value
    Pre_Solving/source_pd_<step>_<corr>.csv     cellID,x,y,z,value  → b
    Pre_Solving/solution_pd_<step>_<corr>.csv   cellID,x,y,z,value  → prior pressure (x0 guess)
    After_Solving/pd_PISO_<step>_<corr>_t<t>.csv  cellID,x,y,z,value  → OF post-PISO field

Mesh is structured 80×80×80 = 512000 cells, dx = 2.5 μm.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


# Resolve relative to repo root: dilu/amgx/bench/<this>.py → repo/
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCH_BASE = _REPO_ROOT / "dilu" / "benchmark"

DATASETS: dict[str, Path] = {
    # csv_dir = directory containing Pre_Solving/ and After_Solving/
    "melting":     _BENCH_BASE / "Melting"     / "Melting",
    "evaporation": _BENCH_BASE / "Evaporation" / "Evaporation",
}

# Currently-selected dataset (mutable global, set via set_dataset)
SENIOR_CSV_BASE: Path = DATASETS["evaporation"]
_CURRENT_DATASET: str = "evaporation"

MESH_NX, MESH_NY, MESH_NZ = 80, 80, 80
MESH_N = MESH_NX * MESH_NY * MESH_NZ  # 512000


def set_dataset(name: str) -> None:
    """Switch globally between 'melting' and 'evaporation'."""
    global SENIOR_CSV_BASE, _CURRENT_DATASET
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}, choose from {list(DATASETS)}")
    SENIOR_CSV_BASE = DATASETS[name]
    _CURRENT_DATASET = name
    _list_available_cached.cache_clear()


def get_dataset() -> str:
    return _CURRENT_DATASET


@dataclass
class SeniorBundle:
    A: csr_matrix          # (N,N) symmetric pd; built from upper-triangle COO + mirror
    b: np.ndarray          # (N,)
    x_ref: np.ndarray      # (N,) — OpenFOAM post-PISO field (loose ref, tol 1e-8)
    x0_pre: np.ndarray     # (N,) — Pre_Solving/solution = prior step's pressure (x0 guess)
    t_seconds: float       # simulation time at this dump (parsed from After_Solving filename)
    step: int
    corr: int
    n: int


def _load_value_csv(path: Path) -> np.ndarray:
    """Load (cellID,x,y,z,value) CSV → values ordered by cellID. Robust to
    Zone.Identifier files and trailing malformed lines."""
    rows = []
    with path.open() as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
                val = float(parts[4])
                rows.append((cid, val))
            except ValueError:
                continue
    if not rows:
        raise RuntimeError(f"no rows parsed from {path}")
    arr = np.array(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]
    return arr[:, 1].astype(np.float64)


def _load_matrix_csv(path: Path, n: int) -> csr_matrix:
    """Load 9-column upper-triangle COO + diagonals; mirror to symmetric CSR.
    Robust to malformed trailing lines."""
    rows_l, cols_l, vals_l = [], [], []
    with path.open() as f:
        next(f)  # header
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

    # Mirror upper triangle into lower (skip diagonals)
    off = rows != cols
    rows_full = np.concatenate([rows, cols[off]])
    cols_full = np.concatenate([cols, rows[off]])
    vals_full = np.concatenate([vals, vals[off]])

    return csr_matrix((vals_full, (rows_full, cols_full)),
                      shape=(n, n)).tocsr()


def _piso_path(base: Path, step: int, corr: int) -> Path | None:
    """Find After_Solving/pd_PISO_<step>_<corr>_t<time>.csv (time varies)."""
    after = base / "After_Solving"
    if not after.exists():
        return None
    matches = list(after.glob(f"pd_PISO_{step}_{corr}_t*.csv"))
    # filter out :Zone.Identifier
    matches = [p for p in matches if not p.name.endswith(":Zone.Identifier")]
    return matches[0] if matches else None


_PISO_T_RE = re.compile(r"_t([0-9.eE+\-]+)\.csv$")


def _parse_t(piso_path: Path) -> float:
    m = _PISO_T_RE.search(piso_path.name)
    return float(m.group(1)) if m else float("nan")


def load_step(step: int, corr: int) -> SeniorBundle | None:
    """Load matrix, source, x_ref (After_Solving), x0_pre (Pre_Solving/solution).
    Returns None if any required file is missing."""
    base = SENIOR_CSV_BASE
    pre = base / "Pre_Solving"
    mp = pre / f"matrix_pd_{step}_{corr}.csv"
    sp = pre / f"source_pd_{step}_{corr}.csv"
    x0p = pre / f"solution_pd_{step}_{corr}.csv"
    pp = _piso_path(base, step, corr)
    if not (mp.exists() and sp.exists() and x0p.exists() and pp is not None):
        return None

    b = _load_value_csv(sp)
    x_ref = _load_value_csv(pp)
    x0_pre = _load_value_csv(x0p)
    n = b.size
    A = _load_matrix_csv(mp, n)
    return SeniorBundle(A=A, b=b, x_ref=x_ref, x0_pre=x0_pre,
                         t_seconds=_parse_t(pp),
                         step=step, corr=corr, n=n)


_STEP_RE = re.compile(r"^matrix_pd_(\d+)_(\d+)\.csv$")


@lru_cache(maxsize=4)
def _list_available_cached(csv_base_str: str) -> tuple[tuple[int, int], ...]:
    base = Path(csv_base_str)
    pre = base / "Pre_Solving"
    if not pre.exists():
        return ()
    out = []
    for p in sorted(pre.iterdir()):
        m = _STEP_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    out.sort()
    return tuple(out)


def list_available() -> list[tuple[int, int]]:
    """List (step, corr) for every matrix in the current dataset."""
    return list(_list_available_cached(str(SENIOR_CSV_BASE)))


def reshape_to_grid(values: np.ndarray) -> np.ndarray:
    """Flat (N,) → (NX,NY,NZ); cellID = i + j*NX + k*NX*NY (i fastest)."""
    assert values.size == MESH_N, f"expected {MESH_N} cells, got {values.size}"
    return values.reshape((MESH_NZ, MESH_NY, MESH_NX)).transpose(2, 1, 0)
