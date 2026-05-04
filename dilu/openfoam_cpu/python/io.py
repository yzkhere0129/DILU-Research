"""Load (A, b, x0, x_final, meta) from a dump directory.

Supports both raw MatrixMarket dumps (A.mm + b.mm + x0.mm + x_final.mm) and
shrink_dump.py's data.npz (A_data/A_indices/A_indptr/b/x_final, no x0).
For npz form without x0, caller must supply one explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix


@dataclass
class Case:
    A: csr_matrix
    b: np.ndarray
    x0: Optional[np.ndarray]
    x_final: Optional[np.ndarray]
    sumA: Optional[np.ndarray]   # OpenFOAM lduMatrix::sumA() with post-fold diag
    norm_factor: Optional[float] # OpenFOAM normFactor at solve entry (matches PBiCG view)
    meta: dict
    path: Path


def _read_mm_array(path: Path) -> np.ndarray:
    arr = mmread(str(path))
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return np.asarray(arr, dtype=np.float64).ravel()


def load_case(directory: str | Path) -> Case:
    directory = Path(directory)
    with (directory / "metadata.json").open() as fh:
        meta = json.load(fh)

    nf = None
    nf_path = directory / "normFactor.txt"
    if nf_path.exists():
        nf = float(nf_path.read_text().strip())

    if (directory / "data.npz").exists():
        z = np.load(directory / "data.npz")
        n = int(z["n_rows"])
        A = csr_matrix(
            (z["A_data"], z["A_indices"], z["A_indptr"]), shape=(n, n)
        )
        b = z["b"]
        xf = z["x_final"] if z["x_final"].size else None
        sa = z["sumA"] if "sumA" in z.files and z["sumA"].size else None
        return Case(A=A, b=b, x0=None, x_final=xf, sumA=sa, norm_factor=nf,
                    meta=meta, path=directory)

    A = mmread(str(directory / "A.mm")).tocsr()
    b = _read_mm_array(directory / "b.mm")
    x0 = _read_mm_array(directory / "x0.mm") if (directory / "x0.mm").exists() else None
    xf = _read_mm_array(directory / "x_final.mm") if (directory / "x_final.mm").exists() else None
    sa = _read_mm_array(directory / "sumA.mm") if (directory / "sumA.mm").exists() else None
    return Case(A=A, b=b, x0=x0, x_final=xf, sumA=sa, norm_factor=nf,
                meta=meta, path=directory)
