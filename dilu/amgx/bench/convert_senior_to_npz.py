"""Convert senior's CSV-format pd matrices into compact, git-friendly npz files.

Reads from `dilu/benchmark/DICPCG_Benchmark_Data/Initial_Period/Solving/`
(6.4 GB raw CSV, ignored by git), writes to
`dilu/benchmark/DICPCG_Benchmark_Data_npz/` (per-bundle ~25 MB compressed,
all <100 MB → git-friendly).

Each output file: `bundle_pd_<step>_<corr>.npz` with keys:
    A_data       float64 (nnz,)
    A_indices    int32   (nnz,)
    A_indptr     int32   (N+1,)
    b            float64 (N,)
    x_ref        float64 (N,)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/yzk/DILU-Research")
from dilu.amgx.bench.senior_data_loader import load_step, list_available


OUT_DIR = Path("/home/yzk/DILU-Research/dilu/benchmark/DICPCG_Benchmark_Data_npz")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    avail = list_available()
    print(f"Converting {len(avail)} bundles to npz...")
    total = 0
    for step, corr in avail:
        b = load_step(step, corr)
        if b is None:
            print(f"  step {step}/{corr}: SKIP (missing CSV)")
            continue
        out = OUT_DIR / f"bundle_pd_{step:02d}_{corr}.npz"
        np.savez_compressed(
            out,
            A_data=b.A.data.astype(np.float64),
            A_indices=b.A.indices.astype(np.int32),
            A_indptr=b.A.indptr.astype(np.int32),
            b=b.b.astype(np.float64),
            x_ref=b.x_ref.astype(np.float64),
            n=np.array([b.n], dtype=np.int32),
        )
        sz = out.stat().st_size
        total += sz
        print(f"  step {step:>2d}/{corr}: wrote {out.name} ({sz/1e6:.1f} MB)")
    print(f"\nTotal: {total/1e6:.1f} MB across {len(avail)} files")


if __name__ == "__main__":
    main()
