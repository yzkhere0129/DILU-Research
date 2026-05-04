"""Convert OpenFOAM-dumped matrix tree from MatrixMarket ASCII to compressed NPZ.

For 1M-cell LPBF cases, ASCII MatrixMarket A.mm files are ~330 MB each.
50 timesteps × 5 matrices = 250 × 330 MB ≈ 80 GB per case → too large for
practical transport.

This script:
  - Walks <src_root>/<time>/<eq>_corr<k>/ subdirectories
  - Filters to a configurable subset of correctors (default: pd_corr0,
    T_corr0, T_corr1) since pd_corr1/2 are nearly identical to pd_corr0
  - Loads A.mm, b.mm, x_final.mm, drops x0.mm
  - Saves data.npz containing CSR (A_data, A_indices, A_indptr) +
    n_rows + b + x_final, all zlib-compressed
  - Copies metadata.json verbatim
  - Runs in parallel via multiprocessing

Compression ratio: ~9× (330 MB → ~37 MB per matrix).

Usage:
    python3 shrink_dump.py <src_root> <out_root> [n_workers=16]

Example:
    python3 shrink_dump.py case/postProcessing/matrices ~/case_npz/ 16
"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import scipy.io as sio

KEEP = {"pd_corr0", "T_corr0", "T_corr1"}


def read_arr(p):
    """Parse a MatrixMarket array file (real, general, N×1)."""
    with open(p) as f:
        f.readline()  # %% header
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        n = int(line.split()[0])
        vals = [float(x) for x in f if x.strip() and not x.startswith("%")]
    return np.array(vals, dtype=np.float64)[:n]


def convert(src, dst):
    """Convert one matrix dir from MM → NPZ."""
    dst.mkdir(parents=True, exist_ok=True)
    A = sio.mmread(src / "A.mm").tocsr()
    xf = read_arr(src / "x_final.mm") if (src / "x_final.mm").exists() else np.zeros(0)
    np.savez_compressed(
        dst / "data.npz",
        A_data=A.data,
        A_indices=A.indices.astype(np.int32),
        A_indptr=A.indptr.astype(np.int32),
        n_rows=np.int32(A.shape[0]),
        b=read_arr(src / "b.mm"),
        x_final=xf,
    )
    (dst / "metadata.json").write_text((src / "metadata.json").read_text())


def task(arg):
    """Worker entry point: takes (src_str, out_root_str, src_root_str)."""
    src_s, out_root_s, src_root_s = arg
    src = Path(src_s)
    out_root = Path(out_root_s)
    src_root = Path(src_root_s)
    rel = src.relative_to(src_root)
    dst = out_root / rel
    if not (dst / "data.npz").exists():
        convert(src, dst)
    return str(rel)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: shrink_dump.py <src_root> <out_root> [n_workers=16]")
        sys.exit(1)
    src_root = Path(sys.argv[1])
    out_root = Path(sys.argv[2])
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16

    dirs = [d for d in sorted(src_root.glob("*/*_corr*")) if d.name in KEEP]
    print(f"Converting {len(dirs)} matrices with {n_workers} workers...", flush=True)
    args = [(str(d), str(out_root), str(src_root)) for d in dirs]
    with Pool(n_workers) as p:
        for i, rel in enumerate(p.imap_unordered(task, args), 1):
            print(f"[{i}/{len(dirs)}] {rel}", flush=True)
    print("Done", flush=True)
