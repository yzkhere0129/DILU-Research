"""Convert 50 selected matrices from MatrixMarket (.mm) ASCII → binary (.npz).

Reads the 50 timesteps from manifest.json, for each:
  - Load A.mm, b.mm, x_final.mm (pd and T)
  - Save A as CSR triplets (indptr, indices, data) into single npz
  - Save b, x_OF inside same npz

Output: <out>/matrices_npz/<phase>/ts_<ts>/{pd.npz, T.npz, meta.json}

After this conversion:
  ASCII per matrix:  ~280 MB (pd) + ~280 MB (T)
  Binary per matrix:  ~30 MB (pd) + ~30 MB (T)
  → 10× smaller, 50× faster to load

Usage (Xeon, once):
  PYTHONPATH=. python dilu/amgx/bench/suite/convert_mm_to_npz.py \\
      --case ~/cases/dense_track_dump_500K \\
      --manifest dilu/amgx/bench/suite/manifest.json \\
      --out ~/benchmark_suite_v1
"""
from __future__ import annotations
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import scipy.io as sio


def load_mm_to_arrays(p: Path):
    """Returns (indptr, indices, data, shape) for A.mm or just flat for b/x."""
    arr = sio.mmread(str(p))
    return arr


def convert_one(case: Path, ts: str, system: str):
    """Returns dict ready for np.savez_compressed."""
    sys_dir = case / "postProcessing" / "matrices" / ts / system
    A = sio.mmread(str(sys_dir / "A.mm")).tocsr()
    b = np.asarray(sio.mmread(str(sys_dir / "b.mm"))).flatten()
    x_OF = np.asarray(sio.mmread(str(sys_dir / "x_final.mm"))).flatten()
    meta = json.loads((sys_dir / "metadata.json").read_text())
    return {
        "indptr":  A.indptr.astype(np.int32),
        "indices": A.indices.astype(np.int32),
        "data":    A.data.astype(np.float64),
        "shape":   np.array(A.shape, dtype=np.int64),
        "nnz":     np.array([A.nnz], dtype=np.int64),
        "b":       b.astype(np.float64),
        "x_OF":    x_OF.astype(np.float64),
        "of_iters":   np.array([meta.get("solver_openfoam", {}).get("iterations", -1)], dtype=np.int32),
        "of_init_res": np.array([meta.get("solver_openfoam", {}).get("initial_residual", -1.0)], dtype=np.float64),
        "of_final_res": np.array([meta.get("solver_openfoam", {}).get("final_residual", -1.0)], dtype=np.float64),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--systems", default="pd_corr0,T_corr0",
                     help="comma-separated systems (one .npz per system per ts)")
    args = ap.parse_args()

    case = Path(args.case).expanduser()
    manifest = json.loads(Path(args.manifest).expanduser().read_text())
    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    matrices_dir = out_root / "matrices_npz"
    matrices_dir.mkdir(exist_ok=True)

    systems = [s.strip() for s in args.systems.split(",")]
    sys_short = {"pd_corr0": "pd", "T_corr0": "T"}

    print(f"# Converting {len(manifest['matrices'])} matrices × {len(systems)} systems")
    print(f"  case: {case}")
    print(f"  out:  {out_root}")

    t_start = time.time()
    total_done = 0
    for i, m in enumerate(manifest["matrices"]):
        ts = m["ts"]
        phase = m["phase"]
        ts_dir = matrices_dir / phase / f"ts_{ts}"
        ts_dir.mkdir(parents=True, exist_ok=True)

        # Copy manifest entry to per-ts meta.json
        (ts_dir / "meta.json").write_text(json.dumps(m, indent=2))

        for sysname in systems:
            short = sys_short.get(sysname, sysname)
            out_npz = ts_dir / f"{short}.npz"
            if out_npz.exists():
                print(f"  [{i+1}/{len(manifest['matrices'])}] {ts}/{sysname}: SKIP exists")
                continue
            try:
                t0 = time.time()
                arrs = convert_one(case, ts, sysname)
                np.savez_compressed(out_npz, **arrs)
                t = time.time() - t0
                sz_mb = out_npz.stat().st_size / 1e6
                total_done += 1
                print(f"  [{i+1}/{len(manifest['matrices'])}] {ts}/{sysname}: "
                      f"{t:.1f}s  out {sz_mb:.1f}MB")
            except FileNotFoundError as e:
                print(f"  [{i+1}/{len(manifest['matrices'])}] {ts}/{sysname}: MISSING ({e})")

    print(f"\n# Done. Converted {total_done} files in {(time.time()-t_start)/60:.1f} min")
    print(f"# Total dir size:")
    import subprocess
    print(subprocess.check_output(["du", "-sh", str(matrices_dir)], text=True).strip())

    # Copy manifest to suite dir
    shutil.copy(args.manifest, out_root / "manifest.json")


if __name__ == "__main__":
    main()
