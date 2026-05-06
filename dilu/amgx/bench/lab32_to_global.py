"""Combine 32 processor*/postProcessing/matrices/<t>/<eq>/{A,b,x_final}
into ONE global x_final field for plotting.

For x_final (the per-cell pressure/temperature field) we just need to
remap each processor's local cell IDs to global cell IDs and concat.
This uses OF's `processor*/constant/polyMesh/cellProcAddressing` (or
its OF parallel reconstructPar conventions).

For (A, b) reconstruction (full lduMatrix), we'd additionally need
faceProcAddressing — deferred unless needed.

For now: only x_final reconstruction, sufficient for visualizing the
melting and evaporation pressure fields on the GLOBAL mesh.

Run:
    python3 -u -m dilu.amgx.bench.lab32_to_global \\
        --case ~/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_lab32_dump \\
        --time 3.2e-07 --eq pd_corr0
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import scipy.io as sio


def load_cell_proc_addressing(processor_dir: Path) -> np.ndarray:
    """Read processor*/constant/polyMesh/cellProcAddressing.

    OF format:
        FoamFile { ... }
        N
        (
        global_id_0
        global_id_1
        ...
        )
    Returns array shape (N,) where entry i = global cell ID of local cell i.
    """
    p = processor_dir / "constant" / "polyMesh" / "cellProcAddressing"
    with p.open() as f:
        text = f.read()
    # find first "(" then first ")"
    body_start = text.find("(", text.find("FoamFile") + 1)
    if body_start < 0:
        raise RuntimeError(f"can't find body in {p}")
    body = text[body_start + 1 : text.rfind(")")]
    # parse integers
    ids = np.fromstring(body, sep=" ", dtype=np.int64)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--time", required=True)
    ap.add_argument("--eq", required=True)
    ap.add_argument("--out", default="/tmp")
    args = ap.parse_args()

    case = Path(args.case).expanduser()
    procs = sorted(case.glob("processor*"),
                    key=lambda p: int(p.name.replace("processor", "")))
    print(f"found {len(procs)} processors")

    # Find global N
    addr_lists = []
    for p in procs:
        addr_lists.append(load_cell_proc_addressing(p))
    n_global = max(arr.max() for arr in addr_lists) + 1
    print(f"global N = {n_global}")

    x_final_global = np.zeros(n_global, dtype=np.float64)
    b_global = np.zeros(n_global, dtype=np.float64)

    for proc, addr in zip(procs, addr_lists):
        dump_dir = proc / "postProcessing" / "matrices" / args.time / args.eq
        x_local = sio.mmread(str(dump_dir / "x_final.mm")).flatten()
        b_local = sio.mmread(str(dump_dir / "b.mm")).flatten()
        if x_local.size != addr.size:
            print(f"  ⚠️  {proc.name}: x_final size {x_local.size} ≠ "
                  f"addressing size {addr.size}, skip")
            continue
        x_final_global[addr] = x_local
        b_global[addr] = b_local
        print(f"  {proc.name}: scattered {x_local.size} cells "
              f"(global IDs {addr.min()}..{addr.max()})")

    out = Path(args.out) / f"global_x_{args.eq}_{args.time}.npz"
    np.savez_compressed(out, x_final=x_final_global, b=b_global,
                          n_global=np.array([n_global], dtype=np.int64))
    print(f"\nWrote {out}")
    print(f"  x_final range: [{x_final_global.min():.3e}, "
          f"{x_final_global.max():.3e}]")
    print(f"  b range:       [{b_global.min():.3e}, "
          f"{b_global.max():.3e}]")


if __name__ == "__main__":
    main()
