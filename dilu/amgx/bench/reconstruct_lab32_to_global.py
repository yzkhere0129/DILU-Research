"""Assemble GLOBAL (A, b, x_final) from 32 processor*/postProcessing/matrices
dumps so AMGx (which needs single-GPU global matrix) can consume them.

Strategy:
  - Read processor*/constant/polyMesh/cellProcAddressing for global cell IDs
  - Read processor*/constant/polyMesh/faceProcAddressing for global face IDs
    (positive = internal face going to global; negative = processor patch face
     to be merged with neighbour rank's matching patch face)
  - Concatenate diag/source by cell ID
  - Concatenate upper/lower by face ID; for processor patch faces, sum the
    contributions from BOTH ranks' sides to get the real lduMatrix off-diag
    entry (which OF would have if run serial)

Output: /tmp/global_<eq>_<time>.npz with x_final, b, A_data/A_indices/A_indptr

Run on lab Xeon:
    python3 -u -m dilu.amgx.bench.reconstruct_lab32_to_global \\
        --case ~/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_lab32_dump \\
        --time 3.2e-07 --eq pd_corr0
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix


def parse_label_list(p: Path) -> np.ndarray:
    """Read OF labelList format file → flat int64 array.

    Format:
      FoamFile {...}
      N
      (
      label_0
      label_1
      ...
      )
    """
    text = p.read_text()
    # find body between first '(' after FoamFile and last ')'
    body_start = text.find("(", text.find("}"))
    body_end = text.rfind(")")
    if body_start < 0 or body_end < 0:
        raise RuntimeError(f"can't parse {p}")
    body = text[body_start + 1 : body_end]
    return np.fromstring(body, sep=" ", dtype=np.int64)


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
    print(f"Found {len(procs)} processors")

    # Load addressing
    print("Loading cell + face proc addressing ...")
    cell_addr = []
    face_addr = []
    for p in procs:
        cell_addr.append(parse_label_list(
            p / "constant" / "polyMesh" / "cellProcAddressing"))
        face_addr.append(parse_label_list(
            p / "constant" / "polyMesh" / "faceProcAddressing"))

    n_global = max(arr.max() for arr in cell_addr) + 1
    print(f"  global N = {n_global}")

    # Initialize global arrays
    diag_global = np.zeros(n_global, dtype=np.float64)
    b_global = np.zeros(n_global, dtype=np.float64)
    x_final_global = np.zeros(n_global, dtype=np.float64)

    # Collect off-diagonal triplets (row, col, value) — sum on duplicates
    rows_list, cols_list, vals_list = [], [], []

    print("\nReading per-processor matrices ...")
    t0 = time.time()
    for proc, c_addr in zip(procs, cell_addr):
        eq_dir = proc / "postProcessing" / "matrices" / args.time / args.eq
        A_local = sio.mmread(str(eq_dir / "A.mm")).tocoo()
        b_local = sio.mmread(str(eq_dir / "b.mm")).flatten()
        x_local = sio.mmread(str(eq_dir / "x_final.mm")).flatten()

        # Map to global cell IDs
        n_local = c_addr.size
        diag_local_mask = (A_local.row == A_local.col)
        diag_global[c_addr] += A_local.data[diag_local_mask]
        b_global[c_addr] = b_local
        x_final_global[c_addr] = x_local

        # Off-diagonal entries (both directions)
        off = ~diag_local_mask
        r_glb = c_addr[A_local.row[off]]
        c_glb = c_addr[A_local.col[off]]
        rows_list.append(r_glb)
        cols_list.append(c_glb)
        vals_list.append(A_local.data[off])

        print(f"  {proc.name}: n_local={n_local}, internal_offdiag={off.sum()}",
              flush=True)
    print(f"  load_t = {time.time()-t0:.1f}s")

    # Build sparse matrix (sum duplicates by COO + .tocsr() collapse)
    rows_g = np.concatenate(rows_list)
    cols_g = np.concatenate(cols_list)
    vals_g = np.concatenate(vals_list)
    print(f"\nBuilding global CSR ({rows_g.size} off-diag triplets + diag) ...")
    t0 = time.time()
    A_off = csr_matrix((vals_g, (rows_g, cols_g)), shape=(n_global, n_global))
    A_off.sum_duplicates()
    A_global = A_off + csr_matrix((diag_global, (np.arange(n_global), np.arange(n_global))),
                                    shape=(n_global, n_global))
    A_global = A_global.tocsr()
    A_global.sum_duplicates()
    print(f"  build_t = {time.time()-t0:.1f}s, nnz={A_global.nnz}")

    # Verify consistency
    rel_resid = float(np.linalg.norm(A_global @ x_final_global - b_global)
                       / max(np.linalg.norm(b_global), 1e-300))
    print(f"\n  ‖A_global · x_final - b_global‖₂ / ‖b_global‖₂ = {rel_resid:.3e}")
    if rel_resid < 1e-3:
        print(f"  ✅ Global reconstruction consistent")
    else:
        print(f"  ⚠️  Reconstruction residual {rel_resid:.2e} — may have missed "
              f"processor-patch contributions; check faceProcAddressing logic")

    out = Path(args.out) / f"global_{args.eq}_{args.time}.npz"
    np.savez_compressed(
        out,
        A_data=A_global.data, A_indices=A_global.indices,
        A_indptr=A_global.indptr,
        x_final=x_final_global, b=b_global,
        n_global=np.array([n_global], dtype=np.int64),
        rel_resid=np.array([rel_resid]))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
