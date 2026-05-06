"""Validate MPI amul == serial amul, byte-exact within ULP × N drift.

Loads our own LPBF_sanity dump (2K cells, fast), splits into N ranks via
1D y-axis decomposition, runs amul_mpi on each rank, gathers the result,
and compares against the serial C++ amul on the un-decomposed matrix.

Acceptance:
  - max|Apsi_mpi - Apsi_serial| / ||Apsi_serial||_inf  ≤  1e-13
    (loose because order of += across halo cells may produce ULP drift,
    but the algorithm is mathematically the same)

Run:
  mpirun --oversubscribe -np 4 \\
      /home/yzk/jax-env/bin/python -m dilu.openfoam_cpu.tests.test_amul_mpi
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from mpi4py import MPI

from dilu.openfoam_cpu.python.ldu import csr_to_ldu
from dilu.openfoam_cpu.python import _kernels_cpp as cpp
from dilu.openfoam_cpu.python.mpi.decompose import decompose_1d
from dilu.openfoam_cpu.python.mpi.amul_mpi import amul_mpi


CASE = Path(
    "/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_sanity/"
    "postProcessing/matrices/2.636507509e-12/pd_corr0"
)
# LPBF_sanity blockMesh from system/blockMeshDict: hex (8 32 8) = 2048 cells
MESH_DIM = (8, 32, 8)


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"=== test_amul_mpi: {size} ranks ===", flush=True)

    A_csr = sio.mmread(str(CASE / "A.mm")).tocsr()
    n_cells = A_csr.shape[0]
    if rank == 0:
        print(f"  N = {n_cells}, mesh assumed {MESH_DIM}, "
              f"{MESH_DIM[0]*MESH_DIM[1]*MESH_DIM[2]} cells", flush=True)
    assert n_cells == MESH_DIM[0] * MESH_DIM[1] * MESH_DIM[2], \
        f"mesh dim mismatch ({MESH_DIM} = {np.prod(MESH_DIM)} ≠ {n_cells})"

    ldu = csr_to_ldu(A_csr)

    # All ranks have the global LDU (cheap for 2K cells).  We build the
    # per-rank LocalLDU; this is decomposition setup.
    locals_all = decompose_1d(ldu, n_ranks=size, mesh_dim=MESH_DIM, axis=1)
    local = locals_all[rank]
    if rank == 0:
        for r, lcl in enumerate(locals_all):
            print(f"    rank {r}: n_local={lcl.n_local}, "
                  f"n_local_faces={lcl.n_local_faces}, "
                  f"n_halos={len(lcl.halos)}, "
                  f"halo_face_count_total={sum(h.local_cell.size for h in lcl.halos)}",
                  flush=True)

    # Random psi_global, identical across ranks (fix seed)
    rng = np.random.default_rng(0xCAFEBABE)
    psi_global = rng.standard_normal(n_cells)
    psi_local = psi_global[local.local_to_global].astype(np.float64).copy()

    # Run MPI amul
    Apsi_local = amul_mpi(local, psi_local, comm)

    # Gather all local Apsi to rank 0
    if rank == 0:
        gathered = [None] * size
        gathered_gids = [None] * size
        gathered[0] = Apsi_local
        gathered_gids[0] = local.local_to_global
        for src in range(1, size):
            n_local_src = comm.recv(source=src, tag=100)
            buf = np.empty(n_local_src, dtype=np.float64)
            comm.Recv(buf, source=src, tag=101)
            gids = np.empty(n_local_src, dtype=np.int64)
            comm.Recv(gids, source=src, tag=102)
            gathered[src] = buf
            gathered_gids[src] = gids
        Apsi_mpi = np.empty(n_cells, dtype=np.float64)
        for buf, gids in zip(gathered, gathered_gids):
            Apsi_mpi[gids] = buf
    else:
        comm.send(local.n_local, dest=0, tag=100)
        comm.Send(Apsi_local, dest=0, tag=101)
        comm.Send(local.local_to_global, dest=0, tag=102)

    # Compare on rank 0
    rc = 0
    if rank == 0:
        Apsi_serial = cpp.amul(ldu.diag, ldu.lower, ldu.upper,
                                ldu.owner, ldu.neighbour, psi_global)
        diff = np.abs(Apsi_mpi - Apsi_serial)
        denom = max(float(np.abs(Apsi_serial).max()), 1e-300)
        max_abs = float(diff.max())
        max_rel = max_abs / denom
        bytes_eq = np.array_equal(Apsi_mpi, Apsi_serial)
        print(f"\n  ‖Apsi_mpi - Apsi_serial‖_∞      = {max_abs:.3e}")
        print(f"  rel (vs ‖Apsi_serial‖_∞)         = {max_rel:.3e}")
        print(f"  byte-equal?                      = {bytes_eq}")
        threshold = 1e-13
        ok = (max_rel <= threshold) or bytes_eq
        print(f"\n  GATE (rel ≤ {threshold:.0e}): {'PASS' if ok else 'FAIL'}")
        rc = 0 if ok else 1

    rc = comm.bcast(rc, root=0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
