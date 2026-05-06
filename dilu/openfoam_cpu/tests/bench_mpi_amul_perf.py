"""Per-rank amul/sum_a wall comparison: pure-Python loop vs C++ kernel.

Run on a 2K-cell mesh with 4 ranks (so each rank has 512 internal cells +
some halo). The pure-Python face-loop dominates total wall in the original
implementation — we replace with cpp.amul. This bench shows the speedup.

Run:
  mpirun --oversubscribe -np 4 \\
      /home/yzk/jax-env/bin/python -m dilu.openfoam_cpu.tests.bench_mpi_amul_perf
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from mpi4py import MPI

from dilu.openfoam_cpu.python.ldu import csr_to_ldu
from dilu.openfoam_cpu.python.mpi.decompose import decompose_1d
from dilu.openfoam_cpu.python.mpi.spmv_mpi import amul_mpi, sum_a_mpi


CASE = Path("/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_sanity/"
            "postProcessing/matrices/2.636507509e-12/pd_corr0")
MESH_DIM = (8, 32, 8)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank(); size = comm.Get_size()
    A = sio.mmread(str(CASE / "A.mm")).tocsr()
    ldu = csr_to_ldu(A)
    locals_all = decompose_1d(ldu, n_ranks=size, mesh_dim=MESH_DIM, axis=1)
    local = locals_all[rank]

    rng = np.random.default_rng(0xCAFEBABE)
    psi_local = rng.standard_normal(local.n_local).copy()

    # warm
    for _ in range(3):
        amul_mpi(local, psi_local, comm)
        sum_a_mpi(local)

    N_REPEAT = 500
    comm.Barrier()
    t0 = time.perf_counter()
    for _ in range(N_REPEAT):
        amul_mpi(local, psi_local, comm)
    comm.Barrier()
    t_amul = (time.perf_counter() - t0) / N_REPEAT

    comm.Barrier()
    t0 = time.perf_counter()
    for _ in range(N_REPEAT):
        sum_a_mpi(local)
    comm.Barrier()
    t_sum = (time.perf_counter() - t0) / N_REPEAT

    if rank == 0:
        print(f"=== bench_mpi_amul_perf: {size} ranks, "
              f"per-rank n_local={local.n_local}, n_faces={local.n_local_faces} ===")
        print(f"  amul_mpi  : {t_amul*1e6:7.1f} μs/call")
        print(f"  sum_a_mpi : {t_sum*1e6:7.1f} μs/call")


if __name__ == "__main__":
    main()
