"""End-to-end MPI byte-exactness: amul + sum_a + reductions all match serial.

Run:
  mpirun --oversubscribe -np 4 \\
      /home/yzk/jax-env/bin/python -m dilu.openfoam_cpu.tests.test_mpi_full
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
from dilu.openfoam_cpu.python.mpi.spmv_mpi import amul_mpi, sum_a_mpi
from dilu.openfoam_cpu.python.mpi.reductions_mpi import (
    gsum_mag, gsum_prod,
)


CASE = Path("/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_sanity/"
            "postProcessing/matrices/2.636507509e-12/pd_corr0")
MESH_DIM = (8, 32, 8)


def gather_to_zero(local_arr, gids, comm, n_global):
    """Helper: pack local arrays into a single global array on rank 0."""
    rank = comm.Get_rank(); size = comm.Get_size()
    if rank == 0:
        out = np.empty(n_global, dtype=np.float64)
        out[gids] = local_arr
        for src in range(1, size):
            n_src = comm.recv(source=src, tag=200)
            buf = np.empty(n_src, dtype=np.float64)
            comm.Recv(buf, source=src, tag=201)
            gid_buf = np.empty(n_src, dtype=np.int64)
            comm.Recv(gid_buf, source=src, tag=202)
            out[gid_buf] = buf
        return out
    else:
        comm.send(local_arr.size, dest=0, tag=200)
        comm.Send(local_arr.astype(np.float64).copy(), dest=0, tag=201)
        comm.Send(gids.astype(np.int64).copy(), dest=0, tag=202)
        return None


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank(); size = comm.Get_size()

    A_csr = sio.mmread(str(CASE / "A.mm")).tocsr()
    n = A_csr.shape[0]
    ldu = csr_to_ldu(A_csr)

    locals_all = decompose_1d(ldu, n_ranks=size, mesh_dim=MESH_DIM, axis=1)
    local = locals_all[rank]

    rng = np.random.default_rng(0xCAFEBABE)
    psi_global = rng.standard_normal(n)
    rA_global = rng.standard_normal(n)
    psi_local = psi_global[local.local_to_global].copy()
    rA_local = rA_global[local.local_to_global].copy()

    # ---- amul ----
    Apsi_local = amul_mpi(local, psi_local, comm)
    Apsi_mpi = gather_to_zero(Apsi_local, local.local_to_global, comm, n)

    # ---- sum_a ----
    sumA_local = sum_a_mpi(local)
    sumA_mpi = gather_to_zero(sumA_local, local.local_to_global, comm, n)

    # ---- gSumMag(rA) ----
    sm_mpi = gsum_mag(rA_local, comm)

    # ---- gSumProd(psi, rA) ----
    sp_mpi = gsum_prod(psi_local, rA_local, comm)

    if rank == 0:
        print(f"\n=== test_mpi_full: {size} ranks ===")
        # Serial reference
        Apsi_serial = cpp.amul(ldu.diag, ldu.lower, ldu.upper,
                                ldu.owner, ldu.neighbour, psi_global)
        sumA_serial = cpp.sum_a(ldu.diag, ldu.lower, ldu.upper,
                                  ldu.owner, ldu.neighbour)
        sm_serial = cpp.sum_abs(rA_global)
        sp_serial = cpp.dot(psi_global, rA_global)

        def check(name, mpi_v, serial_v, denom=None, gate=1e-13):
            if isinstance(mpi_v, np.ndarray):
                d = np.abs(mpi_v - serial_v)
                den = max(float(np.abs(serial_v).max()), 1e-300) if denom is None else denom
                rel = d.max() / den
                exact = bool(np.array_equal(mpi_v, serial_v))
            else:
                d = abs(mpi_v - serial_v)
                den = max(abs(serial_v), 1e-300) if denom is None else denom
                rel = d / den
                exact = (np.float64(mpi_v).tobytes() == np.float64(serial_v).tobytes())
            ok = (rel <= gate) or exact
            tag = "✓" if ok else "✗"
            print(f"  {tag} {name:18s} rel={rel:.3e}  byte_eq={exact}")
            return ok

        ok1 = check("amul",     Apsi_mpi, Apsi_serial)
        ok2 = check("sum_a",    sumA_mpi, sumA_serial)
        ok3 = check("gSumMag",  sm_mpi,   sm_serial)
        ok4 = check("gSumProd", sp_mpi,   sp_serial)
        ok = all([ok1, ok2, ok3, ok4])
        print(f"\n  GATE (rel ≤ 1e-13): {'PASS' if ok else 'FAIL'}")
        rc = 0 if ok else 1
    else:
        rc = 0

    rc = comm.bcast(rc, root=0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
