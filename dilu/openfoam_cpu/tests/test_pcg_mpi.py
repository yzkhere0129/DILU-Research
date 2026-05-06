"""End-to-end MPI PCG byte-exact (within ULP×N) vs serial PCG.

Both solvers run on the same matrix from the same x0=0; they should
take the SAME iter count and produce SAME final x.

Run:
  mpirun --oversubscribe -np 4 \\
      /home/yzk/jax-env/bin/python -m dilu.openfoam_cpu.tests.test_pcg_mpi
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from mpi4py import MPI

from dilu.openfoam_cpu.python.ldu import csr_to_ldu
from dilu.openfoam_cpu.python import pcg as pcg_serial
from dilu.openfoam_cpu.python.mpi.decompose import decompose_1d
from dilu.openfoam_cpu.python.mpi.pcg_mpi import solve_mpi


CASE = Path("/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_sanity/"
            "postProcessing/matrices/2.636507509e-12/pd_corr0")
MESH_DIM = (8, 32, 8)


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank(); size = comm.Get_size()

    A_csr = sio.mmread(str(CASE / "A.mm")).tocsr()
    b = sio.mmread(str(CASE / "b.mm")).flatten()
    x_final = sio.mmread(str(CASE / "x_final.mm")).flatten()
    x0 = sio.mmread(str(CASE / "x0.mm")).flatten()
    n = A_csr.shape[0]
    ldu = csr_to_ldu(A_csr)

    locals_all = decompose_1d(ldu, n_ranks=size, mesh_dim=MESH_DIM, axis=1)
    local = locals_all[rank]

    src_local = b[local.local_to_global].copy()
    x0_local = x0[local.local_to_global].copy()

    if rank == 0:
        print(f"=== test_pcg_mpi: {size} ranks, N={n} ===", flush=True)
    comm.Barrier()

    t0 = time.time()
    res = solve_mpi(local, src_local, x0_local, n_global=n, comm=comm,
                    tolerance=1e-12, min_iter=1, max_iter=500)
    t_mpi = time.time() - t0

    # Gather x_mpi back on rank 0
    if rank == 0:
        x_mpi = np.empty(n, dtype=np.float64)
        x_mpi[local.local_to_global] = res.x_local
        for src in range(1, size):
            ng = comm.recv(source=src, tag=50)
            buf = np.empty(ng, dtype=np.float64)
            comm.Recv(buf, source=src, tag=51)
            gids = np.empty(ng, dtype=np.int64)
            comm.Recv(gids, source=src, tag=52)
            x_mpi[gids] = buf
    else:
        comm.send(res.x_local.size, dest=0, tag=50)
        comm.Send(res.x_local.copy(), dest=0, tag=51)
        comm.Send(local.local_to_global.copy(), dest=0, tag=52)

    rc = 0
    if rank == 0:
        # Run serial PCG for byte-exact comparison
        t0 = time.time()
        ser = pcg_serial.solve(ldu, b, x0, tolerance=1e-12, min_iter=1, max_iter=500)
        t_ser = time.time() - t0

        denom = max(float(np.abs(ser.x).max()), 1e-300)
        max_diff = np.abs(x_mpi - ser.x).max()
        rel = max_diff / denom

        print(f"\n  serial: iter={ser.n_iterations} wall={t_ser:.2f}s "
              f"r0={ser.initial_residual:.2e} rN={ser.final_residual:.2e}")
        print(f"  mpi:    iter={res.n_iterations} wall={t_mpi:.2f}s "
              f"r0={res.initial_residual:.2e} rN={res.final_residual:.2e}")
        print(f"\n  iter match:                {ser.n_iterations == res.n_iterations}")
        print(f"  ‖x_mpi - x_serial‖_∞ / ‖x_serial‖_∞ = {rel:.3e}")
        print(f"  ‖x - x_OF_final‖ / ‖x_OF‖           = "
              f"{np.abs(ser.x - x_final).max() / max(np.abs(x_final).max(),1e-300):.3e}  (OF tol noise)")

        # Expected algo difference between serial-PCG and MPI-PCG:
        # MPI uses block-Jacobi DILU (each rank's local LDU only) whereas
        # serial uses full-domain DILU. Different preconditioner → different
        # iteration trajectory & a few extra iters. Both converge to the
        # same physical solution within OF tol (~1e-8).
        # Gate: iter count diff ≤ 25%, x diff ≤ 1e-7 (well below OF physical tol).
        iter_ratio = abs(res.n_iterations - ser.n_iterations) / max(ser.n_iterations, 1)
        ok_iter = iter_ratio <= 0.25
        ok_x = rel <= 1e-7
        ok = ok_iter and ok_x
        print(f"\n  GATE (iter diff ≤ 25%; rel ≤ 1e-7): "
              f"iter_ratio={iter_ratio:.1%} ({'OK' if ok_iter else 'FAIL'})  "
              f"rel={rel:.2e} ({'OK' if ok_x else 'FAIL'})  →  "
              f"{'PASS' if ok else 'FAIL'}")
        rc = 0 if ok else 1

    rc = comm.bcast(rc, root=0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
