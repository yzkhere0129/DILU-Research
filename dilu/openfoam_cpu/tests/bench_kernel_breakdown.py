"""Per-kernel wall time on one senior pd matrix.

Useful diagnostic for deciding where SIMD / further optimization pays off.
Picks one representative bundle (mid-evaporation, step 95, corr 1) and times:
  amul, sum_a, calc_reciprocal_d, precondition, sum_abs, dot
across N_REPEATS warm runs (best of N reported).

Run:
    /home/yzk/jax-env/bin/python -m dilu.openfoam_cpu.tests.bench_kernel_breakdown
"""
from __future__ import annotations

import time

import numpy as np

from dilu.amgx.bench.senior_data_loader import set_dataset, load_step
from dilu.openfoam_cpu.python.ldu import csr_to_ldu
from dilu.openfoam_cpu.python import _kernels_cpp as cpp


N_REPEATS = 50


def best_of(fn, n=N_REPEATS):
    fn()  # warm-up
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times) * 1e3, np.median(times) * 1e3


def main():
    set_dataset("evaporation")
    bundle = load_step(95, 1)
    if bundle is None:
        bundle = load_step(94, 1)
    ldu = csr_to_ldu(bundle.A)
    print(f"Matrix: N={ldu.n_cells}  faces={ldu.n_faces}  nnz={bundle.A.nnz}")

    diag, lower, upper = ldu.diag, ldu.lower, ldu.upper
    owner, neighbour = ldu.owner, ldu.neighbour

    rng = np.random.default_rng(42)
    psi = rng.standard_normal(ldu.n_cells)
    rA = rng.standard_normal(ldu.n_cells)
    rD = cpp.calc_reciprocal_d(diag, lower, upper, owner, neighbour)
    losort = np.argsort(neighbour, kind='stable').astype(np.int32)

    print()
    print(f"{'kernel':22s} {'best (ms)':>12s} {'median (ms)':>14s}")
    print("-" * 50)

    bench = [
        ("amul",
         lambda: cpp.amul(diag, lower, upper, owner, neighbour, psi)),
        ("tmul",
         lambda: cpp.tmul(diag, lower, upper, owner, neighbour, psi)),
        ("sum_a",
         lambda: cpp.sum_a(diag, lower, upper, owner, neighbour)),
        ("calc_reciprocal_d",
         lambda: cpp.calc_reciprocal_d(diag, lower, upper, owner, neighbour)),
        ("precondition",
         lambda: cpp.precondition(diag, lower, upper, owner, neighbour, rD, rA)),
        ("precondition_t",
         lambda: cpp.precondition_t(diag, lower, upper, owner, neighbour, rD, losort, rA)),
        ("sum_abs",
         lambda: cpp.sum_abs(rA)),
        ("dot",
         lambda: cpp.dot(rA, psi)),
    ]
    for name, fn in bench:
        best, med = best_of(fn)
        print(f"{name:22s} {best:>12.3f} {med:>14.3f}")


if __name__ == "__main__":
    main()
