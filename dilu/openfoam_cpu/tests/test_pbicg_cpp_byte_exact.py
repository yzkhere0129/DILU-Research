"""End-to-end byte-exact test: PBiCG (C++) vs PBiCG (numba+Python).

For every senior pd matrix, run both solvers from x0=0 with the same
tolerance and compare:
  - iter count must be identical
  - final residual must be byte-equal (or within 1 ULP)
  - x must be byte-equal

This is the acceptance gate for the C++ kernels — proves we don't drift
across the full PBiCG iteration (where round-off would compound if any
single op differed).

Run:
    /home/yzk/jax-env/bin/python -m dilu.openfoam_cpu.tests.test_pbicg_cpp_byte_exact
"""
from __future__ import annotations

import sys
import time

import numpy as np

from dilu.amgx.bench.senior_data_loader import set_dataset, list_available, load_step
from dilu.openfoam_cpu.python.ldu import csr_to_ldu
from dilu.openfoam_cpu.python import pbicg, pbicg_cpp


def main() -> int:
    rows = []
    fail = 0

    print("End-to-end PBiCG byte-exact: C++ kernels vs numba+Python\n")
    for ds in ("initial", "evaporation"):
        set_dataset(ds)
        for step, corr in list_available():
            bundle = load_step(step, corr)
            ldu = csr_to_ldu(bundle.A)
            x0 = np.zeros_like(bundle.b)

            t0 = time.time()
            py_res = pbicg.solve(ldu, bundle.b, x0,
                                  tolerance=1e-12, min_iter=1, max_iter=1000)
            t_py = time.time() - t0

            t0 = time.time()
            cpp_res = pbicg_cpp.solve(ldu, bundle.b, x0,
                                        tolerance=1e-12, min_iter=1, max_iter=1000)
            t_cpp = time.time() - t0

            iter_match = (py_res.n_iterations == cpp_res.n_iterations)
            x_byte_eq = bool(py_res.x.tobytes() == cpp_res.x.tobytes())
            r_byte_eq = (np.float64(py_res.final_residual).tobytes()
                         == np.float64(cpp_res.final_residual).tobytes())
            x_max_abs = float(np.abs(py_res.x - cpp_res.x).max())

            ok = iter_match and x_byte_eq and r_byte_eq
            if not ok:
                fail += 1

            label = f"{ds:>11s}/{step:>3d}/{corr}"
            print(f"  {label}  iters py={py_res.n_iterations:>4d} cpp={cpp_res.n_iterations:>4d}"
                  f"  py_t={t_py:>6.2f}s  cpp_t={t_cpp:>6.2f}s  speedup={t_py/max(t_cpp,1e-9):>5.1f}x"
                  f"  x_eq={'Y' if x_byte_eq else 'N'}  r_eq={'Y' if r_byte_eq else 'N'}"
                  + ("" if ok else f"  Δx_max={x_max_abs:.2e}"))
            rows.append(dict(
                ds=ds, step=step, corr=corr,
                iters_py=py_res.n_iterations,
                iters_cpp=cpp_res.n_iterations,
                t_py=t_py, t_cpp=t_cpp,
                x_byte_eq=x_byte_eq, r_byte_eq=r_byte_eq,
                x_max_abs=x_max_abs,
            ))

    print()
    print("=" * 60)
    n = len(rows)
    if n:
        med_py = float(np.median([r["t_py"]  for r in rows]))
        med_cpp = float(np.median([r["t_cpp"] for r in rows]))
        max_iter = max(r["iters_cpp"] for r in rows)
        print(f"Cases tested: {n}")
        print(f"Failures:     {fail}")
        print(f"Median wall (py+numba):  {med_py:.2f}s")
        print(f"Median wall (C++):       {med_cpp:.2f}s")
        print(f"Median speedup:          {med_py/max(med_cpp,1e-9):.1f}x")
        print(f"Max iter:                {max_iter}")
        print(f"P1 perf gate (≤ 250ms median): "
              f"{'PASS' if med_cpp <= 0.250 else 'FAIL'}  ({med_cpp*1e3:.0f}ms)")
    print("=" * 60)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
