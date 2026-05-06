// Byte-exact C++ replicas of the openfoam_cpu/python kernels.
//
// Every loop here is sequential and scalar. We do NOT use FMA, AVX2 with
// reduction reordering, or any associativity-breaking transform — the goal
// is to match the numba @njit Python outputs ULP=0 on the same inputs.
//
// `-fno-fast-math -mno-fma` is enforced in CMakeLists.txt.

#pragma once

#include "ldu.hpp"

namespace ofcpu {

// out = diag * psi  (scaled), then face loop:
//   out[neighbour[f]] += lower[f] * psi[owner[f]]
//   out[owner[f]]     += upper[f] * psi[neighbour[f]]
// Mirrors python/spmv.py:amul.
void amul(const LDUView& A, const double* psi, double* out);

// Same as amul with lower<->upper swapped (transpose A).
void tmul(const LDUView& A, const double* psi, double* out);

// out[i] = diag[i]; then face loop:
//   out[neighbour[f]] += lower[f]
//   out[owner[f]]     += upper[f]
// Mirrors python/spmv.py:sum_a.
void sum_a(const LDUView& A, double* out);

// rD = diag.copy(); face loop with sequential dependency:
//   rD[neighbour[f]] -= upper[f] * lower[f] / rD[owner[f]]
// then rD[i] = 1.0/rD[i] for all i.
// Mirrors python/dilu.py:_calc_rd_kernel.  rD must be size n_cells.
void calc_reciprocal_d(const LDUView& A, double* rD);

// wA[i] = rD[i] * rA[i]; forward face loop:
//   wA[neighbour[f]] -= rD[neighbour[f]] * lower[f] * wA[owner[f]]
// backward (f = nFaces-1..0):
//   wA[owner[f]] -= rD[owner[f]] * upper[f] * wA[neighbour[f]]
// Mirrors python/dilu.py:_precondition_kernel.
void precondition(const LDUView& A, const double* rD, const double* rA,
                  double* wA);

// Transpose preconditioner; uses losort for backward sweep.
//   wT[neighbour[f]] -= rD[neighbour[f]] * upper[f] * wT[owner[f]]   (forward)
//   sf = losort[f]; wT[owner[sf]] -= rD[owner[sf]] * lower[sf] * wT[neighbour[sf]]  (backward)
// Mirrors python/dilu.py:_precondition_t_kernel.
void precondition_t(const LDUView& A, const double* rD, const int32_t* losort,
                    const double* rT, double* wT);

}  // namespace ofcpu
