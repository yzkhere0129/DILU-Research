// Sequential, scalar implementations. Order of operations is identical to
// the numba @njit kernels in python/dilu.py and python/spmv.py — every
// `-=` and `+=` happens in face-index order, which is what makes
// byte-exactness possible.

#include "kernels.hpp"

#include <cstddef>

namespace ofcpu {

void amul(const LDUView& A, const double* psi, double* out) {
    for (std::size_t i = 0; i < A.n_cells; ++i)
        out[i] = A.diag[i] * psi[i];
    for (std::size_t f = 0; f < A.n_faces; ++f) {
        const int32_t o = A.owner[f];
        const int32_t n = A.neighbour[f];
        out[n] += A.lower[f] * psi[o];
        out[o] += A.upper[f] * psi[n];
    }
}

void tmul(const LDUView& A, const double* psi, double* out) {
    for (std::size_t i = 0; i < A.n_cells; ++i)
        out[i] = A.diag[i] * psi[i];
    for (std::size_t f = 0; f < A.n_faces; ++f) {
        const int32_t o = A.owner[f];
        const int32_t n = A.neighbour[f];
        out[n] += A.upper[f] * psi[o];
        out[o] += A.lower[f] * psi[n];
    }
}

void sum_a(const LDUView& A, double* out) {
    for (std::size_t i = 0; i < A.n_cells; ++i)
        out[i] = A.diag[i];
    for (std::size_t f = 0; f < A.n_faces; ++f) {
        out[A.neighbour[f]] += A.lower[f];
        out[A.owner[f]]     += A.upper[f];
    }
}

void calc_reciprocal_d(const LDUView& A, double* rD) {
    for (std::size_t i = 0; i < A.n_cells; ++i)
        rD[i] = A.diag[i];
    for (std::size_t f = 0; f < A.n_faces; ++f) {
        const int32_t o = A.owner[f];
        const int32_t n = A.neighbour[f];
        rD[n] -= A.upper[f] * A.lower[f] / rD[o];
    }
    for (std::size_t i = 0; i < A.n_cells; ++i)
        rD[i] = 1.0 / rD[i];
}

void precondition(const LDUView& A, const double* rD, const double* rA,
                  double* wA) {
    for (std::size_t i = 0; i < A.n_cells; ++i)
        wA[i] = rD[i] * rA[i];
    // Forward sweep
    for (std::size_t f = 0; f < A.n_faces; ++f) {
        const int32_t o = A.owner[f];
        const int32_t n = A.neighbour[f];
        wA[n] -= rD[n] * A.lower[f] * wA[o];
    }
    // Backward sweep
    for (std::size_t f = A.n_faces; f-- > 0; ) {
        const int32_t o = A.owner[f];
        const int32_t n = A.neighbour[f];
        wA[o] -= rD[o] * A.upper[f] * wA[n];
    }
}

void precondition_t(const LDUView& A, const double* rD, const int32_t* losort,
                    const double* rT, double* wT) {
    for (std::size_t i = 0; i < A.n_cells; ++i)
        wT[i] = rD[i] * rT[i];
    // Forward sweep
    for (std::size_t f = 0; f < A.n_faces; ++f) {
        const int32_t o = A.owner[f];
        const int32_t n = A.neighbour[f];
        wT[n] -= rD[n] * A.upper[f] * wT[o];
    }
    // Backward sweep over losort
    for (std::size_t f = A.n_faces; f-- > 0; ) {
        const int32_t sf = losort[f];
        const int32_t o = A.owner[sf];
        const int32_t n = A.neighbour[sf];
        wT[o] -= rD[o] * A.lower[sf] * wT[n];
    }
}

}  // namespace ofcpu
