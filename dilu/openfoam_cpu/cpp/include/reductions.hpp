// Left-fold reductions, byte-exact with python/reductions.py.
//
// Σ_i a[i] (or |a[i]|) computed as `s += a[i]` in index order — same order
// as numba's @njit and OpenFOAM's FieldFunctions.C:478-491.
//
// Do NOT call std::accumulate / std::reduce or BLAS dot here; their order
// is not guaranteed.

#pragma once

#include <cstddef>

namespace ofcpu {

double sum_abs(const double* r, std::size_t n);
double dot    (const double* a, const double* b, std::size_t n);

}  // namespace ofcpu
