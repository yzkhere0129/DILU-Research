#include "reductions.hpp"

#include <cmath>
#include <cstddef>

namespace ofcpu {

double sum_abs(const double* r, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        s += std::fabs(r[i]);
    return s;
}

double dot(const double* a, const double* b, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

}  // namespace ofcpu
