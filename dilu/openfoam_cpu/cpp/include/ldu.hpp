// LDU view — borrows numpy buffers. No ownership.
//
// Mirrors the python LDU dataclass field-for-field. All arrays must be
// C-contiguous double / int32 with the OpenFOAM lduMatrix invariant
//   owner[f] < neighbour[f]  (faces sorted by (owner, neighbour) lex order).
//
// Single source of truth for kernel signatures: kernels expect raw pointers
// + sizes (no pybind types here so the math kernels stay header-only-clean).

#pragma once

#include <cstdint>
#include <cstddef>

namespace ofcpu {

// Plain view; pointers point into externally owned numpy buffers.
struct LDUView {
    const double*  diag;
    const double*  lower;
    const double*  upper;
    const int32_t* owner;
    const int32_t* neighbour;
    std::size_t    n_cells;
    std::size_t    n_faces;
};

}  // namespace ofcpu
