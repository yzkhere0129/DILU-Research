// pybind11 bindings for the openfoam_cpu kernels.
//
// All array arguments use py::array::c_style | py::array::forcecast to
// guarantee contiguous double / int32 buffers — failure here means an
// extra memcpy and (worse) potential ULP drift if dtype was wrong.
//
// Output arrays are allocated and returned (not in-place) for ergonomic
// parity with the python kernels.

#include <cstddef>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "kernels.hpp"
#include "reductions.hpp"

namespace py = pybind11;

using darr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using iarr = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

namespace {

ofcpu::LDUView make_view(const darr& diag, const darr& lower, const darr& upper,
                          const iarr& owner, const iarr& neighbour) {
    if (diag.ndim()      != 1) throw std::runtime_error("diag must be 1D");
    if (lower.ndim()     != 1) throw std::runtime_error("lower must be 1D");
    if (upper.ndim()     != 1) throw std::runtime_error("upper must be 1D");
    if (owner.ndim()     != 1) throw std::runtime_error("owner must be 1D");
    if (neighbour.ndim() != 1) throw std::runtime_error("neighbour must be 1D");
    const std::size_t nC = static_cast<std::size_t>(diag.shape(0));
    const std::size_t nF = static_cast<std::size_t>(upper.shape(0));
    if (static_cast<std::size_t>(lower.shape(0))     != nF
     || static_cast<std::size_t>(owner.shape(0))     != nF
     || static_cast<std::size_t>(neighbour.shape(0)) != nF) {
        throw std::runtime_error("lower/upper/owner/neighbour size mismatch");
    }
    return ofcpu::LDUView{
        diag.data(),  lower.data(), upper.data(),
        owner.data(), neighbour.data(),
        nC, nF
    };
}

darr alloc(std::size_t n) { return darr({static_cast<py::ssize_t>(n)}); }

}  // namespace


PYBIND11_MODULE(_kernels_cpp, m) {
    m.doc() = "Byte-exact C++ kernels for openfoam_cpu replica";

    m.def("amul",
        [](darr diag, darr lower, darr upper, iarr owner, iarr neighbour,
           darr psi) {
            auto V = make_view(diag, lower, upper, owner, neighbour);
            if (static_cast<std::size_t>(psi.shape(0)) != V.n_cells)
                throw std::runtime_error("psi size != n_cells");
            auto out = alloc(V.n_cells);
            ofcpu::amul(V, psi.data(), out.mutable_data());
            return out;
        },
        py::arg("diag"), py::arg("lower"), py::arg("upper"),
        py::arg("owner"), py::arg("neighbour"), py::arg("psi"));

    m.def("tmul",
        [](darr diag, darr lower, darr upper, iarr owner, iarr neighbour,
           darr psi) {
            auto V = make_view(diag, lower, upper, owner, neighbour);
            if (static_cast<std::size_t>(psi.shape(0)) != V.n_cells)
                throw std::runtime_error("psi size != n_cells");
            auto out = alloc(V.n_cells);
            ofcpu::tmul(V, psi.data(), out.mutable_data());
            return out;
        },
        py::arg("diag"), py::arg("lower"), py::arg("upper"),
        py::arg("owner"), py::arg("neighbour"), py::arg("psi"));

    m.def("sum_a",
        [](darr diag, darr lower, darr upper, iarr owner, iarr neighbour) {
            auto V = make_view(diag, lower, upper, owner, neighbour);
            auto out = alloc(V.n_cells);
            ofcpu::sum_a(V, out.mutable_data());
            return out;
        },
        py::arg("diag"), py::arg("lower"), py::arg("upper"),
        py::arg("owner"), py::arg("neighbour"));

    m.def("calc_reciprocal_d",
        [](darr diag, darr lower, darr upper, iarr owner, iarr neighbour) {
            auto V = make_view(diag, lower, upper, owner, neighbour);
            auto rD = alloc(V.n_cells);
            ofcpu::calc_reciprocal_d(V, rD.mutable_data());
            return rD;
        },
        py::arg("diag"), py::arg("lower"), py::arg("upper"),
        py::arg("owner"), py::arg("neighbour"));

    m.def("precondition",
        [](darr diag, darr lower, darr upper, iarr owner, iarr neighbour,
           darr rD, darr rA) {
            auto V = make_view(diag, lower, upper, owner, neighbour);
            if (static_cast<std::size_t>(rD.shape(0)) != V.n_cells)
                throw std::runtime_error("rD size != n_cells");
            if (static_cast<std::size_t>(rA.shape(0)) != V.n_cells)
                throw std::runtime_error("rA size != n_cells");
            auto wA = alloc(V.n_cells);
            ofcpu::precondition(V, rD.data(), rA.data(), wA.mutable_data());
            return wA;
        },
        py::arg("diag"), py::arg("lower"), py::arg("upper"),
        py::arg("owner"), py::arg("neighbour"),
        py::arg("rD"), py::arg("rA"));

    m.def("precondition_t",
        [](darr diag, darr lower, darr upper, iarr owner, iarr neighbour,
           darr rD, iarr losort, darr rT) {
            auto V = make_view(diag, lower, upper, owner, neighbour);
            if (static_cast<std::size_t>(rD.shape(0)) != V.n_cells)
                throw std::runtime_error("rD size != n_cells");
            if (static_cast<std::size_t>(losort.shape(0)) != V.n_faces)
                throw std::runtime_error("losort size != n_faces");
            if (static_cast<std::size_t>(rT.shape(0)) != V.n_cells)
                throw std::runtime_error("rT size != n_cells");
            auto wT = alloc(V.n_cells);
            ofcpu::precondition_t(V, rD.data(), losort.data(), rT.data(),
                                  wT.mutable_data());
            return wT;
        },
        py::arg("diag"), py::arg("lower"), py::arg("upper"),
        py::arg("owner"), py::arg("neighbour"),
        py::arg("rD"), py::arg("losort"), py::arg("rT"));

    m.def("sum_abs", [](darr r) {
        return ofcpu::sum_abs(r.data(), static_cast<std::size_t>(r.shape(0)));
    }, py::arg("r"));

    m.def("dot", [](darr a, darr b) {
        if (a.shape(0) != b.shape(0))
            throw std::runtime_error("dot: size mismatch");
        return ofcpu::dot(a.data(), b.data(),
                          static_cast<std::size_t>(a.shape(0)));
    }, py::arg("a"), py::arg("b"));
}
