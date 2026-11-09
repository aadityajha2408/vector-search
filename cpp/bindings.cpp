// Python bindings (pybind11) for ExactIndex and HNSWIndex. Module: src.vecsearch_cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "exact.hpp"
#include "hnsw.hpp"

namespace py = pybind11;
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

static void check_2d(const FloatArray& a, size_t dim, const char* name) {
    if (a.ndim() != 2) throw std::invalid_argument(std::string(name) + " must be a 2-D array");
    if (static_cast<size_t>(a.shape(1)) != dim)
        throw std::invalid_argument(std::string(name) + " has the wrong dimensionality");
}

static void check_k(size_t k) {
    if (k == 0) throw std::invalid_argument("k must be >= 1");
}

PYBIND11_MODULE(vecsearch_cpp, m) {
    m.doc() = "Exact and HNSW vector search in C++ (inner product on normalised vectors)";

    py::class_<vs::ExactIndex>(m, "ExactIndex")
        .def(py::init<size_t>(), py::arg("dim"))
        .def("add", [](vs::ExactIndex& self, FloatArray vecs) {
            check_2d(vecs, self.dim(), "vectors");
            self.add(vecs.data(), static_cast<size_t>(vecs.shape(0)));
        })
        .def("size", &vs::ExactIndex::size)
        .def("dim", &vs::ExactIndex::dim)
        .def("search", [](const vs::ExactIndex& self, FloatArray queries, size_t k, int num_threads) {
            check_2d(queries, self.dim(), "queries");
            check_k(k);
            const size_t nq = static_cast<size_t>(queries.shape(0));
            std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)};
            py::array_t<int64_t> ids(shape);
            py::array_t<float> scores(shape);
            const float* q_ptr = queries.data();
            int64_t* id_ptr = ids.mutable_data();
            float* score_ptr = scores.mutable_data();
            {
                py::gil_scoped_release release;
                self.search(q_ptr, nq, k, id_ptr, score_ptr, num_threads);
            }
            return py::make_tuple(ids, scores);
        }, py::arg("queries"), py::arg("k"), py::arg("num_threads") = 0);

    py::class_<vs::HNSWIndex>(m, "HNSWIndex")
        .def(py::init<size_t, size_t, size_t, uint32_t, bool>(), py::arg("dim"), py::arg("M") = 16,
             py::arg("ef_construction") = 200, py::arg("seed") = 42, py::arg("use_heuristic") = true)
        .def("add", [](vs::HNSWIndex& self, FloatArray vecs) {
            check_2d(vecs, self.dim(), "vectors");
            const float* ptr = vecs.data();
            const size_t n = static_cast<size_t>(vecs.shape(0));
            py::gil_scoped_release release;
            self.add(ptr, n);
        })
        .def("size", &vs::HNSWIndex::size)
        .def("dim", &vs::HNSWIndex::dim)
        .def("max_level", &vs::HNSWIndex::max_level)
        .def("level0_avg_degree", &vs::HNSWIndex::level0_avg_degree)
        .def("search", [](const vs::HNSWIndex& self, FloatArray queries, size_t k, size_t ef, int num_threads) {
            check_2d(queries, self.dim(), "queries");
            check_k(k);
            const size_t nq = static_cast<size_t>(queries.shape(0));
            std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)};
            py::array_t<int64_t> ids(shape);
            py::array_t<float> scores(shape);
            const float* q_ptr = queries.data();
            int64_t* id_ptr = ids.mutable_data();
            float* score_ptr = scores.mutable_data();
            {
                py::gil_scoped_release release;
                self.search(q_ptr, nq, k, ef, id_ptr, score_ptr, num_threads);
            }
            return py::make_tuple(ids, scores);
        }, py::arg("queries"), py::arg("k"), py::arg("ef") = 50, py::arg("num_threads") = 0);
}
