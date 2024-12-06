#include "ZividPython/Matrix4x4.h"

#include <Zivid/Version.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>

#include <pybind11/stl.h>

namespace py = pybind11;
using Zivid::Matrix4x4;

namespace
{
    // Maps negative indexes to size + index, and throws if index is out of range
    constexpr std::size_t mapIndex(const std::int64_t index, const std::size_t size)
    {
        const auto mappedIndex = (index < 0) ? static_cast<std::size_t>(size + index) : static_cast<std::size_t>(index);
        if(mappedIndex >= size)
        {
            throw std::out_of_range("Matrix index " + std::to_string(index) + " out of range");
        }
        return mappedIndex;
    }

    Matrix4x4 createMatrixFrom4x4Array(
        const std::array<std::array<Matrix4x4::ValueType, Matrix4x4::cols>, Matrix4x4::rows> &arr)
    {
        Matrix4x4 matrix;
        auto iter = matrix.begin();
        for(size_t row = 0; row < Matrix4x4::rows; ++row)
        {
            iter = std::copy_n(arr[row].data(), Matrix4x4::cols, iter);
        }
        return matrix;
    }

    auto getItem(const Matrix4x4 &matrix, const std::tuple<int64_t, int64_t> indexes)
    {
        const auto [row, col] = indexes;
        return matrix(mapIndex(row, Matrix4x4::rows), mapIndex(col, Matrix4x4::cols));
    }

    void setItem(Matrix4x4 &matrix, const std::tuple<int64_t, int64_t> indexes, const Matrix4x4::ValueType value)
    {
        const auto [row, col] = indexes;
        matrix(mapIndex(row, Matrix4x4::rows), mapIndex(col, Matrix4x4::cols)) = value;
    }

    auto bufferInfo(Matrix4x4 &matrix)
    {
        return py::buffer_info{ matrix.data(),
                                sizeof(Matrix4x4::ValueType),
                                py::format_descriptor<Matrix4x4::ValueType>::format(),
                                2,
                                { Matrix4x4::rows, Matrix4x4::cols },
                                { sizeof(Matrix4x4::ValueType) * Matrix4x4::cols, sizeof(Matrix4x4::ValueType) } };
    }
} // namespace

void ZividPython::wrapClass(py::class_<Zivid::Matrix4x4> pyClass)
{
    pyClass.doc() = "Matrix of size 4x4 containing 32 bit floats";
    pyClass.def(py::init(), "Zero initializes all values")
        .def(py::init<const std::array<Matrix4x4::ValueType, Matrix4x4::cols * Matrix4x4::rows> &>())
        .def(py::init<const Matrix4x4 &>())
        .def(py::init(&createMatrixFrom4x4Array))
        .def(py::init<const std::string &>())
        .def_static("identity", &Matrix4x4::identity<Matrix4x4::ValueType>)
        .def("save", &Matrix4x4::save)
        .def("load", &Matrix4x4::load)
        .def("_getitem", &getItem)
        .def("_setitem", &setItem)
        .def(
            "__iter__",
            [](const Matrix4x4 &matrix) { return py::make_iterator(matrix.cbegin(), matrix.cend()); },
            py::keep_alive<0, 1>{})
        .def_property_readonly_static("rows", [](const py::object & /*self*/) { return Matrix4x4::rows; })
        .def_property_readonly_static("cols", [](const py::object & /*self*/) { return Matrix4x4::cols; })
        .def("inverse", &Matrix4x4::inverse<Matrix4x4::ValueType>)
        .def_buffer(&bufferInfo);
}