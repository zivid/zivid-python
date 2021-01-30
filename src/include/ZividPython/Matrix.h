#pragma once

#include "pybind11/eigen.h"
#include <Zivid/Matrix.h>
#include <Zivid/Point.h>

namespace ZividPython::Conversion
{
    template<typename T, int rows, int cols>
    auto toCpp(const Eigen::Matrix<T, rows, cols, Eigen::RowMajor> &source)
    {
        return Zivid::Matrix<T, rows, cols>{ source.data(), source.data() + source.size() };
    }

    template<typename T, size_t rows, size_t cols>
    auto toPy(const Zivid::Matrix<T, rows, cols> &source)
    {
        return Eigen::Matrix<T, rows, cols, Eigen::RowMajor>{ &(source(0, 0)) };
    }

    inline auto toPyVector(const Zivid::PointXYZ &source)
    {
        return Eigen::Vector3f{ source.x, source.y, source.z };
    }
} // namespace ZividPython::Conversion
