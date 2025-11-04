#pragma once

#if (defined(__GNUC__) && !defined(__clang__))
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include "pybind11/eigen.h"

#if (defined(__GNUC__) && !defined(__clang__))
#    pragma GCC diagnostic pop
#endif
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

    inline auto toPyVector(const Zivid::PointXY &source)
    {
        return Eigen::Vector2f{ source.x, source.y };
    }

} // namespace ZividPython::Conversion
