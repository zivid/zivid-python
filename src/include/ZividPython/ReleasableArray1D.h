#pragma once

#include <ZividPython/Releasable.h>
#include <ZividPython/Wrappers.h>

#include <Zivid/UnorganizedPointCloud.h>

namespace ZividPython
{
    template<typename NativeType>
    using ReleasableArray1D = Releasable<Zivid::Array1D<NativeType>>;

    void wrapClass(pybind11::class_<ReleasableArray1D<Zivid::PointXYZ>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray1D<Zivid::ColorRGBA>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray1D<Zivid::ColorBGRA>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray1D<Zivid::ColorRGBA_SRGB>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray1D<Zivid::ColorBGRA_SRGB>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray1D<Zivid::SNR>> pyClass);
} // namespace ZividPython
