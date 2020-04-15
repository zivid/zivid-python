#pragma once

#include <Zivid/PointCloud.h>
#include <ZividPython/Releasable.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    template<typename NativeType>
    using ReleasableArray2D = Releasable<Zivid::Array2D<NativeType>>;

    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::SNR>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::ColorRGBA>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::PointXYZ>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::PointXYZW>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::PointZ>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::PointXYZColorRGBA>> pyClass);

} // namespace ZividPython
