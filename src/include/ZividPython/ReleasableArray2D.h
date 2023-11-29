#pragma once

#include <ZividPython/Releasable.h>
#include <ZividPython/Wrappers.h>

#include <Zivid/Color.h>
#include <Zivid/PointCloud.h>

namespace ZividPython
{
    template<typename NativeType>
    using ReleasableArray2D = Releasable<Zivid::Array2D<NativeType>>;

    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::SNR>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::ColorRGBA>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::ColorBGRA>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::ColorSRGB>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::NormalXYZ>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::PointXYZ>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::PointXYZW>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::PointZ>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::PointXYZColorRGBA>> pyClass);
    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::PointXYZColorBGRA>> pyClass);

} // namespace ZividPython
