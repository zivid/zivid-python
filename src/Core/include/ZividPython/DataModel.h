#pragma once

#include <Zivid/Point.h>
#include <Zivid/Range.h>

#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Range<double>> pyClass);
    void wrapClass(pybind11::class_<Zivid::PointXYZ> pyClass);
} // namespace ZividPython

namespace ZividPython::DataModel
{
    void wrapAsSubmodule(pybind11::module &dest);
} // namespace ZividPython::DataModel
