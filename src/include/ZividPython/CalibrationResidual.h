#pragma once

#include <Zivid/HandEye/Calibrate.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::HandEye::CalibrationResidual> pyClass);
} // namespace ZividPython
