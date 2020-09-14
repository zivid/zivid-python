#pragma once

#include <Zivid/Calibration/HandEye.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Calibration::HandEyeResidual> pyClass);
    void wrapClass(pybind11::class_<Zivid::Calibration::HandEyeOutput> pyClass);
    void wrapClass(pybind11::class_<Zivid::Calibration::HandEyeInput> pyClass);
} // namespace ZividPython
