#pragma once

#include <Zivid/Calibration/MultiCamera.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Calibration::MultiCameraResidual> pyClass);
    void wrapClass(pybind11::class_<Zivid::Calibration::MultiCameraOutput> pyClass);
} // namespace ZividPython
