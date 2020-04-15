#pragma once

#include <Zivid/Calibration/Detector.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Calibration::DetectionResult> pyClass);
} // namespace ZividPython
