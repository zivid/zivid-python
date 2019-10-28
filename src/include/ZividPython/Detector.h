#pragma once

#include <Zivid/HandEye/Detector.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::HandEye::DetectionResult> pyClass);
} // namespace ZividPython
