#pragma once

#include <Zivid/Calibration/Detector.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Calibration::DetectionResult> pyClass);
    void wrapClass(pybind11::class_<Zivid::Calibration::MarkerShape> pyClass);
    void wrapClass(pybind11::class_<Zivid::Calibration::MarkerDictionary> pyClass);
    void wrapClass(pybind11::class_<Zivid::Calibration::DetectionResultFiducialMarkers> pyClass);
} // namespace ZividPython
