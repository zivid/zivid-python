#pragma once

#include <Zivid/Calibration/Detector.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapEnum(pybind11::enum_<Zivid::Calibration::CalibrationBoardDetectionStatus> pyEnum);
    void wrapClass(pybind11::class_<Zivid::Calibration::DetectionResult> pyClass);
    void wrapClass(pybind11::class_<Zivid::Calibration::MarkerShape> pyClass);
    void wrapClass(pybind11::class_<Zivid::Calibration::MarkerDictionary> pyClass);
    void wrapClass(pybind11::class_<Zivid::Calibration::DetectionResultFiducialMarkers> pyClass);
} // namespace ZividPython
