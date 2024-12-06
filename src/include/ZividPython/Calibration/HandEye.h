#pragma once

#include <Zivid/Calibration/HandEye.h>
#include <Zivid/Experimental/Calibration/HandEyeLowDOF.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Calibration::HandEyeResidual> pyClass);
    void wrapClass(pybind11::class_<Zivid::Calibration::HandEyeOutput> pyClass);
    void wrapClass(pybind11::class_<Zivid::Calibration::HandEyeInput> pyClass);
    void wrapClass(
        pybind11::class_<Zivid::Experimental::Calibration::HandEyeLowDOF::FixedPlacementOfFiducialMarker> pyClass);
    void wrapClass(
        pybind11::class_<Zivid::Experimental::Calibration::HandEyeLowDOF::FixedPlacementOfFiducialMarkers> pyClass);
    void wrapClass(
        pybind11::class_<Zivid::Experimental::Calibration::HandEyeLowDOF::FixedPlacementOfCalibrationBoard> pyClass);
    void wrapClass(
        pybind11::class_<Zivid::Experimental::Calibration::HandEyeLowDOF::FixedPlacementOfCalibrationObjects> pyClass);
} // namespace ZividPython
