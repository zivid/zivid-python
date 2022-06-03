#pragma once

#include <Zivid/Experimental/Calibration/InfieldCorrection.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython::InfieldCorrection
{
    void wrapAsSubmodule(pybind11::module &dest);
} // namespace ZividPython::InfieldCorrection

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Experimental::Calibration::InfieldCorrectionInput> pyClass);
    void wrapClass(pybind11::class_<Zivid::Experimental::Calibration::CameraVerification> pyClass);
    void wrapClass(pybind11::class_<Zivid::Experimental::Calibration::AccuracyEstimate> pyClass);
    void wrapClass(pybind11::class_<Zivid::Experimental::Calibration::CameraCorrection> pyClass);
} // namespace ZividPython
