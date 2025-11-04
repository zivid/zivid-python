#pragma once

#include <Zivid/Calibration/InfieldCorrection.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython::InfieldCorrection
{
    void wrapAsSubmodule(pybind11::module &dest);
} // namespace ZividPython::InfieldCorrection

namespace ZividPython
{
    void wrapEnum(pybind11::enum_<Zivid::Calibration::InfieldCorrectionDetectionStatus> pyEnum);
    void wrapClass(pybind11::class_<Zivid::Calibration::InfieldCorrectionInput> pyClass);
    void wrapClass(pybind11::class_<Zivid::Calibration::CameraVerification> pyClass);
    void wrapClass(pybind11::class_<Zivid::Calibration::AccuracyEstimate> pyClass);
    void wrapClass(pybind11::class_<Zivid::Calibration::CameraCorrection> pyClass);
} // namespace ZividPython
