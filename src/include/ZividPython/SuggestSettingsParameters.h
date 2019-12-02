#pragma once

#include <Zivid/CaptureAssistant.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::CaptureAssistant::SuggestSettingsParameters> pyClass);
} // namespace ZividPython
