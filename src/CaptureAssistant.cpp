#include <Zivid/CaptureAssistant.h>

#include <ZividPython/Wrappers.h>
#include <ZividPython/CaptureAssistant.h>
#include <ZividPython/SuggestSettingsParameters.h>
#include <ZividPython/AmbientLightFrequency.h>
#include <ZividPython/ReleasableCamera.h>

#include <pybind11/pybind11.h>

#include <vector>
#include <chrono>

namespace ZividPython::CaptureAssistant
{
    void wrapAsSubmodule(pybind11::module &dest)
    {
        using namespace Zivid::CaptureAssistant;

        ZIVID_PYTHON_WRAP_ENUM_CLASS(dest, AmbientLightFrequency);
        ZIVID_PYTHON_WRAP_CLASS(dest, SuggestSettingsParameters);

        dest.def("suggest_settings",
                 [](ReleasableCamera &camera, const Zivid::CaptureAssistant::SuggestSettingsParameters &suggestSettingsParameters) {
                     return Zivid::CaptureAssistant::suggestSettings(camera.impl(), suggestSettingsParameters);
                 });
    }
} // namespace ZividPython::CaptureAssistant
