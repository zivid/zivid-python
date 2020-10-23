#include <Zivid/CaptureAssistant.h>

#include <ZividPython/CaptureAssistant.h>
#include <ZividPython/DataModelWrapper.h>
#include <ZividPython/ReleasableCamera.h>
#include <ZividPython/Wrappers.h>

#include <pybind11/pybind11.h>

#include <chrono>
#include <vector>

namespace ZividPython::CaptureAssistant
{
    void wrapAsSubmodule(pybind11::module &dest)
    {
        using namespace Zivid::CaptureAssistant;
        ZIVID_PYTHON_WRAP_DATA_MODEL(dest, SuggestSettingsParameters);

        dest.def("suggest_settings",
                 [](ReleasableCamera &camera,
                    const Zivid::CaptureAssistant::SuggestSettingsParameters &suggestSettingsParameters) {
                     return Zivid::CaptureAssistant::suggestSettings(camera.impl(), suggestSettingsParameters);
                 });
    }
} // namespace ZividPython::CaptureAssistant
