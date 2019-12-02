#include <ZividPython/AmbientLightFrequency.h>
#include <ZividPython/CaptureAssistant.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapEnum(pybind11::enum_<Zivid::CaptureAssistant::AmbientLightFrequency> pyEnum)
    {
        pyEnum.value("hz50", Zivid::CaptureAssistant::AmbientLightFrequency::hz50)
            .value("hz60", Zivid::CaptureAssistant::AmbientLightFrequency::hz60)
            .value("none", Zivid::CaptureAssistant::AmbientLightFrequency::none)
            .export_values();
    }
} // namespace ZividPython
