#include <ZividPython/CaptureAssistant.h>
#include <ZividPython/SuggestSettingsParameters.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::CaptureAssistant::SuggestSettingsParameters> pyClass)
    {
        pyClass
            .def(py::init([](std::chrono::milliseconds &budget) {
                     return std::make_unique<Zivid::CaptureAssistant::SuggestSettingsParameters>(budget);
                 }),
                 py::arg("max_capture_time"))
            .def(py::init(
                     [](std::chrono::milliseconds &budget, Zivid::CaptureAssistant::AmbientLightFrequency &frequency) {
                         return std::make_unique<Zivid::CaptureAssistant::SuggestSettingsParameters>(budget, frequency);
                     }),
                 py::arg("max_capture_time"),
                 py::arg("ambient_light_frequency"))
            .def("maxCaptureTime", &Zivid::CaptureAssistant::SuggestSettingsParameters::maxCaptureTime)
            .def("ambientLightFrequency", &Zivid::CaptureAssistant::SuggestSettingsParameters::ambientLightFrequency);
    }
} // namespace ZividPython
