#include <Zivid/HDR.h>

#include <ZividPython/ReleasableCamera.h>
#include <ZividPython/ReleasableFrame.h>

#include <pybind11/pybind11.h>

#include <variant>
#include <vector>

namespace py = pybind11;

namespace ZividPython
{
    MetaData wrapClass(pybind11::class_<ReleasableCamera> pyClass)
    {
        pyClass.def(py::init<>())
            .def(py::self == py::self) // NOLINT
            .def(py::self != py::self) // NOLINT
            .def("connect", &ReleasableCamera::connect, py::arg("settings") = Zivid::Settings{})
            .def("disconnect", &ReleasableCamera::disconnect)
            .def("capture", py::overload_cast<>(&ReleasableCamera::capture))
            .def_property("settings", &ReleasableCamera::settings, &ReleasableCamera::setSettings)
            .def_property_readonly("state", &ReleasableCamera::state)
            .def_property_readonly("model_name", &ReleasableCamera::modelName)
            .def_property_readonly("revision", &ReleasableCamera::revision)
            .def_property_readonly("serial_number",
                                   [](ReleasableCamera &camera) { return camera.serialNumber().toString(); })
            .def_property_readonly("user_data_max_size_bytes", &ReleasableCamera::userDataMaxSizeBytes)
            .def("write_user_data", &ReleasableCamera::writeUserData)
            .def_property_readonly("user_data", &ReleasableCamera::userData)
            .def(
                "capture",
                [](ReleasableCamera &camera, const std::vector<Zivid::Settings> &settingsCollection) {
                    // Todo: This is a workaround for a bug in Zivid SDK, it can be removed when
                    //       Zivid::HDR::combineFrames starts to support empty ranges.
                    if(settingsCollection.empty())
                    {
                        throw std::runtime_error{ "Capture called with empty settings list" };
                    }
                    std::vector<Zivid::Frame> frames;
                    std::transform(std::begin(settingsCollection),
                                   std::end(settingsCollection),
                                   std::back_inserter(frames),
                                   [&](const auto &settings) {
                                       camera.setSettings(settings);
                                       return camera.capture().impl();
                                   });
                    return ReleasableFrame{ Zivid::HDR::combineFrames(begin(frames), end(frames)) };
                },
                py::arg("settings_collection"))
            .def_property_readonly("firmware_version", &ReleasableCamera::firmwareVersion);

        return { R"(Interface to one Zivid camera

See :class:`Settings` for a list of settings that can be configured in the camera.
Capture single frames by calling :func:`capture` or start continuous frame recording
:func:`start_live`.
)" };
    }
} // namespace ZividPython
