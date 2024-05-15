#include <Zivid/CameraInfo.h>
#include <Zivid/Detail/ToolchainDetector.h>

#include <ZividPython/SingletonApplication.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<SingletonApplication> pyClass)
    {
        pyClass
            .def(py::init([] {
                // This method constructs a Zivid::Application and identifies the wrapper as the Zivid Python wrapper.
                // For users of the SDK: please do not use this method and construct the Zivid::Application directly instead.
                return SingletonApplication{ Zivid::Detail::createApplicationForWrapper(
                    Zivid::Detail::EnvironmentInfo::Wrapper::python) };
            }))
            .def("cameras", &SingletonApplication::cameras)
            .def("connect_camera", [](SingletonApplication &application) { return application.connectCamera(); })
            .def(
                "connect_camera",
                [](SingletonApplication &application, const std::string &serialNumber) {
                    return application.connectCamera(Zivid::CameraInfo::SerialNumber{ serialNumber });
                },
                py::arg("serial_number"))
            .def("create_file_camera", &SingletonApplication::createFileCamera, py::arg("frame_file"));
    }
} // namespace ZividPython
