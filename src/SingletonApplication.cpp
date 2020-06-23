#include <Zivid/CameraInfo.h>

#include <ZividPython/SingletonApplication.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<SingletonApplication> pyClass)
    {
        pyClass.def(py::init())
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
