#include <Zivid/SerialNumber.h>

#include <ZividPython/SingletonApplication.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    MetaData wrapClass(pybind11::class_<SingletonApplication> pyClass)
    {
        pyClass.def(py::init<>())
            .def("cameras", &SingletonApplication::cameras)
            .def("connect_camera",
                 py::overload_cast<const Zivid::Settings &>(&SingletonApplication::connectCamera),
                 py::arg("settings") = Zivid::Settings{})
            .def(
                "connect_camera",
                [](SingletonApplication &application,
                   const std::string &serialNumber,
                   const Zivid::Settings &settings) {
                    return application.connectCamera(Zivid::SerialNumber{ serialNumber }, settings);
                },
                py::arg("serial_number"),
                py::arg("settings") = Zivid::Settings{})
            .def("create_file_camera",
                 &SingletonApplication::createFileCamera,
                 py::arg("frame_file"),
                 py::arg("settings") = Zivid::Settings{});

        return { "Manager class for Zivid" };
    }
} // namespace ZividPython
