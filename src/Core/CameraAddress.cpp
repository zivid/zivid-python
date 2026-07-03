#include "ZividPython/CameraAddress.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::CameraAddress> pyClass)
    {
        pyClass
            .def(
                py::init<std::string>(),
                "Construct a CameraAddress from a hostname or IPv4 address string",
                py::arg("value"))
            .def("value", &Zivid::CameraAddress::value, "Get the address value")
            .def("to_string", &Zivid::CameraAddress::toString, "Get a string representation of the address")
            .def("__str__", &Zivid::CameraAddress::toString)
            .def("__repr__", &Zivid::CameraAddress::toString)
            .def("__eq__", &Zivid::CameraAddress::operator==, "Check if two addresses are equal")
            .def("__ne__", &Zivid::CameraAddress::operator!=, "Check if two addresses are not equal");
    }
} // namespace ZividPython
