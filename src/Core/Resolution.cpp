#include "ZividPython/Resolution.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Resolution> pyClass)
    {
        pyClass
            .def(
                py::init<size_t, size_t>(),
                "Construct a Resolution from width and height",
                py::arg("width"),
                py::arg("height"))
            .def("width", &Zivid::Resolution::width, "Get the width value of the resolution")
            .def("height", &Zivid::Resolution::height, "Get the height value of the resolution")
            .def("size", &Zivid::Resolution::size, "Get the size (area) that is the product of the width and height")
            .def("to_string", &Zivid::Resolution::toString, "Get a string representation of the resolution")
            .def("__str__", &Zivid::Resolution::toString)
            .def("__repr__", &Zivid::Resolution::toString)
            .def("__eq__", &Zivid::Resolution::operator==, "Check if two resolutions are equal")
            .def("__ne__", &Zivid::Resolution::operator!=, "Check if two resolutions are not equal");
    }
} // namespace ZividPython
