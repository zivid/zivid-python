#include "ZividPython/VectorXY.h"

#include <Zivid/Vector.h>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using Zivid::VectorXY;

void ZividPython::wrapClass(py::class_<Zivid::VectorXY> pyClass)
{
    pyClass.def(py::init<float, float>(), py::arg("x"), py::arg("y"))
        .def_readwrite("x", &VectorXY::x)
        .def_readwrite("y", &VectorXY::y)
        .def("angle_in_degrees", &VectorXY::angleInDegrees)
        .def("angle_in_radians", &VectorXY::angleInRadians)
        .def("__repr__", &VectorXY::toString)
        .def(py::self == py::self)  // NOLINT
        .def(py::self != py::self); // NOLINT
}
