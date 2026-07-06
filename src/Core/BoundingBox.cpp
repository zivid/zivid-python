#include "ZividPython/BoundingBox.h"

#include <Zivid/BoundingBox.h>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using Zivid::BoundingBox;

void ZividPython::wrapClass(py::class_<Zivid::BoundingBox> pyClass)
{
    pyClass.def(py::init<int, int, int, int>(), py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"))
        .def_readwrite("x", &BoundingBox::x)
        .def_readwrite("y", &BoundingBox::y)
        .def_readwrite("width", &BoundingBox::width)
        .def_readwrite("height", &BoundingBox::height);
}
