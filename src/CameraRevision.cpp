#include <ZividPython/CameraRevision.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    MetaData wrapClass(pybind11::class_<Zivid::CameraRevision> pyClass)
    {
        pyClass.def(py::init<>())
            .def(py::init<int, int>(), py::arg("major"), py::arg("minor"))
            .def(py::self == py::self) // NOLINT
            .def_property_readonly("major", &Zivid::CameraRevision::majorRevision)
            .def_property_readonly("minor", &Zivid::CameraRevision::minorRevision);

        return { "Camera revision" };
    }
} // namespace ZividPython
