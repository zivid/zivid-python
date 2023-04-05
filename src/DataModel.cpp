#include <Zivid/DataModel/NodeType.h>
#include <Zivid/Point.h>
#include <Zivid/Range.h>

#include <ZividPython/DataModel.h>
#include <ZividPython/NodeType.h>
#include <ZividPython/Wrappers.h>

#include "pybind11/numpy.h"
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Range<double>> pyClass)
    {
        using R = Zivid::Range<double>;
        pyClass.def(pybind11::init<>())
            .def(pybind11::init([](double min, double max) {
                return R{ min, max };
            }))
            .def(pybind11::init([](const std::array<double, 2> &array) {
                return R{ array[0], array[1] };
            }))
            .def_property_readonly("min", [](const R &r) { return r.min(); })
            .def_property_readonly("max", [](const R &r) { return r.max(); })
            .def("to_array",
                 [](const R &self) {
                     return std::array<double, 2>{ self.min(), self.max() };
                 })
            .def("__repr__", &R::toString)
            .def("to_string", &R::toString)
            .def(pybind11::self == pybind11::self) // NOLINT
            .def(pybind11::self != pybind11::self) // NOLINT
            .def("is_in_range", &R::isInRange);

        pybind11::implicitly_convertible<pybind11::iterable, R>();
    }

    void wrapClass(pybind11::class_<Zivid::PointXYZ> pyClass)
    {
        using T = Zivid::PointXYZ;
        pyClass.def(pybind11::init<>())
            .def(pybind11::init<float, float, float>())
            .def(pybind11::init([](const std::array<float, 3> &array) {
                return T{ array[0], array[1], array[2] };
            }))
            .def("is_nan", &T::isNaN)
            .def("__repr__", &T::toString)
            .def("to_string", &T::toString)
            .def("to_array",
                 [](const T &self) {
                     return std::array<float, 3>{ self.x, self.y, self.z };
                 })
            .def(pybind11::self == pybind11::self)
            .def(pybind11::self != pybind11::self)
            .def_readwrite("x", &T::x)
            .def_readwrite("y", &T::y)
            .def_readwrite("z", &T::z);

        pybind11::implicitly_convertible<pybind11::iterable, T>();
    }

} // namespace ZividPython

namespace ZividPython::DataModel
{
    void wrapAsSubmodule(pybind11::module &dest)
    {
        using namespace Zivid::DataModel;
        ZIVID_PYTHON_WRAP_ENUM_CLASS(dest, NodeType);

        using Range = Zivid::Range<double>;
        using PointXYZ = Zivid::PointXYZ;
        ZIVID_PYTHON_WRAP_CLASS(dest, Range);
        ZIVID_PYTHON_WRAP_CLASS(dest, PointXYZ);
    }
} // namespace ZividPython::DataModel