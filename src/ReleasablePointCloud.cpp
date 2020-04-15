#include <ZividPython/ReleasablePointCloud.h>

#include <Zivid/PointCloud.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasablePointCloud> pyClass)
    {
        pyClass.def(py::init<>())
            .def("width", &ReleasablePointCloud::width)
            .def("height", &ReleasablePointCloud::height);
    }

} // namespace ZividPython
