#include <ZividPython/ReleasableFrame.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    MetaData wrapClass(pybind11::class_<ReleasableFrame> pyClass)
    {
        pyClass.def(py::init<>())
            .def(py::init<const std::string &>(), py::arg("file_name"))
            .def("save", &ReleasableFrame::save, py::arg("file_name"))
            .def("load", &ReleasableFrame::load, py::arg("file_name"))
            .def_property_readonly("settings", &ReleasableFrame::settings)
            .def_property_readonly("state", &ReleasableFrame::state)
            .def_property_readonly("info", &ReleasableFrame::info)
            .def("get_point_cloud", &ReleasableFrame::getPointCloud);

        return { R"(A frame captured by a Zivid camera

Contains a Compute device point cloud and/or a CPU point cloud as well as
calibration data, settings and state used by the API at time of the frame capture.
)" };
    }
} // namespace ZividPython
