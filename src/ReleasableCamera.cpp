#include <ZividPython/ReleasableCamera.h>
#include <ZividPython/ReleasableFrame.h>

#include <pybind11/pybind11.h>

#include <variant>
#include <vector>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasableCamera> pyClass)
    {
        pyClass.def(py::init())
            .def(py::self == py::self) // NOLINT
            .def(py::self != py::self) // NOLINT
            .def("disconnect", &ReleasableCamera::disconnect)
            .def("connect", &ReleasableCamera::connect)
            .def("capture_2d_3d", &ReleasableCamera::capture2D3D)
            .def("capture_3d", &ReleasableCamera::capture3D)
            .def("capture_2d",
                 py::overload_cast<const Zivid::Settings2D &>(&ReleasableCamera::capture2D),
                 py::arg("settings_2d"))
            .def("capture_2d",
                 py::overload_cast<const Zivid::Settings &>(&ReleasableCamera::capture2D),
                 py::arg("settings"))
            .def("capture", py::overload_cast<const Zivid::Settings &>(&ReleasableCamera::capture), py::arg("settings"))
            .def("capture",
                 py::overload_cast<const Zivid::Settings2D &>(&ReleasableCamera::capture),
                 py::arg("settings_2d"))
            .def_property_readonly("state", &ReleasableCamera::state)
            .def_property_readonly("info", &ReleasableCamera::info)
            .def("write_user_data", &ReleasableCamera::writeUserData)
            .def_property_readonly("user_data", &ReleasableCamera::userData)
            .def_property_readonly("network_configuration", &ReleasableCamera::networkConfiguration)
            .def("apply_network_configuration", &ReleasableCamera::applyNetworkConfiguration)
            .def("measure_scene_conditions", &ReleasableCamera::measureSceneConditions);
    }
} // namespace ZividPython
