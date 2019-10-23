#include <Zivid/Firmware.h>
#include <ZividPython/Firmware.h>
#include <ZividPython/ReleasableCamera.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython::Firmware
{
    void wrapAsSubmodule(pybind11::module &dest)
    {
        dest.def(
                "update",
                [](ReleasableCamera &camera, const Zivid::Firmware::ProgressCallback &callback) {
                    Zivid::Firmware::update(camera.impl(), callback);
                },
                py::arg("camera"),
                py::arg("progress_callback") = Zivid::Firmware::ProgressCallback{})
            .def(
                "is_up_to_date",
                [](ReleasableCamera &camera) { return Zivid::Firmware::isUpToDate(camera.impl()); },
                py::arg("camera"));
    }
} // namespace ZividPython::Firmware
