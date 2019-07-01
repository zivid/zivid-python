#include <Zivid/Firmware.h>
#include <ZividPython/Firmware.h>
#include <ZividPython/ReleasableCamera.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython::Firmware
{
    MetaData wrapAsSubmodule(pybind11::module &dest)
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

        return { "Functions used to query the state and update the camera firmware" };
    }
} // namespace ZividPython::Firmware
