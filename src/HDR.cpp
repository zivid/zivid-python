#include <Zivid/HDR.h>
#include <ZividPython/HDR.h>
#include <ZividPython/ReleasableCamera.h>
#include <ZividPython/ReleasableFrame.h>

#include <pybind11/pybind11.h>

#include <vector>

namespace py = pybind11;

namespace ZividPython::HDR
{
    void wrapAsSubmodule(pybind11::module &dest)
    {
        dest.def(
                "combine_frames",
                [](const std::vector<ReleasableFrame> &releasableFrames) {
                    std::vector<Zivid::Frame> frames;
                    frames.reserve(releasableFrames.size());
                    std::transform(std::begin(releasableFrames),
                                   std::end(releasableFrames),
                                   std::back_inserter(frames),
                                   [](const auto &releaseableFrame) { return releaseableFrame.impl(); });
                    return ReleasableFrame{ Zivid::HDR::combineFrames(begin(frames), end(frames)) };
                },
                py::arg("frame_sequence"))
            .def(
                "capture",
                [](ReleasableCamera &camera, const std::vector<Zivid::Settings> &settings) {
                    return ReleasableFrame{ Zivid::HDR::capture(camera.impl(), settings) };
                },
                py::arg("camera"),
                py::arg("settings_list"));
    }
} // namespace ZividPython::HDR
