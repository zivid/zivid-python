#include <Zivid/HDR.h>
#include <ZividPython/HDR.h>
#include <ZividPython/ReleasableFrame.h>

#include <pybind11/pybind11.h>

#include <vector>

namespace py = pybind11;

namespace ZividPython::HDR
{
    MetaData wrapAsSubmodule(pybind11::module &dest)
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
            py::arg("frame_sequence"));

        return { "Zivid environment, configured through environment variables" };
    }
} // namespace ZividPython::HDR
