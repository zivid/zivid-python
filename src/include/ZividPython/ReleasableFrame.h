#pragma once

#include <Zivid/Frame.h>
#include <ZividPython/Releasable.h>
#include <ZividPython/ReleasableFrame2D.h>
#include <ZividPython/ReleasablePointCloud.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class ReleasableFrame : public Releasable<Zivid::Frame>
    {
    public:
        using Releasable<Zivid::Frame>::Releasable;

        ZIVID_PYTHON_FORWARD_1_ARGS(save, const std::string &, fileName)
        ZIVID_PYTHON_FORWARD_1_ARGS(load, const std::string &, fileName)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasablePointCloud, pointCloud)
        std::optional<ReleasableFrame2D> frame2D()
        {
            auto frame = impl().frame2D();
            if(!frame.has_value())
            {
                return std::nullopt;
            }
            return ReleasableFrame2D{ std::move(frame.value()) };
        }
        ZIVID_PYTHON_FORWARD_0_ARGS(settings)
        ZIVID_PYTHON_FORWARD_0_ARGS(state)
        ZIVID_PYTHON_FORWARD_0_ARGS(info)
        ZIVID_PYTHON_FORWARD_0_ARGS(cameraInfo)
    };

    void wrapClass(pybind11::class_<ReleasableFrame> pyClass);
} // namespace ZividPython
