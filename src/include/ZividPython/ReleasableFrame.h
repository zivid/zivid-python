#pragma once

#include <Zivid/Frame.h>
#include <ZividPython/Releasable.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class ReleasableFrame : public Releasable<Zivid::Frame>
    {
    public:
        using Releasable<Zivid::Frame>::Releasable;

        ZIVID_PYTHON_FORWARD_1_ARGS(save, const std::string &, fileName)
        ZIVID_PYTHON_FORWARD_1_ARGS(load, const std::string &, fileName)
        ZIVID_PYTHON_FORWARD_0_ARGS(getPointCloud)
        ZIVID_PYTHON_FORWARD_0_ARGS(settings)
        ZIVID_PYTHON_FORWARD_0_ARGS(state)
        ZIVID_PYTHON_FORWARD_0_ARGS(info)
    };

    void wrapClass(pybind11::class_<ReleasableFrame> pyClass);
} // namespace ZividPython
