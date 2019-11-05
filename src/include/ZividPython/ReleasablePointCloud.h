#pragma once

#include <Zivid/PointCloud.h>
#include <ZividPython/Releasable.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class ReleasablePointCloud : public Releasable<Zivid::PointCloud>
    {
    public:
        using Releasable<Zivid::PointCloud>::Releasable;

        ZIVID_PYTHON_FORWARD_0_ARGS(width)
        ZIVID_PYTHON_FORWARD_0_ARGS(height)
        ZIVID_PYTHON_FORWARD_0_ARGS(dataPtr)
    };

    void wrapClass(pybind11::class_<ReleasablePointCloud> pyClass);
} // namespace ZividPython
