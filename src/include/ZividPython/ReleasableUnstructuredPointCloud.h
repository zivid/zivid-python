#pragma once

#include <Zivid/UnstructuredPointCloud.h>
#include <ZividPython/Releasable.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class ReleasableUnstructuredPointCloud : public Releasable<Zivid::UnstructuredPointCloud>
    {
    public:
        using Releasable<Zivid::UnstructuredPointCloud>::Releasable;

        ZIVID_PYTHON_FORWARD_0_ARGS(size)
    };

    void wrapClass(pybind11::class_<ReleasableUnstructuredPointCloud> pyClass);
} // namespace ZividPython
