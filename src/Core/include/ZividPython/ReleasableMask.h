#pragma once

#include <ZividPython/Releasable.h>
#include <ZividPython/Wrappers.h>

#include <Zivid/Mask.h>

namespace ZividPython
{
    class ReleasableMask : public Releasable<Zivid::Mask>
    {
    public:
        using Releasable<Zivid::Mask>::Releasable;

        ZIVID_PYTHON_FORWARD_0_ARGS(width)
        ZIVID_PYTHON_FORWARD_0_ARGS(height)
        ZIVID_PYTHON_FORWARD_0_ARGS(size)
        ZIVID_PYTHON_FORWARD_0_ARGS(resolution)
        ZIVID_PYTHON_FORWARD_0_ARGS(toString)
    };

    void wrapClass(pybind11::class_<ReleasableMask> pyClass);

} // namespace ZividPython
