#pragma once

#include <Zivid/Image.h>
#include <ZividPython/Releasable.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class ReleasableImage : public Releasable<Zivid::Image<Zivid::RGBA8>>
    {
    public:
        using Releasable<Zivid::Image<Zivid::RGBA8>>::Releasable;

        ZIVID_PYTHON_FORWARD_1_ARGS(save, const std::string &, fileName)
        ZIVID_PYTHON_FORWARD_0_ARGS(width)
        ZIVID_PYTHON_FORWARD_0_ARGS(height)
        ZIVID_PYTHON_FORWARD_0_ARGS(dataPtr)
    };

    void wrapClass(pybind11::class_<ReleasableImage> pyClass);
} // namespace ZividPython
