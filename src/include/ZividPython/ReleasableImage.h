#pragma once

#include <Zivid/Image.h>
#include <ZividPython/Releasable.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class ReleasableImage : public Releasable<Zivid::Image<Zivid::ColorRGBA>>
    {
    public:
        using Releasable<Zivid::Image<Zivid::ColorRGBA>>::Releasable;

        ZIVID_PYTHON_FORWARD_1_ARGS(save, const std::string &, fileName)
        ZIVID_PYTHON_FORWARD_0_ARGS(width)
        ZIVID_PYTHON_FORWARD_0_ARGS(height)
        //ZIVID_PYTHON_FORWARD_0_ARGS(copyImageRGBA) TODO: operator()?
    };

    void wrapClass(pybind11::class_<ReleasableImage> pyClass);
} // namespace ZividPython
