#pragma once

#include <Zivid/Frame2D.h>
#include <Zivid/Settings2D.h>

#include <ZividPython/Releasable.h>
#include <ZividPython/ReleasableImage.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class ReleasableFrame2D : public Releasable<Zivid::Frame2D>
    {
    public:
        using Releasable<Zivid::Frame2D>::Releasable;

        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageRGBA, imageRGBA)
        ZIVID_PYTHON_FORWARD_0_ARGS(settings)
        ZIVID_PYTHON_FORWARD_0_ARGS(state)
        ZIVID_PYTHON_FORWARD_0_ARGS(info)
    };

    void wrapClass(pybind11::class_<ReleasableFrame2D> pyClass);
} // namespace ZividPython
