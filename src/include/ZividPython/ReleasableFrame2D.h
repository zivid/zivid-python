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

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableFrame2D)

        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageRGBA, imageRGBA)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageBGRA, imageBGRA)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageRGBA_SRGB, imageRGBA_SRGB)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageBGRA_SRGB, imageBGRA_SRGB)
        ZIVID_PYTHON_FORWARD_0_ARGS(settings)
        ZIVID_PYTHON_FORWARD_0_ARGS(state)
        ZIVID_PYTHON_FORWARD_0_ARGS(info)
        ZIVID_PYTHON_FORWARD_0_ARGS(cameraInfo)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableFrame2D, clone, const);
    };

    void wrapClass(pybind11::class_<ReleasableFrame2D> pyClass);
} // namespace ZividPython
