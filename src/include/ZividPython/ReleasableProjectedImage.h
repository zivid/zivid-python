#pragma once

#include <Zivid/Experimental/ProjectedImage.h>
#include <ZividPython/Releasable.h>
#include <ZividPython/ReleasableFrame2D.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class ReleasableProjectedImage : public Releasable<Zivid::Experimental::ProjectedImage>
    {
    public:
        using Releasable<Zivid::Experimental::ProjectedImage>::Releasable;

        ZIVID_PYTHON_FORWARD_0_ARGS(stop)
        ZIVID_PYTHON_FORWARD_0_ARGS(active)
        ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(ReleasableFrame2D, capture, const Zivid::Settings2D &, settings2D)
    };

    void wrapClass(pybind11::class_<ReleasableProjectedImage> pyClass);
} // namespace ZividPython
