#pragma once

#include <Zivid/PointCloud.h>
#include <ZividPython/Releasable.h>
#include <ZividPython/ReleasableImage.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class ReleasablePointCloud : public Releasable<Zivid::PointCloud>
    {
    public:
        using Releasable<Zivid::PointCloud>::Releasable;

        ZIVID_PYTHON_FORWARD_0_ARGS(width)
        ZIVID_PYTHON_FORWARD_0_ARGS(height)
        ZIVID_PYTHON_FORWARD_1_ARGS(transform, const Zivid::Matrix4x4 &, matrix)
        ZIVID_PYTHON_FORWARD_1_ARGS(downsample, Zivid::PointCloud::Downsampling, downsampling)
        ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(ReleasablePointCloud,
                                                downsampled,
                                                Zivid::PointCloud::Downsampling,
                                                downsampling)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageRGBA, copyImageRGBA)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageBGRA, copyImageBGRA)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageSRGB, copyImageSRGB)
    };

    void wrapClass(pybind11::class_<ReleasablePointCloud> pyClass);
} // namespace ZividPython
