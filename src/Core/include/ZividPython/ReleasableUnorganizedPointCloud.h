#pragma once

#include <Zivid/DeviceArray.h>
#include <Zivid/UnorganizedPointCloud.h>
#include <ZividPython/Releasable.h>
#include <ZividPython/ReleasableImageDeviceArray.h>
#include <ZividPython/ReleasablePointCloudDeviceArray.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class ReleasableUnorganizedPointCloud : public Releasable<Zivid::UnorganizedPointCloud>
    {
    public:
        using Releasable<Zivid::UnorganizedPointCloud>::Releasable;

        ZIVID_PYTHON_FORWARD_0_ARGS(size)
        ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(
            ReleasableUnorganizedPointCloud,
            extended,
            const Zivid::UnorganizedPointCloud &,
            other)
        ZIVID_PYTHON_FORWARD_1_ARGS(extend, const Zivid::UnorganizedPointCloud &, other)
        ZIVID_PYTHON_FORWARD_2_ARGS_WRAP_RETURN(
            ReleasableUnorganizedPointCloud,
            voxelDownsampled,
            float,
            voxelSize,
            int,
            minPointsPerVoxel)
        ZIVID_PYTHON_FORWARD_1_ARGS(transform, const Zivid::Matrix4x4 &, matrix)
        ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(
            ReleasableUnorganizedPointCloud,
            transformed,
            const Zivid::Matrix4x4 &,
            matrix)
        ZIVID_PYTHON_FORWARD_0_ARGS(center)
        ZIVID_PYTHON_FORWARD_1_ARGS(paintUniformColor, const Zivid::ColorRGBA &, color)
        ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(
            ReleasableUnorganizedPointCloud,
            paintedUniformColor,
            const Zivid::ColorRGBA &,
            color)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableUnorganizedPointCloud, clone, const)

        ReleasableDeviceArrayPointXYZ deviceArrayPointXYZ(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArrayPointXYZ{ WITH_GIL_UNLOCKED(impl().devicePointsXYZ(streamOrQueue)) };
        }

        ReleasableDeviceArrayPointXYZW deviceArrayPointXYZW(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArrayPointXYZW{ WITH_GIL_UNLOCKED(impl().devicePointsXYZW(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayRGBA deviceArrayColorRGBA(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayRGBA{ WITH_GIL_UNLOCKED(
                impl().deviceColors<Zivid::ColorRGBA>(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayBGRA deviceArrayColorBGRA(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayBGRA{ WITH_GIL_UNLOCKED(
                impl().deviceColors<Zivid::ColorBGRA>(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayRGBA_SRGB deviceArrayColorRGBA_SRGB(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayRGBA_SRGB{ WITH_GIL_UNLOCKED(
                impl().deviceColors<Zivid::ColorRGBA_SRGB>(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayBGRA_SRGB deviceArrayColorBGRA_SRGB(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayBGRA_SRGB{ WITH_GIL_UNLOCKED(
                impl().deviceColors<Zivid::ColorBGRA_SRGB>(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayRGBAf deviceArrayColorRGBAf(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayRGBAf{ WITH_GIL_UNLOCKED(
                impl().deviceColors<Zivid::ColorRGBAf>(streamOrQueue)) };
        }

        ReleasableDeviceArraySNR deviceArraySNR(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArraySNR{ WITH_GIL_UNLOCKED(impl().deviceSNRs(streamOrQueue)) };
        }
    };

    void wrapClass(pybind11::class_<ReleasableUnorganizedPointCloud> pyClass);
} // namespace ZividPython
