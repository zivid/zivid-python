#pragma once

#include <Zivid/PointCloud.h>
#include <Zivid/Settings.h>
#include <ZividPython/Matrix.h>
#include <ZividPython/Releasable.h>
#include <ZividPython/ReleasableImage.h>
#include <ZividPython/ReleasableImageDeviceArray.h>
#include <ZividPython/ReleasablePointCloudDeviceArray.h>
#include <ZividPython/ReleasableUnorganizedPointCloud.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class ReleasablePointCloud : public Releasable<Zivid::PointCloud>
    {
    public:
        using Releasable<Zivid::PointCloud>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasablePointCloud)

        ZIVID_PYTHON_FORWARD_0_ARGS(width)
        ZIVID_PYTHON_FORWARD_0_ARGS(height)
        ZIVID_PYTHON_FORWARD_1_ARGS(transform, const Zivid::Matrix4x4 &, matrix)
        ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(ReleasablePointCloud, transformed, const Zivid::Matrix4x4 &, matrix)

        Eigen::Matrix<float, 4, 4, Eigen::RowMajor> transformationMatrix() const
        {
            return Conversion::toPy(WITH_GIL_UNLOCKED(impl().transformationMatrix()));
        }
        ZIVID_PYTHON_FORWARD_1_ARGS(downsample, Zivid::PointCloud::Downsampling, downsampling)
        ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(
            ReleasablePointCloud,
            downsampled,
            Zivid::PointCloud::Downsampling,
            downsampling)
        ZIVID_PYTHON_FORWARD_1_ARGS(mask, const Zivid::Mask &, mask)
        ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(ReleasablePointCloud, masked, const Zivid::Mask &, mask)
        ZIVID_PYTHON_FORWARD_1_ARGS(maskByRegionOfInterest, const Zivid::Settings::RegionOfInterest::Box &, roiSettings)
        ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(
            ReleasablePointCloud,
            maskedByRegionOfInterest,
            const Zivid::Settings::RegionOfInterest::Box &,
            roiSettings)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageRGBA, copyImageRGBA)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageBGRA, copyImageBGRA)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageRGBA_SRGB, copyImageRGBA_SRGB)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageBGRA_SRGB, copyImageBGRA_SRGB)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableUnorganizedPointCloud, toUnorganizedPointCloud)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasablePointCloud, clone, const)

        // Device buffer methods for point cloud data
        ReleasableDeviceArrayPointXYZ devicePointsXYZ(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArrayPointXYZ{ WITH_GIL_UNLOCKED(impl().devicePointsXYZ(streamOrQueue)) };
        }

        ReleasableDeviceArrayPointXYZW devicePointsXYZW(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArrayPointXYZW{ WITH_GIL_UNLOCKED(impl().devicePointsXYZW(streamOrQueue)) };
        }

        ReleasableDeviceArrayPointZ devicePointsZ(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArrayPointZ{ WITH_GIL_UNLOCKED(impl().devicePointsZ(streamOrQueue)) };
        }

        ReleasableDeviceArraySNR deviceSNRs(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArraySNR{ WITH_GIL_UNLOCKED(impl().deviceSNRs(streamOrQueue)) };
        }

        ReleasableDeviceArrayNormalXYZ deviceNormalsXYZ(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArrayNormalXYZ{ WITH_GIL_UNLOCKED(impl().deviceNormalsXYZ(streamOrQueue)) };
        }

        // Device buffer methods for image data
        ReleasableImageDeviceArrayRGBA deviceImageRGBA(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayRGBA{ WITH_GIL_UNLOCKED(
                impl().imageDeviceArray<Zivid::ColorRGBA>(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayRGBA_SRGB deviceImageRGBA_SRGB(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayRGBA_SRGB{ WITH_GIL_UNLOCKED(
                impl().imageDeviceArray<Zivid::ColorRGBA_SRGB>(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayBGRA deviceImageBGRA(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayBGRA{ WITH_GIL_UNLOCKED(
                impl().imageDeviceArray<Zivid::ColorBGRA>(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayBGRA_SRGB deviceImageBGRA_SRGB(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayBGRA_SRGB{ WITH_GIL_UNLOCKED(
                impl().imageDeviceArray<Zivid::ColorBGRA_SRGB>(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayRGBAf deviceImageRGBAf(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayRGBAf{ WITH_GIL_UNLOCKED(
                impl().imageDeviceArray<Zivid::ColorRGBAf>(streamOrQueue)) };
        }
    };

    void wrapClass(pybind11::class_<ReleasablePointCloud> pyClass);
} // namespace ZividPython
