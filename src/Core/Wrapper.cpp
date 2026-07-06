#include <ZividPython/DataModelWrapper.h>
#include <ZividPython/FrameFileType.h>
#include <ZividPython/Wrappers.h>

#include <ZividPython/BoundingBox.h>
#include <ZividPython/Calibration/Calibration.h>
#include <ZividPython/CameraAddress.h>
#include <ZividPython/CaptureAssistant.h>
#include <ZividPython/CoreWrapper.h>
#include <ZividPython/DataModel.h>
#include <ZividPython/Firmware.h>
#include <ZividPython/InfieldCorrection/InfieldCorrection.h>
#include <ZividPython/Matrix4x4.h>
#include <ZividPython/PixelMapping.h>
#include <ZividPython/Presets.h>
#include <ZividPython/Projection.h>
#include <ZividPython/ReleasableArray1D.h>
#include <ZividPython/ReleasableArray2D.h>
#include <ZividPython/ReleasableCamera.h>
#include <ZividPython/ReleasableComputeDevice.h>
#include <ZividPython/ReleasableDeviceArrayView.h>
#include <ZividPython/ReleasableFrame.h>
#include <ZividPython/ReleasableFrame2D.h>
#include <ZividPython/ReleasableImageDeviceArray.h>
#include <ZividPython/ReleasableMask.h>
#include <ZividPython/ReleasablePointCloud.h>
#include <ZividPython/ReleasablePointCloudDeviceArray.h>
#include <ZividPython/ReleasableProjectedImage.h>
#include <ZividPython/ReleasableUnorganizedPointCloud.h>
#include <ZividPython/Resolution.h>
#include <ZividPython/SingletonApplication.h>
#include <ZividPython/Toolbox/Barcode.h>
#include <ZividPython/Toolbox/PointCloudRegistration.h>
#include <ZividPython/Toolbox/Toolbox.h>
#include <ZividPython/Version.h>

#include <Zivid/Experimental/PointCloudExport.h>
#include <ZividPython/PointCloudExport.h>
#include <pybind11/pybind11.h>

#include "Zivid/CameraIntrinsics.h"

ZIVID_PYTHON_MODULE // NOLINT
{
    module.attr("__version__") = pybind11::str(ZIVID_PYTHON_VERSION);

    using namespace Zivid;

    ZIVID_PYTHON_WRAP_ENUM_CLASS(module, FrameFileType);
    module.def("read_frame_file_type", &Zivid::readFrameFileType, py::arg("file_name"));

    // GPU compute device enums
    ZIVID_PYTHON_WRAP_ENUM_CLASS(module, ComputeBackend);

    // GPU context/stream/queue structs - manually registered (no toString method)
    {
        auto cudaContextPtr = pybind11::class_<CUDAContextPtr>(module, "CUDAContextPtr");
        ZividPython::wrapClass(cudaContextPtr);

        auto cudaStreamPtr = pybind11::class_<CUDAStreamPtr>(module, "CUDAStreamPtr");
        ZividPython::wrapClass(cudaStreamPtr);

        auto openCLCommandQueuePtr = pybind11::class_<OpenCLCommandQueuePtr>(module, "OpenCLCommandQueuePtr");
        ZividPython::wrapClass(openCLCommandQueuePtr);

        auto streamOrQueue = pybind11::class_<StreamOrQueue>(module, "StreamOrQueue");
        ZividPython::wrapClass(streamOrQueue);

        module.def(
            "synchronize_stream",
            &Zivid::synchronizeStream,
            pybind11::arg("stream_or_queue"),
            "Block the host thread until all work previously enqueued on stream_or_queue has completed. "
            "Equivalent to cudaStreamSynchronize on CUDA builds and clFinish on OpenCL builds. "
            "Use this after the sync-free DeviceArray host accessors (copy_to_host_organized_array, "
            "copy_to_host_unorganized_array, to_image) to make their results safe to read.");
    }

    // GPU compute device and image device buffer classes - manually registered (no toString method)
    {
        auto computeDevice = pybind11::class_<ZividPython::ReleasableComputeDevice>(module, "ComputeDevice");
        ZividPython::wrapClass(computeDevice);

        auto imageDeviceArrayRGBA =
            pybind11::class_<ZividPython::ReleasableImageDeviceArrayRGBA>(module, "ImageDeviceArrayRGBA");
        ZividPython::wrapClass(imageDeviceArrayRGBA);

        auto imageDeviceArrayBGRA =
            pybind11::class_<ZividPython::ReleasableImageDeviceArrayBGRA>(module, "ImageDeviceArrayBGRA");
        ZividPython::wrapClass(imageDeviceArrayBGRA);

        auto imageDeviceArrayRGBA_SRGB =
            pybind11::class_<ZividPython::ReleasableImageDeviceArrayRGBA_SRGB>(module, "ImageDeviceArrayRGBA_SRGB");
        ZividPython::wrapClass(imageDeviceArrayRGBA_SRGB);

        auto imageDeviceArrayBGRA_SRGB =
            pybind11::class_<ZividPython::ReleasableImageDeviceArrayBGRA_SRGB>(module, "ImageDeviceArrayBGRA_SRGB");
        ZividPython::wrapClass(imageDeviceArrayBGRA_SRGB);

        auto imageDeviceArrayRGBAf =
            pybind11::class_<ZividPython::ReleasableImageDeviceArrayRGBAf>(module, "ImageDeviceArrayRGBAf");
        ZividPython::wrapClass(imageDeviceArrayRGBAf);

        auto imageDeviceArrayRGB =
            pybind11::class_<ZividPython::ReleasableImageDeviceArrayRGB>(module, "ImageDeviceArrayRGB");
        ZividPython::wrapClass(imageDeviceArrayRGB);

        auto imageDeviceArrayRGB_SRGB =
            pybind11::class_<ZividPython::ReleasableImageDeviceArrayRGB_SRGB>(module, "ImageDeviceArrayRGB_SRGB");
        ZividPython::wrapClass(imageDeviceArrayRGB_SRGB);

        auto imageDeviceArrayBGR =
            pybind11::class_<ZividPython::ReleasableImageDeviceArrayBGR>(module, "ImageDeviceArrayBGR");
        ZividPython::wrapClass(imageDeviceArrayBGR);

        auto imageDeviceArrayBGR_SRGB =
            pybind11::class_<ZividPython::ReleasableImageDeviceArrayBGR_SRGB>(module, "ImageDeviceArrayBGR_SRGB");
        ZividPython::wrapClass(imageDeviceArrayBGR_SRGB);

        auto deviceArrayViewRGBA =
            pybind11::class_<ZividPython::ReleasableDeviceArrayViewRGBA>(module, "DeviceArrayViewRGBA");
        ZividPython::wrapClass(deviceArrayViewRGBA);

        auto deviceArrayViewBGRA =
            pybind11::class_<ZividPython::ReleasableDeviceArrayViewBGRA>(module, "DeviceArrayViewBGRA");
        ZividPython::wrapClass(deviceArrayViewBGRA);

        auto deviceArrayViewRGBA_SRGB =
            pybind11::class_<ZividPython::ReleasableDeviceArrayViewRGBA_SRGB>(module, "DeviceArrayViewRGBA_SRGB");
        ZividPython::wrapClass(deviceArrayViewRGBA_SRGB);

        auto deviceArrayViewBGRA_SRGB =
            pybind11::class_<ZividPython::ReleasableDeviceArrayViewBGRA_SRGB>(module, "DeviceArrayViewBGRA_SRGB");
        ZividPython::wrapClass(deviceArrayViewBGRA_SRGB);

        auto deviceArrayViewRGBAf =
            pybind11::class_<ZividPython::ReleasableDeviceArrayViewRGBAf>(module, "DeviceArrayViewRGBAf");
        ZividPython::wrapClass(deviceArrayViewRGBAf);

        auto deviceArrayViewRGB =
            pybind11::class_<ZividPython::ReleasableDeviceArrayViewRGB>(module, "DeviceArrayViewRGB");
        ZividPython::wrapClass(deviceArrayViewRGB);

        auto deviceArrayViewRGB_SRGB =
            pybind11::class_<ZividPython::ReleasableDeviceArrayViewRGB_SRGB>(module, "DeviceArrayViewRGB_SRGB");
        ZividPython::wrapClass(deviceArrayViewRGB_SRGB);

        auto deviceArrayViewBGR =
            pybind11::class_<ZividPython::ReleasableDeviceArrayViewBGR>(module, "DeviceArrayViewBGR");
        ZividPython::wrapClass(deviceArrayViewBGR);

        auto deviceArrayViewBGR_SRGB =
            pybind11::class_<ZividPython::ReleasableDeviceArrayViewBGR_SRGB>(module, "DeviceArrayViewBGR_SRGB");
        ZividPython::wrapClass(deviceArrayViewBGR_SRGB);

        auto deviceArrayPointXYZ =
            pybind11::class_<ZividPython::ReleasableDeviceArrayPointXYZ>(module, "DeviceArrayPointXYZ");
        ZividPython::wrapClass(deviceArrayPointXYZ);

        auto deviceArrayPointXYZW =
            pybind11::class_<ZividPython::ReleasableDeviceArrayPointXYZW>(module, "DeviceArrayPointXYZW");
        ZividPython::wrapClass(deviceArrayPointXYZW);

        auto deviceArrayPointZ =
            pybind11::class_<ZividPython::ReleasableDeviceArrayPointZ>(module, "DeviceArrayPointZ");
        ZividPython::wrapClass(deviceArrayPointZ);

        auto deviceArraySNR = pybind11::class_<ZividPython::ReleasableDeviceArraySNR>(module, "DeviceArraySNR");
        ZividPython::wrapClass(deviceArraySNR);

        auto deviceArrayNormalXYZ =
            pybind11::class_<ZividPython::ReleasableDeviceArrayNormalXYZ>(module, "DeviceArrayNormalXYZ");
        ZividPython::wrapClass(deviceArrayNormalXYZ);
    }

    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, DataModel);

    ZIVID_PYTHON_WRAP_DATA_MODEL(module, Settings);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, Settings2D);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, CameraHealth);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, CameraState);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, CameraInfo);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, FrameInfo);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, CameraIntrinsics);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, NetworkConfiguration);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, SceneConditions);

    ZIVID_PYTHON_WRAP_CLASS_AS_SINGLETON(module, Application)
    ZIVID_PYTHON_WRAP_CLASS_AS_RELEASABLE(module, Camera);
    ZIVID_PYTHON_WRAP_CLASS_AS_RELEASABLE(module, Frame);
    ZIVID_PYTHON_WRAP_CLASS_AS_RELEASABLE(module, Frame2D);
    ZIVID_PYTHON_WRAP_CLASS_AS_RELEASABLE(module, ProjectedImage);

    ZIVID_PYTHON_WRAP_CLASS(module, BoundingBox);
    ZIVID_PYTHON_WRAP_CLASS(module, CameraAddress);
    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, Mask);
    ZIVID_PYTHON_WRAP_CLASS(module, Resolution);
    ZIVID_PYTHON_WRAP_CLASS_BUFFER(module, Matrix4x4);

    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, ImageRGBA);
    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, ImageBGRA);
    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, ImageRGBA_SRGB);
    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, ImageBGRA_SRGB);
    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, ImageRGB);
    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, ImageRGB_SRGB);
    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, ImageBGR);
    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, ImageBGR_SRGB);
    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, PointCloud);
    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, UnorganizedPointCloud);

    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, ColorRGBA);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, ColorBGRA);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, ColorRGBA_SRGB);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, ColorBGRA_SRGB);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, ColorRGB);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, ColorRGB_SRGB);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, ColorBGR);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, ColorBGR_SRGB);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, NormalXYZ);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, PointXYZ);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, PointXYZW);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, PointZ);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, SNR);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, PointXYZColorRGBA);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, PointXYZColorBGRA);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, PointXYZColorRGBA_SRGB);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, PointXYZColorBGRA_SRGB);

    ZIVID_PYTHON_WRAP_ARRAY1D_BUFFER_AS_RELEASABLE(module, ColorRGBA);
    ZIVID_PYTHON_WRAP_ARRAY1D_BUFFER_AS_RELEASABLE(module, ColorBGRA);
    ZIVID_PYTHON_WRAP_ARRAY1D_BUFFER_AS_RELEASABLE(module, ColorRGBA_SRGB);
    ZIVID_PYTHON_WRAP_ARRAY1D_BUFFER_AS_RELEASABLE(module, ColorBGRA_SRGB);
    ZIVID_PYTHON_WRAP_ARRAY1D_BUFFER_AS_RELEASABLE(module, PointXYZ);
    ZIVID_PYTHON_WRAP_ARRAY1D_BUFFER_AS_RELEASABLE(module, PointXYZW);
    ZIVID_PYTHON_WRAP_ARRAY1D_BUFFER_AS_RELEASABLE(module, SNR);

    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Firmware);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Version);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Calibration);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, CaptureAssistant);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, InfieldCorrection);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Projection);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Presets);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Toolbox);

    using PixelMapping = Zivid::Experimental::PixelMapping;
    ZIVID_PYTHON_WRAP_CLASS(module, PixelMapping);

    namespace PointCloudExport = Zivid::Experimental::PointCloudExport;
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, PointCloudExport);
}
