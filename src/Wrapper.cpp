#include <Zivid/Zivid.h>

#include <ZividPython/DataModelWrapper.h>
#include <ZividPython/Wrappers.h>

#include <ZividPython/Calibration/Calibration.h>
#include <ZividPython/CaptureAssistant.h>
#include <ZividPython/DataModel.h>
#include <ZividPython/Firmware.h>
#include <ZividPython/InfieldCorrection/InfieldCorrection.h>
#include <ZividPython/Matrix4x4.h>
#include <ZividPython/PixelMapping.h>
#include <ZividPython/Presets.h>
#include <ZividPython/Projection.h>
#include <ZividPython/ReleasableArray2D.h>
#include <ZividPython/ReleasableCamera.h>
#include <ZividPython/ReleasableFrame.h>
#include <ZividPython/ReleasableFrame2D.h>
#include <ZividPython/ReleasablePointCloud.h>
#include <ZividPython/ReleasableProjectedImage.h>
#include <ZividPython/SingletonApplication.h>
#include <ZividPython/Version.h>
#include <ZividPython/Wrapper.h>

#include <Zivid/Experimental/PointCloudExport.h>
#include <ZividPython/PointCloudExport.h>
#include <pybind11/pybind11.h>

ZIVID_PYTHON_MODULE // NOLINT
{
    module.attr("__version__") = pybind11::str(ZIVID_PYTHON_VERSION);

    using namespace Zivid;

    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, DataModel);

    ZIVID_PYTHON_WRAP_DATA_MODEL(module, Settings);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, Settings2D);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, CameraState);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, CameraInfo);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, FrameInfo);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, CameraIntrinsics);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, NetworkConfiguration);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, SceneConditions);

    ZIVID_PYTHON_WRAP_CLASS_AS_SINGLETON(module, Application);
    ZIVID_PYTHON_WRAP_CLASS_AS_RELEASABLE(module, Camera);
    ZIVID_PYTHON_WRAP_CLASS_AS_RELEASABLE(module, Frame);
    ZIVID_PYTHON_WRAP_CLASS_AS_RELEASABLE(module, Frame2D);
    ZIVID_PYTHON_WRAP_CLASS_AS_RELEASABLE(module, ProjectedImage);

    ZIVID_PYTHON_WRAP_CLASS_BUFFER(module, Matrix4x4);

    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, ImageRGBA);
    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, ImageBGRA);
    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, ImageSRGB);
    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, PointCloud);

    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, ColorRGBA);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, ColorBGRA);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, ColorSRGB);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, NormalXYZ);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, PointXYZ);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, PointXYZW);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, PointZ);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, SNR);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, PointXYZColorRGBA);
    ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(module, PointXYZColorBGRA);

    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Firmware);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Version);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Calibration);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, CaptureAssistant);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, InfieldCorrection);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Projection);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Presets);

    using PixelMapping = Zivid::Experimental::PixelMapping;
    ZIVID_PYTHON_WRAP_CLASS(module, PixelMapping);

    namespace PointCloudExport = Zivid::Experimental::PointCloudExport;
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, PointCloudExport);
}
