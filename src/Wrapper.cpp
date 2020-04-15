#include <Zivid/Zivid.h>

#include <ZividPython/DataModelWrapper.h>
#include <ZividPython/Wrappers.h>

#include <ZividPython/CaptureAssistant.h>
#include <ZividPython/DataModel.h>
#include <ZividPython/Firmware.h>
#include <ZividPython/Calibration/Calibration.h>
#include <ZividPython/Calibration/Pose.h>
#include <ZividPython/ReleasableCamera.h>
#include <ZividPython/ReleasableFrame.h>
#include <ZividPython/ReleasableFrame2D.h>
#include <ZividPython/ReleasablePointCloud.h>
#include <ZividPython/SingletonApplication.h>
#include <ZividPython/Version.h>
#include <ZividPython/Wrapper.h>

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

    ZIVID_PYTHON_WRAP_CLASS_AS_SINGLETON(module, Application);
    ZIVID_PYTHON_WRAP_CLASS_AS_RELEASABLE(module, Camera);
    ZIVID_PYTHON_WRAP_CLASS_AS_RELEASABLE(module, Frame);
    ZIVID_PYTHON_WRAP_CLASS_AS_RELEASABLE(module, Frame2D);

    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, Image);
    ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(module, PointCloud);

    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Firmware);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Version);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Calibration);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, CaptureAssistant); 
}
