#include <Zivid/Zivid.h>

#include <ZividPython/DataModelWrapper.h>
#include <ZividPython/Wrappers.h>

#include <ZividPython/CameraRevision.h>
#include <ZividPython/Environment.h>
#include <ZividPython/Firmware.h>
#include <ZividPython/HDR.h>
#include <ZividPython/PointCloud.h>
#include <ZividPython/ReleasableCamera.h>
#include <ZividPython/ReleasableFrame.h>
#include <ZividPython/SingletonApplication.h>
#include <ZividPython/Version.h>
#include <ZividPython/Wrapper.h>

#include <pybind11/pybind11.h>

ZIVID_PYTHON_MODULE // NOLINT
{
    module.doc() = "Python bindings for the Zivid camera";
    module.attr("__version__") = pybind11::str(ZIVID_PYTHON_VERSION);

    ZIVID_PYTHON_WRAP_DATA_MODEL(module, Settings);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, CameraState);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, FrameInfo);
    ZIVID_PYTHON_WRAP_DATA_MODEL(module, CameraIntrinsics);

    ZIVID_PYTHON_WRAP_CLASS_AS_SINGLETON(module, Application);
    ZIVID_PYTHON_WRAP_CLASS_AS_RELEASABLE(module, Camera);
    ZIVID_PYTHON_WRAP_CLASS_AS_RELEASABLE(module, Frame);
    ZIVID_PYTHON_WRAP_CLASS(module, CameraRevision);

    ZIVID_PYTHON_WRAP_CLASS_BUFFER(module, PointCloud);

    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Environment);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Firmware);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, HDR);
    ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(module, Version);
}
