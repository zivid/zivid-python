#include <Zivid/Experimental/Toolbox/PointCloudRegistration.h>

#include <ZividPython/DataModelWrapper.h>
#include <ZividPython/Matrix.h>
#include <ZividPython/ReleasableUnorganizedPointCloud.h>
#include <ZividPython/Toolbox/Barcode.h>
#include <ZividPython/Toolbox/PointCloudRegistration.h>
#include <ZividPython/Toolbox/Toolbox.h>

#include <pybind11/pybind11.h>

namespace ZividPython::Toolbox
{
    void wrapAsSubmodule(pybind11::module &dest)
    {
        using namespace Zivid::Experimental::Toolbox;

        ZIVID_PYTHON_WRAP_DATA_MODEL(dest, Zivid::Experimental::LocalPointCloudRegistrationParameters);
        ZIVID_PYTHON_WRAP_CLASS(dest, LocalPointCloudRegistrationResult);

        dest.def(
            "local_point_cloud_registration",
            [](const ZividPython::ReleasableUnorganizedPointCloud &target,
               const ZividPython::ReleasableUnorganizedPointCloud &source,
               const Zivid::Experimental::LocalPointCloudRegistrationParameters &param,
               const Zivid::Calibration::Pose &initialTransform) {
                return Zivid::Experimental::Toolbox::localPointCloudRegistration(
                    target.impl(), source.impl(), param, initialTransform);
            },
            py::arg("target"),
            py::arg("source"),
            py::arg("params"),
            py::arg("initial_transform"));

        ZIVID_PYTHON_WRAP_ENUM_CLASS(dest, LinearBarcodeFormat);
        ZIVID_PYTHON_WRAP_ENUM_CLASS(dest, MatrixBarcodeFormat);
        ZIVID_PYTHON_WRAP_CLASS(dest, LinearBarcodeDetectionResult);
        ZIVID_PYTHON_WRAP_CLASS(dest, MatrixBarcodeDetectionResult);
        ZIVID_PYTHON_WRAP_CLASS_AS_RELEASABLE(dest, BarcodeDetector);
    }
} // namespace ZividPython::Toolbox
