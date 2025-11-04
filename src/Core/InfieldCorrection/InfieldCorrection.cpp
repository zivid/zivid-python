#include <Zivid/Calibration/InfieldCorrection.h>

#include <ZividPython/InfieldCorrection/InfieldCorrection.h>
#include <ZividPython/Matrix.h>
#include <ZividPython/ReleasableCamera.h>

#include <pybind11/pybind11.h>

#include <vector>

namespace py = pybind11;

namespace ZividPython::InfieldCorrection
{
    void wrapAsSubmodule(pybind11::module &dest)
    {
        using namespace Zivid::Calibration;

        ZIVID_PYTHON_WRAP_ENUM_CLASS(dest, InfieldCorrectionDetectionStatus);
        ZIVID_PYTHON_WRAP_CLASS(dest, InfieldCorrectionInput);
        ZIVID_PYTHON_WRAP_CLASS(dest, CameraVerification);
        ZIVID_PYTHON_WRAP_CLASS(dest, AccuracyEstimate);
        ZIVID_PYTHON_WRAP_CLASS(dest, CameraCorrection);

        dest.def("verify_camera", &verifyCamera)
            .def("compute_camera_correction", &computeCameraCorrection)
            .def(
                "write_camera_correction",
                [](ReleasableCamera &releasableCamera, CameraCorrection cameraCorrection) {
                    writeCameraCorrection(releasableCamera.impl(), cameraCorrection);
                })
            .def(
                "reset_camera_correction",
                [](ReleasableCamera &releasableCamera) { resetCameraCorrection(releasableCamera.impl()); })
            .def(
                "has_camera_correction",
                [](ReleasableCamera &releasableCamera) { return hasCameraCorrection(releasableCamera.impl()); })
            .def("camera_correction_timestamp", [](ReleasableCamera &releasableCamera) {
                return cameraCorrectionTimestamp(releasableCamera.impl());
            });
    }
} // namespace ZividPython::InfieldCorrection

namespace ZividPython
{
    using namespace Zivid::Calibration;

    void wrapEnum(pybind11::enum_<InfieldCorrectionDetectionStatus> pyEnum)
    {
        pyEnum.value("ok", Zivid::Calibration::InfieldCorrectionDetectionStatus::ok)
            .value("detection_failed", Zivid::Calibration::InfieldCorrectionDetectionStatus::detectionFailed)
            .value("invalid_alignment", Zivid::Calibration::InfieldCorrectionDetectionStatus::invalidAlignment);
    }

    void wrapClass(pybind11::class_<InfieldCorrectionInput> pyClass)
    {
        pyClass.def(py::init<Zivid::Calibration::DetectionResult>())
            .def("valid", &InfieldCorrectionInput::valid)
            .def("detection_result", &InfieldCorrectionInput::detectionResult)
            .def("status", &InfieldCorrectionInput::status)
            .def("status_description", &InfieldCorrectionInput::statusDescription);
    }

    void wrapClass(pybind11::class_<CameraVerification> pyClass)
    {
        pyClass.def("local_dimension_trueness", &CameraVerification::localDimensionTrueness)
            .def("position", [](const CameraVerification &cameraVerification) {
                return Conversion::toPyVector(cameraVerification.position());
            });
    }

    void wrapClass(pybind11::class_<AccuracyEstimate> pyClass)
    {
        pyClass.def("dimension_accuracy", &AccuracyEstimate::dimensionAccuracy)
            .def("z_min", &AccuracyEstimate::zMin)
            .def("z_max", &AccuracyEstimate::zMax);
    }

    void wrapClass(pybind11::class_<CameraCorrection> pyClass)
    {
        pyClass.def("accuracy_estimate", &CameraCorrection::accuracyEstimate);
    }

} // namespace ZividPython
