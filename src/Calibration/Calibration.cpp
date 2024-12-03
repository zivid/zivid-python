#include <Zivid/Calibration/Detector.h>
#include <Zivid/Calibration/HandEye.h>
#include <Zivid/Calibration/MultiCamera.h>
#include <Zivid/Calibration/Pose.h>
#include <Zivid/Experimental/Calibration.h>
#include <Zivid/Experimental/Calibration/HandEyeLowDOF.h>
#include <Zivid/PointCloud.h>

#include <ZividPython/Calibration/Detector.h>
#include <ZividPython/Calibration/HandEye.h>
#include <ZividPython/Calibration/MultiCamera.h>
#include <ZividPython/Calibration/Pose.h>
#include <ZividPython/ReleasableCamera.h>
#include <ZividPython/ReleasableFrame.h>
#include <ZividPython/ReleasablePointCloud.h>
#include <ZividPython/Wrappers.h>

#include <pybind11/pybind11.h>

#include <vector>

namespace py = pybind11;

namespace ZividPython::Calibration
{
    void wrapAsSubmodule(py::module &dest)
    {
        using namespace Zivid::Calibration;
        using namespace Zivid::Experimental::Calibration::HandEyeLowDOF;

        ZIVID_PYTHON_WRAP_CLASS(dest, Pose);
        ZIVID_PYTHON_WRAP_CLASS(dest, HandEyeOutput);
        ZIVID_PYTHON_WRAP_CLASS(dest, HandEyeInput);
        ZIVID_PYTHON_WRAP_CLASS(dest, DetectionResult);
        ZIVID_PYTHON_WRAP_CLASS(dest, MarkerShape);
        ZIVID_PYTHON_WRAP_CLASS(dest, MarkerDictionary);
        ZIVID_PYTHON_WRAP_CLASS(dest, DetectionResultFiducialMarkers);
        ZIVID_PYTHON_WRAP_CLASS(dest, HandEyeResidual);

        ZIVID_PYTHON_WRAP_CLASS(dest, FixedPlacementOfFiducialMarker);
        ZIVID_PYTHON_WRAP_CLASS(dest, FixedPlacementOfFiducialMarkers);
        ZIVID_PYTHON_WRAP_CLASS(dest, FixedPlacementOfCalibrationBoard);
        ZIVID_PYTHON_WRAP_CLASS(dest, FixedPlacementOfCalibrationObjects);

        ZIVID_PYTHON_WRAP_CLASS(dest, MultiCameraResidual);
        ZIVID_PYTHON_WRAP_CLASS(dest, MultiCameraOutput);

        dest.def("detect_feature_points",
                 [](const ReleasablePointCloud &releasablePointCloud) {
                     return Zivid::Calibration::detectFeaturePoints(releasablePointCloud.impl());
                 })
            .def("detect_calibration_board",
                 [](ReleasableCamera &releasableCamera) {
                     return Zivid::Calibration::detectCalibrationBoard(releasableCamera.impl());
                 })
            .def("detect_calibration_board",
                 [](ReleasableFrame &releasableFrame) {
                     return Zivid::Calibration::detectCalibrationBoard(releasableFrame.impl());
                 })
            .def("capture_calibration_board",
                 [](ReleasableCamera &releasableCamera) {
                     return ReleasableFrame{ Zivid::Calibration::captureCalibrationBoard(releasableCamera.impl()) };
                 })
            .def("detect_markers",
                 [](const ReleasableFrame &releasableFrame,
                    const std::vector<int> &allowedMarkerIds,
                    const MarkerDictionary &markerDictionary) {
                     return detectMarkers(releasableFrame.impl(), allowedMarkerIds, markerDictionary);
                 })
            .def("calibrate_eye_in_hand", &Zivid::Calibration::calibrateEyeInHand)
            .def("calibrate_eye_to_hand", &Zivid::Calibration::calibrateEyeToHand)
            .def(
                "calibrate_eye_in_hand_low_dof",
                [](const std::vector<HandEyeInput> &inputs, const FixedPlacementOfCalibrationObjects &fixedObjects) {
                    return Zivid::Experimental::Calibration::calibrateEyeInHandLowDOF(inputs, fixedObjects);
                },
                py::arg("inputs"),
                py::arg("fixed_objects"))
            .def(
                "calibrate_eye_to_hand_low_dof",
                [](const std::vector<HandEyeInput> &inputs, const FixedPlacementOfCalibrationObjects &fixedObjects) {
                    return Zivid::Experimental::Calibration::calibrateEyeToHandLowDOF(inputs, fixedObjects);
                },
                py::arg("inputs"),
                py::arg("fixed_objects"))
            .def("calibrate_multi_camera", &Zivid::Calibration::calibrateMultiCamera)
            .def(
                "intrinsics",
                [](ReleasableCamera &releasableCamera) {
                    return Zivid::Experimental::Calibration::intrinsics(releasableCamera.impl());
                },
                py::arg("camera"))
            .def(
                "intrinsics",
                [](ReleasableCamera &releasableCamera, const Zivid::Settings &settings) {
                    return Zivid::Experimental::Calibration::intrinsics(releasableCamera.impl(), settings);
                },
                py::arg("camera"),
                py::arg("settings"))
            .def(
                "intrinsics",
                [](ReleasableCamera &releasableCamera, const Zivid::Settings2D &settings_2d) {
                    return Zivid::Experimental::Calibration::intrinsics(releasableCamera.impl(), settings_2d);
                },
                py::arg("camera"),
                py::arg("settings_2d"))
            .def("estimate_intrinsics",
                 [](ReleasableFrame &releasableFrame) {
                     return Zivid::Experimental::Calibration::estimateIntrinsics(releasableFrame.impl());
                 })
            .def(
                "pixel_mapping",
                [](ReleasableCamera &releasableCamera, const Zivid::Settings &settings) {
                    return Zivid::Experimental::Calibration::pixelMapping(releasableCamera.impl(), settings);
                },
                py::arg("camera"),
                py::arg("settings"));
    }
} // namespace ZividPython::Calibration
