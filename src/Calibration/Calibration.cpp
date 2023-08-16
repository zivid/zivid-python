#include <Zivid/Calibration/Detector.h>
#include <Zivid/Calibration/HandEye.h>
#include <Zivid/Calibration/MultiCamera.h>
#include <Zivid/Calibration/Pose.h>
#include <Zivid/Experimental/Calibration.h>
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

        ZIVID_PYTHON_WRAP_CLASS(dest, Pose);
        ZIVID_PYTHON_WRAP_CLASS(dest, HandEyeOutput);
        ZIVID_PYTHON_WRAP_CLASS(dest, HandEyeInput);
        ZIVID_PYTHON_WRAP_CLASS(dest, DetectionResult);
        ZIVID_PYTHON_WRAP_CLASS(dest, HandEyeResidual);

        ZIVID_PYTHON_WRAP_CLASS(dest, MultiCameraResidual);
        ZIVID_PYTHON_WRAP_CLASS(dest, MultiCameraOutput);

        dest.def("detect_feature_points",
                 [](const ReleasablePointCloud &releasablePointCloud) {
                     return Zivid::Calibration::detectFeaturePoints(releasablePointCloud.impl());
                 })
            .def("calibrate_eye_in_hand", &Zivid::Calibration::calibrateEyeInHand)
            .def("calibrate_eye_to_hand", &Zivid::Calibration::calibrateEyeToHand)
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
            .def("estimate_intrinsics", [](ReleasableFrame &releasableFrame) {
                return Zivid::Experimental::Calibration::estimateIntrinsics(releasableFrame.impl());
            });
    }
} // namespace ZividPython::Calibration