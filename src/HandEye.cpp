#include <Zivid/HandEye/Calibrate.h>
#include <Zivid/HandEye/Detector.h>
#include <Zivid/HandEye/Pose.h>
#include <Zivid/PointCloud.h>

#include <ZividPython/Calibrate.h>
#include <ZividPython/CalibrationResidual.h>
#include <ZividPython/Detector.h>
#include <ZividPython/Pose.h>
#include <ZividPython/ReleasablePointCloud.h>
#include <ZividPython/Wrappers.h>

#include <pybind11/pybind11.h>

#include <vector>

namespace ZividPython::HandEye
{
    void wrapAsSubmodule(pybind11::module &dest)
    {
        using namespace Zivid::HandEye;

        ZIVID_PYTHON_WRAP_CLASS(dest, Pose);
        ZIVID_PYTHON_WRAP_CLASS(dest, CalibrationOutput);
        ZIVID_PYTHON_WRAP_CLASS(dest, CalibrationInput);
        ZIVID_PYTHON_WRAP_CLASS(dest, DetectionResult);
        ZIVID_PYTHON_WRAP_CLASS(dest, CalibrationResidual);

        dest.def("detect_feature_points",
                 [](const ReleasablePointCloud &releasablePointCloud) {
                     return Zivid::HandEye::detectFeaturePoints(releasablePointCloud.impl());
                 })
            .def("calibrate_eye_in_hand", &Zivid::HandEye::calibrateEyeInHand)
            .def("calibrate_eye_to_hand", &Zivid::HandEye::calibrateEyeToHand);
    }
} // namespace ZividPython::HandEye
