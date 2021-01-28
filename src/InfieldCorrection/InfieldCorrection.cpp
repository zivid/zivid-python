
#include <Zivid/Experimental/Calibration/InfieldCorrection.h>
#include <ZividPython/InfieldCorrection/InfieldCorrection.h>
#include <ZividPython/ReleasableCamera.h>
#include <pybind11/pybind11.h>

#include <vector>

namespace py = pybind11;

namespace ZividPython::InfieldCorrection
{
    void wrapAsSubmodule(pybind11::module &dest)
    {
        using namespace Zivid::Experimental::Calibration;

        //ZIVID_PYTHON_WRAP_CLASS(dest, Pose);
        //ZIVID_PYTHON_WRAP_CLASS(dest, HandEyeOutput);
        //ZIVID_PYTHON_WRAP_CLASS(dest, HandEyeInput);
        //ZIVID_PYTHON_WRAP_CLASS(dest, DetectionResult);
        //ZIVID_PYTHON_WRAP_CLASS(dest, HandEyeResidual);
        //ZIVID_PYTHON_WRAP_CLASS(dest, MultiCameraResidual);
        //ZIVID_PYTHON_WRAP_CLASS(dest, MultiCameraOutput);

        ZIVID_PYTHON_WRAP_CLASS(dest, InfieldCorrectionInput);
        ZIVID_PYTHON_WRAP_CLASS(dest, CameraVerification);

        dest.def("detect_feature_points_2",
                 [](ReleasableCamera &releasableCamera) {
                     return Zivid::Experimental::Calibration::detectFeaturePoints(releasableCamera.impl());
                 })
            .def("verify_camera", &Zivid::Experimental::Calibration::verifyCamera)
            .def("compute_camera_correction", &Zivid::Experimental::Calibration::computeCameraCorrection)
            .def("write_camera_correction", &Zivid::Experimental::Calibration::writeCameraCorrection);
    }
} // namespace ZividPython::InfieldCorrection

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Experimental::Calibration::InfieldCorrectionInput> pyClass)
    {
        pyClass.def(py::init<Zivid::Calibration::DetectionResult>())
            .def("valid", &Zivid::Experimental::Calibration::InfieldCorrectionInput::valid);
    }

    void wrapClass(pybind11::class_<Zivid::Experimental::Calibration::CameraVerification> pyClass)
    {
        pyClass.def("local_dimension_trueness",
                    &Zivid::Experimental::Calibration::CameraVerification::localDimensionTrueness);
    }

} // namespace ZividPython
