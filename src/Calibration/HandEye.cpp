#include <Zivid/Calibration/HandEye.h>

#include <ZividPython/Calibration/Detector.h>
#include <ZividPython/Calibration/HandEye.h>
#include <ZividPython/Matrix.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Calibration::HandEyeResidual> pyClass)
    {
        pyClass.def("rotation", &Zivid::Calibration::HandEyeResidual::rotation)
            .def("translation", &Zivid::Calibration::HandEyeResidual::translation);
    }

    void wrapClass(pybind11::class_<Zivid::Calibration::HandEyeOutput> pyClass)
    {
        pyClass.def("valid", &Zivid::Calibration::HandEyeOutput::valid)
            .def("transform",
                 [](const Zivid::Calibration::HandEyeOutput &calibrationOutput) {
                     return Conversion::toPy(calibrationOutput.transform());
                 })
            .def("residuals", &Zivid::Calibration::HandEyeOutput::residuals);
    }

    void wrapClass(pybind11::class_<Zivid::Calibration::HandEyeInput> pyClass)
    {
        pyClass.def(py::init<Zivid::Calibration::Pose, Zivid::Calibration::DetectionResult>())
            .def(py::init<Zivid::Calibration::Pose, Zivid::Calibration::DetectionResultFiducialMarkers>())
            .def("robot_pose",
                 [](const Zivid::Calibration::HandEyeInput &handEyeInput) {
                     return Conversion::toPy(handEyeInput.robotPose().toMatrix());
                 })
            .def("detection_result", &Zivid::Calibration::HandEyeInput::detectionResult);
    }

    void wrapClass(
        pybind11::class_<Zivid::Experimental::Calibration::HandEyeLowDOF::FixedPlacementOfFiducialMarker> pyClass)
    {
        using T = Zivid::Experimental::Calibration::HandEyeLowDOF::FixedPlacementOfFiducialMarker;
        pyClass.def(py::init<int, const Zivid::PointXYZ &>())
            .def_property_readonly("id", &T::id)
            .def_property_readonly("position", &T::position);
    }

    void wrapClass(
        pybind11::class_<Zivid::Experimental::Calibration::HandEyeLowDOF::FixedPlacementOfFiducialMarkers> pyClass)
    {
        using T = Zivid::Experimental::Calibration::HandEyeLowDOF::FixedPlacementOfFiducialMarkers;
        pyClass.def(
            py::init<const Zivid::Calibration::MarkerDictionary &,
                     const std::vector<Zivid::Experimental::Calibration::HandEyeLowDOF::FixedPlacementOfFiducialMarker>
                         &>());
    }

    void wrapClass(
        pybind11::class_<Zivid::Experimental::Calibration::HandEyeLowDOF::FixedPlacementOfCalibrationBoard> pyClass)
    {
        pyClass.def(py::init<const Zivid::PointXYZ &>()).def(py::init<const Zivid::Calibration::Pose &>());
    }

    void wrapClass(
        pybind11::class_<Zivid::Experimental::Calibration::HandEyeLowDOF::FixedPlacementOfCalibrationObjects> pyClass)
    {
        pyClass
            .def(py::init<const Zivid::Experimental::Calibration::HandEyeLowDOF::FixedPlacementOfFiducialMarkers &>())
            .def(py::init<const Zivid::Experimental::Calibration::HandEyeLowDOF::FixedPlacementOfCalibrationBoard &>());
    }
} // namespace ZividPython
