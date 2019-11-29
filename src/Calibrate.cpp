#include <Zivid/HandEye/Calibrate.h>

#include <ZividPython/Calibrate.h>
#include <ZividPython/CalibrationResidual.h>
#include <ZividPython/Matrix.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::HandEye::CalibrationOutput> pyClass)
    {
        pyClass.def("valid", &Zivid::HandEye::CalibrationOutput::valid)
            .def("handEyeTransform",
                 [](const Zivid::HandEye::CalibrationOutput &calibrationOutput) {
                     return Conversion::toPy(calibrationOutput.handEyeTransform());
                 })
            .def("perPoseCalibrationResiduals", &Zivid::HandEye::CalibrationOutput::perPoseCalibrationResiduals);
    }

    void wrapClass(pybind11::class_<Zivid::HandEye::CalibrationInput> pyClass)
    {
        pyClass.def(py::init<Zivid::HandEye::Pose, Zivid::HandEye::DetectionResult>());
    }
} // namespace ZividPython
