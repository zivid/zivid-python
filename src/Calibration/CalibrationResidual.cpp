#include <Zivid/Calibration/HandEye.h>

#include <ZividPython/Calibration/CalibrationResidual.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Calibration::HandEyeResidual> pyClass)
    {
        pyClass.def("rotation", &Zivid::Calibration::HandEyeResidual::rotation)
            .def("translation", &Zivid::Calibration::HandEyeResidual::translation);
    }
} // namespace ZividPython
