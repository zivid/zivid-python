#include <Zivid/HandEye/Calibrate.h>

#include <ZividPython/CalibrationResidual.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::HandEye::CalibrationResidual> pyClass)
    {
        pyClass.def("rotation", &Zivid::HandEye::CalibrationResidual::rotation)
            .def("translation", &Zivid::HandEye::CalibrationResidual::translation);
    }
} // namespace ZividPython
