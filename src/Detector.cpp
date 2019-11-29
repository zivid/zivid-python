#include <Zivid/HandEye/Detector.h>

#include <ZividPython/Detector.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::HandEye::DetectionResult> pyClass)
    {
        pyClass.def("valid", &Zivid::HandEye::DetectionResult::valid);
    }
} // namespace ZividPython
