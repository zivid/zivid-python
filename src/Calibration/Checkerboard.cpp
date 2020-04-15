// TODO: Header/class mismatch
#include <Zivid/HandEye/Checkerboard.h>
#include <Zivid/PointCloud.h>
#include <Zivid/Vector.h>

#include <ZividPython/Checkerboard.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython::Calibration
{
    void wrapClass(pybind11::class_<Zivid::HandEye::CheckerboardDetector> pyClass)
    {
        pyClass.def(py::init<size_t, size_t>()).def("detect", &Zivid::HandEye::CheckerboardDetector::detect);
    }
} // namespace ZividPython::Calibration
