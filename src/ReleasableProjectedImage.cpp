#include <ZividPython/ReleasableProjectedImage.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasableProjectedImage> pyClass)
    {
        pyClass.def("stop", &ReleasableProjectedImage::stop)
            .def("active", &ReleasableProjectedImage::active)
            .def("capture", &ReleasableProjectedImage::capture);
    }
} // namespace ZividPython
