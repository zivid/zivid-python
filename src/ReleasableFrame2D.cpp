#include <ZividPython/ReleasableFrame2D.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasableFrame2D> pyClass)
    {
        pyClass.def_property_readonly("settings", &ReleasableFrame2D::settings)
            .def_property_readonly("state", &ReleasableFrame2D::state)
            .def_property_readonly("info", &ReleasableFrame2D::info)
            .def_property_readonly("camera_info", &ReleasableFrame2D::cameraInfo)
            .def("image_rgba", &ReleasableFrame2D::imageRGBA)
            .def("image_bgra", &ReleasableFrame2D::imageBGRA)
            .def("image_srgb", &ReleasableFrame2D::imageSRGB);
    }
} // namespace ZividPython
