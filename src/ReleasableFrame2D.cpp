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
            .def("image_rgba", &ReleasableFrame2D::imageRGBA);
    }
} // namespace ZividPython
