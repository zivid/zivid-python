#include <Zivid/Experimental/PixelMapping.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Experimental::PixelMapping> pyClass)
    {
        pyClass.def(pybind11::init(), "Initializes all values to their defaults")
            .def(pybind11::init<int, int, float, float>())
            .def("row_stride", &Zivid::Experimental::PixelMapping::rowStride)
            .def("col_stride", &Zivid::Experimental::PixelMapping::colStride)
            .def("row_offset", &Zivid::Experimental::PixelMapping::rowOffset)
            .def("col_offset", &Zivid::Experimental::PixelMapping::colOffset);
    }
} // namespace ZividPython
