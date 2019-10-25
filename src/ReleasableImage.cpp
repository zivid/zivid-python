#include <ZividPython/ReleasableImage.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace
{
    py::buffer_info makeBufferInfo(ZividPython::ReleasableImage &image)
    {
        const auto data = image.dataPtr();

        return py::buffer_info{
            const_cast<Zivid::RGBA8 *>(
                data), // TODO: Const casting this until pybind11 has newer version than 2.4.3 has been released
            sizeof(Zivid::RGBA8),
            py::format_descriptor<Zivid::RGBA8>::format(),
            2,
            { image.height(), image.width() },
            { sizeof(Zivid::RGBA8) * image.width(), sizeof(Zivid::RGBA8) }
        };
    }
} // namespace

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasableImage> pyClass)
    {
        PYBIND11_NUMPY_DTYPE(Zivid::RGBA8, r, g, b, a);

        pyClass.def_buffer(makeBufferInfo)
            .def("save", &ReleasableImage::save, py::arg("file_name"))
            .def("width", &ReleasableImage::width)
            .def("height", &ReleasableImage::height);
    }
} // namespace ZividPython
