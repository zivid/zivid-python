#include <ZividPython/ReleasableMask.h>

#include <Zivid/Resolution.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasableMask> pyClass)
    {
        pyClass.def(py::init<>(), "Create an empty mask")
            .def(
                py::init<const Zivid::Resolution &>(),
                "Create a zero-filled mask with the specified resolution",
                py::arg("resolution"))
            .def(
                py::init<const Zivid::Resolution &, const uint8_t *, const uint8_t *>(),
                "Create a mask from raw data",
                py::arg("resolution"),
                py::arg("data_begin"),
                py::arg("data_end"))
            .def(
                py::init([](const Zivid::Resolution &resolution, py::bytes data) {
                    py::buffer_info info(py::buffer(data).request());
                    if(info.format != py::format_descriptor<uint8_t>::format())
                    {
                        throw std::invalid_argument("Data must be uint8 bytes");
                    }
                    return ReleasableMask(
                        Zivid::Mask(
                            resolution,
                            static_cast<const uint8_t *>(info.ptr),
                            static_cast<const uint8_t *>(info.ptr) + info.size));
                }),
                "Create a mask from Python bytes data",
                py::arg("resolution"),
                py::arg("data"))
            .def("width", &ReleasableMask::width, "Get the width of the mask")
            .def("height", &ReleasableMask::height, "Get the height of the mask")
            .def("size", &ReleasableMask::size, "Get the size of the mask (width * height)")
            .def("resolution", &ReleasableMask::resolution, "Get the resolution of the mask")
            .def("to_string", &ReleasableMask::toString, "Get string representation of the mask")
            .def(
                "impl",
                static_cast<Zivid::Mask &(ReleasableMask::*)()>(&ReleasableMask::impl),
                "Get the underlying native Mask implementation",
                py::return_value_policy::reference_internal)
            .def_buffer([](ReleasableMask &mask) -> py::buffer_info {
                return py::buffer_info(
                    const_cast<uint8_t *>(mask.impl().data()),                 // Pointer to buffer
                    sizeof(uint8_t),                                           // Size of one scalar
                    py::format_descriptor<uint8_t>::format(),                  // Python struct-style format descriptor
                    2,                                                         // Number of dimensions
                    { mask.impl().height(), mask.impl().width() },             // Buffer dimensions
                    { sizeof(uint8_t) * mask.impl().width(), sizeof(uint8_t) } // Strides (in bytes) for each index
                );
            });
    }
} // namespace ZividPython
