#include <ZividPython/ReleasableImage.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace
{
    py::buffer_info imageRGBADataBuffer(ZividPython::ReleasableImageRGBA &image)
    {
        using WrapperType = uint8_t;
        using NativeType = Zivid::ColorRGBA;
        constexpr size_t dim = 3;
        constexpr size_t depth = 4;

        static_assert(sizeof(WrapperType) * depth == sizeof(NativeType));
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::r)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::g)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::b)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::a)>);

        auto *dataPtr = static_cast<void *>(const_cast<NativeType *>(image.impl().data()));

        return py::buffer_info{
            dataPtr,
            sizeof(WrapperType),
            py::format_descriptor<WrapperType>::format(),
            dim,
            { image.impl().height(), image.impl().width(), depth },
            { sizeof(WrapperType) * image.impl().width() * depth, sizeof(WrapperType) * depth, sizeof(WrapperType) }
        };
    }
} // namespace

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasableImageRGBA> pyClass)
    {
        pyClass.def_buffer(imageRGBADataBuffer)
            .def("save", &ReleasableImageRGBA::save, py::arg("file_name"))
            .def("width", &ReleasableImageRGBA::width)
            .def("height", &ReleasableImageRGBA::height);
    }
} // namespace ZividPython
