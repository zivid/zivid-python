#include <ZividPython/ReleasableImage.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace
{
    py::buffer_info imageRGBADataBuffer(ZividPython::ReleasableImageRGBA &image)
    {
        using WrapperType = uint8_t;
        using NativeType = Zivid::ColorRGBA;

        constexpr py::ssize_t dim = 3;
        constexpr py::ssize_t depth = 4;
        constexpr py::ssize_t dataSize = sizeof(WrapperType);

        static_assert(dataSize * depth == sizeof(NativeType));
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::r)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::g)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::b)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::a)>);

        auto *dataPtr = static_cast<void *>(const_cast<NativeType *>(image.impl().data()));
        const auto height = static_cast<py::ssize_t>(image.impl().height());
        const auto width = static_cast<py::ssize_t>(image.impl().width());

        return py::buffer_info{ dataPtr,
                                dataSize,
                                py::format_descriptor<WrapperType>::format(),
                                dim,
                                { height, width, depth },
                                { dataSize * width * depth, dataSize * depth, dataSize } };
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
