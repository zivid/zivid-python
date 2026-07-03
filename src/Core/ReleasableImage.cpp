#include <ZividPython/ReleasableImage.h>

#include <Zivid/Color.h>

#include <pybind11/buffer_info.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace
{
    template<typename ImageType, typename NativeType>
    py::buffer_info imageDataBuffer(ImageType &image)
    {
        using WrapperType = uint8_t;

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

    py::buffer_info imageRGBADataBuffer(ZividPython::ReleasableImageRGBA &image)
    {
        return imageDataBuffer<ZividPython::ReleasableImageRGBA, Zivid::ColorRGBA>(image);
    }

    py::buffer_info imageBGRADataBuffer(ZividPython::ReleasableImageBGRA &image)
    {
        return imageDataBuffer<ZividPython::ReleasableImageBGRA, Zivid::ColorBGRA>(image);
    }

    py::buffer_info imageRGBA_SRGBDataBuffer(ZividPython::ReleasableImageRGBA_SRGB &image)
    {
        return imageDataBuffer<ZividPython::ReleasableImageRGBA_SRGB, Zivid::ColorRGBA_SRGB>(image);
    }

    py::buffer_info imageBGRA_SRGBDataBuffer(ZividPython::ReleasableImageBGRA_SRGB &image)
    {
        return imageDataBuffer<ZividPython::ReleasableImageBGRA_SRGB, Zivid::ColorBGRA_SRGB>(image);
    }

    template<typename ImageType, typename NativeType>
    py::buffer_info imageData3ChannelBuffer(ImageType &image)
    {
        using WrapperType = uint8_t;

        constexpr py::ssize_t dim = 3;
        constexpr py::ssize_t depth = 3;
        constexpr py::ssize_t dataSize = sizeof(WrapperType);

        static_assert(dataSize * depth == sizeof(NativeType));
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::r)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::g)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::b)>);

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

    py::buffer_info imageRGBDataBuffer(ZividPython::ReleasableImageRGB &image)
    {
        return imageData3ChannelBuffer<ZividPython::ReleasableImageRGB, Zivid::ColorRGB>(image);
    }

    py::buffer_info imageRGB_SRGBDataBuffer(ZividPython::ReleasableImageRGB_SRGB &image)
    {
        return imageData3ChannelBuffer<ZividPython::ReleasableImageRGB_SRGB, Zivid::ColorRGB_SRGB>(image);
    }

    py::buffer_info imageBGRDataBuffer(ZividPython::ReleasableImageBGR &image)
    {
        return imageData3ChannelBuffer<ZividPython::ReleasableImageBGR, Zivid::ColorBGR>(image);
    }

    py::buffer_info imageBGR_SRGBDataBuffer(ZividPython::ReleasableImageBGR_SRGB &image)
    {
        return imageData3ChannelBuffer<ZividPython::ReleasableImageBGR_SRGB, Zivid::ColorBGR_SRGB>(image);
    }
} // namespace

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasableImageRGBA> pyClass)
    {
        pyClass.def_buffer(imageRGBADataBuffer)
            .def("save", &ReleasableImageRGBA::save, py::arg("file_name"))
            .def("width", &ReleasableImageRGBA::width)
            .def("height", &ReleasableImageRGBA::height)
            .def(py::init<const std::string &>(), "Load image from file");
    }

    void wrapClass(pybind11::class_<ReleasableImageBGRA> pyClass)
    {
        pyClass.def_buffer(imageBGRADataBuffer)
            .def("save", &ReleasableImageBGRA::save, py::arg("file_name"))
            .def("width", &ReleasableImageBGRA::width)
            .def("height", &ReleasableImageBGRA::height)
            .def(py::init<const std::string &>(), "Load image from file");
    }

    void wrapClass(pybind11::class_<ReleasableImageRGBA_SRGB> pyClass)
    {
        pyClass.def_buffer(imageRGBA_SRGBDataBuffer)
            .def("save", &ReleasableImageRGBA_SRGB::save, py::arg("file_name"))
            .def("width", &ReleasableImageRGBA_SRGB::width)
            .def("height", &ReleasableImageRGBA_SRGB::height)
            .def(py::init<const std::string &>(), "Load image from file");
    }

    void wrapClass(pybind11::class_<ReleasableImageBGRA_SRGB> pyClass)
    {
        pyClass.def_buffer(imageBGRA_SRGBDataBuffer)
            .def("save", &ReleasableImageBGRA_SRGB::save, py::arg("file_name"))
            .def("width", &ReleasableImageBGRA_SRGB::width)
            .def("height", &ReleasableImageBGRA_SRGB::height)
            .def(py::init<const std::string &>(), "Load image from file");
    }

    void wrapClass(pybind11::class_<ReleasableImageRGB> pyClass)
    {
        pyClass.def_buffer(imageRGBDataBuffer)
            .def("save", &ReleasableImageRGB::save, py::arg("file_name"))
            .def("width", &ReleasableImageRGB::width)
            .def("height", &ReleasableImageRGB::height)
            .def(py::init<const std::string &>(), "Load image from file");
    }

    void wrapClass(pybind11::class_<ReleasableImageRGB_SRGB> pyClass)
    {
        pyClass.def_buffer(imageRGB_SRGBDataBuffer)
            .def("save", &ReleasableImageRGB_SRGB::save, py::arg("file_name"))
            .def("width", &ReleasableImageRGB_SRGB::width)
            .def("height", &ReleasableImageRGB_SRGB::height)
            .def(py::init<const std::string &>(), "Load image from file");
    }

    void wrapClass(pybind11::class_<ReleasableImageBGR> pyClass)
    {
        pyClass.def_buffer(imageBGRDataBuffer)
            .def("save", &ReleasableImageBGR::save, py::arg("file_name"))
            .def("width", &ReleasableImageBGR::width)
            .def("height", &ReleasableImageBGR::height)
            .def(py::init<const std::string &>(), "Load image from file");
    }

    void wrapClass(pybind11::class_<ReleasableImageBGR_SRGB> pyClass)
    {
        pyClass.def_buffer(imageBGR_SRGBDataBuffer)
            .def("save", &ReleasableImageBGR_SRGB::save, py::arg("file_name"))
            .def("width", &ReleasableImageBGR_SRGB::width)
            .def("height", &ReleasableImageBGR_SRGB::height)
            .def(py::init<const std::string &>(), "Load image from file");
    }
} // namespace ZividPython
