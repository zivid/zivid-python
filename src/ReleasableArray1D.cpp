#include <ZividPython/ReleasableArray1D.h>
#include <ZividPython/ReleasableUnorganizedPointCloud.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace
{

    template<typename NativeType, typename WrapperType, size_t channels>
    py::buffer_info pointCloudDataBuffer(ZividPython::ReleasableArray1D<NativeType> &arrayWrapper)
    {
        static_assert(channels > 0, "Channels must be one or higher");
        static_assert(sizeof(WrapperType) * channels == sizeof(NativeType), "Unexpected data format");

        constexpr py::ssize_t dim = (channels == 1) ? 1 : 2;
        const auto shape = { arrayWrapper.impl().size(), channels };
        const auto strides = { sizeof(WrapperType) * channels, sizeof(WrapperType) };
        auto *dataPtr = static_cast<void *>(const_cast<NativeType *>(arrayWrapper.impl().data()));
        return py::buffer_info(
            dataPtr,
            sizeof(WrapperType),
            py::format_descriptor<WrapperType>::format(),
            dim,
            std::vector<py::ssize_t>(shape.begin(), shape.begin() + dim),
            std::vector<py::ssize_t>(strides.begin(), strides.begin() + dim));
    }

    template<typename NativeType>
    std::unique_ptr<ZividPython::ReleasableArray1D<NativeType>> pointCloudDataCopier(
        ZividPython::ReleasableUnorganizedPointCloud &pointCloud)
    {
        return std::make_unique<ZividPython::ReleasableArray1D<NativeType>>(pointCloud.impl().copyData<NativeType>());
    }

    template<typename NativeType>
    void wrapColorClass(pybind11::class_<ZividPython::ReleasableArray1D<NativeType>> pyClass)
    {
        using WrapperType = uint8_t;
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::r)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::g)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::b)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::a)>);

        pyClass.def(py::init<>(&pointCloudDataCopier<NativeType>))
            .def_buffer(pointCloudDataBuffer<NativeType, WrapperType, 4>);
    }
} // namespace

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasableArray1D<Zivid::PointXYZ>> pyClass)
    {
        using WrapperType = float;
        using NativeType = Zivid::PointXYZ;
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::x)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::y)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::z)>);

        pyClass.def(py::init<>(&pointCloudDataCopier<NativeType>))
            .def_buffer(pointCloudDataBuffer<NativeType, WrapperType, 3>);
    }

    void wrapClass(pybind11::class_<ReleasableArray1D<Zivid::ColorRGBA>> pyClass)
    {
        wrapColorClass<Zivid::ColorRGBA>(pyClass);
    }

    void wrapClass(pybind11::class_<ReleasableArray1D<Zivid::ColorBGRA>> pyClass)
    {
        wrapColorClass<Zivid::ColorBGRA>(pyClass);
    }

    void wrapClass(pybind11::class_<ReleasableArray1D<Zivid::ColorRGBA_SRGB>> pyClass)
    {
        wrapColorClass<Zivid::ColorRGBA_SRGB>(pyClass);
    }

    void wrapClass(pybind11::class_<ReleasableArray1D<Zivid::ColorBGRA_SRGB>> pyClass)
    {
        wrapColorClass<Zivid::ColorBGRA_SRGB>(pyClass);
    }

    void wrapClass(pybind11::class_<ReleasableArray1D<Zivid::SNR>> pyClass)
    {
        using WrapperType = float;
        using NativeType = Zivid::SNR;
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::value)>);

        pyClass.def(py::init<>(&pointCloudDataCopier<NativeType>))
            .def_buffer(pointCloudDataBuffer<NativeType, WrapperType, 1>);
    }
} // namespace ZividPython
