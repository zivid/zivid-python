#include <ZividPython/ReleasableArray2D.h>

#include <Zivid/PointCloud.h>
#include <ZividPython/ReleasablePointCloud.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace
{
#pragma pack(push)
    struct PointXYZColorRGBA
    {
        float x, y, z;
        uint8_t r, g, b, a;
    };
#pragma pack(pop)

    template<typename NativeType, typename WrapperType, size_t depth>
    py::buffer_info pointCloudDataBuffer(ZividPython::ReleasableArray2D<NativeType> &arrayWrapper)
    {
        static_assert(depth > 0, "Depth of array must be one or higher");
        static_assert(sizeof(WrapperType) * depth == sizeof(NativeType), "Unexpected data format");

        constexpr py::ssize_t dim = (depth == 1) ? 2 : 3;
        const auto shape = { arrayWrapper.impl().height(), arrayWrapper.impl().width(), depth };
        const auto strides = { sizeof(WrapperType) * arrayWrapper.impl().width() * depth,
                               sizeof(WrapperType) * depth,
                               sizeof(WrapperType) };
        auto *dataPtr = static_cast<void *>(const_cast<NativeType *>(arrayWrapper.impl().data()));
        return py::buffer_info(dataPtr,
                               sizeof(WrapperType),
                               py::format_descriptor<WrapperType>::format(),
                               dim,
                               std::vector<py::ssize_t>(shape.begin(), shape.begin() + dim),
                               std::vector<py::ssize_t>(strides.begin(), strides.begin() + dim));
    }

    template<typename NativeType>
    std::unique_ptr<ZividPython::ReleasableArray2D<NativeType>> pointCloudDataCopier(
        ZividPython::ReleasablePointCloud &pointCloud)
    {
        return std::make_unique<ZividPython::ReleasableArray2D<NativeType>>(pointCloud.impl().copyData<NativeType>());
    }

} // namespace

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::SNR>> pyClass)
    {
        using WrapperType = float;
        using NativeType = Zivid::SNR;
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::value)>);

        pyClass.def(py::init<>(&pointCloudDataCopier<NativeType>))
            .def_buffer(pointCloudDataBuffer<NativeType, WrapperType, 1>);
    }

    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::ColorRGBA>> pyClass)
    {
        using WrapperType = uint8_t;
        using NativeType = Zivid::ColorRGBA;
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::r)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::g)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::b)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::a)>);

        pyClass.def(py::init<>(&pointCloudDataCopier<NativeType>))
            .def_buffer(pointCloudDataBuffer<NativeType, WrapperType, 4>);
    }

    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::PointXYZ>> pyClass)
    {
        using WrapperType = float;
        using NativeType = Zivid::PointXYZ;
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::x)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::y)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::z)>);

        pyClass.def(py::init<>(&pointCloudDataCopier<NativeType>))
            .def_buffer(pointCloudDataBuffer<NativeType, WrapperType, 3>);
    }

    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::PointXYZW>> pyClass)
    {
        using WrapperType = float;
        using NativeType = Zivid::PointXYZW;
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::x)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::y)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::z)>);
        static_assert(std::is_same_v<WrapperType, decltype(NativeType::w)>);

        pyClass.def(py::init<>(&pointCloudDataCopier<NativeType>))
            .def_buffer(pointCloudDataBuffer<NativeType, WrapperType, 4>);
    }

    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::PointZ>> pyClass)
    {
        using WrapperType = float;
        using NativeType = Zivid::PointZ;

        static_assert(std::is_same_v<WrapperType, decltype(NativeType::z)>);

        pyClass.def(py::init<>(&pointCloudDataCopier<NativeType>))
            .def_buffer(pointCloudDataBuffer<NativeType, WrapperType, 1>);
    }

    void wrapClass(pybind11::class_<ReleasableArray2D<Zivid::PointXYZColorRGBA>> pyClass)
    {
        using WrapperType = PointXYZColorRGBA;
        using NativeType = Zivid::PointXYZColorRGBA;
        using NativePoint = decltype(NativeType::point);
        using NativeColor = decltype(NativeType::color);
        static_assert(std::is_same_v<decltype(WrapperType::x), decltype(NativePoint::x)>);
        static_assert(std::is_same_v<decltype(WrapperType::y), decltype(NativePoint::y)>);
        static_assert(std::is_same_v<decltype(WrapperType::z), decltype(NativePoint::z)>);
        static_assert(std::is_same_v<decltype(WrapperType::r), decltype(NativeColor::r)>);
        static_assert(std::is_same_v<decltype(WrapperType::g), decltype(NativeColor::g)>);
        static_assert(std::is_same_v<decltype(WrapperType::b), decltype(NativeColor::b)>);
        static_assert(std::is_same_v<decltype(WrapperType::a), decltype(NativeColor::a)>);
        // Additional layout checks for composite types
        static_assert(offsetof(NativeType, point) == offsetof(WrapperType, x));
        static_assert(offsetof(NativeType, color) == offsetof(WrapperType, r));
        static_assert(offsetof(NativePoint, x) == offsetof(WrapperType, x));
        static_assert(offsetof(NativePoint, y) == offsetof(WrapperType, y));
        static_assert(offsetof(NativePoint, z) == offsetof(WrapperType, z));
        static_assert(offsetof(NativeColor, r) == 0);
        static_assert(offsetof(NativeColor, g) == offsetof(WrapperType, g) - offsetof(WrapperType, r));
        static_assert(offsetof(NativeColor, b) == offsetof(WrapperType, b) - offsetof(WrapperType, r));
        static_assert(offsetof(NativeColor, a) == offsetof(WrapperType, a) - offsetof(WrapperType, r));

        PYBIND11_NUMPY_DTYPE(WrapperType, x, y, z, r, g, b, a);
        pyClass.def(py::init<>(&pointCloudDataCopier<NativeType>))
            .def_buffer(pointCloudDataBuffer<NativeType, WrapperType, 1>);
    }
} // namespace ZividPython
