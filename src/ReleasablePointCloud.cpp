#include <ZividPython/ReleasablePointCloud.h>

#include <Zivid/PointCloud.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

#define IS_SAME_MEMBER(T1, T2, m)                                                                                      \
    (std::is_same_v<decltype(T1::m), decltype(T2::m)> && offsetof(T1, m) == offsetof(T2, m))

namespace
{
#pragma pack(push)
    struct DataType
    {
        float x, y, z;
        uint8_t b, g, r, a;
    };
#pragma pack(pop)

    py::buffer_info makeBufferInfo(ZividPython::ReleasablePointCloud &pointCloud)
    {
        auto dataArray =
            pointCloud.impl()
                .copyData<Zivid::PointXYZColorRGBA>(); // TODO: double copy, both here and when returning buffer_info
        // try to use latest pybind11 to get around const data pointers https://github.com/pybind/pybind11/issues/1993
        auto *data = const_cast<Zivid::PointXYZColorRGBA *>(dataArray.data());

        using NativeDataType = std::remove_pointer_t<decltype(data)>;

        static_assert(sizeof(NativeDataType) == sizeof(DataType), "Unexpected point cloud format");
        // TODO: reintroduce static_asserts
        //static_assert(IS_SAME_MEMBER(NativeDataType, DataType, x), "Unexpected point cloud format");
        //static_assert(IS_SAME_MEMBER(NativeDataType, DataType, y), "Unexpected point cloud format");
        //static_assert(IS_SAME_MEMBER(NativeDataType, DataType, z), "Unexpected point cloud format");
        //static_assert(IS_SAME_MEMBER(NativeDataType, DataType, contrast), "Unexpected point cloud format");
        //static_assert(offsetof(NativeDataType, rgba) == offsetof(DataType, b), "Unexpected point cloud format");
        //static_assert(sizeof(NativeDataType::rgba)
        //                  == sizeof(DataType::r) + sizeof(DataType::g) + sizeof(DataType::b) + sizeof(DataType::a),
        //              "Unexpected point cloud format");

        return py::buffer_info{ data,
                                sizeof(DataType),
                                py::format_descriptor<DataType>::format(),
                                2,
                                { pointCloud.height(), pointCloud.width() },
                                { sizeof(DataType) * pointCloud.width(), sizeof(DataType) } };
    }
} // namespace

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasablePointCloud> pyClass)
    {
        PYBIND11_NUMPY_DTYPE(DataType, x, y, z, b, g, r, a);

        pyClass.def(py::init<>())
            .def_buffer(makeBufferInfo)
            .def("width", &ReleasablePointCloud::width)
            .def("height", &ReleasablePointCloud::height);
    }
} // namespace ZividPython
