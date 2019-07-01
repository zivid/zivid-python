#include <ZividPython/PointCloud.h>

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
        float contrast;
        uint8_t b, g, r, a;
    };
#pragma pack(pop)

    py::buffer_info makeBufferInfo(Zivid::PointCloud &pointCloud)
    {
        const auto data = pointCloud.dataPtr();

        using NativeDataType = std::remove_pointer_t<decltype(data)>;

        static_assert(sizeof(NativeDataType) == sizeof(DataType), "Unexpected point cloud format");
        static_assert(IS_SAME_MEMBER(NativeDataType, DataType, x), "Unexpected point cloud format");
        static_assert(IS_SAME_MEMBER(NativeDataType, DataType, y), "Unexpected point cloud format");
        static_assert(IS_SAME_MEMBER(NativeDataType, DataType, z), "Unexpected point cloud format");
        static_assert(IS_SAME_MEMBER(NativeDataType, DataType, contrast), "Unexpected point cloud format");
        static_assert(offsetof(NativeDataType, rgba) == offsetof(DataType, b), "Unexpected point cloud format");
        static_assert(sizeof(NativeDataType::rgba)
                          == sizeof(DataType::r) + sizeof(DataType::g) + sizeof(DataType::b) + sizeof(DataType::a),
                      "Unexpected point cloud format");

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
    MetaData wrapClass(pybind11::class_<Zivid::PointCloud> pyClass)
    {
        PYBIND11_NUMPY_DTYPE(DataType, x, y, z, contrast, b, g, r, a);

        pyClass.def(py::init<>()).def_buffer(makeBufferInfo);

        return { R"(A point cloud with x,y,z, contrast and color data laid out on a 2D grid

This class implements the `Buffer Protocol <docs.python.org/c-api/buffer.html>`_ and can
be used in combinatins with modules supporting that protocol.

NumPy is supported by using the `numpy.array` class like this:

.. code-block:: python

   np_point_cloud = numpy.array(point_cloud))" };
    }
} // namespace ZividPython
