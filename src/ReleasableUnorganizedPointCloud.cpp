#include <ZividPython/Matrix.h>
#include <ZividPython/ReleasableUnorganizedPointCloud.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasableUnorganizedPointCloud> pyClass)
    {
        pyClass.def(py::init())
            .def("size", &ReleasableUnorganizedPointCloud::size)
            .def(
                "extended",
                [](ReleasableUnorganizedPointCloud &pc, const ReleasableUnorganizedPointCloud &other) {
                    return pc.extended(other.impl());
                },
                py::arg("other"))
            .def(
                "extend",
                [](ReleasableUnorganizedPointCloud &pc, const ReleasableUnorganizedPointCloud &other) {
                    pc.extend(other.impl());
                },
                py::arg("other"))
            .def(
                "voxel_downsampled",
                &ReleasableUnorganizedPointCloud::voxelDownsampled,
                py::arg("voxel_size"),
                py::arg("min_points_per_voxel"))
            .def(
                "transform",
                [](ReleasableUnorganizedPointCloud &pointCloud,
                   const Eigen::Matrix<float, 4, 4, Eigen::RowMajor> &matrix) {
                    pointCloud.transform(Conversion::toCpp(matrix));
                })
            .def(
                "transformed",
                [](ReleasableUnorganizedPointCloud &pointCloud,
                   const Eigen::Matrix<float, 4, 4, Eigen::RowMajor> &matrix) {
                    return pointCloud.transformed(Conversion::toCpp(matrix));
                })
            .def("center", [](ReleasableUnorganizedPointCloud &pointCloud) { pointCloud.center(); })
            .def(
                "centroid",
                [](const ReleasableUnorganizedPointCloud &pointCloud) -> std::optional<Eigen::Vector3f> {
                    auto optionalPointXYZ = pointCloud.impl().centroid();
                    if(optionalPointXYZ.has_value())
                    {
                        return Conversion::toPyVector(optionalPointXYZ.value());
                    }
                    return std::nullopt;
                })
            .def("clone", &ReleasableUnorganizedPointCloud::clone);
    }
} // namespace ZividPython
