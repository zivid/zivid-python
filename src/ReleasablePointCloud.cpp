#include <ZividPython/Matrix.h>
#include <ZividPython/ReleasablePointCloud.h>

#include <Zivid/PointCloud.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasablePointCloud> pyClass)
    {
        pyClass.def(py::init<>())
            .def("width", &ReleasablePointCloud::width)
            .def("height", &ReleasablePointCloud::height)
            .def(
                "transform",
                [](ReleasablePointCloud &pointCloud, const Eigen::Matrix<float, 4, 4, Eigen::RowMajor> &matrix) {
                    pointCloud.transform(Conversion::toCpp(matrix));
                })
            .def(
                "transformed",
                [](ReleasablePointCloud &pointCloud, const Eigen::Matrix<float, 4, 4, Eigen::RowMajor> &matrix) {
                    return pointCloud.transformed(Conversion::toCpp(matrix));
                })
            .def("transformation_matrix", &ReleasablePointCloud::transformationMatrix)
            .def(
                "downsample",
                [](ReleasablePointCloud &pointCloud, Zivid::PointCloud::Downsampling downsampling) {
                    pointCloud.downsample(downsampling);
                })
            .def(
                "downsampled",
                [](ReleasablePointCloud &pointCloud, Zivid::PointCloud::Downsampling downsampling) {
                    return pointCloud.downsampled(downsampling);
                })
            .def("copy_image_rgba", &ReleasablePointCloud::copyImageRGBA)
            .def("copy_image_bgra", &ReleasablePointCloud::copyImageBGRA)
            .def("copy_image_rgba_srgb", &ReleasablePointCloud::copyImageRGBA_SRGB)
            .def("copy_image_bgra_srgb", &ReleasablePointCloud::copyImageBGRA_SRGB)
            .def("to_unorganized_point_cloud", &ReleasablePointCloud::toUnorganizedPointCloud)
            .def("clone", &ReleasablePointCloud::clone);

        py::enum_<Zivid::PointCloud::Downsampling>{ pyClass, "Downsampling" }
            .value("by2x2", Zivid::PointCloud::Downsampling::by2x2)
            .value("by3x3", Zivid::PointCloud::Downsampling::by3x3)
            .value("by4x4", Zivid::PointCloud::Downsampling::by4x4)
            .export_values();
    }

} // namespace ZividPython
