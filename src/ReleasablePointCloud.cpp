#include <ZividPython/ReleasablePointCloud.h>

#include <Zivid/PointCloud.h>
#include <ZividPython/Matrix.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasablePointCloud> pyClass)
    {
        pyClass.def(py::init<>())
            .def("width", &ReleasablePointCloud::width)
            .def("height", &ReleasablePointCloud::height)
            .def("transform",
                 [](ReleasablePointCloud &pointCloud, const Eigen::Matrix<float, 4, 4, Eigen::RowMajor> &matrix) {
                     pointCloud.transform(Conversion::toCpp(matrix));
                 });
    }

} // namespace ZividPython
