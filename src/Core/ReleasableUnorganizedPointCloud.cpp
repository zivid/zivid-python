#include <ZividPython/Matrix.h>
#include <ZividPython/ReleasableUnorganizedPointCloud.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace
{
    std::array<uint8_t, 4> toCppColor(py::object pyObject)
    {
        std::array<uint8_t, 4> color = { 0, 0, 0, 0 };

        if(py::isinstance<py::array>(pyObject))
        {
            py::array_t<uint8_t, py::array::c_style> pyArray(pyObject);

            if(pyArray.ndim() == 1 && pyArray.shape(0) == 4)
            {
                auto r = pyArray.unchecked<1>();
                for(size_t i = 0; i < static_cast<size_t>(r.shape(0)); i++)
                {
                    color[i] = r(i);
                }
            }
            else if(pyArray.ndim() == 2 && pyArray.shape(0) == 1 && pyArray.shape(1) == 4)
            {
                auto r = pyArray.unchecked<2>();
                for(size_t i = 0; i < static_cast<size_t>(r.shape(1)); i++)
                {
                    color[i] = r(0, i);
                }
            }
            else
            {
                throw std::runtime_error("Expected shape (4,) or (1,4) with dtype=uint8");
            }
        }
        else if(py::isinstance<py::list>(pyObject))
        {
            py::list pyList = py::cast<py::list>(pyObject);
            if(pyList.size() != 4)
            {
                throw std::runtime_error("Expected a list of length 4");
            }
            for(size_t i = 0; i < pyList.size(); i++)
            {
                int v = pyList[i].cast<int>();
                if(v < 0 || v > 255)
                {
                    throw std::runtime_error("List values must be between 0 and 255");
                }
                color[i] = static_cast<uint8_t>(v);
            }
        }
        else
        {
            throw std::runtime_error(
                "Expected a list of 4 integers or a NumPy array of shape (4,) or (1,4) with dtype=uint8");
        }

        return color;
    }
} // namespace

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
            .def(
                "paint_uniform_color",
                [](ReleasableUnorganizedPointCloud &pointCloud, const py::object &pyColor) {
                    const auto color = toCppColor(pyColor);
                    pointCloud.paintUniformColor(Zivid::ColorRGBA{ color[0], color[1], color[2], color[3] });
                })
            .def(
                "painted_uniform_color",
                [](ReleasableUnorganizedPointCloud &pointCloud, const py::object &pyColor) {
                    const auto color = toCppColor(pyColor);
                    return pointCloud.paintedUniformColor(Zivid::ColorRGBA{ color[0], color[1], color[2], color[3] });
                })
            .def("clone", &ReleasableUnorganizedPointCloud::clone);
    }
} // namespace ZividPython
