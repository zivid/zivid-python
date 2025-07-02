#include <ZividPython/PointCloudExport.h>
#include <ZividPython/ReleasableFrame.h>
#include <ZividPython/ReleasableUnorganizedPointCloud.h>

#include <utility>

namespace py = pybind11;

namespace
{
    template<typename FileFormat>
    auto wrapFileFormat(py::class_<FileFormat> pyClass)
    {
        return pyClass.def(py::init<const std::string &>(), py::arg("file_name"))
            .def("to_string", &FileFormat::toString);
    }

    template<typename FileFormat>
    auto exportFrame(const ZividPython::ReleasableFrame &frame, const FileFormat &fileFormat)
    {
        return Zivid::Experimental::PointCloudExport::exportFrame(frame.impl(), fileFormat);
    }

    template<typename FileFormat>
    auto exportUnorganizedPointCloud(
        const ZividPython::ReleasableUnorganizedPointCloud &upc,
        const FileFormat &fileFormat)
    {
        return Zivid::Experimental::PointCloudExport::exportUnorganizedPointCloud(upc.impl(), fileFormat);
    }
} // namespace

namespace ZividPython
{
    void wrapEnum(py::enum_<Zivid::Experimental::PointCloudExport::ColorSpace> pyEnum)
    {
        pyEnum.value("srgb", Zivid::Experimental::PointCloudExport::ColorSpace::sRGB)
            .value("linear_rgb", Zivid::Experimental::PointCloudExport::ColorSpace::linearRGB);
    }

    void wrapEnum(pybind11::enum_<Zivid::Experimental::PointCloudExport::IncludeNormals> pyEnum)
    {
        pyEnum.value("no", Zivid::Experimental::PointCloudExport::IncludeNormals::no)
            .value("yes", Zivid::Experimental::PointCloudExport::IncludeNormals::yes);
    }

    void wrapClass(py::class_<Zivid::Experimental::PointCloudExport::FileFormat::ZDF> pyClass)
    {
        wrapFileFormat(std::move(pyClass));
    }

    void wrapClass(py::class_<Zivid::Experimental::PointCloudExport::FileFormat::PLY> pyClass)
    {
        using Layout = Zivid::Experimental::PointCloudExport::FileFormat::PLY::Layout;
        ZIVID_PYTHON_WRAP_ENUM_CLASS(pyClass, Layout);

        wrapFileFormat(std::move(pyClass))
            .def(
                py::init<
                    const std::string &,
                    Layout,
                    Zivid::Experimental::PointCloudExport::ColorSpace,
                    Zivid::Experimental::PointCloudExport::IncludeNormals>(),
                py::arg("file_name"),
                py::arg("layout"),
                py::arg("color_space"),
                py::arg("include_normals"));
    }

    void wrapEnum(py::enum_<Zivid::Experimental::PointCloudExport::FileFormat::PLY::Layout> pyEnum)
    {
        pyEnum.value("ordered", Zivid::Experimental::PointCloudExport::FileFormat::PLY::Layout::ordered)
            .value("unordered", Zivid::Experimental::PointCloudExport::FileFormat::PLY::Layout::unordered);
    }

    void wrapClass(py::class_<Zivid::Experimental::PointCloudExport::FileFormat::XYZ> pyClass)
    {
        wrapFileFormat(std::move(pyClass))
            .def(
                py::init<const std::string &, Zivid::Experimental::PointCloudExport::ColorSpace>(),
                py::arg("file_name"),
                py::arg("color_space"));
    }

    void wrapEnum(py::enum_<Zivid::Experimental::PointCloudExport::FileFormat::PCD::Layout> pyEnum)
    {
        pyEnum.value("organized", Zivid::Experimental::PointCloudExport::FileFormat::PCD::Layout::organized)
            .value("unorganized", Zivid::Experimental::PointCloudExport::FileFormat::PCD::Layout::unorganized);
    }

    void wrapClass(py::class_<Zivid::Experimental::PointCloudExport::FileFormat::PCD> pyClass)
    {
        using Layout = Zivid::Experimental::PointCloudExport::FileFormat::PCD::Layout;
        ZIVID_PYTHON_WRAP_ENUM_CLASS(pyClass, Layout);

        wrapFileFormat(std::move(pyClass))
            .def(
                py::init<
                    const std::string &,
                    Zivid::Experimental::PointCloudExport::ColorSpace,
                    Zivid::Experimental::PointCloudExport::IncludeNormals,
                    Layout>(),
                py::arg("file_name"),
                py::arg("color_space"),
                py::arg("include_normals"),
                py::arg("layout"));
    }

    namespace PointCloudExport
    {
        namespace FileFormat
        {
            void wrapAsSubmodule(py::module &dest)
            {
                using ZDF = Zivid::Experimental::PointCloudExport::FileFormat::ZDF;
                ZIVID_PYTHON_WRAP_CLASS(dest, ZDF);

                using PLY = Zivid::Experimental::PointCloudExport::FileFormat::PLY;
                ZIVID_PYTHON_WRAP_CLASS(dest, PLY);

                using XYZ = Zivid::Experimental::PointCloudExport::FileFormat::XYZ;
                ZIVID_PYTHON_WRAP_CLASS(dest, XYZ);

                using PCD = Zivid::Experimental::PointCloudExport::FileFormat::PCD;
                ZIVID_PYTHON_WRAP_CLASS(dest, PCD);
            }
        } // namespace FileFormat

        void wrapAsSubmodule(py::module &dest)
        {
            using ColorSpace = Zivid::Experimental::PointCloudExport::ColorSpace;
            using IncludeNormals = Zivid::Experimental::PointCloudExport::IncludeNormals;
            ZIVID_PYTHON_WRAP_ENUM_CLASS(dest, ColorSpace);
            ZIVID_PYTHON_WRAP_ENUM_CLASS(dest, IncludeNormals);

            wrapNamespaceAsSubmodule(dest, FileFormat::wrapAsSubmodule, "FileFormat");

            dest.def(
                    "export_frame",
                    py::overload_cast<
                        const ReleasableFrame &,
                        const Zivid::Experimental::PointCloudExport::FileFormat::ZDF &>(
                        &exportFrame<Zivid::Experimental::PointCloudExport::FileFormat::ZDF>))
                .def(
                    "export_frame",
                    py::overload_cast<
                        const ReleasableFrame &,
                        const Zivid::Experimental::PointCloudExport::FileFormat::PLY &>(
                        &exportFrame<Zivid::Experimental::PointCloudExport::FileFormat::PLY>))
                .def(
                    "export_frame",
                    py::overload_cast<
                        const ReleasableFrame &,
                        const Zivid::Experimental::PointCloudExport::FileFormat::XYZ &>(
                        &exportFrame<Zivid::Experimental::PointCloudExport::FileFormat::XYZ>))
                .def(
                    "export_frame",
                    py::overload_cast<
                        const ReleasableFrame &,
                        const Zivid::Experimental::PointCloudExport::FileFormat::PCD &>(
                        &exportFrame<Zivid::Experimental::PointCloudExport::FileFormat::PCD>))
                .def(
                    "export_unorganized_point_cloud",
                    py::overload_cast<
                        const ReleasableUnorganizedPointCloud &,
                        const Zivid::Experimental::PointCloudExport::FileFormat::PLY &>(
                        &exportUnorganizedPointCloud<Zivid::Experimental::PointCloudExport::FileFormat::PLY>))
                .def(
                    "export_unorganized_point_cloud",
                    py::overload_cast<
                        const ReleasableUnorganizedPointCloud &,
                        const Zivid::Experimental::PointCloudExport::FileFormat::XYZ &>(
                        &exportUnorganizedPointCloud<Zivid::Experimental::PointCloudExport::FileFormat::XYZ>))
                .def(
                    "export_unorganized_point_cloud",
                    py::overload_cast<
                        const ReleasableUnorganizedPointCloud &,
                        const Zivid::Experimental::PointCloudExport::FileFormat::PCD &>(
                        &exportUnorganizedPointCloud<Zivid::Experimental::PointCloudExport::FileFormat::PCD>));
        }
    } // namespace PointCloudExport
} // namespace ZividPython
