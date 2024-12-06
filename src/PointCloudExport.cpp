#include <ZividPython/PointCloudExport.h>

#include <ZividPython/ReleasableFrame.h>

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
} // namespace

namespace ZividPython
{
    void wrapEnum(py::enum_<Zivid::Experimental::PointCloudExport::ColorSpace> pyEnum)
    {
        pyEnum.value("srgb", Zivid::Experimental::PointCloudExport::ColorSpace::sRGB)
            .value("linear_rgb", Zivid::Experimental::PointCloudExport::ColorSpace::linearRGB)
            .export_values();
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
            .def(py::init<const std::string &, Layout, Zivid::Experimental::PointCloudExport::ColorSpace>(),
                 py::arg("file_name"),
                 py::arg("layout"),
                 py::arg("color_space"));
    }

    void wrapEnum(py::enum_<Zivid::Experimental::PointCloudExport::FileFormat::PLY::Layout> pyEnum)
    {
        pyEnum.value("ordered", Zivid::Experimental::PointCloudExport::FileFormat::PLY::Layout::ordered)
            .value("unordered", Zivid::Experimental::PointCloudExport::FileFormat::PLY::Layout::unordered)
            .export_values();
    }

    void wrapClass(py::class_<Zivid::Experimental::PointCloudExport::FileFormat::XYZ> pyClass)
    {
        wrapFileFormat(std::move(pyClass))
            .def(py::init<const std::string &, Zivid::Experimental::PointCloudExport::ColorSpace>(),
                 py::arg("file_name"),
                 py::arg("color_space"));
    }

    void wrapClass(py::class_<Zivid::Experimental::PointCloudExport::FileFormat::PCD> pyClass)
    {
        wrapFileFormat(std::move(pyClass))
            .def(py::init<const std::string &, Zivid::Experimental::PointCloudExport::ColorSpace>(),
                 py::arg("file_name"),
                 py::arg("color_space"));
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
            ZIVID_PYTHON_WRAP_ENUM_CLASS(dest, ColorSpace);

            wrapNamespaceAsSubmodule(dest, FileFormat::wrapAsSubmodule, "FileFormat");

            dest.def("export_frame",
                     py::overload_cast<const ReleasableFrame &,
                                       const Zivid::Experimental::PointCloudExport::FileFormat::ZDF &>(
                         &exportFrame<Zivid::Experimental::PointCloudExport::FileFormat::ZDF>))
                .def("export_frame",
                     py::overload_cast<const ReleasableFrame &,
                                       const Zivid::Experimental::PointCloudExport::FileFormat::PLY &>(
                         &exportFrame<Zivid::Experimental::PointCloudExport::FileFormat::PLY>))
                .def("export_frame",
                     py::overload_cast<const ReleasableFrame &,
                                       const Zivid::Experimental::PointCloudExport::FileFormat::XYZ &>(
                         &exportFrame<Zivid::Experimental::PointCloudExport::FileFormat::XYZ>))
                .def("export_frame",
                     py::overload_cast<const ReleasableFrame &,
                                       const Zivid::Experimental::PointCloudExport::FileFormat::PCD &>(
                         &exportFrame<Zivid::Experimental::PointCloudExport::FileFormat::PCD>));
        }
    } // namespace PointCloudExport
} // namespace ZividPython