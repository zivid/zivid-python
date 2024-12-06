#pragma once

#include <Zivid/Experimental/PointCloudExport.h>
#include <ZividPython/Wrappers.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapEnum(pybind11::enum_<Zivid::Experimental::PointCloudExport::ColorSpace> pyEnum);

    void wrapClass(pybind11::class_<Zivid::Experimental::PointCloudExport::FileFormat::ZDF> pyClass);

    void wrapClass(pybind11::class_<Zivid::Experimental::PointCloudExport::FileFormat::PLY> pyClass);
    void wrapEnum(pybind11::enum_<Zivid::Experimental::PointCloudExport::FileFormat::PLY::Layout> pyEnum);

    void wrapClass(pybind11::class_<Zivid::Experimental::PointCloudExport::FileFormat::XYZ> pyClass);

    void wrapClass(pybind11::class_<Zivid::Experimental::PointCloudExport::FileFormat::PCD> pyClass);

    namespace PointCloudExport
    {
        namespace FileFormat
        {
            void wrapAsSubmodule(pybind11::module &dest);
        } // namespace FileFormat

        void wrapAsSubmodule(pybind11::module &dest);
    } // namespace PointCloudExport
} // namespace ZividPython
