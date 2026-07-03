#include <ZividPython/ReleasableFrame2D.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasableFrame2D> pyClass)
    {
        pyClass.def(py::init())
            .def(py::init<const std::string &>(), py::arg("file_name"))
            .def("save", &ReleasableFrame2D::save, py::arg("file_name"))
            .def("load", &ReleasableFrame2D::load, py::arg("file_name"))
            .def_property_readonly("settings", &ReleasableFrame2D::settings)
            .def_property_readonly("state", &ReleasableFrame2D::state)
            .def_property_readonly("info", &ReleasableFrame2D::info)
            .def_property_readonly("camera_info", &ReleasableFrame2D::cameraInfo)
            .def("image_rgba", &ReleasableFrame2D::imageRGBA)
            .def("image_bgra", &ReleasableFrame2D::imageBGRA)
            .def("image_rgba_srgb", &ReleasableFrame2D::imageRGBA_SRGB)
            .def("image_bgra_srgb", &ReleasableFrame2D::imageBGRA_SRGB)
            .def("image_rgb", &ReleasableFrame2D::imageRGB)
            .def("image_rgb_srgb", &ReleasableFrame2D::imageRGB_SRGB)
            .def("image_bgr", &ReleasableFrame2D::imageBGR)
            .def("image_bgr_srgb", &ReleasableFrame2D::imageBGR_SRGB)
            .def("clone", &ReleasableFrame2D::clone)
            .def("image_device_array_rgba", &ReleasableFrame2D::imageDeviceArrayRGBA, py::arg("stream_or_queue"))
            .def("image_device_array_bgra", &ReleasableFrame2D::imageDeviceArrayBGRA, py::arg("stream_or_queue"))
            .def(
                "image_device_array_rgba_srgb",
                &ReleasableFrame2D::imageDeviceArrayRGBA_SRGB,
                py::arg("stream_or_queue"))
            .def(
                "image_device_array_bgra_srgb",
                &ReleasableFrame2D::imageDeviceArrayBGRA_SRGB,
                py::arg("stream_or_queue"))
            .def("image_device_array_rgba_float", &ReleasableFrame2D::imageDeviceArrayRGBAf, py::arg("stream_or_queue"))
            .def("image_device_array_rgb", &ReleasableFrame2D::imageDeviceArrayRGB, py::arg("stream_or_queue"))
            .def(
                "image_device_array_rgb_srgb", &ReleasableFrame2D::imageDeviceArrayRGB_SRGB, py::arg("stream_or_queue"))
            .def("image_device_array_bgr", &ReleasableFrame2D::imageDeviceArrayBGR, py::arg("stream_or_queue"))
            .def(
                "image_device_array_bgr_srgb", &ReleasableFrame2D::imageDeviceArrayBGR_SRGB, py::arg("stream_or_queue"))
            .def(
                "image_device_array_rgba_fill",
                &ReleasableFrame2D::imageDeviceArrayRGBAFill,
                py::arg("stream_or_queue"),
                py::arg("destination_buffer"))
            .def(
                "image_device_array_bgra_fill",
                &ReleasableFrame2D::imageDeviceArrayBGRAFill,
                py::arg("stream_or_queue"),
                py::arg("destination_buffer"))
            .def(
                "image_device_array_rgba_srgb_fill",
                &ReleasableFrame2D::imageDeviceArrayRGBA_SRGBFill,
                py::arg("stream_or_queue"),
                py::arg("destination_buffer"))
            .def(
                "image_device_array_bgra_srgb_fill",
                &ReleasableFrame2D::imageDeviceArrayBGRA_SRGBFill,
                py::arg("stream_or_queue"),
                py::arg("destination_buffer"))
            .def(
                "image_device_array_rgba_float_fill",
                &ReleasableFrame2D::imageDeviceArrayRGBAfFill,
                py::arg("stream_or_queue"),
                py::arg("destination_buffer"))
            .def(
                "image_device_array_rgb_fill",
                &ReleasableFrame2D::imageDeviceArrayRGBFill,
                py::arg("stream_or_queue"),
                py::arg("destination_buffer"))
            .def(
                "image_device_array_rgb_srgb_fill",
                &ReleasableFrame2D::imageDeviceArrayRGB_SRGBFill,
                py::arg("stream_or_queue"),
                py::arg("destination_buffer"))
            .def(
                "image_device_array_bgr_fill",
                &ReleasableFrame2D::imageDeviceArrayBGRFill,
                py::arg("stream_or_queue"),
                py::arg("destination_buffer"))
            .def(
                "image_device_array_bgr_srgb_fill",
                &ReleasableFrame2D::imageDeviceArrayBGR_SRGBFill,
                py::arg("stream_or_queue"),
                py::arg("destination_buffer"));
    }
} // namespace ZividPython
