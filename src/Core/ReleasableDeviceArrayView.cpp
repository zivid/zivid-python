#include <ZividPython/ReleasableDeviceArrayView.h>

#include <pybind11/pybind11.h>

#include <cstdint>

namespace ZividPython
{
    namespace
    {
        // Helper template to add common DeviceArrayView methods to a pybind11 class
        template<typename ReleasableType>
        void addCommonDeviceArrayViewMethods(pybind11::class_<ReleasableType> &pyClass)
        {
            pyClass.def_property_readonly("shape", &ReleasableType::shape)
                .def_property_readonly("strides", &ReleasableType::strides)
                .def_property_readonly("strides_in_bytes", &ReleasableType::stridesInBytes)
                .def_property_readonly("size_bytes", &ReleasableType::sizeInBytes)
                .def_property_readonly("backend", &ReleasableType::backend)
                .def_property_readonly("is_valid", &ReleasableType::isValid)
                .def_property_readonly("is_empty", &ReleasableType::isEmpty)
                .def("device_pointer", [](const ReleasableType &self) {
                    return reinterpret_cast<std::uintptr_t>(self.devicePointer());
                });
        }
    } // namespace

    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewRGBA> pyClass)
    {
        addCommonDeviceArrayViewMethods(pyClass);
        pyClass.def("to_array_2d", &ReleasableDeviceArrayViewRGBA::toArray2D, pybind11::arg("stream_or_queue"));
        pyClass.def("to_image", &ReleasableDeviceArrayViewRGBA::toImage, pybind11::arg("stream_or_queue"));
    }

    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewBGRA> pyClass)
    {
        addCommonDeviceArrayViewMethods(pyClass);
        pyClass.def("to_array_2d", &ReleasableDeviceArrayViewBGRA::toArray2D, pybind11::arg("stream_or_queue"));
        pyClass.def("to_image", &ReleasableDeviceArrayViewBGRA::toImage, pybind11::arg("stream_or_queue"));
    }

    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewRGBA_SRGB> pyClass)
    {
        addCommonDeviceArrayViewMethods(pyClass);
        pyClass.def("to_array_2d", &ReleasableDeviceArrayViewRGBA_SRGB::toArray2D, pybind11::arg("stream_or_queue"));
        pyClass.def("to_image", &ReleasableDeviceArrayViewRGBA_SRGB::toImage, pybind11::arg("stream_or_queue"));
    }

    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewBGRA_SRGB> pyClass)
    {
        addCommonDeviceArrayViewMethods(pyClass);
        pyClass.def("to_array_2d", &ReleasableDeviceArrayViewBGRA_SRGB::toArray2D, pybind11::arg("stream_or_queue"));
        pyClass.def("to_image", &ReleasableDeviceArrayViewBGRA_SRGB::toImage, pybind11::arg("stream_or_queue"));
    }

    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewRGBAf> pyClass)
    {
        addCommonDeviceArrayViewMethods(pyClass);
        pyClass.def("to_array_2d", &ReleasableDeviceArrayViewRGBAf::toArray2D, pybind11::arg("stream_or_queue"));
        // Note: Float format buffer does not have to_image() - use device_pointer() for GPU interop
    }

    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewRGB> pyClass)
    {
        addCommonDeviceArrayViewMethods(pyClass);
        pyClass.def("to_array_2d", &ReleasableDeviceArrayViewRGB::toArray2D, pybind11::arg("stream_or_queue"));
        pyClass.def("to_image", &ReleasableDeviceArrayViewRGB::toImage, pybind11::arg("stream_or_queue"));
    }

    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewRGB_SRGB> pyClass)
    {
        addCommonDeviceArrayViewMethods(pyClass);
        pyClass.def("to_array_2d", &ReleasableDeviceArrayViewRGB_SRGB::toArray2D, pybind11::arg("stream_or_queue"));
        pyClass.def("to_image", &ReleasableDeviceArrayViewRGB_SRGB::toImage, pybind11::arg("stream_or_queue"));
    }

    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewBGR> pyClass)
    {
        addCommonDeviceArrayViewMethods(pyClass);
        pyClass.def("to_array_2d", &ReleasableDeviceArrayViewBGR::toArray2D, pybind11::arg("stream_or_queue"));
        pyClass.def("to_image", &ReleasableDeviceArrayViewBGR::toImage, pybind11::arg("stream_or_queue"));
    }

    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewBGR_SRGB> pyClass)
    {
        addCommonDeviceArrayViewMethods(pyClass);
        pyClass.def("to_array_2d", &ReleasableDeviceArrayViewBGR_SRGB::toArray2D, pybind11::arg("stream_or_queue"));
        pyClass.def("to_image", &ReleasableDeviceArrayViewBGR_SRGB::toImage, pybind11::arg("stream_or_queue"));
    }
} // namespace ZividPython
