#include <ZividPython/ReleasableImageDeviceArray.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    namespace
    {
        // Helper template to add common ImageDeviceArray methods to a pybind11 class
        template<typename ReleasableType>
        void addCommonImageDeviceArrayMethods(pybind11::class_<ReleasableType> &pyClass)
        {
            pyClass.def_property_readonly("shape", &ReleasableType::shape)
                .def_property_readonly("strides", &ReleasableType::strides)
                .def_property_readonly("strides_in_bytes", &ReleasableType::stridesInBytes)
                .def_property_readonly("size_bytes", &ReleasableType::sizeInBytes)
                .def_property_readonly("backend", &ReleasableType::backend)
                .def_property_readonly("is_valid", &ReleasableType::isValid)
                .def_property_readonly("is_empty", &ReleasableType::isEmpty)
                .def("to_array_2d", &ReleasableType::toArray2D, pybind11::arg("stream_or_queue"))
                .def("to_array_1d", &ReleasableType::toArray1D, pybind11::arg("stream_or_queue"))
                .def("device_pointer", [](const ReleasableType &self) {
                    return reinterpret_cast<std::uintptr_t>(self.devicePointer());
                });
        }
    } // namespace

    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayRGBA> pyClass)
    {
        addCommonImageDeviceArrayMethods(pyClass);
        pyClass.def("to_image", &ReleasableImageDeviceArrayRGBA::toImage, pybind11::arg("stream_or_queue"));
    }

    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayBGRA> pyClass)
    {
        addCommonImageDeviceArrayMethods(pyClass);
        pyClass.def("to_image", &ReleasableImageDeviceArrayBGRA::toImage, pybind11::arg("stream_or_queue"));
    }

    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayRGBA_SRGB> pyClass)
    {
        addCommonImageDeviceArrayMethods(pyClass);
        pyClass.def("to_image", &ReleasableImageDeviceArrayRGBA_SRGB::toImage, pybind11::arg("stream_or_queue"));
    }

    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayBGRA_SRGB> pyClass)
    {
        addCommonImageDeviceArrayMethods(pyClass);
        pyClass.def("to_image", &ReleasableImageDeviceArrayBGRA_SRGB::toImage, pybind11::arg("stream_or_queue"));
    }

    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayRGBAf> pyClass)
    {
        addCommonImageDeviceArrayMethods(pyClass);
        // Note: Float format buffer does not have toImage() - use device_pointer() for GPU interop
    }

    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayRGB> pyClass)
    {
        addCommonImageDeviceArrayMethods(pyClass);
        pyClass.def("to_image", &ReleasableImageDeviceArrayRGB::toImage, pybind11::arg("stream_or_queue"));
    }

    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayRGB_SRGB> pyClass)
    {
        addCommonImageDeviceArrayMethods(pyClass);
        pyClass.def("to_image", &ReleasableImageDeviceArrayRGB_SRGB::toImage, pybind11::arg("stream_or_queue"));
    }

    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayBGR> pyClass)
    {
        addCommonImageDeviceArrayMethods(pyClass);
        pyClass.def("to_image", &ReleasableImageDeviceArrayBGR::toImage, pybind11::arg("stream_or_queue"));
    }

    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayBGR_SRGB> pyClass)
    {
        addCommonImageDeviceArrayMethods(pyClass);
        pyClass.def("to_image", &ReleasableImageDeviceArrayBGR_SRGB::toImage, pybind11::arg("stream_or_queue"));
    }
} // namespace ZividPython
