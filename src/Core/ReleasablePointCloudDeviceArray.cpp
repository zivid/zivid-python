#include <ZividPython/ReleasablePointCloudDeviceArray.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    namespace
    {
        // Helper template to add common DeviceArray methods to a pybind11 class
        template<typename ReleasableType>
        void addCommonDeviceArrayMethods(pybind11::class_<ReleasableType> &pyClass)
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

    void wrapClass(pybind11::class_<ReleasableDeviceArrayPointXYZ> pyClass)
    {
        addCommonDeviceArrayMethods(pyClass);
    }

    void wrapClass(pybind11::class_<ReleasableDeviceArrayPointXYZW> pyClass)
    {
        addCommonDeviceArrayMethods(pyClass);
    }

    void wrapClass(pybind11::class_<ReleasableDeviceArrayPointZ> pyClass)
    {
        addCommonDeviceArrayMethods(pyClass);
    }

    void wrapClass(pybind11::class_<ReleasableDeviceArraySNR> pyClass)
    {
        addCommonDeviceArrayMethods(pyClass);
    }

    void wrapClass(pybind11::class_<ReleasableDeviceArrayNormalXYZ> pyClass)
    {
        addCommonDeviceArrayMethods(pyClass);
    }
} // namespace ZividPython
