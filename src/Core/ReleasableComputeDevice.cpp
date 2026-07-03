#include <ZividPython/ReleasableComputeDevice.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapEnum(pybind11::enum_<Zivid::ComputeBackend> pyEnum)
    {
        pyEnum.value("cuda", Zivid::ComputeBackend::cuda).value("opencl", Zivid::ComputeBackend::opencl);
    }

    void wrapClass(pybind11::class_<Zivid::CUDAContextPtr> pyClass)
    {
        pyClass.def(py::init<>())
            .def(
                py::init(
                    [](std::uintptr_t context) { return Zivid::CUDAContextPtr{ reinterpret_cast<void *>(context) }; }),
                py::arg("context"))
            .def_property(
                "context",
                [](const Zivid::CUDAContextPtr &self) { return reinterpret_cast<std::uintptr_t>(self.context); },
                [](Zivid::CUDAContextPtr &self, std::uintptr_t value) {
                    self.context = reinterpret_cast<void *>(value);
                });
    }

    void wrapClass(pybind11::class_<Zivid::CUDAStreamPtr> pyClass)
    {
        pyClass.def(py::init<>())
            .def(
                py::init(
                    [](std::uintptr_t stream) { return Zivid::CUDAStreamPtr{ reinterpret_cast<void *>(stream) }; }),
                py::arg("stream"))
            .def_property(
                "stream",
                [](const Zivid::CUDAStreamPtr &self) { return reinterpret_cast<std::uintptr_t>(self.stream); },
                [](Zivid::CUDAStreamPtr &self, std::uintptr_t value) {
                    self.stream = reinterpret_cast<void *>(value);
                });
    }

    void wrapClass(pybind11::class_<Zivid::OpenCLCommandQueuePtr> pyClass)
    {
        pyClass.def(py::init<>())
            .def(
                py::init([](std::uintptr_t commandQueue) {
                    return Zivid::OpenCLCommandQueuePtr{ reinterpret_cast<void *>(commandQueue) };
                }),
                py::arg("command_queue"))
            .def_property(
                "command_queue",
                [](const Zivid::OpenCLCommandQueuePtr &self) {
                    return reinterpret_cast<std::uintptr_t>(self.commandQueue);
                },
                [](Zivid::OpenCLCommandQueuePtr &self, std::uintptr_t value) {
                    self.commandQueue = reinterpret_cast<void *>(value);
                });
    }

    void wrapClass(pybind11::class_<Zivid::StreamOrQueue> pyClass)
    {
        pyClass.def(py::init<Zivid::CUDAStreamPtr>(), py::arg("stream"))
            .def(py::init<Zivid::OpenCLCommandQueuePtr>(), py::arg("command_queue"))
            .def_readwrite("stream", &Zivid::StreamOrQueue::stream)
            .def_readwrite("command_queue", &Zivid::StreamOrQueue::commandQueue);
        py::implicitly_convertible<Zivid::CUDAStreamPtr, Zivid::StreamOrQueue>();
        py::implicitly_convertible<Zivid::OpenCLCommandQueuePtr, Zivid::StreamOrQueue>();
    }

    void wrapClass(pybind11::class_<ReleasableComputeDevice> pyClass)
    {
        pyClass.def_property_readonly("model", &ReleasableComputeDevice::model)
            .def_property_readonly("vendor", &ReleasableComputeDevice::vendor)
            .def_property_readonly("backend", &ReleasableComputeDevice::backend)
            .def(
                "native_context",
                [](const ReleasableComputeDevice &self) {
                    return reinterpret_cast<std::uintptr_t>(self.nativeContext());
                })
            .def(
                "native_stream_handle",
                [](const ReleasableComputeDevice &self) {
                    return reinterpret_cast<std::uintptr_t>(self.nativeStreamHandle());
                })
            .def("sdk_stream_or_queue", &ReleasableComputeDevice::sdkStreamOrQueue)
            .def(
                "create_device_array_view_rgba",
                &ReleasableComputeDevice::createDeviceArrayViewRGBA,
                py::arg("cuda_pointer"),
                py::arg("width"),
                py::arg("height"),
                py::arg("stream_or_queue"))
            .def(
                "create_device_array_view_bgra",
                &ReleasableComputeDevice::createDeviceArrayViewBGRA,
                py::arg("cuda_pointer"),
                py::arg("width"),
                py::arg("height"),
                py::arg("stream_or_queue"))
            .def(
                "create_device_array_view_rgba_srgb",
                &ReleasableComputeDevice::createDeviceArrayViewRGBA_SRGB,
                py::arg("cuda_pointer"),
                py::arg("width"),
                py::arg("height"),
                py::arg("stream_or_queue"))
            .def(
                "create_device_array_view_bgra_srgb",
                &ReleasableComputeDevice::createDeviceArrayViewBGRA_SRGB,
                py::arg("cuda_pointer"),
                py::arg("width"),
                py::arg("height"),
                py::arg("stream_or_queue"))
            .def(
                "create_device_array_view_rgbaf",
                &ReleasableComputeDevice::createDeviceArrayViewRGBAf,
                py::arg("cuda_pointer"),
                py::arg("width"),
                py::arg("height"),
                py::arg("stream_or_queue"))
            .def(
                "create_device_array_view_rgb",
                &ReleasableComputeDevice::createDeviceArrayViewRGB,
                py::arg("cuda_pointer"),
                py::arg("width"),
                py::arg("height"),
                py::arg("stream_or_queue"))
            .def(
                "create_device_array_view_rgb_srgb",
                &ReleasableComputeDevice::createDeviceArrayViewRGB_SRGB,
                py::arg("cuda_pointer"),
                py::arg("width"),
                py::arg("height"),
                py::arg("stream_or_queue"))
            .def(
                "create_device_array_view_bgr",
                &ReleasableComputeDevice::createDeviceArrayViewBGR,
                py::arg("cuda_pointer"),
                py::arg("width"),
                py::arg("height"),
                py::arg("stream_or_queue"))
            .def(
                "create_device_array_view_bgr_srgb",
                &ReleasableComputeDevice::createDeviceArrayViewBGR_SRGB,
                py::arg("cuda_pointer"),
                py::arg("width"),
                py::arg("height"),
                py::arg("stream_or_queue"))
            .def_property_readonly("cuda_runtime_library_name", &ReleasableComputeDevice::cudaRuntimeLibraryName)
            .def("__str__", &ReleasableComputeDevice::toString)
            .def("__repr__", &ReleasableComputeDevice::toString);
    }
} // namespace ZividPython
