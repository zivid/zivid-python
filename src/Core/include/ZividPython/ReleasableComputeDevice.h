#pragma once

#include <Zivid/ComputeDevice.h>

#include <ZividPython/Releasable.h>
#include <ZividPython/ReleasableDeviceArrayView.h>
#include <ZividPython/Wrappers.h>

#include <cstddef>
#include <cstdint>

namespace ZividPython
{
    class ReleasableComputeDevice : public Releasable<Zivid::ComputeDevice>
    {
    public:
        using Releasable<Zivid::ComputeDevice>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableComputeDevice)

        ZIVID_PYTHON_FORWARD_0_ARGS(model, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(vendor, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(backend, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(nativeContext, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(nativeStreamHandle, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(sdkStreamOrQueue, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(cudaRuntimeLibraryName, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(toString, const)

        ReleasableDeviceArrayViewRGBA createDeviceArrayViewRGBA(
            std::uintptr_t cudaPointer,
            size_t width,
            size_t height,
            const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArrayViewRGBA{ WITH_GIL_UNLOCKED(
                impl().createDeviceArrayView<Zivid::ColorRGBA>(
                    Zivid::CUDADevicePointer{ reinterpret_cast<void *>(cudaPointer) },
                    width,
                    height,
                    streamOrQueue.stream)) };
        }

        ReleasableDeviceArrayViewBGRA createDeviceArrayViewBGRA(
            std::uintptr_t cudaPointer,
            size_t width,
            size_t height,
            const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArrayViewBGRA{ WITH_GIL_UNLOCKED(
                impl().createDeviceArrayView<Zivid::ColorBGRA>(
                    Zivid::CUDADevicePointer{ reinterpret_cast<void *>(cudaPointer) },
                    width,
                    height,
                    streamOrQueue.stream)) };
        }

        ReleasableDeviceArrayViewRGBA_SRGB createDeviceArrayViewRGBA_SRGB(
            std::uintptr_t cudaPointer,
            size_t width,
            size_t height,
            const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArrayViewRGBA_SRGB{ WITH_GIL_UNLOCKED(
                impl().createDeviceArrayView<Zivid::ColorRGBA_SRGB>(
                    Zivid::CUDADevicePointer{ reinterpret_cast<void *>(cudaPointer) },
                    width,
                    height,
                    streamOrQueue.stream)) };
        }

        ReleasableDeviceArrayViewBGRA_SRGB createDeviceArrayViewBGRA_SRGB(
            std::uintptr_t cudaPointer,
            size_t width,
            size_t height,
            const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArrayViewBGRA_SRGB{ WITH_GIL_UNLOCKED(
                impl().createDeviceArrayView<Zivid::ColorBGRA_SRGB>(
                    Zivid::CUDADevicePointer{ reinterpret_cast<void *>(cudaPointer) },
                    width,
                    height,
                    streamOrQueue.stream)) };
        }

        ReleasableDeviceArrayViewRGBAf createDeviceArrayViewRGBAf(
            std::uintptr_t cudaPointer,
            size_t width,
            size_t height,
            const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArrayViewRGBAf{ WITH_GIL_UNLOCKED(
                impl().createDeviceArrayView<Zivid::ColorRGBAf>(
                    Zivid::CUDADevicePointer{ reinterpret_cast<void *>(cudaPointer) },
                    width,
                    height,
                    streamOrQueue.stream)) };
        }

        ReleasableDeviceArrayViewRGB createDeviceArrayViewRGB(
            std::uintptr_t cudaPointer,
            size_t width,
            size_t height,
            const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArrayViewRGB{ WITH_GIL_UNLOCKED(
                impl().createDeviceArrayView<Zivid::ColorRGB>(
                    Zivid::CUDADevicePointer{ reinterpret_cast<void *>(cudaPointer) },
                    width,
                    height,
                    streamOrQueue.stream)) };
        }

        ReleasableDeviceArrayViewRGB_SRGB createDeviceArrayViewRGB_SRGB(
            std::uintptr_t cudaPointer,
            size_t width,
            size_t height,
            const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArrayViewRGB_SRGB{ WITH_GIL_UNLOCKED(
                impl().createDeviceArrayView<Zivid::ColorRGB_SRGB>(
                    Zivid::CUDADevicePointer{ reinterpret_cast<void *>(cudaPointer) },
                    width,
                    height,
                    streamOrQueue.stream)) };
        }

        ReleasableDeviceArrayViewBGR createDeviceArrayViewBGR(
            std::uintptr_t cudaPointer,
            size_t width,
            size_t height,
            const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArrayViewBGR{ WITH_GIL_UNLOCKED(
                impl().createDeviceArrayView<Zivid::ColorBGR>(
                    Zivid::CUDADevicePointer{ reinterpret_cast<void *>(cudaPointer) },
                    width,
                    height,
                    streamOrQueue.stream)) };
        }

        ReleasableDeviceArrayViewBGR_SRGB createDeviceArrayViewBGR_SRGB(
            std::uintptr_t cudaPointer,
            size_t width,
            size_t height,
            const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableDeviceArrayViewBGR_SRGB{ WITH_GIL_UNLOCKED(
                impl().createDeviceArrayView<Zivid::ColorBGR_SRGB>(
                    Zivid::CUDADevicePointer{ reinterpret_cast<void *>(cudaPointer) },
                    width,
                    height,
                    streamOrQueue.stream)) };
        }
    };

    void wrapEnum(pybind11::enum_<Zivid::ComputeBackend> pyEnum);
    void wrapClass(pybind11::class_<Zivid::CUDAContextPtr> pyClass);
    void wrapClass(pybind11::class_<Zivid::CUDAStreamPtr> pyClass);
    void wrapClass(pybind11::class_<Zivid::OpenCLCommandQueuePtr> pyClass);
    void wrapClass(pybind11::class_<Zivid::StreamOrQueue> pyClass);
    void wrapClass(pybind11::class_<ReleasableComputeDevice> pyClass);
} // namespace ZividPython
