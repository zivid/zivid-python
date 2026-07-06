#pragma once

#include <Zivid/DeviceArray.h>

#include "ZividPython/Releasable.h"
#include "ZividPython/ReleasableArray1D.h"
#include "ZividPython/ReleasableArray2D.h"
#include "ZividPython/ReleasableImage.h"
#include "ZividPython/Wrappers.h"

namespace ZividPython
{
    // Releasable wrappers for each ImageDeviceArray type
    class ReleasableImageDeviceArrayRGBA : public Releasable<Zivid::DeviceArray<Zivid::ColorRGBA>>
    {
    public:
        using Releasable<Zivid::DeviceArray<Zivid::ColorRGBA>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableImageDeviceArrayRGBA)

        ZIVID_PYTHON_FORWARD_0_ARGS(shape, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(strides, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(stridesInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(sizeInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(backend, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isValid, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isEmpty, const)

        ReleasableImageRGBA toImage(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableImageRGBA{ WITH_GIL_UNLOCKED(Zivid::toImage(impl(), streamOrQueue)) };
        }

        ReleasableArray2D<Zivid::ColorRGBA> toArray2D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray2D<Zivid::ColorRGBA>{ WITH_GIL_UNLOCKED(impl().toArray2D(streamOrQueue)) };
        }

        ReleasableArray1D<Zivid::ColorRGBA> toArray1D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray1D<Zivid::ColorRGBA>{ WITH_GIL_UNLOCKED(impl().toArray1D(streamOrQueue)) };
        }

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    class ReleasableImageDeviceArrayBGRA : public Releasable<Zivid::DeviceArray<Zivid::ColorBGRA>>
    {
    public:
        using Releasable<Zivid::DeviceArray<Zivid::ColorBGRA>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableImageDeviceArrayBGRA)

        ZIVID_PYTHON_FORWARD_0_ARGS(shape, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(strides, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(stridesInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(sizeInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(backend, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isValid, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isEmpty, const)

        ReleasableImageBGRA toImage(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableImageBGRA{ WITH_GIL_UNLOCKED(Zivid::toImage(impl(), streamOrQueue)) };
        }

        ReleasableArray2D<Zivid::ColorBGRA> toArray2D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray2D<Zivid::ColorBGRA>{ WITH_GIL_UNLOCKED(impl().toArray2D(streamOrQueue)) };
        }

        ReleasableArray1D<Zivid::ColorBGRA> toArray1D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray1D<Zivid::ColorBGRA>{ WITH_GIL_UNLOCKED(impl().toArray1D(streamOrQueue)) };
        }

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    class ReleasableImageDeviceArrayRGBA_SRGB : public Releasable<Zivid::DeviceArray<Zivid::ColorRGBA_SRGB>>
    {
    public:
        using Releasable<Zivid::DeviceArray<Zivid::ColorRGBA_SRGB>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableImageDeviceArrayRGBA_SRGB)

        ZIVID_PYTHON_FORWARD_0_ARGS(shape, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(strides, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(stridesInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(sizeInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(backend, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isValid, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isEmpty, const)

        ReleasableImageRGBA_SRGB toImage(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableImageRGBA_SRGB{ WITH_GIL_UNLOCKED(Zivid::toImage(impl(), streamOrQueue)) };
        }

        ReleasableArray2D<Zivid::ColorRGBA_SRGB> toArray2D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray2D<Zivid::ColorRGBA_SRGB>{ WITH_GIL_UNLOCKED(impl().toArray2D(streamOrQueue)) };
        }

        ReleasableArray1D<Zivid::ColorRGBA_SRGB> toArray1D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray1D<Zivid::ColorRGBA_SRGB>{ WITH_GIL_UNLOCKED(impl().toArray1D(streamOrQueue)) };
        }

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    class ReleasableImageDeviceArrayBGRA_SRGB : public Releasable<Zivid::DeviceArray<Zivid::ColorBGRA_SRGB>>
    {
    public:
        using Releasable<Zivid::DeviceArray<Zivid::ColorBGRA_SRGB>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableImageDeviceArrayBGRA_SRGB)

        ZIVID_PYTHON_FORWARD_0_ARGS(shape, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(strides, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(stridesInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(sizeInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(backend, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isValid, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isEmpty, const)

        ReleasableImageBGRA_SRGB toImage(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableImageBGRA_SRGB{ WITH_GIL_UNLOCKED(Zivid::toImage(impl(), streamOrQueue)) };
        }

        ReleasableArray2D<Zivid::ColorBGRA_SRGB> toArray2D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray2D<Zivid::ColorBGRA_SRGB>{ WITH_GIL_UNLOCKED(impl().toArray2D(streamOrQueue)) };
        }

        ReleasableArray1D<Zivid::ColorBGRA_SRGB> toArray1D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray1D<Zivid::ColorBGRA_SRGB>{ WITH_GIL_UNLOCKED(impl().toArray1D(streamOrQueue)) };
        }

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    // DeviceArray<ColorRGBAf> (float format)
    class ReleasableImageDeviceArrayRGBAf : public Releasable<Zivid::DeviceArray<Zivid::ColorRGBAf>>
    {
    public:
        using Releasable<Zivid::DeviceArray<Zivid::ColorRGBAf>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableImageDeviceArrayRGBAf)

        ZIVID_PYTHON_FORWARD_0_ARGS(shape, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(strides, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(stridesInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(sizeInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(backend, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isValid, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isEmpty, const)

        ReleasableArray2D<Zivid::ColorRGBAf> toArray2D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray2D<Zivid::ColorRGBAf>{ WITH_GIL_UNLOCKED(impl().toArray2D(streamOrQueue)) };
        }

        ReleasableArray1D<Zivid::ColorRGBAf> toArray1D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray1D<Zivid::ColorRGBAf>{ WITH_GIL_UNLOCKED(impl().toArray1D(streamOrQueue)) };
        }

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    // DeviceArray<ColorRGB> (3-channel linear u8, no alpha)
    class ReleasableImageDeviceArrayRGB : public Releasable<Zivid::DeviceArray<Zivid::ColorRGB>>
    {
    public:
        using Releasable<Zivid::DeviceArray<Zivid::ColorRGB>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableImageDeviceArrayRGB)

        ZIVID_PYTHON_FORWARD_0_ARGS(shape, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(strides, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(stridesInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(sizeInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(backend, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isValid, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isEmpty, const)

        ReleasableImageRGB toImage(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableImageRGB{ WITH_GIL_UNLOCKED(Zivid::toImage(impl(), streamOrQueue)) };
        }

        ReleasableArray2D<Zivid::ColorRGB> toArray2D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray2D<Zivid::ColorRGB>{ WITH_GIL_UNLOCKED(impl().toArray2D(streamOrQueue)) };
        }

        ReleasableArray1D<Zivid::ColorRGB> toArray1D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray1D<Zivid::ColorRGB>{ WITH_GIL_UNLOCKED(impl().toArray1D(streamOrQueue)) };
        }

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    // DeviceArray<ColorRGB_SRGB> (3-channel sRGB u8, no alpha)
    class ReleasableImageDeviceArrayRGB_SRGB : public Releasable<Zivid::DeviceArray<Zivid::ColorRGB_SRGB>>
    {
    public:
        using Releasable<Zivid::DeviceArray<Zivid::ColorRGB_SRGB>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableImageDeviceArrayRGB_SRGB)

        ZIVID_PYTHON_FORWARD_0_ARGS(shape, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(strides, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(stridesInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(sizeInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(backend, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isValid, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isEmpty, const)

        ReleasableImageRGB_SRGB toImage(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableImageRGB_SRGB{ WITH_GIL_UNLOCKED(Zivid::toImage(impl(), streamOrQueue)) };
        }

        ReleasableArray2D<Zivid::ColorRGB_SRGB> toArray2D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray2D<Zivid::ColorRGB_SRGB>{ WITH_GIL_UNLOCKED(impl().toArray2D(streamOrQueue)) };
        }

        ReleasableArray1D<Zivid::ColorRGB_SRGB> toArray1D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray1D<Zivid::ColorRGB_SRGB>{ WITH_GIL_UNLOCKED(impl().toArray1D(streamOrQueue)) };
        }

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    // DeviceArray<ColorBGR> (3-channel linear u8, no alpha, channel-swapped)
    class ReleasableImageDeviceArrayBGR : public Releasable<Zivid::DeviceArray<Zivid::ColorBGR>>
    {
    public:
        using Releasable<Zivid::DeviceArray<Zivid::ColorBGR>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableImageDeviceArrayBGR)

        ZIVID_PYTHON_FORWARD_0_ARGS(shape, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(strides, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(stridesInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(sizeInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(backend, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isValid, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isEmpty, const)

        ReleasableImageBGR toImage(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableImageBGR{ WITH_GIL_UNLOCKED(Zivid::toImage(impl(), streamOrQueue)) };
        }

        ReleasableArray2D<Zivid::ColorBGR> toArray2D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray2D<Zivid::ColorBGR>{ WITH_GIL_UNLOCKED(impl().toArray2D(streamOrQueue)) };
        }

        ReleasableArray1D<Zivid::ColorBGR> toArray1D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray1D<Zivid::ColorBGR>{ WITH_GIL_UNLOCKED(impl().toArray1D(streamOrQueue)) };
        }

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    // DeviceArray<ColorBGR_SRGB> (3-channel sRGB u8, no alpha, channel-swapped)
    class ReleasableImageDeviceArrayBGR_SRGB : public Releasable<Zivid::DeviceArray<Zivid::ColorBGR_SRGB>>
    {
    public:
        using Releasable<Zivid::DeviceArray<Zivid::ColorBGR_SRGB>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableImageDeviceArrayBGR_SRGB)

        ZIVID_PYTHON_FORWARD_0_ARGS(shape, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(strides, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(stridesInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(sizeInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(backend, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isValid, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isEmpty, const)

        ReleasableImageBGR_SRGB toImage(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableImageBGR_SRGB{ WITH_GIL_UNLOCKED(Zivid::toImage(impl(), streamOrQueue)) };
        }

        ReleasableArray2D<Zivid::ColorBGR_SRGB> toArray2D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray2D<Zivid::ColorBGR_SRGB>{ WITH_GIL_UNLOCKED(impl().toArray2D(streamOrQueue)) };
        }

        ReleasableArray1D<Zivid::ColorBGR_SRGB> toArray1D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray1D<Zivid::ColorBGR_SRGB>{ WITH_GIL_UNLOCKED(impl().toArray1D(streamOrQueue)) };
        }

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    // Function declarations for wrapping
    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayRGBA> pyClass);
    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayBGRA> pyClass);
    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayRGBA_SRGB> pyClass);
    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayBGRA_SRGB> pyClass);
    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayRGBAf> pyClass);
    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayRGB> pyClass);
    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayRGB_SRGB> pyClass);
    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayBGR> pyClass);
    void wrapClass(pybind11::class_<ReleasableImageDeviceArrayBGR_SRGB> pyClass);
} // namespace ZividPython
