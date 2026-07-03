#pragma once

#include <Zivid/DeviceArray.h>

#include "ZividPython/Releasable.h"
#include "ZividPython/ReleasableArray1D.h"
#include "ZividPython/ReleasableArray2D.h"
#include "ZividPython/ReleasableImage.h"
#include "ZividPython/Wrappers.h"

namespace ZividPython
{
    // Releasable wrappers for non-owning DeviceArrayView<Format> over caller-owned GPU memory.
    // Created via ComputeDevice.create_device_array_view_*, mirroring the owning ImageDeviceArray wrappers.
    class ReleasableDeviceArrayViewRGBA : public Releasable<Zivid::DeviceArrayView<Zivid::ColorRGBA>>
    {
    public:
        using Releasable<Zivid::DeviceArrayView<Zivid::ColorRGBA>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableDeviceArrayViewRGBA)

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

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    class ReleasableDeviceArrayViewBGRA : public Releasable<Zivid::DeviceArrayView<Zivid::ColorBGRA>>
    {
    public:
        using Releasable<Zivid::DeviceArrayView<Zivid::ColorBGRA>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableDeviceArrayViewBGRA)

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

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    class ReleasableDeviceArrayViewRGBA_SRGB : public Releasable<Zivid::DeviceArrayView<Zivid::ColorRGBA_SRGB>>
    {
    public:
        using Releasable<Zivid::DeviceArrayView<Zivid::ColorRGBA_SRGB>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableDeviceArrayViewRGBA_SRGB)

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

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    class ReleasableDeviceArrayViewBGRA_SRGB : public Releasable<Zivid::DeviceArrayView<Zivid::ColorBGRA_SRGB>>
    {
    public:
        using Releasable<Zivid::DeviceArrayView<Zivid::ColorBGRA_SRGB>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableDeviceArrayViewBGRA_SRGB)

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

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    class ReleasableDeviceArrayViewRGBAf : public Releasable<Zivid::DeviceArrayView<Zivid::ColorRGBAf>>
    {
    public:
        using Releasable<Zivid::DeviceArrayView<Zivid::ColorRGBAf>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableDeviceArrayViewRGBAf)

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

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    class ReleasableDeviceArrayViewRGB : public Releasable<Zivid::DeviceArrayView<Zivid::ColorRGB>>
    {
    public:
        using Releasable<Zivid::DeviceArrayView<Zivid::ColorRGB>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableDeviceArrayViewRGB)

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

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    class ReleasableDeviceArrayViewRGB_SRGB : public Releasable<Zivid::DeviceArrayView<Zivid::ColorRGB_SRGB>>
    {
    public:
        using Releasable<Zivid::DeviceArrayView<Zivid::ColorRGB_SRGB>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableDeviceArrayViewRGB_SRGB)

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

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    class ReleasableDeviceArrayViewBGR : public Releasable<Zivid::DeviceArrayView<Zivid::ColorBGR>>
    {
    public:
        using Releasable<Zivid::DeviceArrayView<Zivid::ColorBGR>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableDeviceArrayViewBGR)

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

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    class ReleasableDeviceArrayViewBGR_SRGB : public Releasable<Zivid::DeviceArrayView<Zivid::ColorBGR_SRGB>>
    {
    public:
        using Releasable<Zivid::DeviceArrayView<Zivid::ColorBGR_SRGB>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableDeviceArrayViewBGR_SRGB)

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

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewRGBA> pyClass);
    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewBGRA> pyClass);
    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewRGBA_SRGB> pyClass);
    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewBGRA_SRGB> pyClass);
    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewRGBAf> pyClass);
    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewRGB> pyClass);
    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewRGB_SRGB> pyClass);
    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewBGR> pyClass);
    void wrapClass(pybind11::class_<ReleasableDeviceArrayViewBGR_SRGB> pyClass);
} // namespace ZividPython
