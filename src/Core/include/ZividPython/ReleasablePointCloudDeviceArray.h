#pragma once

#include <Zivid/DeviceArray.h>

#include "ZividPython/Releasable.h"
#include "ZividPython/ReleasableArray1D.h"
#include "ZividPython/ReleasableArray2D.h"
#include "ZividPython/Wrappers.h"

namespace ZividPython
{
    class ReleasableDeviceArrayPointXYZ : public Releasable<Zivid::DeviceArray<Zivid::PointXYZ>>
    {
    public:
        using Releasable<Zivid::DeviceArray<Zivid::PointXYZ>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableDeviceArrayPointXYZ)

        ZIVID_PYTHON_FORWARD_0_ARGS(shape, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(strides, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(stridesInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(sizeInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(backend, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isValid, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isEmpty, const)

        ReleasableArray2D<Zivid::PointXYZ> toArray2D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray2D<Zivid::PointXYZ>{ WITH_GIL_UNLOCKED(impl().toArray2D(streamOrQueue)) };
        }

        ReleasableArray1D<Zivid::PointXYZ> toArray1D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray1D<Zivid::PointXYZ>{ WITH_GIL_UNLOCKED(impl().toArray1D(streamOrQueue)) };
        }

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    class ReleasableDeviceArrayPointXYZW : public Releasable<Zivid::DeviceArray<Zivid::PointXYZW>>
    {
    public:
        using Releasable<Zivid::DeviceArray<Zivid::PointXYZW>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableDeviceArrayPointXYZW)

        ZIVID_PYTHON_FORWARD_0_ARGS(shape, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(strides, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(stridesInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(sizeInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(backend, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isValid, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isEmpty, const)

        ReleasableArray2D<Zivid::PointXYZW> toArray2D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray2D<Zivid::PointXYZW>{ WITH_GIL_UNLOCKED(impl().toArray2D(streamOrQueue)) };
        }

        ReleasableArray1D<Zivid::PointXYZW> toArray1D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray1D<Zivid::PointXYZW>{ WITH_GIL_UNLOCKED(impl().toArray1D(streamOrQueue)) };
        }

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    class ReleasableDeviceArrayPointZ : public Releasable<Zivid::DeviceArray<Zivid::PointZ>>
    {
    public:
        using Releasable<Zivid::DeviceArray<Zivid::PointZ>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableDeviceArrayPointZ)

        ZIVID_PYTHON_FORWARD_0_ARGS(shape, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(strides, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(stridesInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(sizeInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(backend, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isValid, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isEmpty, const)

        ReleasableArray2D<Zivid::PointZ> toArray2D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray2D<Zivid::PointZ>{ WITH_GIL_UNLOCKED(impl().toArray2D(streamOrQueue)) };
        }

        ReleasableArray1D<Zivid::PointZ> toArray1D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray1D<Zivid::PointZ>{ WITH_GIL_UNLOCKED(impl().toArray1D(streamOrQueue)) };
        }

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    class ReleasableDeviceArraySNR : public Releasable<Zivid::DeviceArray<Zivid::SNR>>
    {
    public:
        using Releasable<Zivid::DeviceArray<Zivid::SNR>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableDeviceArraySNR)

        ZIVID_PYTHON_FORWARD_0_ARGS(shape, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(strides, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(stridesInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(sizeInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(backend, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isValid, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isEmpty, const)

        ReleasableArray2D<Zivid::SNR> toArray2D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray2D<Zivid::SNR>{ WITH_GIL_UNLOCKED(impl().toArray2D(streamOrQueue)) };
        }

        ReleasableArray1D<Zivid::SNR> toArray1D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray1D<Zivid::SNR>{ WITH_GIL_UNLOCKED(impl().toArray1D(streamOrQueue)) };
        }

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    class ReleasableDeviceArrayNormalXYZ : public Releasable<Zivid::DeviceArray<Zivid::NormalXYZ>>
    {
    public:
        using Releasable<Zivid::DeviceArray<Zivid::NormalXYZ>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableDeviceArrayNormalXYZ)

        ZIVID_PYTHON_FORWARD_0_ARGS(shape, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(strides, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(stridesInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(sizeInBytes, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(backend, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isValid, const)
        ZIVID_PYTHON_FORWARD_0_ARGS(isEmpty, const)

        ReleasableArray2D<Zivid::NormalXYZ> toArray2D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray2D<Zivid::NormalXYZ>{ WITH_GIL_UNLOCKED(impl().toArray2D(streamOrQueue)) };
        }

        ReleasableArray1D<Zivid::NormalXYZ> toArray1D(Zivid::StreamOrQueue streamOrQueue) const
        {
            return ReleasableArray1D<Zivid::NormalXYZ>{ WITH_GIL_UNLOCKED(impl().toArray1D(streamOrQueue)) };
        }

        void *devicePointer() const
        {
            return WITH_GIL_UNLOCKED(impl().devicePointer());
        }
    };

    // Function declarations for wrapping
    void wrapClass(pybind11::class_<ReleasableDeviceArrayPointXYZ> pyClass);
    void wrapClass(pybind11::class_<ReleasableDeviceArrayPointXYZW> pyClass);
    void wrapClass(pybind11::class_<ReleasableDeviceArrayPointZ> pyClass);
    void wrapClass(pybind11::class_<ReleasableDeviceArraySNR> pyClass);
    void wrapClass(pybind11::class_<ReleasableDeviceArrayNormalXYZ> pyClass);
} // namespace ZividPython
