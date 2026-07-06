#pragma once

#include <Zivid/Frame2D.h>
#include <Zivid/Settings2D.h>

#include <ZividPython/Releasable.h>
#include <ZividPython/ReleasableDeviceArrayView.h>
#include <ZividPython/ReleasableImage.h>
#include <ZividPython/ReleasableImageDeviceArray.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class ReleasableFrame2D : public Releasable<Zivid::Frame2D>
    {
    public:
        using Releasable<Zivid::Frame2D>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableFrame2D)

        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageRGBA, imageRGBA)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageBGRA, imageBGRA)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageRGBA_SRGB, imageRGBA_SRGB)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageBGRA_SRGB, imageBGRA_SRGB)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageRGB, imageRGB)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageRGB_SRGB, imageRGB_SRGB)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageBGR, imageBGR)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableImageBGR_SRGB, imageBGR_SRGB)
        ZIVID_PYTHON_FORWARD_0_ARGS(settings)
        ZIVID_PYTHON_FORWARD_0_ARGS(state)
        ZIVID_PYTHON_FORWARD_0_ARGS(info)
        ZIVID_PYTHON_FORWARD_0_ARGS(cameraInfo)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableFrame2D, clone, const)
        ZIVID_PYTHON_FORWARD_1_ARGS(save, const std::string &, fileName)
        ZIVID_PYTHON_FORWARD_1_ARGS(load, const std::string &, fileName)

        ReleasableImageDeviceArrayRGBA imageDeviceArrayRGBA(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayRGBA{ WITH_GIL_UNLOCKED(
                impl().imageDeviceArray<Zivid::ColorRGBA>(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayBGRA imageDeviceArrayBGRA(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayBGRA{ WITH_GIL_UNLOCKED(
                impl().imageDeviceArray<Zivid::ColorBGRA>(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayRGBA_SRGB imageDeviceArrayRGBA_SRGB(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayRGBA_SRGB{ WITH_GIL_UNLOCKED(
                impl().imageDeviceArray<Zivid::ColorRGBA_SRGB>(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayBGRA_SRGB imageDeviceArrayBGRA_SRGB(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayBGRA_SRGB{ WITH_GIL_UNLOCKED(
                impl().imageDeviceArray<Zivid::ColorBGRA_SRGB>(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayRGBAf imageDeviceArrayRGBAf(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayRGBAf{ WITH_GIL_UNLOCKED(
                impl().imageDeviceArray<Zivid::ColorRGBAf>(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayRGB imageDeviceArrayRGB(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayRGB{ WITH_GIL_UNLOCKED(
                impl().imageDeviceArray<Zivid::ColorRGB>(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayRGB_SRGB imageDeviceArrayRGB_SRGB(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayRGB_SRGB{ WITH_GIL_UNLOCKED(
                impl().imageDeviceArray<Zivid::ColorRGB_SRGB>(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayBGR imageDeviceArrayBGR(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayBGR{ WITH_GIL_UNLOCKED(
                impl().imageDeviceArray<Zivid::ColorBGR>(streamOrQueue)) };
        }

        ReleasableImageDeviceArrayBGR_SRGB imageDeviceArrayBGR_SRGB(const Zivid::StreamOrQueue &streamOrQueue) const
        {
            return ReleasableImageDeviceArrayBGR_SRGB{ WITH_GIL_UNLOCKED(
                impl().imageDeviceArray<Zivid::ColorBGR_SRGB>(streamOrQueue)) };
        }

        void imageDeviceArrayRGBAFill(
            const Zivid::StreamOrQueue &streamOrQueue,
            const ReleasableDeviceArrayViewRGBA &destinationBuffer) const
        {
            WITH_GIL_UNLOCKED(impl().imageDeviceArray<Zivid::ColorRGBA>(destinationBuffer.impl(), streamOrQueue));
        }

        void imageDeviceArrayBGRAFill(
            const Zivid::StreamOrQueue &streamOrQueue,
            const ReleasableDeviceArrayViewBGRA &destinationBuffer) const
        {
            WITH_GIL_UNLOCKED(impl().imageDeviceArray<Zivid::ColorBGRA>(destinationBuffer.impl(), streamOrQueue));
        }

        void imageDeviceArrayRGBA_SRGBFill(
            const Zivid::StreamOrQueue &streamOrQueue,
            const ReleasableDeviceArrayViewRGBA_SRGB &destinationBuffer) const
        {
            WITH_GIL_UNLOCKED(impl().imageDeviceArray<Zivid::ColorRGBA_SRGB>(destinationBuffer.impl(), streamOrQueue));
        }

        void imageDeviceArrayBGRA_SRGBFill(
            const Zivid::StreamOrQueue &streamOrQueue,
            const ReleasableDeviceArrayViewBGRA_SRGB &destinationBuffer) const
        {
            WITH_GIL_UNLOCKED(impl().imageDeviceArray<Zivid::ColorBGRA_SRGB>(destinationBuffer.impl(), streamOrQueue));
        }

        void imageDeviceArrayRGBAfFill(
            const Zivid::StreamOrQueue &streamOrQueue,
            const ReleasableDeviceArrayViewRGBAf &destinationBuffer) const
        {
            WITH_GIL_UNLOCKED(impl().imageDeviceArray<Zivid::ColorRGBAf>(destinationBuffer.impl(), streamOrQueue));
        }

        void imageDeviceArrayRGBFill(
            const Zivid::StreamOrQueue &streamOrQueue,
            const ReleasableDeviceArrayViewRGB &destinationBuffer) const
        {
            WITH_GIL_UNLOCKED(impl().imageDeviceArray<Zivid::ColorRGB>(destinationBuffer.impl(), streamOrQueue));
        }

        void imageDeviceArrayRGB_SRGBFill(
            const Zivid::StreamOrQueue &streamOrQueue,
            const ReleasableDeviceArrayViewRGB_SRGB &destinationBuffer) const
        {
            WITH_GIL_UNLOCKED(impl().imageDeviceArray<Zivid::ColorRGB_SRGB>(destinationBuffer.impl(), streamOrQueue));
        }

        void imageDeviceArrayBGRFill(
            const Zivid::StreamOrQueue &streamOrQueue,
            const ReleasableDeviceArrayViewBGR &destinationBuffer) const
        {
            WITH_GIL_UNLOCKED(impl().imageDeviceArray<Zivid::ColorBGR>(destinationBuffer.impl(), streamOrQueue));
        }

        void imageDeviceArrayBGR_SRGBFill(
            const Zivid::StreamOrQueue &streamOrQueue,
            const ReleasableDeviceArrayViewBGR_SRGB &destinationBuffer) const
        {
            WITH_GIL_UNLOCKED(impl().imageDeviceArray<Zivid::ColorBGR_SRGB>(destinationBuffer.impl(), streamOrQueue));
        }
    };

    void wrapClass(pybind11::class_<ReleasableFrame2D> pyClass);
} // namespace ZividPython
