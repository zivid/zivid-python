#pragma once

#include <Zivid/Image.h>
#include <ZividPython/Releasable.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class ReleasableImageRGBA : public Releasable<Zivid::Image<Zivid::ColorRGBA>>
    {
    public:
        using Releasable<Zivid::Image<Zivid::ColorRGBA>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableImageRGBA)

        ZIVID_PYTHON_FORWARD_1_ARGS(save, const std::string &, fileName)
        ZIVID_PYTHON_FORWARD_0_ARGS(width)
        ZIVID_PYTHON_FORWARD_0_ARGS(height)
    };

    class ReleasableImageBGRA : public Releasable<Zivid::Image<Zivid::ColorBGRA>>
    {
    public:
        using Releasable<Zivid::Image<Zivid::ColorBGRA>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableImageBGRA)

        ZIVID_PYTHON_FORWARD_1_ARGS(save, const std::string &, fileName)
        ZIVID_PYTHON_FORWARD_0_ARGS(width)
        ZIVID_PYTHON_FORWARD_0_ARGS(height)
    };

    class ReleasableImageRGBA_SRGB : public Releasable<Zivid::Image<Zivid::ColorRGBA_SRGB>>
    {
    public:
        using Releasable<Zivid::Image<Zivid::ColorSRGB>>::Releasable;

        ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(ReleasableImageRGBA_SRGB)

        ZIVID_PYTHON_FORWARD_1_ARGS(save, const std::string &, fileName)
        ZIVID_PYTHON_FORWARD_0_ARGS(width)
        ZIVID_PYTHON_FORWARD_0_ARGS(height)
    };

    class ReleasableImageBGRA_SRGB : public Releasable<Zivid::Image<Zivid::ColorBGRA_SRGB>>
    {
    public:
        using Releasable<Zivid::Image<Zivid::ColorBGRA_SRGB>>::Releasable;

        ZIVID_PYTHON_FORWARD_1_ARGS(save, const std::string &, fileName)
        ZIVID_PYTHON_FORWARD_0_ARGS(width)
        ZIVID_PYTHON_FORWARD_0_ARGS(height)
    };

    void wrapClass(pybind11::class_<ReleasableImageRGBA> pyClass);
    void wrapClass(pybind11::class_<ReleasableImageBGRA> pyClass);
    void wrapClass(pybind11::class_<ReleasableImageRGBA_SRGB> pyClass);
    void wrapClass(pybind11::class_<ReleasableImageBGRA_SRGB> pyClass);
} // namespace ZividPython
