#pragma once

#include <Zivid/Experimental/PixelMapping.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Experimental::PixelMapping> pyClass);
} // namespace ZividPython
