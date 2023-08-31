#pragma once

#include <Zivid/Presets.h>
#include <ZividPython/Wrappers.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Presets::Preset> pyClass);

    void wrapClass(pybind11::class_<Zivid::Presets::Category> pyClass);

    namespace Presets
    {
        void wrapAsSubmodule(pybind11::module &dest);
    } // namespace Presets
} // namespace ZividPython
