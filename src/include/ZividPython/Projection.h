#pragma once

#include <ZividPython/Wrappers.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace ZividPython::Projection
{
    void wrapAsSubmodule(pybind11::module &dest);
} // namespace ZividPython::Projection
