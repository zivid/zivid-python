#pragma once

#include <Zivid/Resolution.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Resolution> pyClass);
}
