#pragma once

#include <Zivid/Matrix.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Matrix4x4> pyClass);
}